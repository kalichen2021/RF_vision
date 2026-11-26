"""平面检测 / 四边形切分 / 数字融合与景深失焦。

本模块只做图像处理，不打印任何东西（输出由 CLI 负责）。
"""

from __future__ import annotations

from typing import Sequence

from configs.path import PathConfig
import cv2
import itertools
import sys
import numpy as np
import random
import math
from numpy.typing import NDArray
from .debug_tools import cv_show, read_image
from .logging_tools import get_logger

_logger = get_logger(__name__)
from .geometry import order_quad, QuadSegment
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _smooth_mask(gray: NDArray[np.uint8], win: int = 15,
                 var_thresh: float = 8.0, iter: int = 1) -> NDArray[np.uint8]:
  """基于局部方差的平滑区域掩膜

  在 win×win 窗口内计算灰度方差，方差小于 var_thresh 的像素视为平滑区域，
  再通过形态学开 / 闭运算去除噪点并填补空洞。

  Args:
      gray (NDArray): 单通道灰度图像
      win (int, optional): 计算局部方差的窗口大小（px）。 Defaults to 15.
      var_thresh (float, optional): 判定为平滑区域的方差阈值，
        方差越小越平滑。 Defaults to 8.0.

  Returns:
      NDArray[np.uint8]: 二值掩膜，平滑区域为 255，其余为 0
  """
  g = gray.astype(np.float32)
  mu = cv2.blur(g, (win, win))
  var = cv2.blur(g * g, (win, win)) - mu * mu
  mask = (var < var_thresh).astype(np.uint8) * 255
  k = np.ones((win, win), np.uint8)
  for _ in range(iter):
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.dilate(mask, k, iterations=1)
  return mask


def _get_maxquad(cnt: NDArray[np.int32]) -> NDArray[np.float64] | None:
  """获取轮廓的近似四边形

  策略（逐级兜底，保证大的平坦区域不会因为轮廓不够规整被整块丢掉）：

  1. 用递增 epsilon 的 `approxPolyDP`，取第一个 4 点凸多边形；
  2. 得不到 4 点时，把轮廓粗近似成 5~12 个顶点，枚举 4 点子集取面积最大的
     凸四边形（近似「最大内接四边形」）；
  3. 仍失败时退回 `minAreaRect` 的旋转外接矩形，交给 `is_planar_quad`
     继续做平面性筛选。

  Args:
      cnt (NDArray): 输入轮廓点，通常来自 cv2.findContours

  Returns:
      NDArray[np.float64] | None: 近似四边形顶点坐标，形状为 (4, 2)；
        轮廓退化时返回 None
  """
  peri = cv2.arcLength(cnt, True)
  if peri <= 0:
    return None

  for k in np.arange(0.01, 0.11, 0.005):
    approx = cv2.approxPolyDP(cnt, k * peri, True)
    if len(approx) == 4 and cv2.isContourConvex(approx):
      return approx.reshape(4, 2).astype(np.float64)

  # 兜底一：粗近似 + 4 点子集枚举，取最大凸四边形
  for k in np.arange(0.02, 0.16, 0.01):
    points = cv2.approxPolyDP(cnt, k * peri, True).reshape(-1, 2)
    if not 5 <= len(points) <= 12:
      continue
    best, best_area = None, 0.0
    for combo in itertools.combinations(range(len(points)), 4):
      cand = points[list(combo)].astype(np.float32)
      if not cv2.isContourConvex(cand):
        continue
      area = abs(cv2.contourArea(cand))
      if area > best_area:
        best_area, best = area, cand.astype(np.float64)
    if best is not None:
      return best

  # 兜底二：旋转外接矩形（可能带入少量非平坦像素，交给平面性校验过滤）
  if len(cnt) >= 3:
    return cv2.boxPoints(cv2.minAreaRect(cnt)).astype(np.float64)
  return None


def _get_threhold_edge(img: NDArray[np.uint8]) -> NDArray[np.uint8]:
  kernel = np.ones((3, 3), np.uint8)
  dilated = cv2.dilate(img, kernel, iterations=5)
  res = cv2.subtract(dilated, img)
  return res


def _is_acceptable(sub_quad: NDArray[np.floating],
                   distortion_limit: float) -> bool:
  """判断切出来的小四边形是否可接受（畸变是否在容忍范围内）"""
  # 计算四条边
  edges = [
      np.linalg.norm(sub_quad[0] - sub_quad[1]),
      np.linalg.norm(sub_quad[1] - sub_quad[2]),
      np.linalg.norm(sub_quad[2] - sub_quad[3]),
      np.linalg.norm(sub_quad[3] - sub_quad[0])
  ]

  min_edge = min(edges)
  # 如果最短边太小，说明退化成了三角形或极小碎块，直接舍弃
  if min_edge < 1e-4:
    return False

  max_edge = max(edges)
  aspect_ratio = max_edge / min_edge

  # 将 distortion_limit (0~1) 映射为可接受的最大长宽比阈值
  # distortion_limit=0 时，只接受接近正方形的 (比例 <= 1.5)
  # distortion_limit=1 时，接受较扁的 (比例 <= 5.0)
  allowed_ratio = 1.5 + distortion_limit * 3.5

  if aspect_ratio > allowed_ratio:
    return False

  return True


def _get_mask(offset_img: NDArray[np.uint8],
              offset: tuple[float, float] | None = None) -> NDArray[np.uint8]:
  """把贴图二值化后铺到整幅环境图尺寸的掩膜上

  Args:
      offset_img (NDArray[np.uint8]): 前景白、背景黑的贴图；
        若是 `proj_tran` 的返回值，则自带 `offset` 元数据
      offset (tuple[float, float] | None): 贴图在环境图中的左上角坐标 `(x, y)`；
        缺省时取 `offset_img.offset`

  Raises:
      ValueError: 既没有传入 offset，贴图也没有 `offset` 元数据

  Returns:
      NDArray[np.uint8]: 与「贴图落点 + 贴图尺寸」等大的单通道掩膜
  """
  if offset is None:
    offset = getattr(offset_img, "offset", None)
  if offset is None:
    raise ValueError("缺少 offset：请显式传入 offset，或使用 proj_tran 的返回值")
  dx, dy = int(round(float(offset[0]))), int(round(float(offset[1])))
  oh, ow = offset_img.shape[:2]
  out = np.zeros((max(dy + oh, 0), max(dx + ow, 0)), np.uint8)
  font_mask = cv2.threshold(offset_img, 1, 255, cv2.THRESH_BINARY)[1]
  # 贴图可能有一部分落在画布外（dx / dy 为负），只写回可见区域
  x0, y0 = max(dx, 0), max(dy, 0)
  src = font_mask[max(0, -dy):, max(0, -dx):]
  out[y0:y0 + src.shape[0], x0:x0 + src.shape[1]] = src

  return out


def segment_quad(quad: NDArray[np.floating],
                 distortion: tuple[float, float],
                 normalize: tuple[float, float],
                 area_range: tuple[float, float] = (.3, 3000),
                 gap: float = 0.12,
                 board_aspect: float | None = None,
                 cell_jitter: float = 0.18,
                 area_scale: float = 1.0,
                 min_cells: int = 1) -> list[QuadSegment]:
  """按「目标面积」将四边形分割成多个四边形，通过面积 / 畸变过滤舍弃碎片。

  切分数量不单独指定：先按 `area_range` 采样的目标面积确定格子数，
  再用平坦区域总面积除以它得到行列数，所以板子要多大就切多少块。

  **面积连续化**：行列数是整数，原始格子面积只能按 `1/(cols*rows)` 跳变
  （例如 154k 的区域只能切出 154k / 38.5k …），只靠整数网格会出现「目标 80k
  却只能拿到 38k」的量化误差，大区间永远落不到实处。这里给每块板单独算一个
  向内收缩的间隙，把面积连续调到 `area_range` 内，因此调大 `area_range`
  时板位面积会真正跟着变大。

  板子看着像正方形，实际是长宽未知的矩形：每次调用会在畸变容忍度允许的
  比例范围内随机取一个长宽比，再据此解出行列数，不把格子切成等宽等高。

  Args:
    quad (NDArray[np.float64]): 输入的四边形，(4, 2) 的顶点坐标 [左上, 右上, 右下, 左下]
    distortion (float | tuple): 畸变容忍度（0~1），越小要求越严格（丢弃越多的畸变块）；
        同时决定板位长宽比的允许范围（1.5 + 3.5*d）。
    normalize (float | tuple): 归一化程度（0~1），值越大内部网格越接近菱形分布。
    area_range (tuple): 单块板位的目标面积范围（像素²），<1 的值按占平坦区域面积的比例解释；
        同时决定切分数量。
    gap (float): 板位向内收缩的**最小**间隙 (0~1，0 为无缝)；实际间隙会在
        `gap`~0.45 之间逐块自适应，用来把格子面积调到目标区间。
    board_aspect (float | tuple | None): 板位长宽比（宽/高）；tuple 时在区间内随机取，
        None 时取畸变容忍度允许的整个范围。Defaults to None.
    cell_jitter (float): 切割线位置的抖动幅度（0~1）；>0 时同一区域里每块板的长宽都不同。
        Defaults to 0.18.
    area_scale (float): `area_range` 与板位面积的换算系数。当 `area_range` 描述的是
        「数字区域」面积时传 `(1 - 2*留边_横) * (1 - 2*留边_纵)`，函数内部会把板位目标放大到
        `area_range / area_scale`，使数字内缩后的面积正好落回区间。默认 1（描述板位）。
    min_cells (int): 期望至少切出的板位数。用于「每张图放 N 个数字」——多需要几块
        就优先选更细的网格，而不是只切一块最大的板。默认 1。
  """
  # 1. 处理范围参数
  if isinstance(distortion, tuple):
    distortion = random.uniform(distortion[0], distortion[1])
  if isinstance(normalize, tuple):
    normalize = random.uniform(normalize[0], normalize[1])

  # 2. 计算物理空间的平均宽高 (用于自适应分配行列数，防止长条)
  width_top = np.linalg.norm(quad[1] - quad[0])
  width_bottom = np.linalg.norm(quad[2] - quad[3])
  height_left = np.linalg.norm(quad[3] - quad[0])
  height_right = np.linalg.norm(quad[2] - quad[1])

  avg_w = (width_top + width_bottom) / 2.0
  avg_h = (height_left + height_right) / 2.0
  quad_area = abs(cv2.contourArea(np.asarray(quad, dtype=np.float32)))
  if quad_area <= 1.0:
    return []

  # 3. 目标面积区间（<1 按比例解释；再换算到「板位」侧）
  raw_lo, raw_hi = sorted(float(v) * quad_area if float(v) < 1 else float(v)
                          for v in area_range)
  scale = float(area_scale) if area_scale else 1.0
  area_lo = max(raw_lo / scale, 1.0)
  area_hi = max(raw_hi / scale, area_lo)

  gap_min = float(np.clip(gap, 0.0, 0.45))
  gap_max = 0.45
  # 收缩后面积 = 原始格子面积 * (1-gap)^2：原始格子至少要 >= area_lo（不收缩），
  # 又要在 gap_max 下能收到 <= area_hi
  pre_hi = area_hi / ((1.0 - gap_max) ** 2)

  # 板子看着接近正方形，实际长宽未知：在畸变容忍允许的范围内随机取一个长宽比
  max_ratio = 1.5 + distortion * 3.5        # 与 _is_acceptable 的上限保持一致
  if board_aspect is None:
    board_ratio = random.uniform(1.0 / max_ratio, max_ratio)
  elif isinstance(board_aspect, (tuple, list)):
    board_ratio = random.uniform(
        float(board_aspect[0]), float(board_aspect[1]))
  else:
    board_ratio = float(board_aspect)

  # 行列数要同时满足「格子宽高比接近 board_ratio」与「格子原始面积能覆盖目标区间」：
  # 先按 sqrt 求理想列数，再枚举该列数附近可行的行数，最后按两个误差择优。
  aspect_ratio = max(avg_w, 1e-6) / max(avg_h, 1e-6)
  min_cells = max(1, int(min_cells))
  target_ref = random.uniform(area_lo, min(area_hi, quad_area))
  if min_cells > 1:
    # 需要更多板位时，把目标格子面积压到「刚好能分出 min_cells 块」
    target_ref = min(target_ref, quad_area / min_cells)
  ideal_cols = max(1.0, math.sqrt(
      max(quad_area, 1.0) / max(target_ref, 1.0) * aspect_ratio / board_ratio))

  candidates = []          # (cols, rows, 原始格子面积, 格子宽高比)
  for c in range(1, max(2, int(round(ideal_cols)) + 4)):
    r_max = max(1, int(math.floor(quad_area / (c * max(area_lo, 1.0)))))
    for r in range(1, r_max + 1):
      pre_area = quad_area / (c * r)
      if pre_area < area_lo or pre_area > pre_hi:
        continue
      cell_ratio = aspect_ratio * r / c
      if not 1.0 / max_ratio <= cell_ratio <= max_ratio:
        continue
      candidates.append((c, r, pre_area, cell_ratio))

  if not candidates:        # 这块区域放不下既合面积、又合长宽比的板子
    return []

  # 需求块数优先：只在能切出 >= min_cells 块的网格里挑（实在做不到时退回全部）
  fit = [cand for cand in candidates if cand[0] * cand[1] >= min_cells]
  pool = fit or candidates
  # 优先用「原始格子面积 >= 目标」的网格（能用收缩把面积连续调到目标，
  # 不会被整数行列量化卡在某个固定档位）；都达不到时取最大的那一档
  above = [cand for cand in pool if cand[2] >= target_ref]
  cols, rows, _, _ = min(
      above or pool,
      key=lambda cand: abs(math.log(cand[2] / target_ref))
      + abs(math.log(cand[3] / board_ratio)))

  # 3. 生成内部网格点（切割线带抖动：同一区域里每块板的长宽都不完全相同）
  jitter = max(0.0, min(0.45, float(cell_jitter)))

  def _cuts(n: int) -> np.ndarray:
    """0~1 上 n 段切割线；每段权重 1±jitter，保证板子不全是同一个尺寸"""
    if n <= 1 or jitter <= 0.0:
      return np.linspace(0.0, 1.0, n + 1)
    w = np.clip(np.array([1.0 + random.uniform(-jitter, jitter)
                          for _ in range(n)]), 1.0 - jitter, None)
    return np.concatenate([[0.0], np.cumsum(w) / w.sum()])

  u_vals = _cuts(cols)
  v_vals = _cuts(rows)

  grid_points = np.zeros((rows + 1, cols + 1, 2), dtype=np.float64)

  for j in range(rows + 1):
    for i in range(cols + 1):
      u = u_vals[i]
      v = v_vals[j]

      p_top = (1 - u) * quad[0] + u * quad[1]
      p_bot = (1 - u) * quad[3] + u * quad[2]
      p_base = (1 - v) * p_top + v * p_bot

      # 内部网格正则化处理 (normalize 控制菱形分布)
      if 0 < i < cols and 0 < j < rows:
        u_center = u - 0.5
        v_center = v - 0.5

        if normalize > 0.5:
          w_diamond = (normalize - 0.5) * 2.0
          p_base[0] += u_center * v_center * w_diamond * avg_w * 0.5
          p_base[1] += (u_center**2 - v_center**2) * w_diamond * avg_h * 0.5

      grid_points[j, i] = p_base

  # 4. 切片提取、逐块采样目标面积并自适应 gap
  segments = []
  for j in range(rows):
    for i in range(cols):
      p_tl = grid_points[j, i]
      p_tr = grid_points[j, i + 1]
      p_br = grid_points[j + 1, i + 1]
      p_bl = grid_points[j + 1, i]

      center = (p_tl + p_tr + p_br + p_bl) / 4.0
      raw_quad = np.array([p_tl, p_tr, p_br, p_bl], dtype=np.float64)
      pre_area = abs(cv2.contourArea(raw_quad.astype(np.float32)))
      if pre_area < area_lo:      # 这一块本身已经小于下限，收缩后更不可能达标
        continue

      # 每块板独立采样目标面积：把面积从「整数网格的离散档位」改成连续取值，
      # 这样调大 area_range 才能真的得到更大的板位
      target = random.uniform(area_lo, min(area_hi, pre_area))
      g = 1.0 - math.sqrt(max(target, 1e-6) / pre_area)
      if pre_area > area_hi:      # 原始格子超过上限，至少要缩到上限以内
        g = max(g, 1.0 - math.sqrt(area_hi / pre_area))
      g = float(np.clip(max(g, gap_min), 0.0, gap_max))

      p_tl = p_tl + (center - p_tl) * g
      p_tr = p_tr + (center - p_tr) * g
      p_br = p_br + (center - p_br) * g
      p_bl = p_bl + (center - p_bl) * g

      sub_quad = np.array([p_tl, p_tr, p_br, p_bl], dtype=np.float64)
      area = abs(cv2.contourArea(sub_quad.astype(np.float32)))
      # 长宽比超过 distortion 允许的阈值、或面积不在范围内的碎片直接丢弃；
      # 面积上下限留 2% 余量：抖动 / 透视会让实测面积与估算略有出入
      if (_is_acceptable(sub_quad, distortion)
              and area_lo * 0.98 <= area <= area_hi * 1.02):
        # 坐标必须保持 float64（绝对图像坐标），不能用 OffsetImg 承载
        segments.append(QuadSegment(sub_quad))

  return segments


def _warp_to_square(img: NDArray[np.uint8], quad: NDArray[np.floating],
                    size: int = 64) -> NDArray[np.uint8]:
  """把四边形区域透视变换成 size×size 的正方形，便于在同一坐标系下做检查"""
  size = int(size)
  dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
                 dtype=np.float32)
  M = cv2.getPerspectiveTransform(np.asarray(quad, dtype=np.float32), dst)
  return cv2.warpPerspective(img, M, (size, size), flags=cv2.INTER_LINEAR)


def _line_angles(gray: NDArray[np.uint8], canny: tuple[int, int] = (50, 150),
                 min_len: int = 40, max_gap: int = 8) -> NDArray[np.float64]:
  """检测图中的长直线方向（0~180°，不区分正反）"""
  edges = cv2.Canny(gray, int(canny[0]), int(canny[1]))
  lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=60,
                          minLineLength=int(min_len), maxLineGap=int(max_gap))
  if lines is None:
    return np.zeros(0, dtype=np.float64)
  seg = np.asarray(lines, dtype=np.float64).reshape(-1, 4)
  return np.degrees(np.arctan2(seg[:, 3] - seg[:, 1],
                               seg[:, 2] - seg[:, 0])) % 180.0


def _angle_gap(a: float, b: float) -> float:
  """两个 0~180° 方向之间的最小夹角"""
  d = abs(float(a) - float(b)) % 180.0
  return min(d, 180.0 - d)


def _plane_residual(patch: NDArray[np.uint8]) -> float:
  """最小二乘拟合平面（ax + by + c）后的残差 RMS，单位是灰度级"""
  p = np.asarray(patch, dtype=np.float32)
  h, w = p.shape
  yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
  A = np.stack([xx.ravel(), yy.ravel(), np.ones(h * w, np.float32)], axis=1)
  coef, *_ = np.linalg.lstsq(A, p.ravel(), rcond=None)
  r = A @ coef - p.ravel()
  return float(np.sqrt((r ** 2).mean()))


def is_planar_quad(gray: NDArray[np.uint8], quad: NDArray[np.floating],
                   flat_mask: NDArray[np.uint8] | None = None,
                   line_angles: NDArray[np.float64] | None = None,
                   size: int = 64,
                   min_flat_ratio: float = 0.85, max_edge_ratio: float = 0.05,
                   edge_thresh: float = 45.0, max_plane_rms: float = 8.0,
                   angle_tol: float = 15.0, min_aligned_edges: int = 2,
                   min_lines_for_angle_check: int = 6) -> bool:
  """判断四边形区域是否像一块「平板」（可以往上贴数字）

  四项检查：
  1. **内部平整**：把区域正射到正方形后统计强边缘像素占比，折痕 / 跨物体都会超标；
  2. **平坦覆盖**：四边形内落在平坦掩膜里的比例 >= `min_flat_ratio`；
  3. **平面拟合**：用 ax + by + c 拟合亮度，残差 RMS 超过 `max_plane_rms`
     说明区域内有折痕或两块不同表面；
  4. **透视一致**：四边形的边方向应与图中检测到的长直线方向一致
     （墙 / 柜面 / 桌面的真实棱边不会乱飘），至少 `min_aligned_edges` 条边
     能在 `angle_tol` 内对齐；图中直线太少（< `min_lines_for_angle_check`）
     时该项信息不足，直接跳过。

  Args:
      gray (NDArray[np.uint8]): 灰度环境图
      quad (NDArray[np.float64]): 四边形，(4, 2)，顺序 [左上, 右上, 右下, 左下]
      flat_mask (NDArray[np.uint8] | None): 平坦区域掩膜（255 = 平坦）
      line_angles (NDArray[np.float64] | None): `_line_angles` 的输出
      size (int): 正射检查用的正方形边长
      min_flat_ratio (float): 四边形内平坦像素的最低占比
      max_edge_ratio (float): 四边形内强边缘像素允许的最大占比
      edge_thresh (float): 判定强边缘的梯度幅值阈值
      max_plane_rms (float): 平面拟合残差 RMS 上限（灰度级）
      angle_tol (float): 边方向与场景直线方向的对齐容差（度）
      min_aligned_edges (int): 至少多少条边对齐
      min_lines_for_angle_check (int): 少于这么多条场景直线时跳过透视一致性检查

  Returns:
      bool: 是否像一块平板
  """
  q = np.asarray(quad, dtype=np.float64).reshape(4, 2)

  patch = _warp_to_square(gray, q, size)
  gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
  gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
  mag = cv2.magnitude(gx, gy)[2:-2, 2:-2]
  if float((mag > edge_thresh).mean()) > max_edge_ratio:
    return False

  core = patch[2:-2, 2:-2]
  if _plane_residual(core) > max_plane_rms:
    return False

  if flat_mask is not None:
    m = _warp_to_square(flat_mask, q, size)[2:-2, 2:-2]
    if float((m > 127).mean()) < min_flat_ratio:
      return False

  if line_angles is not None and len(line_angles) >= min_lines_for_angle_check:
    angles = np.asarray(line_angles, dtype=np.float64)
    aligned = 0
    for i in range(4):
      d = q[(i + 1) % 4] - q[i]
      a = float(np.degrees(np.arctan2(d[1], d[0])) % 180.0)
      if np.min(np.abs([_angle_gap(a, b) for b in angles])) <= angle_tol:
        aligned += 1
    if aligned < min_aligned_edges:
      return False

  return True


def get_flat_quad(img: NDArray[np.uint8],
                  planar_check: bool = True) -> list[NDArray[np.float64]]:
  """获得图片平滑区域，返回四边形

  Args:
    img (NDArray[np.uint8]): 输入的图像，可以bgr或gray，数据类型为 uint8
    planar_check (bool): 是否做平面性校验（剔除非平面 / 跨物体的四边形）
  Returns:
    NDArray[np.float64]: 四边形顶点坐标，形状为 (4, 2)
  """
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
  mask = _smooth_mask(gray)
  # cv2.imencode('.png', mask)[1].tofile(
  #    f"{LOG_DIR}/{img.shape}_debug_0_mask.png")

  contours, _ = cv2.findContours(
      mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  if not contours:
    _logger.debug("未检测到平滑区域")
    return None

  _img = cv2.drawContours(img.copy(), contours, -1, (0, 255, 0), 2)
  # cv2.imencode('.png', _img)[1].tofile(
  #    f"{LOG_DIR}/{img.shape}_debug_1_contours.png")

  quads = [_get_maxquad(cnt) for cnt in contours]
  quads = [np.array(q, dtype=np.int32) for q in quads if q is not None]
  _img = cv2.drawContours(img.copy(), np.array(
      quads, dtype=np.int32), -1, (0, 255, 0), 2)
  # cv2.imencode('.png', _img)[1].tofile(
  #    f"{LOG_DIR}/{img.shape}_debug_1_quads.png")

  quads = [order_quad(q, False) for q in quads]
  _img = cv2.drawContours(img.copy(), np.array(
      quads, dtype=np.int32), -1, (0, 255, 0), 2)
  # cv2.imencode('.png', _img)[1].tofile(
  #    f"{LOG_DIR}/{img.shape}_debug_2_contours.png")

  if planar_check and quads:
    angles = _line_angles(gray)
    ex_keep = []
    for q in quads:
      if is_planar_quad(gray, q, mask, angles):
        ex_keep.append(q)
    _logger.debug("平面性校验: 保留 %d / 剔除 %d", len(ex_keep), len(quads) - len(ex_keep))
    quads = ex_keep
    _img = cv2.drawContours(img.copy(), np.array(
        quads, dtype=np.int32), -1, (0, 255, 0), 2) if quads else img.copy()
    # cv2.imencode('.png', _img)[1].tofile(
    #    f"{LOG_DIR}/{img.shape}_debug_3_planar.png")

  return quads


# ---------------------------------------------------------------------------
# 数字融合 / 景深失焦
# ---------------------------------------------------------------------------

# 默认的「数字 / 环境」对比度区间（|数字 - 背景| / 255）
CONTRAST_RANGE: tuple[float, float] = (0.35, 0.70)


def pick_contrast(contrast: float | tuple[float, float] | Sequence[float]
                  = CONTRAST_RANGE) -> float:
  """取本次使用目标对比度

  Args:
      contrast (float | tuple[float, float] | Sequence[float]): 固定值 / 二元区间 / 候选集合，
        取值 0~1，表示 |数字灰度 - 背景亮度| / 255

  Returns:
      float: 0~1 的目标对比度
  """
  if isinstance(contrast, (int, float, np.integer, np.floating)) and not isinstance(contrast, bool):
    return float(np.clip(contrast, 0.0, 1.0))
  vals = [float(v) for v in contrast]
  if not vals:
    raise ValueError("pick_contrast: 候选对比度为空")
  if isinstance(contrast, tuple) and len(vals) == 2:
    return float(np.clip(random.uniform(*sorted(vals)), 0.0, 1.0))
  return float(np.clip(random.choice(vals), 0.0, 1.0))


def contrast_gray(bg: float, contrast: float = 0.5,
                  polarity: str = "any") -> int:
  """按目标对比度给数字定色：使 |数字灰度 - 背景亮度| ≈ contrast * 255

  `polarity` 控制方向：`"down"` 只压暗（亮板/白纸 → 黑字）、
  `"up"` 只提亮（暗板 → 白字）、`"any"` 优先朝有余量的方向取
  （暗背景提亮、亮背景压暗）。指定方向越界时自动回退到另一侧，
  两侧都越界时取 0 / 255（此时实际对比度会被截断）。

  Args:
      bg (float): 贴图落点处的背景亮度 0~255
      contrast (float): 目标对比度 0~1
      polarity (str): `"any"` / `"down"` / `"up"`

  Returns:
      int: 数字灰度 0~255
  """
  bg = float(bg)
  c = float(np.clip(contrast, 0.0, 1.0)) * 255.0
  up, down = bg + c, bg - c
  up_ok, down_ok = up <= 255.0, down >= 0.0
  if polarity == "down" and down_ok:
    return int(round(down))          # 只压暗：白板黑字
  if polarity == "up" and up_ok:
    return int(round(up))            # 只提亮：黑板白字
  if up_ok and down_ok:
    return int(round(up if bg < 127.0 else down))
  if up_ok:
    return int(round(up))
  if down_ok:
    return int(round(down))
  return 255 if bg < 127.0 else 0


def pick_alpha(alpha: float | tuple[float, float] | Sequence[float]
               = (0.5, 1.0)) -> float:
  """取数字与环境图的融合透明度

  Args:
      alpha (float | tuple[float, float] | Sequence[float]): 固定值 / 二元区间 / 候选集合

  Returns:
      float: 0~1，越大数字越「实」
  """
  if isinstance(alpha, (int, float, np.integer, np.floating)) and not isinstance(alpha, bool):
    return float(np.clip(alpha, 0.0, 1.0))
  vals = [float(v) for v in alpha]
  if not vals:
    raise ValueError("pick_alpha: 候选透明度为空")
  if isinstance(alpha, tuple) and len(vals) == 2:
    return float(np.clip(random.uniform(*sorted(vals)), 0.0, 1.0))
  return float(np.clip(random.choice(vals), 0.0, 1.0))


def _glyph_coverage(glyph: NDArray[np.uint8]) -> NDArray[np.float32]:
  """把数字贴图转成 0~1 的前景覆盖度（保留抗锯齿 / 模糊产生的软边）

  约定贴图是「白字黑底」。若整幅偏亮（平均灰度 > 127），认为它是黑字白底，自动取反。
  """
  g = glyph if glyph.ndim == 2 else cv2.cvtColor(glyph, cv2.COLOR_BGR2GRAY)
  g = g.astype(np.float32)
  if float(g.mean()) > 127.0:
    g = 255.0 - g
  return g / 255.0


def _clip_window(offset: tuple[float, float], h: int, w: int, H: int, W: int
                 ) -> tuple[int, int, int, int, int, int] | None:
  """把贴图窗口裁到画布内

  Returns:
      tuple[int, int, int, int, int, int] | None: (dx0, dy0, dx1, dy1, sx0, sy0)，
      即画布上的目标矩形与贴图内的源起点；完全落在画布外时返回 None
  """
  x0, y0 = int(round(float(offset[0]))), int(round(float(offset[1])))
  dx0, dy0 = max(x0, 0), max(y0, 0)
  dx1, dy1 = min(x0 + w, W), min(y0 + h, H)
  if dx1 <= dx0 or dy1 <= dy0:
    return None
  return dx0, dy0, dx1, dy1, dx0 - x0, dy0 - y0


def local_background(img_env: NDArray[np.uint8], glyph: NDArray[np.uint8],
                     offset: tuple[float, float] | None = None) -> float:
  """贴图落点处环境图的背景亮度（中位数），用于按对比度给数字定色

  Args:
      img_env (NDArray[np.uint8]): 环境图
      glyph (NDArray[np.uint8]): 数字贴图（用来确定落点尺寸）
      offset (tuple[float, float] | None): 贴图左上角坐标，缺省取 `glyph.offset`

  Raises:
      ValueError: 既没有传入 offset，贴图也没有 offset 元数据

  Returns:
      float: 0~255 的背景亮度
  """
  if offset is None:
    offset = getattr(glyph, "offset", None)
  if offset is None:
    raise ValueError("缺少 offset：请显式传入贴图左上角坐标，或使用 proj_tran 的返回值")
  env = np.asarray(img_env)
  H, W = env.shape[:2]
  win = _clip_window(offset, glyph.shape[0], glyph.shape[1], H, W)
  if win is None:
    return float(env.mean())
  dx0, dy0, dx1, dy1, _, _ = win
  return float(np.median(env[dy0:dy1, dx0:dx1]))


def fuse_img(img_env: NDArray[np.uint8], glyph: NDArray[np.uint8],
             offset: tuple[float, float] | None = None,
             gray: int | None = None, contrast: float = 0.5,
             alpha: float = 1.0) -> NDArray[np.uint8]:
  """将「白字黑底」数字贴图按对比度与透明度融合进环境图（第一步：融合）

  逐像素 alpha 合成：`out = env * (1 - a) + gray * a`，其中 `a = 覆盖度 * alpha`，
  覆盖度取自贴图灰度，因此失焦后的软边会被自然保留。

  数字颜色不再是固定色阶，而是按「与局部背景的对比度」自动确定：
  `|gray - bg| ≈ contrast * 255`，保证数字在任何底色上都保持一致的可见度。

  Args:
      img_env (NDArray[np.uint8]): 环境图，BGR 或灰度
      glyph (NDArray[np.uint8]): 白字黑底的数字贴图，通常由 `proj_tran` 得到
      offset (tuple[float, float] | None): 贴图左上角在环境图中的绝对坐标；
        缺省时取 `glyph.offset`（`proj_tran` 的返回值自带）
      gray (int | None): 直接指定数字灰度；给定时忽略 `contrast`
      contrast (float): 目标对比度 0~1，缺省 0.5，用于自动定色
      alpha (float): 融合强度 0~1，越大数字越实

  Raises:
      ValueError: 既没有传入 offset，贴图也没有 offset 元数据

  Returns:
      NDArray[np.uint8]: 融合后的新图（不修改入参）
  """
  if offset is None:
    offset = getattr(glyph, "offset", None)
  if offset is None:
    raise ValueError("缺少 offset：请显式传入贴图左上角坐标，或使用 proj_tran 的返回值")

  cov = _glyph_coverage(glyph)
  h, w = cov.shape
  out = np.array(img_env, copy=True)
  H, W = out.shape[:2]
  win = _clip_window(offset, h, w, H, W)
  if win is None:
    return out
  dx0, dy0, dx1, dy1, sx0, sy0 = win

  if gray is None:
    gray = contrast_gray(local_background(img_env, glyph, offset), contrast)
  gray = int(np.clip(gray, 0, 255))
  alpha = float(np.clip(alpha, 0.0, 1.0))

  tile = cov[sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)]
  a = tile * alpha
  region = out[dy0:dy1, dx0:dx1].astype(np.float32)
  color = np.full_like(region, float(gray))
  if region.ndim == 3:
    a = a[..., None]
  out[dy0:dy1, dx0:dx1] = np.clip(
      region * (1.0 - a) + color * a, 0, 255).astype(np.uint8)

  return out


def _gaussian(glyph: NDArray[np.uint8], sigma: float | None
              ) -> NDArray[np.uint8]:
  """固定 sigma 的整块高斯模糊（sigma <= 0 时原样返回）"""
  if sigma is None or float(sigma) <= 0:
    return glyph
  sigma = float(sigma)
  k = int(2 * math.ceil(3.0 * sigma) + 1)
  return cv2.GaussianBlur(glyph, (k, k), sigma)


def _relative_taper(near_len: float, far_len: float) -> float:
  """(近边 - 远边) / 长边，范围 [-1, 1]，表示该方向的透视收缩强度"""
  m = max(abs(near_len), abs(far_len), 1e-6)
  return float((near_len - far_len) / m)


def quad_perspective_ramp(quad: NDArray[np.floating], size: tuple[int, int]
                          ) -> NDArray[np.float32]:
  """按四边形的透视收缩关系生成 0（近）~1（远）的平滑深度渐变图

  透视规律：同一平面上越远的部分在图像里越小。于是分别比较上/下两条边、
  左/右两条边，较短的一侧视为更远，两方向合成为四个角点的相对远近，
  再双线性插值成与贴图同尺寸的渐变图（贴图正是该四边形的透视变换结果）。

  Args:
      quad (NDArray[np.float64]): 四边形，(4, 2)，顺序 [左上, 右上, 右下, 左下]
      size (tuple[int, int]): 贴图尺寸 (宽, 高)

  Returns:
      NDArray[np.float32]: 形状 (h, w) 的渐变图，0 = 最近，1 = 最远；
      四边形是正矩形（无透视线索）时返回常数 0.5
  """
  q = np.asarray(quad, dtype=np.float64).reshape(4, 2)
  w, h = int(size[0]), int(size[1])
  h, w = max(h, 1), max(w, 1)

  w_top = float(np.linalg.norm(q[1] - q[0]))
  w_bot = float(np.linalg.norm(q[2] - q[3]))
  h_left = float(np.linalg.norm(q[3] - q[0]))
  h_right = float(np.linalg.norm(q[2] - q[1]))

  # 底边比顶边长 => 上远下近（正值 = 上远）；右边比左边长 => 左远右近（正值 = 左远）
  s_vert = _relative_taper(w_bot, w_top)
  s_horz = _relative_taper(h_right, h_left)

  tl = (s_vert + s_horz) / 2.0
  tr = (s_vert - s_horz) / 2.0
  bl = (-s_vert + s_horz) / 2.0
  br = (-s_vert - s_horz) / 2.0
  corners = np.array([[tl, tr], [bl, br]], np.float32)

  peak = float(np.abs(corners).max())
  if peak < 1e-6:
    return np.full((h, w), 0.5, np.float32)   # 正矩形，没有可用的透视线索
  corners /= peak

  ramp = cv2.resize(corners, (w, h), interpolation=cv2.INTER_LINEAR)
  return np.clip((ramp + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)


def quad_point(quad: NDArray[np.floating], u: float, v: float
               ) -> NDArray[np.float64]:
  """按双线性参数在四边形内取点（u / v ∈ [0, 1]，对应 [左上, 右上, 右下, 左下]）"""
  q = np.asarray(quad, dtype=np.float64).reshape(4, 2)
  p_top = (1.0 - u) * q[0] + u * q[1]
  p_bot = (1.0 - u) * q[3] + u * q[2]
  return (1.0 - v) * p_top + v * p_bot


# TODO: 使margin支持负数， 由于源数字也有原始边距
def inset_quad(quad: NDArray[np.floating],
               margin: float | Sequence[float] = 0.18
               ) -> NDArray[np.float64]:
  """把四边形按比例向内收缩，得到数字在板面上占据的区域

  Args:
      quad (NDArray[np.float64]): 板面四边形，(4, 2)，[左上, 右上, 右下, 左下]
      margin (float | Sequence[float]): 内缩比例（0~0.45）。
          标量 = 横纵相同；``(横, 纵)`` = 两个方向分别内缩
          （横作用在 u 方向 = 左→右，纵作用在 v 方向 = 上→下）

  Returns:
      NDArray[np.float64]: 内缩后的四边形，(4, 2)

  Raises:
      ValueError: ``margin`` 元素个数既不是 1 也不是 2。
  """
  vals = np.atleast_1d(np.asarray(margin, dtype=np.float64)).ravel()
  if vals.size == 1:
    mx = my = float(vals[0])
  elif vals.size == 2:
    mx, my = float(vals[0]), float(vals[1])
  else:
    raise ValueError(f"margin 应为标量或 (横, 纵)，收到 {margin!r}")
  mx = float(np.clip(mx, 0.0, 0.45))
  my = float(np.clip(my, 0.0, 0.45))
  u0, u1 = mx, 1.0 - mx
  v0, v1 = my, 1.0 - my
  return np.array([quad_point(quad, u0, v0), quad_point(quad, u1, v0),
                   quad_point(quad, u1, v1), quad_point(quad, u0, v1)],
                  dtype=np.float64)


def _pick_tone(tone: float | tuple[float, float] | Sequence[float]) -> float:
  """取一个亮度偏移：固定值 / 二元区间 / 候选序列"""
  if isinstance(tone, (int, float, np.integer, np.floating)) and not isinstance(tone, bool):
    return float(tone)
  vals = [float(v) for v in tone]
  if not vals:
    raise ValueError("_pick_tone: 候选亮度偏移为空")
  if isinstance(tone, tuple) and len(vals) == 2:
    lo, hi = sorted(vals)
    return random.uniform(lo, hi)
  return random.choice(vals)


def _ramp_full(quad: NDArray[np.floating], H: int,
               W: int) -> NDArray[np.float32]:
  """把四边形内部的 0(近)~1(远) 透视渐变铺到整幅画布上（四边形外取 0.5）"""
  ramp_full = np.full((H, W), 0.5, np.float32)
  q = np.asarray(quad, dtype=np.float64).reshape(4, 2)
  x0 = max(int(np.floor(q[:, 0].min())), 0)
  y0 = max(int(np.floor(q[:, 1].min())), 0)
  x1 = min(int(np.ceil(q[:, 0].max())) + 1, W)
  y1 = min(int(np.ceil(q[:, 1].max())) + 1, H)
  if x1 > x0 and y1 > y0:
    ramp_full[y0:y1, x0:x1] = quad_perspective_ramp(q, (x1 - x0, y1 - y0))
  return ramp_full


def _defocus_mask(mask: NDArray[np.uint8], quad: NDArray[np.floating],
                  sigma: float | None, spread: float = 0.5,
                  base: float = 0.0) -> NDArray[np.float32]:
  """对二值掩膜做与数字一致的失焦，返回 0~1 的软覆盖度

  Args:
      mask (NDArray[np.uint8]): 0/255 二值掩膜
      quad (NDArray[np.float64]): 掩膜对应的四边形（用于块内的近清远糊渐变）
      sigma (float): 基准高斯 sigma（对应近端），<= 0 表示不失焦
      spread (float): 远近两端 sigma 差异比例
      base (float): 基础柔边 sigma（叠加在 sigma 上）

  Returns:
      NDArray[np.float32]: 0~1 的覆盖度，尺寸与 `mask` 一致
  """
  base = max(float(base), 0.0)
  sigma = 0.0 if sigma is None else max(float(sigma), 0.0)
  if sigma <= 0 and base <= 0:
    return mask.astype(np.float32) / 255.0
  if sigma <= 0:
    return cv2.GaussianBlur(mask, (0, 0), base).astype(np.float32) / 255.0

  spread = float(np.clip(spread, 0.0, 0.9))
  near = cv2.GaussianBlur(mask, (0, 0), sigma * (1.0 - spread) + base)
  far = cv2.GaussianBlur(mask, (0, 0), sigma * (1.0 + spread) + base)
  H, W = mask.shape[:2]
  wmap = _ramp_full(quad, H, W)
  return np.clip((near.astype(np.float32) * (1.0 - wmap) +
                  far.astype(np.float32) * wmap) / 255.0, 0.0, 1.0)


def draw_board(img_env: NDArray[np.uint8], quad: NDArray[np.floating],
               tone_delta: float | tuple[float, float] = 18.0,
               feather: float = 2.0,
               shadow: bool = True,
               shadow_offset: tuple[int, int] = (4, 6),
               shadow_blur: float = 9.0, shadow_alpha: float = 0.35,
               sigma: float = 0.0, persp_spread: float = 0.5,
               target_gray: float | tuple[float, float] | None = None
               ) -> NDArray[np.uint8]:
  """在环境图上渲染一块「板」：整体调色 + 失焦柔边 + 软投影

  数字就贴在这块板上。板面和数字使用同一套失焦：`sigma` 越大，
  板边越糊（同样带四边形内部的近清远糊渐变），这样离基准平面越远的板
  连轮廓都不会是“刀切”般清晰。

  Args:
      img_env (NDArray[np.uint8]): 环境图（BGR 或灰度）
      quad (NDArray[np.float64]): 板面四边形，(4, 2)
      tone_delta (float | tuple | Sequence): 板面与背景的亮度差（正 = 更亮）；
          仅在 `target_gray` 为 None 时生效
      feather (float): 基础边缘柔化 sigma（像素）
      shadow (bool): 是否画软投影
      shadow_offset (tuple[float, float]): 投影相对板面的偏移
      shadow_blur (float): 投影额外的模糊 sigma
      shadow_alpha (float): 投影不透明度
      sigma (float): 失焦量（与数字使用同一个值），0 表示只做基础柔边
      persp_spread (float): 板内「近端 → 远端」的失焦差异比例
      target_gray (float | tuple | Sequence | None): 板面的**绝对**目标灰度；
          给定时忽略 `tone_delta`（相对背景的偏移）。暗环境里也能画出
          真正的「白纸/白板」，是稳定得到「白纸黑字」的关键

  Returns:
      NDArray[np.uint8]: 画好板的新图（不修改入参）
  """
  out = np.array(img_env, copy=True)
  H, W = out.shape[:2]
  mask = np.zeros((H, W), np.uint8)
  cv2.fillConvexPoly(mask, np.round(quad).astype(np.int32), 255)
  if not mask.any():
    return out

  if shadow:
    sh = np.zeros_like(mask)
    M = np.float32([[1, 0, float(shadow_offset[0])],
                    [0, 1, float(shadow_offset[1])]])
    cv2.warpAffine(mask, M, (W, H), dst=sh, borderValue=0)
    # 投影跟着板一起失焦
    sh_sigma = max(float(shadow_blur), 0.0) + max(float(sigma), 0.0)
    sa = _defocus_mask(sh, quad, 0.0, base=sh_sigma)
    sa = sa * float(np.clip(shadow_alpha, 0.0, 1.0))
    if out.ndim == 3:
      sa = sa[..., None]
    out = np.clip(out.astype(np.float32) * (1.0 - sa), 0, 255).astype(np.uint8)

  bg = float(np.median(out[mask > 127]))
  if target_gray is None:
    target = float(np.clip(bg + _pick_tone(tone_delta), 0, 255))
  else:
    # 绝对目标灰度：不依赖背景亮度，暗场景也能出「白板」
    target = float(np.clip(_pick_tone(target_gray), 0, 255))
  a = _defocus_mask(mask, quad, sigma, persp_spread, base=feather)
  if out.ndim == 3:
    a = a[..., None]
  out = np.clip(out.astype(np.float32) * (1.0 - a) + target * a,
                0, 255).astype(np.uint8)
  return out


def defocus_glyph(glyph: NDArray[np.uint8], sigma: float | None,
                  ramp: NDArray[np.float32] | None = None,
                  spread: float = 0.5) -> NDArray[np.uint8]:
  """对数字贴图做高斯模糊，模拟失焦

  不给 ramp 时是整块均匀模糊；给了 ramp（0 = 近，1 = 远）时，按
  「近端清晰、远端更糊」在 `sigma * (1 ∓ spread)` 两档之间渐变，
  从而保留数字所在平面的透视关系，而不是整块一起糊掉。

  Args:
      glyph (NDArray[np.uint8]): 数字贴图
      sigma (float): 基准高斯 sigma（对应近端），<= 0 时原样返回
      ramp (NDArray[np.float32] | None): `quad_perspective_ramp` 的输出
      spread (float): 远近两端的 sigma 差异比例，0 = 均匀失焦，0.5 = 远端 1.5 倍

  Returns:
      NDArray[np.uint8]: 模糊后的贴图（sigma <= 0 时为原对象）
  """
  if sigma is None or float(sigma) <= 0:
    return glyph
  sigma = float(sigma)
  if ramp is None:
    return _gaussian(glyph, sigma)

  spread = float(np.clip(spread, 0.0, 0.9))
  near = _gaussian(glyph, sigma * (1.0 - spread))
  far = _gaussian(glyph, sigma * (1.0 + spread))

  wmap = np.asarray(ramp, dtype=np.float32)
  if wmap.shape[:2] != glyph.shape[:2]:
    wmap = cv2.resize(wmap, (glyph.shape[1], glyph.shape[0]),
                      interpolation=cv2.INTER_LINEAR)
  wmap = wmap if glyph.ndim == 2 else wmap[..., None]

  return np.clip(near.astype(np.float32) * (1.0 - wmap) + far.astype(np.float32) * wmap,
                 0, 255).astype(np.uint8)


def _minmax_norm(values: Sequence[float] | NDArray[np.floating]
                 ) -> NDArray[np.float64]:
  """批次内 min-max 归一化；所有值相同时统一取 0.5"""
  v = np.asarray(values, dtype=np.float64)
  if v.size == 0:
    return v
  lo, hi = float(v.min()), float(v.max())
  if hi - lo < 1e-9:
    return np.full_like(v, 0.5)
  return (v - lo) / (hi - lo)


def estimate_depths(quads: Sequence[NDArray[np.floating]],
                    img_shape: tuple[int, ...],
                    size_weight: float = 0.55, position_weight: float = 0.30,
                    taper_weight: float = 0.15) -> NDArray[np.float64]:
  """估计每块四边形所在平面的相对深度：0 = 最近，1 = 最远

  三个线索（都先在本批次内 min-max 归一化再加权）：
  1. 表观尺寸：√面积 越小越远（远小近大）
  2. 画面纵向位置：越靠上越远（地面 / 桌面场景先验）
  3. 透视收缩：底边 / 顶边 比值越大（上窄下宽）越远

  Args:
      quads (list[NDArray]): 四边形列表，形状 (4, 2)，顺序 [左上, 右上, 右下, 左下]
      img_shape (tuple): 环境图尺寸，取 `img_shape[0]` 作为画面高度
      size_weight (float): 尺寸线索权重
      position_weight (float): 纵向位置线索权重
      taper_weight (float): 透视收缩线索权重

  Returns:
      NDArray[np.float64]: 形状 (N,) 的深度，0 最近、1 最远
  """
  qs = [np.asarray(q, dtype=np.float64).reshape(4, 2) for q in quads]
  if not qs:
    return np.zeros(0, dtype=np.float64)

  height = max(float(img_shape[0]), 1.0)
  area = np.array([abs(cv2.contourArea(q.astype(np.float32))) for q in qs])
  center_y = np.array([q[:, 1].mean() for q in qs])
  w_top = np.array([np.linalg.norm(q[1] - q[0]) for q in qs])
  w_bot = np.array([np.linalg.norm(q[2] - q[3]) for q in qs])
  taper = w_bot / np.maximum(w_top, 1e-6)

  f_size = 1.0 - _minmax_norm(np.sqrt(np.maximum(area, 0.0)))    # 越小越远
  f_pos = 1.0 - _minmax_norm(center_y / height)                  # 越上越远
  f_taper = _minmax_norm(np.log(np.clip(taper, 1e-3, 1e3)))      # 收缩越强越远

  depth = (size_weight * f_size + position_weight * f_pos +
           taper_weight * f_taper)
  return _minmax_norm(depth)


def pick_base_plane(depths: Sequence[float], tol: float = 0.12
                    ) -> tuple[float, NDArray[np.bool_]]:
  """根据深度分布挑选基准平面：取深度最集中的一簇（簇内差 <= tol）

  以每个深度为候选中心统计成员数，成员最多者胜出；成员数相同时取更紧凑的一簇。

  Args:
      depths (Sequence[float]): `estimate_depths` 的输出
      tol (float): 判定「同一平面」的深度容差

  Returns:
      tuple[float, NDArray[np.bool_]]: (基准平面深度, 每块是否在基准平面上)

  Raises:
      ValueError: depths 为空
  """
  d = np.asarray(depths, dtype=np.float64).ravel()
  if d.size == 0:
    raise ValueError("pick_base_plane: depths 为空")

  best_key = None
  best_center = float(d[0])
  best_mask = np.ones(d.shape, dtype=bool)
  for c in d:
    mask = np.abs(d - c) <= tol
    key = (int(mask.sum()), -float(np.abs(d[mask] - c).mean()))
    if best_key is None or key > best_key:
      best_key, best_center, best_mask = key, float(d[mask].mean()), mask
  return best_center, best_mask


def defocus_sigma(depth: float, base_depth: float, strength: float = 1.0,
                  max_sigma: float = 12.0) -> float:
  """按「与基准平面的深度差」换算高斯 sigma（失焦程度可调）

  Args:
      depth (float): 该数字的深度
      base_depth (float): 基准平面的深度
      strength (float): 失焦强度倍率，0 表示全部清晰，越大越糊
      max_sigma (float): 允许的最大 sigma（像素），同时也是封顶值

  Returns:
      float: 高斯 sigma，0 表示不模糊
  """
  gap = abs(float(depth) - float(base_depth))
  return float(np.clip(float(strength) * float(max_sigma) * gap,
                       0.0, float(max_sigma)))


def fuse_scene(img_env: NDArray[np.uint8],
               glyphs: Sequence[NDArray[np.uint8]],
               segments: Sequence[QuadSegment],
               contrast: float | tuple[float, float] = CONTRAST_RANGE,
               alpha: float | tuple[float, float] = (0.55, 0.95),
               base_depth: float | None = None, plane_tol: float = 0.12,
               defocus_strength: float = 1.0, max_sigma: float = 12.0,
               persp_spread: float = 0.5,
               digit_quads: Sequence[NDArray[np.floating]] | None = None,
               board_tone: float | tuple[float, float] | None = None,
               board_shadow: bool = True, board_feather: float = 2.0,
               size_ratio: float = 0.0,
               board_gray: tuple[int, int] | None = None,
               board_gray_flip: tuple[int, int] | None = None,
               polarity_flip_rate: float = 0.0,
               min_visible: float = 0.0
               ) -> tuple[NDArray[np.uint8], list[dict]]:
  """把一批数字贴图融合进环境图，并按基准平面做景深失焦

  完整流程：先给每块四边形画上一块“板” → 估计各块板的深度 → 确定基准平面
  → 非基准平面上的数字按深度差做高斯模糊（块内再按四边形透视关系做近清远糊）
  → 按与局部背景的对比度定色并融合进环境图。

  Args:
      img_env (NDArray[np.uint8]): 环境图，BGR 或灰度
      glyphs (list[NDArray[np.uint8]]): 白字黑底的数字贴图（已经是 `digit_quads` 的透视结果）
      segments (list[QuadSegment]): 板面四边形（提供 `quad`）
      contrast (float | tuple | Sequence[float]): 数字/背景对比度，见 `pick_contrast`
      alpha (float | tuple | Sequence[float]): 融合透明度，见 `pick_alpha`
      base_depth (float | None): 手动指定基准平面深度；None 表示自动挑选最集中的深度簇
      plane_tol (float): 判定「同一平面」的深度容差
      defocus_strength (float): 失焦强度倍率（0 = 全部清晰）
      max_sigma (float): 单块贴图允许的最大高斯 sigma（像素）
      persp_spread (float): 单块内部「近端 → 远端」的失焦差异比例，0 = 均匀
      digit_quads (list[NDArray] | None): 数字实际占据的四边形（通常用
        `inset_quad` 从板面收缩而来）；None 表示数字铺满板面
      board_tone (float | tuple | Sequence | None): 板面亮度差；None 表示不画板
      board_shadow (bool): 是否给板画软投影
      board_feather (float): 板边柔化 sigma
      size_ratio (float): 失焦上限 = `size_ratio * min(数字贴图宽, 高)`，
        用于保证远处的小数字仍可辨认；0 表示不限
      board_gray (float | tuple | Sequence | None): 板面的**绝对**目标灰度区间；
        给定时不再用 `board_tone`（相对背景），暗环境里也能画出「白纸/白板」
      board_gray_flip (float | tuple | Sequence | None): 反极性样本的板面灰度
        （深色板）；None 表示反极性时仍用 `board_gray`
      polarity_flip_rate (float): 反极性（深板 + 亮字）样本比例，0 = 全部白纸黑字
      min_visible (float): 融合后的最小可见对比度 `|数字-背景|*alpha/255`；
        低于该值时拉满 alpha 并提高对比度，避免「看不见但仍有标签」的噪声样本

  Raises:
      ValueError: glyphs / segments / digit_quads 数量不一致

  Returns:
      tuple[NDArray[np.uint8], list[dict]]: (融合后的图, 每块数字的
      digit_quad / quad / depth / sigma / contrast / gray / bg / alpha / on_base 信息)
  """
  gs = list(glyphs)
  segs = list(segments)
  if len(gs) != len(segs):
    raise ValueError(f"glyphs({len(gs)}) 与 segments({len(segs)}) 数量不一致")
  dquads = [np.asarray(q, dtype=np.float64).reshape(4, 2)
            for q in ([s.quad for s in segs] if digit_quads is None else digit_quads)]
  if len(dquads) != len(segs):
    raise ValueError(
        f"digit_quads({len(dquads)}) 与 segments({len(segs)}) 数量不一致")

  out = np.array(img_env, copy=True)
  depths = estimate_depths([s.quad for s in segs], out.shape)
  if base_depth is None:
    base_depth, on_base = pick_base_plane(depths, plane_tol)
  else:
    base_depth = float(base_depth)
    on_base = np.abs(depths - base_depth) <= plane_tol
  sigmas = [defocus_sigma(d, base_depth, defocus_strength, max_sigma)
            for d in depths]
  if size_ratio > 0:
    # 失焦封顶随数字尺寸缩放：小数字不应该被糊到认不出来
    sigmas = [min(s, float(size_ratio) * float(min(g.shape[:2])))
              for s, g in zip(sigmas, gs)]

  # 极性：board_gray 给定时默认「浅板 + 深字」（白纸黑字），
  # 再按 polarity_flip_rate 混入少量反极性样本（深板 + 亮字）
  if board_gray is None:
    polarity = "any"                       # 旧行为：由背景亮度决定
    tone_range = None
  else:
    flip = bool(polarity_flip_rate) and random.random() < float(polarity_flip_rate)
    polarity = "up" if flip else "down"
    tone_range = board_gray_flip if (flip and board_gray_flip is not None) \
        else board_gray

  # 先把板画上（板面和数字用同一个 sigma 失焦），数字才是“写在板上”的
  if board_tone is not None or tone_range is not None:
    for seg, sg in zip(segs, sigmas):
      out = draw_board(out, seg.quad, tone_delta=board_tone,
                       target_gray=None if tone_range is None
                       else _pick_tone(tone_range),
                       feather=board_feather, shadow=board_shadow,
                       sigma=sg, persp_spread=persp_spread)

  infos = []
  for i, (glyph, seg, dquad) in enumerate(zip(gs, segs, dquads)):
    sigma = sigmas[i]
    # 贴图就是该四边形的透视变换结果，用四边形形状决定块内的远近方向
    ramp = quad_perspective_ramp(dquad, (glyph.shape[1], glyph.shape[0]))
    layer = defocus_glyph(glyph, sigma, ramp, persp_spread)

    offset = (float(dquad[:, 0].min()), float(dquad[:, 1].min()))
    contrast_v = pick_contrast(contrast)
    bg = local_background(out, layer, offset)
    gray_v = contrast_gray(bg, contrast_v, polarity=polarity)
    alpha_v = pick_alpha(alpha)
    # 可见性保底：低对比 × 低透明度再叠加失焦，会产出“几乎看不见但仍有标签”
    # 的噪声样本（直接抬高 heatmap / offset 的 loss 地板）；太淡就拉满透明度
    if min_visible > 0 and abs(gray_v - bg) * alpha_v < float(min_visible) * 255.0:
      alpha_v = 1.0
      gray_v = contrast_gray(bg, max(float(contrast_v), 0.5), polarity=polarity)

    out = fuse_img(out, layer, offset, gray=gray_v, alpha=alpha_v)
    infos.append({
        "depth": float(depths[i]),
        "sigma": float(sigma),
        "contrast": float(contrast_v),
        "gray": int(gray_v),
        "bg": float(bg),
        "alpha": float(alpha_v),
        "on_base": bool(on_base[i]),
        "offset": offset,
        "quad": seg.quad.tolist(),
        "digit_quad": dquad.tolist(),
    })

  return out, infos


if __name__ == "__main__":

  img_path = RAW_DIR / "env/0036.jpg"
  img = read_image(img_path)
  assert img is not None, f"无法读取图片: {img_path}"

  quads = get_flat_quad(img)

  if quads is None:
    _logger.debug("未检测到符合条件的四边形区域")
  else:
    vis = img.copy()
    for i, quad in enumerate(quads):
      segs = segment_quad(quad, distortion=(0.1, 0.3), normalize=(
          0.5, 0.8), area_range=(1000, 30000))
      print(f"Quad {i+1}: {len(segs)} segments")
      for seg in segs:
        cv2.polylines(vis, [np.round(seg.quad).astype(np.int32)],
                      True, (0, 255, 0), 2)
    cv_show(vis, f"Quad {len(quads)} segments")
