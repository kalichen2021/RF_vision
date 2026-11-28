"""板面姿态采样：按「等效相机距离 + 三轴旋转」重采样板面四边形（仅用于生成）。

现有链路里，板面四边形直接来自**环境图里检测到的平坦区域**，所以它的透视强弱
完全由照片本身的拍摄距离决定，无法调节。本模块把这块形状"重算一遍"：

1. 取原四边形的中心与面积（保持不变）；
2. 在 3D 中摆一个与它面积相同、长宽比相同的矩形；
3. 按 `rot_x / rot_y / rot_z` 旋转，放在距离 `distance × 短边` 的等效相机前；
4. 用针孔投影得到四边形，再缩放/平移对齐原来的中心与面积。

于是：

- `distance` 越大 → 越像"长焦远距离拍摄"，透视越弱（灭点越远）；
- `rot_x / rot_y / rot_z` 越大 → 板面倾斜越明显（俯仰 / 偏航 / 面内滚转）。

只改板面四边形的**形状**，标签仍由它派生，所以 `labels.json` 格式完全不变。
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from numpy.typing import NDArray

from src.utils.geometry import order_quad

# 三轴旋转（度）→ 3x3 旋转矩阵（x=俯仰, y=偏航, z=面内滚转；顺序 Rz @ Ry @ Rx）
def _rot_matrix(rx: float, ry: float, rz: float) -> NDArray[np.float64]:
  """把三轴角度（弧度）合成一个旋转矩阵。

  Args:
      rx: 绕 X 轴（俯仰）。
      ry: 绕 Y 轴（偏航）。
      rz: 绕 Z 轴（面内滚转）。

  Returns:
      `(3, 3)` 旋转矩阵 `Rz @ Ry @ Rx`。
  """
  cx, sx = math.cos(rx), math.sin(rx)
  cy, sy = math.cos(ry), math.sin(ry)
  cz, sz = math.cos(rz), math.sin(rz)
  mx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], np.float64)
  my = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], np.float64)
  mz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], np.float64)
  return mz @ my @ mx


def _is_convex(quad: NDArray[np.floating], eps: float = 1e-9) -> bool:
  """判断四个点是否构成凸多边形（相邻边叉积同号）。

  Args:
      quad: `(4, 2)` 点列（顺序需为环形）。
      eps: 判定为零的阈值。

  Returns:
      True 表示凸（含退化到一条线的情况返回 False）。
  """
  q = np.asarray(quad, np.float64)
  signs = []
  for i in range(4):
    a = q[(i + 1) % 4] - q[i]
    b = q[(i + 2) % 4] - q[(i + 1) % 4]
    cross = float(a[0] * b[1] - a[1] * b[0])
    if abs(cross) <= eps:
      return False
    signs.append(cross > 0)
  return all(signs) or not any(signs)


def _shrink_to_bounds(quad: NDArray[np.floating], wh: tuple[int, int],
                      margin: float = 1.0) -> NDArray[np.float64]:
  """绕质心等比缩小四边形，使其全部落在图像内（保持形状不变）。

  Args:
      quad: `(4, 2)` 点列。
      wh: 图像尺寸 `(宽, 高)`。
      margin: 距边界的留白（像素）。

  Returns:
      缩小后（或原样）的四边形。
  """
  q = np.asarray(quad, np.float64)
  w, h = float(wh[0]), float(wh[1])
  c = q.mean(axis=0)
  d = q - c
  lo_x, hi_x = margin, max(w - 1.0 - margin, margin)
  lo_y, hi_y = margin, max(h - 1.0 - margin, margin)
  s = 1.0
  for dx, dy in d:
    if dx > 1e-9:
      s = min(s, (hi_x - c[0]) / dx)
    elif dx < -1e-9:
      s = min(s, (lo_x - c[0]) / dx)
    if dy > 1e-9:
      s = min(s, (hi_y - c[1]) / dy)
    elif dy < -1e-9:
      s = min(s, (lo_y - c[1]) / dy)
  s = float(np.clip(s, 1e-3, 1.0))
  return c + d * s


def resample_pose(quad: NDArray[np.floating],
                  distance: tuple[float, float] = (3.0, 9.0),
                  rot_x: tuple[float, float] = (0.0, 0.0),
                  rot_y: tuple[float, float] = (0.0, 0.0),
                  rot_z: tuple[float, float] = (0.0, 0.0),
                  area_jitter: tuple[float, float] = (1.0, 1.0),
                  area_mode: str = "keep",
                  blend: float = 1.0,
                  bounds: tuple[int, int] | None = None,
                  rng: np.random.Generator | None = None,
                  tries: int = 4) -> NDArray[np.float64]:
  """把任意四边形重采样成「等效距离 + 三轴倾角」下的矩形透视投影。

  中心保持与原四边形一致；面积由 `area_mode` 决定：

  - `"keep"`：把面积拉回原板面（只乘 `area_jitter`），板面大小分布稳定；
  - `"natural"`：不做面积补偿，保留透视压缩（大角度偏航/俯仰会真的变窄变偏）。

  Args:
      quad: 输入的板面四边形 `(4, 2)`，顺序 [左上, 右上, 右下, 左下]。
      distance: 相机距离 / 板面特征尺寸（sqrt(面积)）的区间。
          越大透视越弱（灭点越远、越像长焦）；越小透视越强。取值须 > 1，否则板会穿过相机。
      rot_x: 俯仰角区间（度），正负号决定往哪边倒。
      rot_y: 偏航角区间（度）。
      rot_z: 面内滚转区间（度）。
      area_jitter: 面积倍率区间（`"keep"` 时是相对原面积；`"natural"` 时是额外缩放）。
      area_mode: `"keep"`（默认）或 `"natural"`，见上图说明。
      blend: 0 = 完全沿用原形状，1 = 完全使用新形状；中间值做顶点线性插值。
      bounds: 图像尺寸 `(宽, 高)`；给出时会把结果缩进画布内。
      rng: 随机数发生器，None 时新建（可传入以保证可复现）。
      tries: 采样失败（退化/非凸/出界）时的重试次数。

  Returns:
      新的板面四边形 `(4, 2)`；全部失败时返回原四边形（形状已排序）。

  Raises:
      ValueError: `area_mode` 不是 `keep` / `natural`。
  """
  mode = str(area_mode).lower()
  if mode not in ("keep", "natural"):
    raise ValueError(f"area_mode 应为 'keep' 或 'natural'，收到 {area_mode!r}")
  rng = np.random.default_rng() if rng is None else rng
  q0 = order_quad(np.asarray(quad, np.float64), reset=False)
  c0 = q0.mean(axis=0)
  a0 = abs(cv2.contourArea(q0.astype(np.float32)))
  if a0 < 4.0:                       # 太小的区域不值得折腾
    return q0

  # 用对边平均长度估计长宽比；面积随后统一归一，所以这里只需要比例
  w0 = 0.5 * (float(np.linalg.norm(q0[1] - q0[0]))
              + float(np.linalg.norm(q0[2] - q0[3])))
  h0 = 0.5 * (float(np.linalg.norm(q0[3] - q0[0]))
              + float(np.linalg.norm(q0[2] - q0[1])))
  rect_w = max(w0, 1e-6)
  rect_h = max(h0, 1e-6)
  rect = np.array([[0.0, 0.0, 0.0], [rect_w, 0.0, 0.0],
                   [rect_w, rect_h, 0.0], [0.0, rect_h, 0.0]], np.float64)
  rect -= np.array([rect_w / 2.0, rect_h / 2.0, 0.0])

  feature = max(math.sqrt(a0), 1e-6)          # 板面特征尺寸
  blend = float(np.clip(blend, 0.0, 1.0))

  for _ in range(max(1, int(tries))):
    d_lo, d_hi = sorted(float(v) for v in distance)
    d = float(rng.uniform(d_lo, d_hi))
    if d <= 1.05:                # 相机贴在板前会投影到无穷远，直接判废
      continue
    # 相机在 z 方向退开 d*feature，焦距取同样的值 → 正对时投影尺寸≈1
    f = d * feature
    rot = _rot_matrix(
        math.radians(float(rng.uniform(*sorted(float(v) for v in rot_x)))),
        math.radians(float(rng.uniform(*sorted(float(v) for v in rot_y)))),
        math.radians(float(rng.uniform(*sorted(float(v) for v in rot_z)))))
    p = rect @ rot.T                          # (4,3)
    z = p[:, 2] + f
    if np.any(z <= 1e-6):
      continue
    proj = np.stack([f * p[:, 0] / z, -f * p[:, 1] / z], axis=1)   # 图像 y 向下
    q = order_quad(proj, reset=False)
    if not _is_convex(q):
      continue
    aq = abs(cv2.contourArea(q.astype(np.float32)))
    if aq < 1e-6:
      continue

    j_lo, j_hi = sorted(float(v) for v in area_jitter)
    jitter = float(rng.uniform(j_lo, j_hi))
    if mode == "keep":
      # 面积对齐原四边形：大角度旋转带来的透视压缩会被这里的放大抵消
      k = math.sqrt(a0 * jitter / aq)
    else:
      # 保留透视压缩：只应用面积抖动，转得越偏就真的越小
      k = math.sqrt(jitter)
    q = (q - q.mean(axis=0)) * k + c0        # 中心对齐原四边形
    if blend < 1.0:
      q = (1.0 - blend) * q0 + blend * q
    if bounds is not None:
      q = _shrink_to_bounds(q, (int(bounds[0]), int(bounds[1])))
    return q

  return q0
