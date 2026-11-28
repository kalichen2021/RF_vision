import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from configs.log import LogConfig
from configs.path import PathConfig as pathcfg
from configs.preprocess_cfg import PreprocessConfig as ppcfg
from src.utils.debug_tools import cv_show, hstack_images
from src.utils.feature_extration import (fuse_scene, get_flat_quad, inset_quad,
                                        segment_quad)
from src.utils.font_processor import font2img, get_font_path
from src.utils.geometry import QuadSegment, proj_tran, quad2M
from src.utils.logging_tools import get_logger
from src.utils.typing_tools import is_number, random_choose
from src.preprocess.perspective import resample_pose

# 本次运行的标识（用作产物目录名）
_UUID = LogConfig.UUID
# 只发日志不 print；调试图目录由调用方（CLI）通过 debug_dir 传入，
# 模块被 import 时不再创建任何目录。
_logger = get_logger(__name__)


def build_record(name: str, img: NDArray[np.uint8], labels: list[dict],
                 extra: dict | None = None) -> dict:
  """构造一条样本标签记录（`image` / `width` / `height` / extra / `objects`）

  Args:
      name (str): 图片文件名（含扩展名）
      img (NDArray[np.uint8]): 样本图（用它的尺寸填 width / height）
      labels (list[dict]): `Preprocess.process` 返回的标签
      extra (dict | None): 附加元信息（如源图名、参数快照）

  Returns:
      dict: 可直接写进合并标签文件的记录
  """
  h, w = img.shape[:2]
  record = {"image": str(name), "width": int(w), "height": int(h)}
  if extra:
    record.update(extra)
  record["objects"] = labels

  return record


def save_sample(out_dir: Path | str, name: str, img: NDArray[np.uint8],
                labels: list[dict], extra: dict | None = None,
                write_labels: bool = True,
                size: tuple[int, int] | None = None,
                ext: str = "png") -> tuple[Path, Path | None]:
  """保存一个训练样本：灰度 PNG（可选：单个 JSON 标签）

  Args:
      out_dir (Path | str): 输出目录
      name (str): 文件名（不含扩展名）
      img (NDArray[np.uint8]): 合成图（三通道会自动转灰度）
      labels (list[dict]): `Preprocess.process` 返回的标签（digit / quad / ...）
      extra (dict | None): 附加元信息（如源图名、参数快照）
      write_labels (bool): 是否额外写一个同名 JSON；批量生成时置 False，
          改用 `save_dataset` 把所有标签合成一个文件

  Returns:
      tuple[Path, Path | None]: (图片路径, 标签路径)；`write_labels=False` 时标签路径为 None
  """
  out_dir = Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  if size is not None:
    # cv2.resize 的尺寸顺序是 (宽, 高)；已经等于目标尺寸就不重复缩放
    target = (int(size[1]), int(size[0]))
    if (gray.shape[1], gray.shape[0]) != target:
      gray = cv2.resize(gray, target, interpolation=cv2.INTER_AREA)
  ext = str(ext).lower().lstrip(".")
  img_path = out_dir / f"{name}.{ext}"
  params = [cv2.IMWRITE_JPEG_QUALITY, 95] if ext in ("jpg", "jpeg") else []
  cv2.imencode(f".{ext}", gray, params)[1].tofile(str(img_path))

  lbl_path = None
  if write_labels:
    lbl_path = out_dir / f"{name}.json"
    lbl_path.write_text(
        json.dumps(build_record(img_path.name, gray, labels, extra),
                   ensure_ascii=False, indent=2),
        encoding="utf-8")

  return img_path, lbl_path


def save_dataset(out_dir: Path | str, records: list[dict],
                 filename: str = "labels.json") -> Path:
  """把整批样本的标签合并写成一个 JSON（`{"count": n, "samples": [...]}`）

  Args:
      out_dir (Path | str): 输出目录（与图片同目录）
      records (list[dict]): `build_record` 产出的记录列表
      filename (str): 合并标签文件名

  Returns:
      Path: 合并标签文件路径
  """
  out_dir = Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  lbl_path = out_dir / filename
  lbl_path.write_text(
      json.dumps({"count": len(records), "samples": list(records)},
                 ensure_ascii=False, indent=2),
      encoding="utf-8")

  return lbl_path


def _parse_margin_range(value: float | tuple[float, float]
                        | tuple[tuple[float, float], tuple[float, float]]
                        ) -> tuple[tuple[float, float], tuple[float, float]]:
  """把留边参数统一成 `((横_lo, 横_hi), (纵_lo, 纵_hi))`。

  支持三种写法（都做 0~0.45 的夹取，并保证区间端点有序）：

  - 标量 ``0.1`` → ``((0.1, 0.1), (0.1, 0.1))``
  - ``(lo, hi)`` → 横、纵共用同一区间
  - ``((横_lo, 横_hi), (纵_lo, 纵_hi))`` → 分别指定

  Args:
      value: 上面任意一种写法。

  Returns:
      `((横_lo, 横_hi), (纵_lo, 纵_hi))`。

  Raises:
      ValueError: 结构无法识别。
  """
  def _clip(pair: tuple[float, float]) -> tuple[float, float]:
    lo, hi = sorted(float(v) for v in pair)
    return max(0.0, lo), min(0.45, hi)

  if isinstance(value, (int, float)) and not isinstance(value, bool):
    v = float(value)
    return _clip((v, v)), _clip((v, v))
  seq = list(value)
  if len(seq) == 2 and all(isinstance(v, (int, float)) for v in seq):
    rng = _clip((seq[0], seq[1]))          # type: ignore[arg-type]
    return rng, rng
  if len(seq) == 2 and all(isinstance(v, (tuple, list)) for v in seq):
    x = _clip(tuple(seq[0]))               # type: ignore[arg-type]
    y = _clip(tuple(seq[1]))               # type: ignore[arg-type]
    return x, y
  raise ValueError(
      f"board_margin 应为标量 / (lo, hi) / ((横lo, 横hi), (纵lo, 纵hi))，"
      f"收到 {value!r}")


class Preprocess:
  """把数字贴到环境图的平面上（合成样本生成器）

  Args:
      board_margin (float | tuple): 数字在板面内的留边比例。三种写法：

          - 标量 ``0.1``：横纵都是 0.1；
          - ``(lo, hi)``：横纵同一个区间；
          - ``((横_lo, 横_hi), (纵_lo, 纵_hi))``：宽、高**分别**给区间。

          每张图横/纵各抽一档，同一张图内所有数字共用。
  """

  def __init__(self, distortion: tuple[float, float],
               normalize: tuple[float, float],
               area_range: tuple[float, float],
               quads_select_rate: float,
               contrast_range: tuple[float, float] = ppcfg.CONTRAST_RANGE,
               fuse_alpha: tuple[float, float] = ppcfg.FUSE_ALPHA,
               defocus_strength: float = ppcfg.DEFOCUS_STRENGTH,
               defocus_max_sigma: float = ppcfg.DEFOCUS_MAX_SIGMA,
               persp_spread: float = ppcfg.DEFOCUS_PERSPECTIVE,
               base_depth: float | None = None,
               plane_tol: float = ppcfg.PLANE_TOL,
               board_tone: tuple[float, float] = ppcfg.BOARD_TONE,
               board_margin: float | tuple[float, float]
               | tuple[tuple[float, float], tuple[float, float]]
               = ppcfg.BOARD_MARGIN,
               board_shadow: bool = ppcfg.BOARD_SHADOW,
               defocus_size_ratio: float = ppcfg.DEFOCUS_SIZE_RATIO,
               digit_count_range: tuple[int, int] | None = ppcfg.SAMPLE_DIGIT_RANGE,
               board_gray: tuple[int, int] | None = ppcfg.BOARD_GRAY,
               board_gray_flip: tuple[int, int] = ppcfg.BOARD_GRAY_FLIP,
               polarity_flip_rate: float = ppcfg.POLARITY_FLIP_RATE,
               min_visible_contrast: float = ppcfg.MIN_VISIBLE_CONTRAST,
               pose_resample: bool = ppcfg.POSE_RESAMPLE,
               pose_distance: tuple[float, float] = ppcfg.POSE_DISTANCE,
               pose_rot_x: tuple[float, float] = ppcfg.POSE_ROT_X,
               pose_rot_y: tuple[float, float] = ppcfg.POSE_ROT_Y,
               pose_rot_z: tuple[float, float] = ppcfg.POSE_ROT_Z,
               pose_area_jitter: tuple[float, float] = ppcfg.POSE_AREA_JITTER,
               pose_area_mode: str = ppcfg.POSE_AREA_MODE,
               pose_blend: float = ppcfg.POSE_BLEND
               ) -> None:
    self.config = {
        "distortion": distortion,
        "normalize": normalize,
        "area_range": area_range
    }
    self.quads_select_rate = quads_select_rate
    # 每张样本的数字个数范围（含两端）；None 表示用 quads_select_rate 的旧逻辑
    if digit_count_range is None:
      self.digit_count_range = None
    elif isinstance(digit_count_range, (int, float)):
      self.digit_count_range = (int(digit_count_range),
                                int(digit_count_range))
    else:
      lo, hi = sorted(int(v) for v in digit_count_range)
      self.digit_count_range = (max(1, lo), max(1, hi))
    # 数字融合 / 景深失焦 配置
    self.contrast_range = contrast_range
    self.fuse_alpha = fuse_alpha
    self.defocus_strength = defocus_strength
    self.defocus_max_sigma = defocus_max_sigma
    self.persp_spread = persp_spread
    self.base_depth = base_depth
    self.plane_tol = plane_tol
    # 板面渲染配置
    self.board_tone = board_tone
    self.board_margin = board_margin
    self.board_shadow = board_shadow
    self.defocus_size_ratio = defocus_size_ratio
    # 板面 / 极性：默认以「白纸黑字」为主（BOARD_GRAY 是绝对灰度，不受环境亮度影响）
    self.board_gray = board_gray
    self.board_gray_flip = board_gray_flip
    self.polarity_flip_rate = float(polarity_flip_rate)
    self.min_visible_contrast = float(min_visible_contrast)
    # 板面姿态（透视强弱 + 三轴旋转）：只影响板面四边形形状
    self.pose_resample = bool(pose_resample)
    self.pose_distance = pose_distance
    self.pose_rot_x = pose_rot_x
    self.pose_rot_y = pose_rot_y
    self.pose_rot_z = pose_rot_z
    self.pose_area_jitter = pose_area_jitter
    self.pose_area_mode = str(pose_area_mode).lower()
    self.pose_blend = float(pose_blend)
    # 数字留边：横/纵分别可以是固定值或区间；每张图各抽一档，同一张图内保持一致
    self.margin_range = _parse_margin_range(board_margin)
    self.board_margin = board_margin
    # AREA_RANGE 描述的是「板位（板面）」面积，不再随留边放大：
    # 留边只决定数字在板内的大小
    # （数字面积 = 板位面积 × (1-2*横留边) × (1-2*纵留边)），
    # 这样换留边不会把板撑大、不会把数字个数稀释掉。
    self.area_scale = 1.0

  def _pick_margin(self) -> tuple[float, float]:
    """每张图抽一档数字留边（同一张图内所有数字共用）。

    Returns:
        ``(横向留边, 纵向留边)``，均已在 ``margin_range`` 范围内；
        区间退化时直接返回固定值。
    """
    (ux_lo, ux_hi), (vy_lo, vy_hi) = self.margin_range
    mx = ux_lo if ux_hi <= ux_lo else random.uniform(ux_lo, ux_hi)
    my = vy_lo if vy_hi <= vy_lo else random.uniform(vy_lo, vy_hi)
    return mx, my

  def detect_quads(self, env_img: NDArray[np.uint8]) -> list[NDArray[np.float64]]:
    """检测环境图上的平坦区域（同一张图结果确定，批量生成时可缓存复用）

    Args:
        env_img (NDArray[np.uint8]): 环境图（BGR 或灰度）

    Returns:
        list[NDArray[np.float64]]: 平坦区域四边形列表，可能为空
    """
    gray = env_img if env_img.ndim == 2 else cv2.cvtColor(
        env_img, cv2.COLOR_BGR2GRAY)
    return get_flat_quad(gray)

  def _quad_capacity(self, quads: list[NDArray[np.float64]],
                     area_scale: float | None = None
                     ) -> list[tuple[int, NDArray[np.float64]]]:
    """估算各平坦区域最多能放几块板（按最小板位面积）

    Returns:
        list[tuple[int, NDArray]]: `(容量, 四边形)`，按容量从大到小排序
    """
    raw_lo = float(min(self.config["area_range"]))
    scale = self.area_scale if area_scale is None else float(area_scale)
    capacity = []
    for q in quads:
      quad_area = abs(cv2.contourArea(np.asarray(q, dtype=np.float32)))
      board_lo = (raw_lo * quad_area if raw_lo < 1 else raw_lo) / scale
      # 最小 gap(0.12) 会再吃掉约 23% 面积，再留 10% 给切割线抖动
      footprint = max(board_lo, 1.0) / ((1 - 0.12) ** 2)
      cap = int(quad_area // max(footprint * 1.1, 1.0))
      if cap > 0:
        capacity.append((cap, q))
    capacity.sort(key=lambda item: item[0], reverse=True)
    return capacity

  def _build_segments(self, quads: list[NDArray[np.float64]],
                      target_count: int,
                      capacity: list[tuple[int, NDArray[np.float64]]] | None = None,
                      area_scale: float | None = None
                      ) -> list:
    """按「这张图要放 target_count 个数字」分配板位

    每个平坦区域的容量 ≈ 区域面积 / 单块最小板位面积。按容量从大到小分配
    名额，再让 `segment_quad(min_cells=...)` 切出足够多的板，避免“只想切
    一块大板”导致数字个数总是偏少。

    Returns:
        list[QuadSegment]: 切出的板位（数量可能多于或少于 target_count）
    """
    if capacity is None:
      capacity = self._quad_capacity(quads, area_scale)
    scale = self.area_scale if area_scale is None else float(area_scale)

    segments = []
    for cap, q in capacity:
      remaining = int(target_count) - len(segments)
      if remaining <= 0:
        break
      want = min(cap, remaining)
      # 切分带随机性（长宽比 / 网格 / gap），同一块区域多抽几次数出最多的一次，
      # 避免一次抽到偏粗的网格导致「只要 6 块却只切出 3 块」
      best: list = []
      for _ in range(3 if want > 1 else 1):
        got = segment_quad(q, **self.config, area_scale=scale,
                           min_cells=want)
        if len(got) > len(best):
          best = got
        if len(best) >= want:
          break
      if best:
        segments.extend(best)
    return segments

  def process(self, _font_imgs: list[NDArray[np.uint8]],
              env_img: NDArray[np.uint8],
              show_cnt: bool = False,
              quads: list[NDArray[np.float64]] | None = None,
              digit_ids: list[int] | None = None,
              debug_dir: Path | None = None
              ) -> tuple[NDArray[np.uint8] | None, list[dict]]:
    """把数字贴到环境图的平面上，返回 (灰度合成图, 标签列表)

    标签按“板面 → 数字内缩四边形”的顺序给出，每个元素包含：
    `digit`（数字类别，取自 `digit_ids`）、`quad`（数字的透视四边形，
    [左上, 右上, 右下, 左下]，绝对像素坐标）、`quad_norm`（归一化到 0~1）、
    `board_quad`（板面四边形）、`depth` / `sigma`（合成时用的深度与失焦量）。

    Args:
        _font_imgs (list[NDArray[np.uint8]]): 白字黑底的数字贴图池
        env_img (NDArray[np.uint8]): 环境图（BGR 或灰度）
        show_cnt (bool, optional): 是否把中间结果写进日志目录。Defaults to False.
        quads (list[NDArray[np.float64]] | None, optional): 预先算好的平坦区域（可缓存复用）；
            缺省时在本次调用里检测。Defaults to None.
        digit_ids (list[int] | None, optional): `_font_imgs` 每张图对应的数字类别
            （一个数字可以有多个字形变体）；缺省时直接用池下标，即「池下标 = 类别」。
            Defaults to None.

    Returns:
        tuple[NDArray[np.uint8] | None, list[dict]]: (灰度合成图, 标签)；
        找不到可用平面时返回 `(None, [])`
    """
    gray = env_img if env_img.ndim == 2 else cv2.cvtColor(
        env_img, cv2.COLOR_BGR2GRAY)

    # 缺省契约：池下标就是数字类别；池子里有同数字的多个字形时必须显式传 digit_ids
    if digit_ids is None:
      digit_ids = list(range(len(_font_imgs)))
    elif len(digit_ids) != len(_font_imgs):
      raise ValueError(
          f"digit_ids 长度({len(digit_ids)})与贴图池({len(_font_imgs)})不一致")

    if quads is None:
      quads = get_flat_quad(gray)
    if not quads:
      _logger.debug("未检测到符合条件的平面区域")
      return None, []

    # 每张图横/纵各抽一档留边：同一张图的板边留白一致；板位大小不受留边影响
    margin = self._pick_margin()
    area_scale = 1.0

    if self.digit_count_range is None:
      # 旧逻辑：按面积切出全部板位，再按比例挑一部分
      segments = [s for q in quads
                  for s in segment_quad(q, **self.config,
                                        area_scale=area_scale)]
      if not segments:
        _logger.debug("分割后没有符合 distortion/area 条件的四边形")
        return None, []
      boxed_text_count = min(len(segments), int(
          len(segments) * self.quads_select_rate + 1))
      segments = random_choose(segments, boxed_text_count)
    else:
      # 新逻辑：先按容量算这张图最多能放几个，再在可行区间内取目标个数
      k_lo, k_hi = self.digit_count_range
      capacity = self._quad_capacity(quads, area_scale)
      max_count = sum(cap for cap, _ in capacity)
      if max_count < k_lo:
        _logger.debug("板位不足：需要 %d~%d 个数字，容量只有 %d 块",
                      k_lo, k_hi, max_count)
        return None, []
      target_count = random.randint(k_lo, min(k_hi, max_count))
      segments = self._build_segments(quads, target_count, capacity, area_scale)
      if len(segments) < k_lo:
        # 高目标放不下时，至少保证能满足区间下限（再不行就换图重试）
        segments = self._build_segments(quads, k_lo, capacity, area_scale)
      if len(segments) < k_lo:
        _logger.debug("板位不足：需要 %d~%d 个数字，只能切出 %d 块",
                      k_lo, k_hi, len(segments))
        return None, []
      segments = random_choose(segments, min(target_count, len(segments)))

    # 板面姿态重采样：把「环境图里检测到的平坦区域形状」换成
    # 「等效相机距离 + 三轴倾角」下的矩形透视投影（中心/面积基本不变）。
    # 只改板面四边形的形状，数字四边形仍由 inset_quad 从它派生，标签格式不变。
    if self.pose_resample and segments:
      h_img, w_img = gray.shape[:2]
      segments = [QuadSegment(resample_pose(
          seg.quad,
          distance=self.pose_distance,
          rot_x=self.pose_rot_x, rot_y=self.pose_rot_y,
          rot_z=self.pose_rot_z,
          area_jitter=self.pose_area_jitter,
          area_mode=self.pose_area_mode,
          blend=self.pose_blend,
          bounds=(w_img, h_img))) for seg in segments]

    # 数字也从贴图池里随机取（类别由 digit_ids 给出）
    boxed_text_count = len(segments)
    font_idx = random_choose(list(range(len(_font_imgs))), boxed_text_count)
    font_imgs = [_font_imgs[i] for i in font_idx]

    # 数字贴在板面内部（按本图抽到的 margin 留边），并按内缩四边形做透视变换
    digit_quads = [inset_quad(seg.quad, margin) for seg in segments]
    traned_font_imgs = [
        np.array(proj_tran(_img, quad2M(dq, *_img.shape)))
        for _img, dq in zip(font_imgs, digit_quads)
    ]

    # 画板 + 融合 + 按基准平面做景深失焦
    fused, infos = fuse_scene(
        gray, traned_font_imgs, segments,
        contrast=self.contrast_range, alpha=self.fuse_alpha,
        base_depth=self.base_depth, plane_tol=self.plane_tol,
        defocus_strength=self.defocus_strength,
        max_sigma=self.defocus_max_sigma,
        persp_spread=self.persp_spread,
        digit_quads=digit_quads,
        board_tone=self.board_tone,
        board_gray=self.board_gray,
        board_gray_flip=self.board_gray_flip,
        polarity_flip_rate=self.polarity_flip_rate,
        min_visible=self.min_visible_contrast,
        board_shadow=self.board_shadow,
        size_ratio=self.defocus_size_ratio)

    h, w = fused.shape[:2]
    labels = []
    for fidx, info in zip(font_idx, infos):
      q = np.asarray(info["digit_quad"], dtype=np.float64)
      labels.append({
          "digit": int(digit_ids[fidx]),
          "quad": [[round(float(x), 2), round(float(y), 2)] for x, y in q],
          "quad_norm": [[round(float(x) / max(w - 1, 1), 4),
                         round(float(y) / max(h - 1, 1), 4)] for x, y in q],
          "board_quad": [[round(float(x), 2), round(float(y), 2)]
                         for x, y in np.asarray(info["quad"], dtype=np.float64)],
          "center": [round(float(q[:, 0].mean()), 2),
                     round(float(q[:, 1].mean()), 2)],
          "depth": round(float(info["depth"]), 4),
          "sigma": round(float(info["sigma"]), 3),
      })

    # DEBUG: 显示板面 / 数字四边形与失焦强度（debug_dir 由 CLI 传入，不传则不落盘）
    if show_cnt and debug_dir is not None:
      debug_dir = Path(debug_dir)
      debug_dir.mkdir(parents=True, exist_ok=True)
      vis_env = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
      for info in infos:
        # 绿 = 基准平面（清晰），红 = 非基准平面（按 sigma 失焦）
        color = (0, 255, 0) if info["on_base"] else (0, 0, 255)
        board = np.round(np.asarray(info["quad"])).astype(np.int32)
        digit = np.round(np.asarray(info["digit_quad"])).astype(np.int32)
        cv2.polylines(vis_env, [board], True, color, 1, cv2.LINE_AA)
        cv2.polylines(vis_env, [digit], True, (255, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(vis_env, f"s={info['sigma']:.1f}",
                    (int(info["offset"][0]), max(
                        int(info["offset"][1]) - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
      cv2.imencode('.png', vis_env)[1].tofile(
          f"{debug_dir}/{vis_env.shape}_env_quads.png")
      cv2.imencode('.png', fused)[1].tofile(
          f"{debug_dir}/{fused.shape}_env_fused.png")

      for i, _ in enumerate(traned_font_imgs):
        # proj_tran 的输出画布 = 目标 quad 的外接矩形, 且已把坐标平移 (0,0),
        # 所以要先减去外接矩形左上角, 得到画布内的局部坐标才能画上框
        q = digit_quads[i]
        q_local = np.round(q - q.min(axis=0)).astype(np.int32)
        det = np.array(traned_font_imgs[i], copy=True)
        if det.ndim == 2:
          det = cv2.cvtColor(det, cv2.COLOR_GRAY2BGR)
        cv2.polylines(det, [q_local], True, (0, 0, 255), 1, cv2.LINE_AA)
        _img = hstack_images([font_imgs[i], det], pad=5)
        cv2.imencode('.png', _img)[1].tofile(
            f"{debug_dir}/{_img.shape}_projed_{i}.png")

    return fused, labels


if __name__ == "__main__":
  PP = Preprocess(ppcfg.DISTORTION, ppcfg.NORMALIZE, ppcfg.AREA_RANGE,
                  ppcfg.QUADS_SELECT_RATE,
                  ppcfg.CONTRAST_RANGE, ppcfg.FUSE_ALPHA, ppcfg.DEFOCUS_STRENGTH,
                  ppcfg.DEFOCUS_MAX_SIGMA, ppcfg.DEFOCUS_PERSPECTIVE,
                  ppcfg.BASE_DEPTH, ppcfg.PLANE_TOL,
                  ppcfg.BOARD_TONE, ppcfg.BOARD_MARGIN, ppcfg.BOARD_SHADOW,
                  ppcfg.DEFOCUS_SIZE_RATIO)
  env_img_path = pathcfg.RAW_DIR / "env/0072.jpg"
  font_imgs = [font2img(get_font_path(0), str(i), thickness=1)
               for i in range(10)]
  env_img = cv2.imdecode(np.fromfile(
      str(env_img_path), dtype=np.uint8), cv2.IMREAD_COLOR)

  fused, labels = PP.process(font_imgs, env_img, True)
  if fused is not None:
    img_path, lbl_path = save_sample(
        pathcfg.PROCESSED_DIR / "fused", env_img_path.stem, fused, labels,
        extra={"source": env_img_path.name})
    print(f"样本: {img_path}\n标签: {lbl_path} ({len(labels)} 个数字)")
