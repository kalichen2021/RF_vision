"""摄像头实时检测：cv2 采集 → 裁切/缩放 → CenterNet 推理 → 解码 → 画面叠加。

推理交给 `src/inference/infer.py::Inferer`，本模块只负责摄像头读写、
单帧预处理、结果绘制和主循环。

**显示与模型输入是同一张图**：先按 `rotate` 旋转，再按 `crop_square` 居中裁成
正方形，裁剪后的正方形才送进模型。预览默认只显示这块正方形（`view="crop"`），
所以窗口长宽比始终是 1:1，和模型实际看到的画面一致；另外两种视图
（`"model"` / `"full"`）只影响显示，不影响送进模型的图。

用法::

    inf = Inferer(CenterNetDigit(num_classes=10, dropout=DROPOUT), SELECT_MODEL)
    with CameraDetector(inf, index=0, thr=0.3) as cam:
        cam.run()        # 按 q / ESC 退出，按 s 存当前帧，按 r / e 旋转画面

命令行入口见 `src/cli/realtime_infer.py`。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from configs.realtime_cfg import RealtimeConfig
from src.utils.debug_tools import draw_quads, save_image, tensor2img
from src.utils.image import center_origin, crop_square, to_tensor
from src.utils.logging_tools import get_logger

# 送进模型的通道模式与插值全部由 configs/realtime_cfg.py 提供
rtcfg = RealtimeConfig
_logger = get_logger(__name__)

# 预览模式：crop = 只显示裁切后的正方形（默认）、model = 显示模型真正看到的画面、
# full = 显示整幅画面并压暗裁切区域以外的部分
_VIEW_MODES: tuple[str, ...] = ("crop", "model", "full")


class CameraDetector:
  """实时摄像头检测器（采集 + 推理 + 显示）

  Args:
      inferrer: `src.inference.infer.Inferer` 实例（提供 `predict`）
      index (int): 摄像头序号
      img_size (tuple[int, int]): 送进网络的尺寸，必须与训练一致
      width (int | None): 采集宽度，None 用摄像头默认
      height (int | None): 采集高度
      thr (float): 中心分数阈值
      topk (int): 每帧最多检出多少个数字
      show (bool): 是否 `cv2.imshow` 弹窗；False 时只推理不显示
      window_name (str): 窗口标题
      max_fps (float | None): 限制处理帧率，None 表示不限
      save_dir (Path | None): 按 `s` 时的快照目录，None 表示不快照
      thickness (int): 框线宽
      font_scale (float): 标签字号
      rotate (int): 初始顺时针旋转角度（0/90/180/270），显示与推理共用
      input_mode (str | None): 送进模型的通道模式，`"gray"`（灰度复制成 3 通道）
          或 `"rgb"`（保留彩色）；None 时取 `configs.realtime_cfg.INPUT_MODE`
      crop_square (bool | None): 是否把画面居中裁成正方形再送进模型；
          None 时取 `configs.realtime_cfg.CROP_SQUARE`
      view (str | None): 预览模式，`"crop"`（默认，只显示裁切后的正方形，
          长宽比与模型输入一致）/ `"model"`（显示缩放到 IMG_SIZE 后模型
          真正看到的画面）/ `"full"`（显示整幅画面并压暗裁切区域以外）；
          None 时取 `configs.realtime_cfg.VIEW_MODE`
      capture: 自定义 VideoCapture（测试注入用），None 时按 `index` 打开
  """

  def __init__(self, inferrer: Any, index: int = 0,
               img_size: tuple[int, int] = (384, 384),
               width: int | None = 1280, height: int | None = 720,
               thr: float = 0.3, topk: int = 30, show: bool = True,
               window_name: str = "digits_tracker",
               max_fps: float | None = None,
               save_dir: Path | None = None, thickness: int = 2,
               font_scale: float = 0.55, rotate: int = 0,
               input_mode: str | None = None,
               crop_square: bool | None = None,
               view: str | None = None,
               capture: Any = None) -> None:
    self.inferrer = inferrer
    self.img_size = tuple(img_size)
    self.thr = float(thr)
    self.topk = int(topk)
    self.show = bool(show)
    self.window_name = window_name
    self.max_fps = None if max_fps is None else float(max_fps)
    self.save_dir = Path(save_dir) if save_dir else None
    self.thickness = int(thickness)
    self.font_scale = float(font_scale)
    self.rotate = self.normalize_rotate(rotate)
    # 送进模型的通道模式：gray = 灰度复制 3 通道（同训练样本），rgb = 保留彩色
    self.input_mode = str(
        rtcfg.INPUT_MODE if input_mode is None else input_mode).lower()
    # 是否先把画面居中裁成正方形再缩放（False = 直接拉伸，画面会被压扁）
    self.crop_square = bool(
        rtcfg.CROP_SQUARE if crop_square is None else crop_square)
    # 预览模式：默认只显示送进模型的那块正方形，避免 16:9 画面被误认为原样输入
    self.view = str(rtcfg.VIEW_MODE if view is None else view).lower()
    if self.view not in _VIEW_MODES:
      raise ValueError(f"view 应为 {_VIEW_MODES} 之一，收到 {view!r}")

    self.cap = capture if capture is not None else self._open(index, width, height)
    self.frames = 0                 # 已处理帧数
    self.last_frame = None          # 最近一帧（带叠加），供快照使用
    self._t_prev = time.perf_counter()
    self._fps = 0.0

  # ---------------- 资源管理 ----------------
  @staticmethod
  def _open(index: int, width: int | None,
            height: int | None) -> cv2.VideoCapture:
    """打开摄像头并按需设置分辨率（有些相机忽略分辨率设置，属正常）"""
    cap = cv2.VideoCapture(int(index))
    if width:
      cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    if height:
      cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    return cap

  @property
  def is_open(self) -> bool:
    return bool(self.cap is not None and self.cap.isOpened())

  @property
  def fps(self) -> float:
    """最近的处理帧率（指数滑动平均）"""
    return self._fps

  def close(self) -> None:
    """释放摄像头并关闭窗口（可重复调用）"""
    if self.cap is not None:
      self.cap.release()
      self.cap = None
    if self.show:
      cv2.destroyAllWindows()

  def __enter__(self) -> "CameraDetector":
    return self

  def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
    self.close()
    return False

  # ---------------- 旋转 ----------------
  @staticmethod
  def normalize_rotate(angle: int) -> int:
    """把任意角度归一化到 0/90/180/270（顺时针）"""
    angle = int(angle) % 360
    if angle % 90:
      raise ValueError(f"旋转角度必须是 90 的倍数，收到 {angle}")
    return angle

  def set_rotate(self, angle: int) -> int:
    """设置顺时针旋转角度并返回生效值（0/90/180/270）"""
    self.rotate = self.normalize_rotate(angle)
    return self.rotate

  def rotate_frame(self, frame: np.ndarray) -> np.ndarray:
    """按 `self.rotate` 顺时针旋转 BGR 帧；0° 时原样返回

    显示与推理都必须经过这里，保证「看到的就是模型读到的」。
    """
    if self.rotate == 90:
      return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if self.rotate == 180:
      return cv2.rotate(frame, cv2.ROTATE_180)
    if self.rotate == 270:
      return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame

  # ---------------- 单帧处理 ----------------
  def crop_for_model(self, frame: np.ndarray
                     ) -> tuple[np.ndarray, tuple[int, int, int]]:
    """按 `crop_square` 把画面居中裁成正方形（与生成数据集时的裁切一致）

    Args:
        frame: 已按 `self.rotate` 旋转过的 BGR 帧。

    Returns:
        `(模型输入图, (x0, y0, side))`；不需要裁切（关闭裁切 / 画面已方）时
        返回原图与全图区域 `(0, 0, min(H, W))`。
    """
    h, w = frame.shape[:2]
    side = min(h, w)
    if not self.crop_square or h == w:
      return frame, (0, 0, side)
    x0, y0 = center_origin(frame)
    cropped, _ = crop_square(frame, origin=(x0, y0))
    return cropped, (x0, y0, side)

  def frame2tensor(self, frame: np.ndarray) -> torch.Tensor:
    """BGR 帧 → `(1, 3, H, W)` 归一化张量（与训练预处理一致）。

    通道模式（`gray` / `rgb`）与插值都由 `configs.realtime_cfg` 决定，
    具体实现复用 `src.utils.image.to_tensor`。传进来的帧应该已经过
    :meth:`crop_for_model`（正方形），此时内部的 `crop="center"` 是空操作。

    Args:
        frame: `cv2.VideoCapture.read` 拿到的 BGR 帧（或裁切后的正方形）。

    Returns:
        `(1, 3, H, W)` float32 张量。
    """
    return to_tensor(frame, self.img_size, self.input_mode,
                     rtcfg.RESIZE_INTERP,
                     crop="center" if self.crop_square else "none")

  def detect(self, frame: np.ndarray
             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """单帧推理 + 解码（传进来的帧需已按 `self.rotate` 旋转过）

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: `(pts_norm(K,4,2),
        digits(K,), scores(K,))`，坐标为 0~1 归一化
    """
    return self.inferrer.predict(self.frame2tensor(frame),
                                 thr=self.thr, topk=self.topk)[0]

  def draw(self, frame: np.ndarray,
           result: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
           crop_box: tuple[int, int, int] | None = None) -> np.ndarray:
    """把检测结果画到帧上，返回新图（不改入参）

    Args:
        frame: 要绘制的画面（BGR）。
        result: `(pts_norm(K,4,2), digits(K,), scores(K,))`，坐标是**模型输入
            正方形**内的 0~1 归一化值。
        crop_box: `crop_for_model` 返回的 `(x0, y0, side)`；给出且与 `frame`
            尺寸不一致时把坐标映射回整幅画面（`view="full"` 用）。

    Returns:
        np.ndarray: 画好框的 BGR 图。
    """
    pts, digits, scores = result
    pts_np = pts.numpy()
    if crop_box is not None:
      x0, y0, side = crop_box
      h, w = frame.shape[:2]
      # 只有「画面 != 裁切区域」时才需要把 0~1 坐标翻译回整幅画面
      if (x0, y0) != (0, 0) or side != min(h, w):
        norm = np.array([max(w - 1, 1), max(h - 1, 1)], np.float64)
        pts_np = (pts_np * float(side)
                  + np.array([x0, y0], np.float64)) / norm
    return draw_quads(frame, pts_np, digits.tolist(), scores.tolist(),
                      color=(0, 255, 0), tag="pred", thickness=self.thickness,
                      font_scale=self.font_scale)

  @staticmethod
  def mark_crop_box(vis: np.ndarray, box: tuple[int, int, int],
                    dim: float = 0.35) -> np.ndarray:
    """在整幅画面上标出「送进模型的正方形」，框外压暗（返回新图）"""
    x0, y0, side = box
    h, w = vis.shape[:2]
    if (x0, y0) == (0, 0) and side >= min(h, w):
      return vis
    out = vis.astype(np.float32)
    out[:y0] *= dim                       # 上
    out[y0 + side:] *= dim                # 下
    out[y0:y0 + side, :x0] *= dim          # 左
    out[y0:y0 + side, x0 + side:] *= dim   # 右
    out = out.astype(np.uint8)
    cv2.rectangle(out, (x0, y0), (x0 + side - 1, y0 + side - 1),
                  (0, 255, 255), 2)
    return out

  def render(self, frame: np.ndarray, model_img: np.ndarray,
             tensor: torch.Tensor, box: tuple[int, int, int],
             result: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
             ) -> np.ndarray:
    """按 `self.view` 生成预览画面（检测框始终画在对应位置上）

    "crop"：默认，只显示送进模型的那块正方形，长宽比与模型输入一致；
    "model"：显示缩放到 `img_size` 后模型真正看到的画面；
    "full"：显示整幅画面，裁切区域外压暗。

    Args:
        frame: 旋转后的整幅画面。
        model_img: 裁切后的正方形（送进模型的图）。
        tensor: `frame2tensor(model_img)` 的结果。
        box: `crop_for_model` 返回的裁切区域。
        result: `Inferer.predict` 的解码结果。

    Returns:
        np.ndarray: 可显示的 BGR 图。
    """
    if self.view == "model":
      return self.draw(tensor2img(tensor[0]), result)
    if self.view == "full":
      return self.mark_crop_box(self.draw(frame, result, crop_box=box), box)
    return self.draw(model_img, result)

  # ---------------- 主循环 ----------------
  def run(self, max_frames: int | None = None) -> int:
    """持续检测直到退出，返回处理的帧数

    退出条件：按 `q` / `ESC`、摄像头断流、或达到 `max_frames`。

    快捷键：`s` 快照、`r` 顺时针转 90°、`e` 逆时针转 90°
    （旋转同时作用于显示画面和送进模型的图像）。

    Args:
        max_frames (int | None): 处理多少帧后退出，None 表示一直跑
    """
    if not self.is_open:
      raise RuntimeError("摄像头打开失败：检查 index 是否被占用，或换一个摄像头序号")

    if self.show:
      # WINDOW_AUTOSIZE：窗口不可缩放，画面按采集到的原始分辨率 1:1 显示
      cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

    while True:
      ok, frame = self.cap.read()
      if not ok or frame is None:
        _logger.warning("摄像头读帧失败，退出")
        break

      self._update_fps()
      # 先旋转再裁切：显示画面与模型输入是同一张正方形图，
      # 检测框坐标不用来回换算（见 render/crop_for_model）
      frame = self.rotate_frame(frame)
      model_img, box = self.crop_for_model(frame)
      tensor = self.frame2tensor(model_img)
      result = self.inferrer.predict(tensor, thr=self.thr, topk=self.topk)[0]
      vis = self.render(frame, model_img, tensor, box, result)
      self.last_frame = vis
      self.frames += 1

      if self.show:
        cv2.putText(vis, f"{self._fps:5.1f} FPS   n={len(result[0])}"
                    f"   rot={self.rotate}   {vis.shape[1]}x{vis.shape[0]}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, "r/e: rotate  s: save  q: quit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1,
                    cv2.LINE_AA)
        cv2.imshow(self.window_name, vis)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
          break
        if key == ord("s"):
          self.snapshot()
        elif key == ord("r"):
          self.set_rotate((self.rotate + 90) % 360)
        elif key == ord("e"):
          self.set_rotate((self.rotate - 90) % 360)

      if max_frames is not None and self.frames >= int(max_frames):
        break
      self._limit_fps()

    return self.frames

  def snapshot(self, img: np.ndarray | None = None) -> Path | None:
    """把当前画面存盘（默认存最近一帧）"""
    img = self.last_frame if img is None else img
    if self.save_dir is None:
      _logger.warning("未配置 save_dir，跳过快照")
      return None
    if img is None:
      _logger.warning("还没有可保存的画面")
      return None

    name = f"frame_{time.strftime('%Y%m%d_%H%M%S')}_{self.frames:05d}.png"
    path = save_image(img, self.save_dir / name)
    _logger.info("已保存快照: %s", path)
    return path

  # ---------------- 内部工具 ----------------
  def _update_fps(self) -> None:
    """指数滑动平均，避免帧率数字乱跳"""
    now = time.perf_counter()
    dt = now - self._t_prev
    self._t_prev = now
    if dt > 0:
      inst = 1.0 / dt
      self._fps = inst if self._fps == 0.0 else 0.9 * self._fps + 0.1 * inst

  def _limit_fps(self) -> None:
    """按 max_fps 让出多余时间（相机出帧快于推理时很有用）"""
    if not self.max_fps:
      return
    target = 1.0 / self.max_fps
    elapsed = time.perf_counter() - self._t_prev
    if elapsed < target:
      time.sleep(target - elapsed)
