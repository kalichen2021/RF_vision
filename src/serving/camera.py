"""摄像头实时检测：cv2 采集 → CenterNet 推理 → 解码 → 画面叠加。

推理交给 ``src/inference/infer.py::Inferer``，本模块只负责摄像头读写、
单帧预处理、结果绘制和主循环。

用法::

    inf = Inferer(CenterNetDigit(num_classes=10, dropout=DROPOUT), SELECT_MODEL)
    with CameraDetector(inf, index=0, thr=0.3) as cam:
        cam.run()        # 按 q / ESC 退出，按 s 存当前帧，按 r / e 旋转画面

命令行入口见 ``src/cli/realtime_infer.py``。
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import torch

from configs import realtime_cfg as rtcfg
from src.utils.debug_tools import draw_quads, save_image

# 与 DigitDataset / build_transform 一致的 ImageNet 归一化（这里直接对 BGR 帧做，
# 省掉一次 PIL 转换，实时性更好）
_MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255.0
_STD = np.array([0.229, 0.224, 0.225], np.float32) * 255.0

# realtime_cfg.RESIZE_INTERP（字符串）→ cv2 插值常量
_INTERP = {
    "area": cv2.INTER_AREA,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
}


class CameraDetector:
  """实时摄像头检测器（采集 + 推理 + 显示）

  Args:
      inferrer: ``src.inference.infer.Inferer`` 实例（提供 ``predict``）
      index (int): 摄像头序号
      img_size (tuple[int, int]): 送进网络的尺寸，必须与训练一致
      width (int | None): 采集宽度，None 用摄像头默认
      height (int | None): 采集高度
      thr (float): 中心分数阈值
      topk (int): 每帧最多检出多少个数字
      show (bool): 是否 ``cv2.imshow`` 弹窗；False 时只推理不显示
      window_name (str): 窗口标题
      max_fps (float | None): 限制处理帧率，None 表示不限
      save_dir (Path | None): 按 ``s`` 时的快照目录，None 表示不快照
      thickness (int): 框线宽
      font_scale (float): 标签字号
      rotate (int): 初始顺时针旋转角度（0/90/180/270），显示与推理共用
      input_mode (str | None): 送进模型的通道模式，``"gray"``（灰度复制成 3 通道）
          或 ``"rgb"``（保留彩色）；None 时取 ``configs.realtime_cfg.INPUT_MODE``
      capture: 自定义 VideoCapture（测试注入用），None 时按 ``index`` 打开
  """

  def __init__(self, inferrer, index=0, img_size=(384, 384), width=1280,
               height=720, thr=0.3, topk=30, show=True,
               window_name="digits_tracker", max_fps=None, save_dir=None,
               thickness=2, font_scale=0.55, rotate=0, input_mode=None,
               capture=None):
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

    self.cap = capture if capture is not None else self._open(index, width, height)
    self.frames = 0                 # 已处理帧数
    self.last_frame = None          # 最近一帧（带叠加），供快照使用
    self._t_prev = time.perf_counter()
    self._fps = 0.0

  # ---------------- 资源管理 ----------------
  @staticmethod
  def _open(index, width, height):
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

  def __exit__(self, exc_type, exc, tb) -> bool:
    self.close()
    return False

  # ---------------- 旋转 ----------------
  @staticmethod
  def normalize_rotate(angle) -> int:
    """把任意角度归一化到 0/90/180/270（顺时针）"""
    angle = int(angle) % 360
    if angle % 90:
      raise ValueError(f"旋转角度必须是 90 的倍数，收到 {angle}")
    return angle

  def set_rotate(self, angle) -> int:
    """设置顺时针旋转角度并返回生效值（0/90/180/270）"""
    self.rotate = self.normalize_rotate(angle)
    return self.rotate

  def rotate_frame(self, frame):
    """按 ``self.rotate`` 顺时针旋转 BGR 帧；0° 时原样返回

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
  def frame2tensor(self, frame) -> torch.Tensor:
    """BGR 帧 → ``(1, 3, H, W)`` 归一化张量（与训练预处理一致）

    通道模式由 ``self.input_mode``（默认取 ``configs.realtime_cfg.INPUT_MODE``）决定：
    ``"gray"`` 先转灰度再复制成 3 通道（与灰度训练样本一致）；
    ``"rgb"`` 保留摄像头原始颜色。两种模式输出都是 3 通道。

    Args:
        frame (np.ndarray): ``cv2.VideoCapture.read`` 拿到的 BGR 帧

    Returns:
        torch.Tensor: ``(1, 3, H, W)`` float32

    Raises:
        ValueError: ``input_mode`` 不是 ``"gray"`` / ``"rgb"``
    """
    mode = self.input_mode
    if mode == "gray":
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    elif mode == "rgb":
      img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
      raise ValueError(
          f"input_mode 应为 'gray' 或 'rgb'，收到 {self.input_mode!r}")

    # cv2.resize 的参数顺序是 (宽, 高)，而 self.img_size 存的是 (高, 宽)；
    # 默认是正方形所以看不出来，非正方形时不转置就会把画面拉错方向
    interp = _INTERP.get(str(rtcfg.RESIZE_INTERP).lower(), cv2.INTER_AREA)
    img = cv2.resize(img, (self.img_size[1], self.img_size[0]),
                     interpolation=interp)
    arr = (img.astype(np.float32) - _MEAN) / _STD
    return torch.from_numpy(arr.transpose(2, 0, 1)).contiguous().unsqueeze(0)

  def detect(self, frame):
    """单帧推理 + 解码（传进来的帧需已按 ``self.rotate`` 旋转过）

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ``(pts_norm(K,4,2),
        digits(K,), scores(K,))``，坐标为 0~1 归一化
    """
    return self.inferrer.predict(self.frame2tensor(frame),
                                 thr=self.thr, topk=self.topk)[0]

  def draw(self, frame, result):
    """把检测结果画到帧上，返回新图（不改入参）"""
    pts, digits, scores = result
    return draw_quads(frame, pts.numpy(), digits.tolist(), scores.tolist(),
                      color=(0, 255, 0), tag="pred", thickness=self.thickness,
                      font_scale=self.font_scale)

  # ---------------- 主循环 ----------------
  def run(self, max_frames=None) -> int:
    """持续检测直到退出，返回处理的帧数

    退出条件：按 ``q`` / ``ESC``、摄像头断流、或达到 ``max_frames``。

    快捷键：``s`` 快照、``r`` 顺时针转 90°、``e`` 逆时针转 90°
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
        print("摄像头读帧失败，退出")
        break

      self._update_fps()
      # 先旋转再推理：显示与模型输入是同一张图
      frame = self.rotate_frame(frame)
      result = self.detect(frame)
      vis = self.draw(frame, result)
      self.last_frame = vis
      self.frames += 1

      if self.show:
        cv2.putText(vis, f"{self._fps:5.1f} FPS   n={len(result[0])}"
                    f"   rot={self.rotate}",
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

  def snapshot(self, img=None) -> Path | None:
    """把当前画面存盘（默认存最近一帧）"""
    img = self.last_frame if img is None else img
    if self.save_dir is None:
      print("未配置 save_dir，跳过快照")
      return None
    if img is None:
      print("还没有可保存的画面")
      return None

    name = f"frame_{time.strftime('%Y%m%d_%H%M%S')}_{self.frames:05d}.png"
    path = save_image(img, self.save_dir / name)
    print(f"已保存快照: {path}")
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
