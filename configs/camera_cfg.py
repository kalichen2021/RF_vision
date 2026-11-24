"""摄像头实时检测的默认配置。

`src/cli/realtime_infer.py` 的命令行参数会覆盖这里的默认值。
"""

from __future__ import annotations


class CameraConfig:
  """摄像头采集 / 检测 / 绘制的默认参数。"""

  # ---- 摄像头 / 窗口 ----
  CAMERA_INDEX: int = 0      # 0 = 默认摄像头；外接摄像头可以试 1、2
  FRAME_WIDTH: int = 1280    # 采集分辨率；None 表示用摄像头默认
  FRAME_HEIGHT: int = 720
  # 预览窗口按采集到的原始分辨率 1:1 显示（WINDOW_AUTOSIZE），不缩放也不翻转
  WINDOW_NAME: str = "digits_tracker"
  # 画面顺时针旋转角度：0 / 90 / 180 / 270，运行中按 r 顺时针、e 逆时针各 90°
  ROTATE: int = 0

  # ---- 检测 ----
  THR: float = 0.3           # 中心热度阈值，越高越保守（漏检多、误检少）
  TOPK: int = 30             # 每帧最多保留多少个中心
  MAX_FPS: float = 30.0      # 限制处理帧率；None 表示不限

  # ---- 绘制 ----
  SHOW: bool = True          # False 表示不弹窗（服务器 / 无显示器）
  DRAW_THICKNESS: int = 2
  FONT_SCALE: float = 0.55
