"""摄像头实时检测的默认配置。

``src/cli/realtime_infer.py`` 的命令行参数会覆盖这里的默认值。
"""

# ---- 摄像头 / 窗口 ----
CAMERA_INDEX = 0            # 0 = 默认摄像头；外接摄像头可以试 1、2
FRAME_WIDTH = 1280          # 采集分辨率；None 表示用摄像头默认
FRAME_HEIGHT = 720
# 预览窗口按采集到的原始分辨率 1:1 显示（WINDOW_AUTOSIZE），既不缩放也不左右翻转
WINDOW_NAME = "digits_tracker"

# ---- 检测 ----
THR = 0.3                   # 中心热度阈值，越高越保守（漏检多、误检少）
TOPK = 30                   # 每帧最多保留多少个中心
MAX_FPS = 30.0              # 限制处理帧率；None 表示不限（相机 60fps 时不限会白烧显卡）

# ---- 绘制 ----
SHOW = True                 # False 表示不弹窗（服务器 / 无显示器）
DRAW_THICKNESS = 2
FONT_SCALE = 0.55
