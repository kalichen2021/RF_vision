"""实时推理（摄像头）的输入预处理配置。

决定「摄像头画面以什么形式送进模型」：

- ``"gray"``（默认）：BGR → 灰度 → 复制成 3 通道（R=G=B）。
  训练集 ``DigitDataset`` 读的就是灰度 PNG 再复制成 3 通道，所以这是
  与训练分布最一致的选项。
- ``"rgb"``：BGR → RGB，保留摄像头原始颜色。
  摄像头若能提供有用颜色（例如彩色板/标记），可以试这个，但会和
  灰度训练的模型存在一定分布差异。

命令行 ``--input-mode`` 可临时覆盖 ``INPUT_MODE``。
"""

# 送进模型的通道模式："gray" 或 "rgb"
INPUT_MODE = "gray"

# resize 插值（cv2 常量名，小写）：
#   "linear" - 与训练 DigitDataset 的 cv2.INTER_LINEAR 一致
#   "area"   - 缩小尺寸时抗锯齿更好（相机分辨率通常远大于 384）
#   "cubic"  - 锐利一些，计算更慢
RESIZE_INTERP = "area"
