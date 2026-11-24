"""实时推理（摄像头）的输入预处理配置。

决定「摄像头画面以什么形式送进模型」：

- `"gray"`（默认）：BGR → 灰度 → 复制成 3 通道（R=G=B）。
  训练集 `DigitDataset` 读的就是灰度 PNG 再复制成 3 通道，所以这是
  与训练分布最一致的选项。
- `"rgb"`：BGR → RGB，保留摄像头原始颜色。
  摄像头若能提供有用颜色（例如彩色板 / 标记），可以试这个，但会和
  灰度训练的模型存在一定分布差异。

命令行 `--input-mode` 可临时覆盖 `INPUT_MODE`。
"""

from __future__ import annotations


class RealtimeConfig:
  """实时推理的输入预处理参数。"""

  # 送进模型的通道模式："gray" 或 "rgb"
  INPUT_MODE: str = "gray"

  # 是否先把摄像头画面**居中裁成正方形**再缩放（True：保持画面比例；
  # False：直接拉伸成 IMG_SIZE，画面会被压扁）
  CROP_SQUARE: bool = True

  # 预览窗口显示什么（只影响显示，不影响送进模型的图）：
  #   "crop"  = 只显示裁切后的正方形（默认，长宽比与模型输入一致）
  #   "model" = 显示缩放到 IMG_SIZE 后模型真正看到的画面
  #   "full"  = 显示整幅画面，裁切区域以外压暗
  VIEW_MODE: str = "crop"

  # resize 插值："linear"（与训练 DigitDataset 一致）/ "area"（缩小更好）/
  # "cubic"（更锐利，计算更慢）
  RESIZE_INTERP: str = "area"
