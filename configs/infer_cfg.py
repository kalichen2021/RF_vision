

"""推理 / 权重选择配置。"""

from __future__ import annotations


class InferConfig:
  """推理侧参数。

  `SELECT_MODEL_ID` 的取值：

  - `"0"`：自动取 `models/` 下最新的 `ckpt_*.pt`；
  - 具体文件名：直接使用该权重。
  """

  # SELECT_MODEL_ID: str = "ckpt_20261123_225153_554.pt"
  SELECT_MODEL_ID: str = "0"

  # 推理时是否先居中裁成正方形再缩放（False = 直接拉伸，会把画面压扁）
  CROP_SQUARE: bool = True