"""推理包：把训练好的 CenterNetDigit 权重变成可用的数字检测器。"""

from .infer import Inferer, build_transform, load_image

__all__ = ["Inferer", "build_transform", "load_image"]
