"""图像读取、缩放、归一化与张量化的统一入口。

训练（`DigitDataset`）、推理（`Inferer` / 命令行）和摄像头实时检测
都必须用同一套预处理，否则训练与推理的分布会不一致。这里把「读图」
「通道处理」「Resize」「ImageNet 归一化」收敛成一份实现，各处只调用。

约定：

- 内部统一使用 **BGR**（OpenCV 语义）或单通道灰度；
- 送进网络前统一变成 `(3, H, W)` float32，并按 ImageNet 均值方差归一化；
- 尺寸参数一律写成 `(高, 宽)`，与 `cv2.resize` 的 `(宽, 高)` 区分开。
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import cv2
import numpy as np

# 注意：本模块会被 DataLoader worker 导入（DigitDataset 用 read_gray / resize），
# 所以**顶层不 import torch**（每个 worker 会多背 1GB+ 提交内存）。
# 需要张量的函数在函数内部惰性 import。

# ImageNet 归一化参数（ResNet 预训练权重的统计量）
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

# 通道模式：gray = 灰度复制成 3 通道（与训练样本一致），rgb = 保留彩色
ChannelMode = Literal["gray", "rgb"]
# 缩放插值名
InterpMode = Literal["area", "linear", "cubic"]
# 裁切模式：none = 不裁；center = 居中裁正方形；random = 随机起点裁正方形
CropMode = Literal["none", "center", "random"]

_INTERP_FLAGS: dict[str, int] = {
    "area": cv2.INTER_AREA,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
}

# (3, 1, 1) 形状的 float32 均值 / 方差，便于对 (3, H, W) 直接广播
_MEAN_CHW: np.ndarray = np.asarray(IMAGENET_MEAN, np.float32)[:, None, None]
_STD_CHW: np.ndarray = np.asarray(IMAGENET_STD, np.float32)[:, None, None]


def read_image(path: Path | str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
  """按 OpenCV 的方式读图（支持中文路径）。

  用 `np.fromfile + cv2.imdecode` 而不是 `cv2.imread`：Windows 下
  `cv2.imread` 读不了含中文的路径。

  Args:
      path: 图片路径。
      flags: `cv2.imread` 的读取模式，默认彩色（BGR）。

  Returns:
      解码后的图像数组。

  Raises:
      ValueError: 文件不存在或解码失败。
  """
  p = Path(path)
  data = np.fromfile(str(p), dtype=np.uint8)
  img = cv2.imdecode(data, flags) if data.size else None
  if img is None:
    raise ValueError(f"图片读取失败: {p}")
  return img


def read_gray(path: Path | str) -> np.ndarray:
  """读成单通道灰度图。

  Args:
      path: 图片路径。

  Returns:
      `(H, W)` uint8 灰度图。
  """
  return read_image(path, cv2.IMREAD_GRAYSCALE)


def to_three_channels(img: np.ndarray, mode: ChannelMode = "gray") -> np.ndarray:
  """把任意图转成 3 通道 RGB（网络输入要求 3 通道）。

  Args:
      img: 灰度 `(H, W)` 或彩色 `(H, W, 3)`（BGR 顺序）图像。
      mode: `"gray"` 先转灰度再复制成 3 通道（与训练样本一致）；
          `"rgb"` 保留颜色，仅把 BGR 换成 RGB。

  Returns:
      `(H, W, 3)` uint8，RGB 顺序。

  Raises:
      ValueError: `mode` 取值非法，或输入通道数不支持。
  """
  if img.ndim == 2:
    gray = img
  elif img.ndim == 3 and img.shape[2] == 3:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  elif img.ndim == 3 and img.shape[2] == 4:
    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
  else:
    raise ValueError(f"不支持的图像形状: {img.shape}")

  if mode == "gray":
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
  if mode == "rgb":
    if img.ndim == 2 or img.shape[2] != 3:
      return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  raise ValueError(f"mode 应为 'gray' 或 'rgb'，收到 {mode!r}")


def resize(img: np.ndarray, size_hw: tuple[int, int],
           interp: str = "area") -> np.ndarray:
  """等比无关的强制缩放（按 `(高, 宽)` 给尺寸，内部换成 `(宽, 高)`）。

  Args:
      img: 输入图像；单通道 / 三通道均可。
      size_hw: 目标尺寸 `(高, 宽)`。
      interp: `"area"` / `"linear"` / `"cubic"`。

  Returns:
      缩放后的图像。

  Raises:
      ValueError: `interp` 取值非法。
  """
  flag = _INTERP_FLAGS.get(str(interp).lower())
  if flag is None:
    raise ValueError(
        f"interp 应为 {sorted(_INTERP_FLAGS)} 之一，收到 {interp!r}")
  h, w = int(size_hw[0]), int(size_hw[1])
  # cv2.resize 的参数顺序是 (宽, 高)
  return cv2.resize(img, (w, h), interpolation=flag)


def normalize(img_rgb: np.ndarray) -> np.ndarray:
  """`(H, W, 3)` uint8 RGB → `(3, H, W)` float32（ImageNet 归一化）。

  Args:
      img_rgb: `(H, W, 3)` uint8，RGB 顺序。

  Returns:
      `(3, H, W)` float32。
  """
  arr = img_rgb.astype(np.float32) / 255.0
  arr = np.ascontiguousarray(arr.transpose(2, 0, 1))
  return (arr - _MEAN_CHW) / _STD_CHW


def crop_square(img: np.ndarray, origin: tuple[int, int] | None = None,
                rng: np.random.Generator | None = None
                ) -> tuple[np.ndarray, tuple[int, int]]:
  """裁出最大的正方形（边长 = min(H, W)），避免后续缩放把画面拉扁。

  Args:
      img: 任意图像（灰度 / 彩色均可）。
      origin: 正方形左上角 `(x, y)`；None 表示随机取起点（“起点任意”），
          越界会自动夹到合法范围。
      rng: 随机数发生器；None 时新建 `np.random.default_rng()`。

  Returns:
      `(裁切后的正方形图像, (x0, y0))`；输入已是正方形时原样返回 `(0, 0)`。
  """
  h, w = img.shape[:2]
  side = min(h, w)
  if h == w:
    return img, (0, 0)
  if origin is None:
    rng = np.random.default_rng() if rng is None else rng
    x0 = int(rng.integers(0, w - side + 1))
    y0 = int(rng.integers(0, h - side + 1))
  else:
    x0 = int(np.clip(origin[0], 0, w - side))
    y0 = int(np.clip(origin[1], 0, h - side))
  return img[y0:y0 + side, x0:x0 + side], (x0, y0)


def center_origin(img: np.ndarray) -> tuple[int, int]:
  """居中裁正方形时，正方形左上角的坐标 `(x0, y0)`。

  Args:
      img: 任意图像。

  Returns:
      `(x0, y0)`；输入已是正方形时为 `(0, 0)`。
  """
  h, w = img.shape[:2]
  side = min(h, w)
  return (w - side) // 2, (h - side) // 2


def to_tensor(img: np.ndarray, size_hw: tuple[int, int] = (384, 384),
              mode: ChannelMode = "gray", interp: str = "area",
              crop: CropMode = "none") -> torch.Tensor:
  """任意图像 → `(1, 3, H, W)` 归一化张量（训练 / 推理共用的唯一入口）。

  Args:
      img: 灰度或 BGR 图像（numpy）。
      size_hw: 送进网络的目标尺寸 `(高, 宽)`。
      mode: `"gray"`（灰度复制 3 通道，与训练样本一致）或 `"rgb"`。
      interp: 缩放插值。
      crop: `"none"` 直接缩放；`"center"` 先居中裁正方形；
          `"random"` 先随机起点裁正方形（相当于位置增强）。

  Returns:
      `(1, 3, H, W)` float32 张量（仍在 CPU 上）。

  Raises:
      ValueError: `crop` 取值非法。
  """
  if crop == "center":
    img, _ = crop_square(img, origin=center_origin(img))
  elif crop == "random":
    img, _ = crop_square(img)
  elif crop != "none":
    raise ValueError(
        f"crop 应为 'none' / 'center' / 'random'，收到 {crop!r}")
  import torch      # 惰性导入：worker 不转张量，不需要 torch

  rgb = to_three_channels(img, mode)
  rgb = resize(rgb, size_hw, interp)
  return torch.from_numpy(normalize(rgb)).unsqueeze(0)


def load_tensor(path: Path | str, size_hw: tuple[int, int] = (384, 384),
                mode: ChannelMode = "gray", interp: str = "area",
                crop: CropMode = "none") -> torch.Tensor:
  """读一张图片文件并转成 `(1, 3, H, W)` 归一化张量。

  Args:
      path: 图片路径（支持中文）。
      size_hw: 送进网络的目标尺寸 `(高, 宽)`。
      mode: 通道模式，见 :func:`to_tensor`。
      interp: 缩放插值。
      crop: 裁切模式，见 :func:`to_tensor`。

  Returns:
      `(1, 3, H, W)` float32 张量。
  """
  return to_tensor(read_image(path), size_hw, mode, interp, crop)


def denormalize(tensor: torch.Tensor) -> np.ndarray:
  """`(3, H, W)` 归一化张量 → `(H, W, 3)` uint8 BGR 图（可视化用）。

  Args:
      tensor: `(3, H, W)`、`(1, 3, H, W)` 或 `(H, W)` 张量。

  Returns:
      BGR uint8 图像；单通道输入返回 `(H, W)` 灰度图。
  """
  t = tensor.detach().float().cpu()
  if t.ndim == 4:
    t = t[0]
  if t.ndim == 2:
    return (t.numpy() * 255).round().clip(0, 255).astype(np.uint8)
  if t.shape[0] == 1:
    arr = t[0].numpy() * IMAGENET_STD[0] + IMAGENET_MEAN[0]
    return (arr.clip(0, 1) * 255).round().astype(np.uint8)

  arr = t.numpy() * _STD_CHW + _MEAN_CHW          # (3, H, W) -> 0~1
  arr = (arr.clip(0, 1) * 255).round().astype(np.uint8).transpose(1, 2, 0)
  return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
