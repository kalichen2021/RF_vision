"""训练 / 验证数据集：读 labels.json + 灰度样本图，产出 `(img, quads, digits)`。

预处理统一走 `src.utils.image`（与推理、摄像头同一份实现），
本模块只负责索引、切分和拼 batch。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np

from src.utils.image import IMAGENET_MEAN, IMAGENET_STD, read_gray, resize

# ImageNet 归一化参数（与 src.utils.image 共用一份，避免各处重复定义）
_MEAN = np.asarray(IMAGENET_MEAN, np.float32)[:, None, None]
_STD = np.asarray(IMAGENET_STD, np.float32)[:, None, None]

DatasetMode = Literal["training", "testing"]
# 正方形裁切策略：True 总是裁；False 从不裁；"auto" 仅在裁完不丢数字时裁
CropSquare = bool | Literal["auto"]


def collate_fn(batch: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> tuple:
  """把 numpy 样本拼成 batch（worker 里**不 import torch**）。

  每个 worker 一旦 import torch，提交内存会多 1GB+；`NUM_WORKERS=10` 时
  足以把 40G 内存吃爆（表现为训练还没开始就 OOM）。这里只做 numpy 堆叠，
  转张量交给主进程的 :func:`batch_to_torch`。
  """
  imgs = np.stack([np.asarray(b[0], np.float32) for b in batch])
  quads = [np.asarray(b[1], np.float32) for b in batch]
  digits = [np.asarray(b[2], np.int64) for b in batch]
  return imgs, quads, digits


def batch_to_torch(batch: tuple) -> tuple:
  """把 ``collate_fn`` 产出的 numpy batch 转成 torch batch（**只在主进程调**）。

  Args:
      batch: ``(imgs (B,3,H,W), quads list, digits list)``，均为 numpy。

  Returns:
      ``(imgs torch.Tensor, quads list[torch.Tensor], digits list[torch.Tensor])``。
  """
  import torch

  imgs, quads, digits = batch
  return (torch.from_numpy(imgs),
          [torch.from_numpy(q) for q in quads],
          [torch.from_numpy(d) for d in digits])


class DigitDataset:
  """读 `labels.json` + 灰度样本图，返回 `(img[3,H,W], quad_norm[N,4,2], digit[N])`

  读图 / 缩放用 cv2（灰度 PNG 直接解码，再 repeat 成 3 通道），比 PIL
  `convert("RGB") -> BICUBIC Resize` 快 2~3 倍；单样本实测约 7ms。

  **返回值是 numpy 而不是 torch 张量**：`collate_fn` 里再转 torch。
  这样 worker 进程不需要在本模块顶层加载 torch / torchvision，能显著降低
  每个 worker 的常驻内存（曾因 NUM_WORKERS=10 时每 worker ≈1GB 把 16G 内存吃爆）。
  """

  def __init__(self, json_path: Path | str, img_dir: Path | str,
               trainingset_ratio: float = 1.0,
               img_size: tuple[int, int] = (384, 384),
               convert: str = "RGB", mode: DatasetMode = "training",
               crop_square: CropSquare = "auto") -> None:
    """_summary_

    Args:
        json_path (Path):  json 路径
        img_dir (Path): 图片文件夹路径
        trainingset_ratio (float, optional): 取前百分之多少的样本作为训练集 (0~1]. Defaults to 1.0.
        img_size (tuple, optional): 图片输入尺寸 (H, W). Defaults to (384, 384).
        convert (str, optional): "RGB"（3 通道，ResNet18 输入）或 "L"（灰度单通道）. Defaults to "RGB".
        mode (str, optional): 数据集模式，可以是 "training" 或 "testing". Defaults to "training".
        crop_square (bool | "auto", optional): 样本不是正方形时的处置：
            True = 居中裁正方形（可能丢边缘数字）；False = 直接拉伸；
            "auto"（默认）= 只有裁切不会丢任何数字时才裁，否则保持拉伸。
    """
    with open(json_path, 'r') as f:
      self.data = json.load(f)
    self.img_dir = Path(img_dir)
    ratio = float(trainingset_ratio)
    if not 0 < ratio <= 1:
      raise ValueError(f"trainingset_ratio 应在 (0, 1] 内，收到 {trainingset_ratio}")
    self.trainingset_ratio = ratio
    self.mode = mode
    # 注意：这里截断过样本，所以 __len__ 必须用 len(self.samples)
    total = len(self.data["samples"])
    cut = max(1, int(total * ratio))
    if self.mode == "training":
      raw = self.data["samples"][:cut]
    elif self.mode == "testing":
      # ratio=1.0 时 cut 等于总数，切片会是空的 -> 宁可报错也不要静默返回 0 条样本
      if cut >= total:
        raise ValueError(
            f"mode='testing' 但 trainingset_ratio={ratio} 把全部 {total} 条都划给了训练集，"
            "测试集为空；请传入与训练时一致的比例（如 0.8）")
      raw = self.data["samples"][cut:]
    else:
      raise ValueError(f"mode 应为 'training' 或 'testing'，收到 {mode}")

    # 只保留训练 / 推理真正用的字段：labels.json 里还有 board_quad / quad /
    # center / depth / sigma 等，DataLoader 用 spawn 起 worker 时每个子进程
    # 都要 pickle 一份数据集，字段越多内存越紧张（曾导致 worker MemoryError）
    self.samples = [self._slim(s) for s in raw]
    del self.data                 # 完整标签不再需要，尽早释放
    
    self.img_size = img_size
    self.convert = convert
    # cv2.resize 的参数顺序是 (宽, 高)，这里统一存成 (h, w) 免得每张图都去猜
    self._img_hw = (int(img_size[0]), int(img_size[1]))
    self._channels = 3 if str(convert).upper() == "RGB" else 1
    # 非正方形样本的处置：True = 居中裁正方形，False = 直接拉伸，"auto" = 不丢数字才裁
    self.crop_square = crop_square

  def __len__(self) -> int:
    # 必须返回「实际可用样本数」：self.samples 已被 trainingset_ratio 截断，
    # 用 data["count"]（原始总数）会让 DataLoader 取到越界下标 -> IndexError
    return len(self.samples)

  @staticmethod
  def _slim(sample: dict[str, Any]) -> dict[str, Any]:
    """只保留 image + objects 里的 digit / quad_norm / quad。

    ``quad``（绝对像素坐标）也要留：样本不是正方形时要靠它把标签换算到
    裁剪后的坐标系里。
    """
    return {"image": sample["image"],
            "objects": [{"digit": int(o["digit"]),
                         "quad_norm": o["quad_norm"],
                         "quad": o.get("quad")}
                        for o in sample.get("objects", [])]}

  @staticmethod
  def _crop_square_labels(gray: np.ndarray, objects: list[dict]
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """居中裁出最大正方形，并把标签换算到裁剪后的归一化坐标。

    中心落在裁剪范围外的数字会被丢弃（图里已经不包含它了）。

    Args:
        gray: 原始灰度图（非正方形）。
        objects: 样本的 objects 列表（需要 ``digit`` 与 ``quad``）。

    Returns:
        ``(裁剪后的灰度图, quads[N,4,2], digits[N])``。
    """
    h, w = gray.shape[:2]
    side = min(h, w)
    x0, y0 = (w - side) // 2, (h - side) // 2
    cropped = gray[y0:y0 + side, x0:x0 + side]
    scale = float(max(side - 1, 1))

    quads: list[np.ndarray] = []
    digits: list[int] = []
    for o in objects:
      quad = o.get("quad")
      if not quad:          # 老数据没有绝对坐标：退回 quad_norm 估算中心
        qn = np.asarray(o["quad_norm"], np.float64)
        cx, cy = qn[:, 0].mean() * w, qn[:, 1].mean() * h
        if not (x0 <= cx < x0 + side and y0 <= cy < y0 + side):
          continue
        shift = np.array([x0 / max(w - 1, 1), y0 / max(h - 1, 1)], np.float64)
        ratio = np.array([(w - 1) / scale, (h - 1) / scale], np.float64)
        quads.append(((qn - shift) * ratio).clip(0.0, 1.0).astype(np.float32))
      else:
        q = np.asarray(quad, np.float64) - np.array([x0, y0], np.float64)
        cx, cy = q[:, 0].mean(), q[:, 1].mean()
        if not (0.0 <= cx < side and 0.0 <= cy < side):
          continue
        quads.append((q / scale).clip(0.0, 1.0).astype(np.float32))
      digits.append(int(o["digit"]))

    quads_np = (np.asarray(quads, np.float32).reshape(-1, 4, 2)
                if quads else np.zeros((0, 4, 2), np.float32))
    return cropped, quads_np, np.asarray(digits, np.int64)

  @staticmethod
  def _crop_keeps_all(shape: tuple[int, ...], objects: list[dict]) -> bool:
    """判断居中裁正方形后是否所有数字的中心都还在画面内。

    Args:
        shape: 原图尺寸（至少含 H、W）。
        objects: 样本 objects 列表。

    Returns:
        True 表示裁完不会丢任何数字。
    """
    h, w = int(shape[0]), int(shape[1])
    side = min(h, w)
    x0, y0 = (w - side) // 2, (h - side) // 2
    for o in objects:
      quad = o.get("quad")
      if quad:
        q = np.asarray(quad, np.float64)
        cx, cy = float(q[:, 0].mean()), float(q[:, 1].mean())
      else:
        qn = np.asarray(o["quad_norm"], np.float64)
        cx, cy = float(qn[:, 0].mean()) * w, float(qn[:, 1].mean()) * h
      if not (x0 <= cx < x0 + side and y0 <= cy < y0 + side):
        return False
    return True

  def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读一张灰度样本 -> (归一化张量, quads[N,4,2], digits[N])

    用 cv2 代替 PIL：样本是灰度 PNG，PIL 的 `convert("RGB")` 会先复制 3 份
    再 BICUBIC 缩放；这里直接灰度解码 + `INTER_LINEAR` 缩放后再 repeat，
    单样本耗时约为原来的 1/2~1/3（读图是 DataLoader 的主要 CPU 开销）。
    """
    s = self.samples[idx]
    path = self.img_dir / s["image"]
    # 读图统一走 src.utils.image（read_gray 内部已处理中文路径）
    gray = read_gray(path)

    # 样本不是正方形时：能裁就裁成正方形（而不是拉伸），标签同步换算
    do_crop = False
    if gray.shape[0] != gray.shape[1] and self.crop_square:
      do_crop = (self.crop_square is True
                 or self._crop_keeps_all(gray.shape, s["objects"]))
    if do_crop:
      gray, quads, digits = self._crop_square_labels(gray, s["objects"])
    else:
      quads = np.asarray([o["quad_norm"] for o in s["objects"]],
                         dtype=np.float32).reshape(-1, 4, 2)
      digits = np.asarray([o["digit"] for o in s["objects"]], dtype=np.int64)

    # 已经是目标尺寸就不重复缩放（生成侧已按 SAVE_SIZE 存 384）
    if gray.shape[:2] != self._img_hw:
      gray = resize(gray, self._img_hw, interp="linear")

    img = gray.astype(np.float32) / 255.0
    if self._channels == 3:
      img = np.repeat(img[None], 3, axis=0)
      mean, std = _MEAN, _STD
    else:
      img = img[None]
      mean, std = _MEAN[:1], _STD[:1]
    img = np.ascontiguousarray((img - mean) / std)

    return img, quads, digits
