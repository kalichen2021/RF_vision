import json
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from pathlib import Path


def collate_fn(batch):
  imgs = torch.stack([b[0] for b in batch])
  quads = [b[1] for b in batch]
  digits = [b[2] for b in batch]
  return imgs, quads, digits


class DigitDataset(Dataset):
  def __init__(self, json_path, img_dir, trainingset_ratio=1.0, img_size=(384, 384), convert="RGB", mode="training"):
    """_summary_

    Args:
        json_path (Path):  json 路径
        img_dir (Path): 图片文件夹路径
        trainingset_ratio (float, optional): 取前百分之多少的样本作为训练集 (0~1]. Defaults to 1.0.
        img_size (tuple, optional): 图片输入尺寸. Defaults to (384, 384).
        convert (str, optional): 图片读取通道，应该为“RGB”三通道，因为r18模型的输入为三通道. Defaults to "RGB".
        mode (str, optional): 数据集模式，可以是 "training" 或 "testing". Defaults to "training".
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
      self.samples = self.data["samples"][:cut]
    elif self.mode == "testing":
      # ratio=1.0 时 cut 等于总数，切片会是空的 -> 宁可报错也不要静默返回 0 条样本
      if cut >= total:
        raise ValueError(
            f"mode='testing' 但 trainingset_ratio={ratio} 把全部 {total} 条都划给了训练集，"
            "测试集为空；请传入与训练时一致的比例（如 0.8）")
      self.samples = self.data["samples"][cut:]
    else:
      raise ValueError(f"mode 应为 'training' 或 'testing'，收到 {mode}")
    
    self.img_size = img_size
    self.convert = convert
    self.tf = T.Compose([
        T.Resize(self.img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

  def __len__(self):
    # 必须返回「实际可用样本数」：self.samples 已被 trainingset_ratio 截断，
    # 用 data["count"]（原始总数）会让 DataLoader 取到越界下标 -> IndexError
    return len(self.samples)

  def __getitem__(self, idx):
    from PIL import Image
    s = self.samples[idx]

    img = Image.open(self.img_dir / s["image"]).convert(self.convert)
    img = self.tf(img)
    quads = torch.tensor([o["quad_norm"]
                         for o in s["objects"]], dtype=torch.float32)
    digits = torch.tensor([o["digit"]
                          for o in s["objects"]], dtype=torch.long)
    return img, quads, digits
