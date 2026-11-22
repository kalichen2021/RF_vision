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
  def __init__(self, json_path, img_dir, img_size=(384, 384), convert="RGB"):
    """_summary_

    Args:
        json_path (Path):  json 路径
        img_dir (Path): 图片文件夹路径
        img_size (tuple, optional): 图片输入尺寸. Defaults to (384, 384).
        convert (str, optional): 图片读取通道，应该为“RGB”三通道，因为r18模型的输入为三通道. Defaults to "RGB".
    """
    with open(json_path, 'r') as f:
      self.data = json.load(f)
    self.img_dir = Path(img_dir)
    self.samples = self.data["samples"]
    self.img_size = img_size
    self.convert = convert
    self.tf = T.Compose([
        T.Resize(self.img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

  def __len__(self):
    return self.data["count"]

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
