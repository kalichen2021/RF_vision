"""临时：用最新权重生成一张热力图预览，确认新功能可用（跑完即删）。"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.path import PathConfig
from configs.training_cfg import TrainingConfig as tc
from src.inference.infer import Inferer
from src.serving.camera import CameraDetector
from src.training.dataset import DigitDataset, batch_to_torch, collate_fn
from src.training.model import CenterNetDigit
from src.utils.image import read_gray, to_tensor
from torch.utils.data import DataLoader

dev = "cuda" if torch.cuda.is_available() else "cpu"
model = CenterNetDigit(num_classes=10, dropout=tc.DROPOUT)
inf = Inferer(model, PathConfig.SELECT_MODEL, device=dev)

ds = DigitDataset(PathConfig.DATASET_LABEL_JSON, PathConfig.SELECT_IMGSET,
                  tc.TRAININGSET_RATIO, mode="testing")
sample = ds.samples[0]["image"]
gray = read_gray(PathConfig.SELECT_IMGSET / sample)

cam = CameraDetector(inf, show=False, show_heatmap=True,
                     capture=type("C", (), {"isOpened": lambda s: True})())
frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
model_img, box = cam.crop_for_model(frame)
tensor = cam.frame2tensor(model_img)
result, hm = cam.infer(tensor)
vis = cam.draw(model_img, result)
himg = cam.heatmap_image(hm, (vis.shape[1], vis.shape[0]), result)

print("权重:", PathConfig.SELECT_MODEL.name, "| 样本:", sample)
print("热力图:", None if hm is None else hm.shape,
      "| max=%.4f mean=%.5f" % (hm.max(), hm.mean()),
      "| 超过 0.3 的格子:", int((hm > 0.3).sum()))
print("检出:", len(result[0]), "个")
combo = np.hstack([vis, himg])
cv2.imwrite("dataset/log/_heatmap_check.png", combo)
print("已保存 dataset/log/_heatmap_check.png")
