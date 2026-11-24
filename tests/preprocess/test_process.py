"""`src/preprocess/process.py` 的单元测试。"""

from __future__ import annotations

import json

import cv2
import numpy as np

from src.preprocess.process import save_sample


def test_save_sample_writes_gray_image_and_json(tmp_path) -> None:
  """训练样本落盘：灰度 PNG + 与之一一对应的 JSON 标签"""
  img = np.full((32, 48, 3), 200, np.uint8)
  labels = [{"digit": 3, "quad": [[1, 2], [3, 4], [5, 6], [7, 8]]}]

  img_path, lbl_path = save_sample(tmp_path, "s0", img, labels,
                                   extra={"source": "x.jpg"})

  assert img_path.name == "s0.png" and lbl_path.name == "s0.json"
  gray = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8),
                      cv2.IMREAD_UNCHANGED)
  assert gray.ndim == 2 and gray.shape == (32, 48)

  data = json.loads(lbl_path.read_text(encoding="utf-8"))
  assert data["image"] == "s0.png"
  assert data["width"] == 48 and data["height"] == 32
  assert data["source"] == "x.jpg"
  assert data["objects"][0]["digit"] == 3
  assert data["objects"][0]["quad"][0] == [1, 2]
