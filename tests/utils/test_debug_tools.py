"""``src/utils/debug_tools.py`` 的单元测试。"""

from __future__ import annotations

import cv2
import numpy as np
import torch

from utils.debug_tools import draw_quads, save_image, tensor2img


def _bg(h: int = 64, w: int = 96) -> np.ndarray:
  """灰底三通道图，方便检查哪里被画上了"""
  return np.full((h, w, 3), 128, np.uint8)


def test_draw_quads_does_not_mutate_input() -> None:
  img = _bg()
  before = img.copy()
  quads = np.array([[[0.1, 0.1], [0.4, 0.1], [0.4, 0.4], [0.1, 0.4]]])

  vis = draw_quads(img, quads, labels=[3], scores=[0.9], color=(0, 255, 0))

  assert vis.shape == img.shape and vis.dtype == np.uint8
  assert np.array_equal(img, before), "默认应返回副本，不改入参"
  assert (vis != img).any(), "图上应该有被画上的像素"


def test_draw_quads_normalized_maps_to_image_scale() -> None:
  img = _bg(h=100, w=200)
  quads = np.array([[[0.25, 0.20], [0.75, 0.20], [0.75, 0.80], [0.25, 0.80]]])

  vis = draw_quads(img, quads, color=(0, 0, 255))
  diff = (vis != img).any(axis=2)
  ys, xs = np.nonzero(diff)

  # x: 0.25*199≈50, 0.75*199≈149；y: 0.20*99≈20, 0.80*99≈79
  assert abs(int(xs.min()) - 50) <= 2 and abs(int(xs.max()) - 149) <= 2
  assert abs(int(ys.min()) - 20) <= 2 and abs(int(ys.max()) - 79) <= 2


def test_draw_quads_accepts_pixel_coords_and_gray_input() -> None:
  gray = np.full((64, 64), 128, np.uint8)
  quads = np.array([[[10, 10], [50, 10], [50, 50], [10, 50]]], np.float32)

  vis = draw_quads(gray, quads, normalized=False, color=(0, 255, 0))

  assert vis.ndim == 3 and vis.shape == (64, 64, 3), "灰度图应被转成三通道"
  assert (vis != cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)).any()


def test_draw_quads_handles_empty() -> None:
  img = _bg()

  vis = draw_quads(img, np.zeros((0, 4, 2)))

  assert np.array_equal(vis, img), "没有框时应原样返回"


def test_tensor2img_denormalizes() -> None:
  # 反归一化：(0 * 0.5 + 0.5) * 255 = 127.5 -> 128
  t = torch.zeros(3, 8, 8)

  img = tensor2img(t, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

  assert img.shape == (8, 8, 3) and img.dtype == np.uint8
  assert np.allclose(img, 128, atol=1)


def test_tensor2img_accepts_batch_and_single_channel() -> None:
  assert tensor2img(torch.zeros(1, 3, 8, 8)).shape == (8, 8, 3)
  assert tensor2img(torch.zeros(1, 8, 8), mean=(0.5,), std=(0.5,)).shape == (8, 8)


def test_save_image_creates_parent_dirs(tmp_path) -> None:
  path = save_image(_bg(), tmp_path / "infer" / "a.png")

  assert path.exists() and path.stat().st_size > 0
  back = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
  assert back.shape == (64, 96, 3)


def test_save_image_defaults_to_png(tmp_path) -> None:
  path = save_image(_bg(), tmp_path / "no_suffix")

  assert path.suffix == ".png" and path.exists()
