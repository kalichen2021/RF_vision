"""摄像头预览与裁切逻辑的单元测试（不真正打开摄像头）。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.serving.camera import CameraDetector


class _FakeCapture:
  """最小 VideoCapture 替身：固定返回同一帧，记录 release。"""

  def __init__(self, frame: np.ndarray) -> None:
    self.frame = frame
    self.released = False

  def isOpened(self) -> bool:      # noqa: N802  (cv2 接口风格)
    return not self.released

  def set(self, *_args: Any) -> bool:
    return True

  def read(self) -> tuple[bool, np.ndarray]:
    return True, self.frame.copy()

  def release(self) -> None:
    self.released = True


class _FakeInferrer:
  """替身 Inferer：返回固定的假检测结果（坐标为模型输入内的 0~1）。"""

  def __init__(self) -> None:
    self.pts: list[list[list[float]]] = [[[0.4, 0.4], [0.6, 0.4],
                                          [0.6, 0.6], [0.4, 0.6]]]
    self.last_shape: tuple[int, ...] | None = None

  def predict(self, tensor: Any, thr: float = 0.3,
              topk: int = 30) -> tuple[Any, ...]:
    import torch
    self.last_shape = tuple(tensor.shape)
    pts = torch.tensor(self.pts, dtype=torch.float32)
    n = len(self.pts)
    return (pts, torch.tensor([3] * n), torch.tensor([0.9] * n)),


def _detector(view: str, h: int = 480, w: int = 640) -> CameraDetector:
  frame = np.full((h, w, 3), 128, np.uint8)
  return CameraDetector(_FakeInferrer(), show=False, view=view,
                        capture=_FakeCapture(frame))


def test_crop_for_model_center_square() -> None:
  cam = _detector("crop")
  frame = np.zeros((480, 640, 3), np.uint8)
  cropped, box = cam.crop_for_model(frame)
  assert cropped.shape[:2] == (480, 480)          # 正方形，边长取较小边
  assert box == (80, 0, 480)                      # 水平居中


def test_crop_for_model_noop_when_disabled() -> None:
  cam = _detector("crop")
  cam.crop_square = False
  frame = np.zeros((480, 640, 3), np.uint8)
  cropped, box = cam.crop_for_model(frame)
  assert cropped.shape == frame.shape
  assert box == (0, 0, 480)


def test_invalid_view_raises() -> None:
  with pytest.raises(ValueError):
    _detector("stretch")


def test_render_crop_view_is_square_and_matches_model_input() -> None:
  cam = _detector("crop")
  frame = np.zeros((480, 640, 3), np.uint8)
  model_img, box = cam.crop_for_model(frame)
  tensor = cam.frame2tensor(model_img)
  result = cam.inferrer.predict(tensor)[0]
  vis = cam.render(frame, model_img, tensor, box, result)
  assert vis.shape[:2] == (480, 480)              # 预览是正方形，与比例一致
  assert tensor.shape == (1, 3, 384, 384)


def test_render_full_view_maps_boxes_back_to_frame() -> None:
  cam = _detector("full")
  frame = np.zeros((480, 640, 3), np.uint8)
  # 贴着裁切区左边界的一条框：x=0 → 整幅画面里应该落在 x≈80（水平居中偏移）
  cam.inferrer.pts = [[[0.0, 0.4], [0.2, 0.4], [0.2, 0.6], [0.0, 0.6]]]
  model_img, box = cam.crop_for_model(frame)
  tensor = cam.frame2tensor(model_img)
  result = cam.inferrer.predict(tensor)[0]
  vis = cam.render(frame, model_img, tensor, box, result)
  assert vis.shape[:2] == (480, 640)              # 整幅画面
  band = slice(200, 280)                         # 只看检测框所在的中间行
  assert vis[band, 76:84].sum() > 0               # 左边线画在 x≈80
  assert vis[band, 0:10].sum() == 0               # 没有被错误地画到 x≈0
  assert vis[band, 300:360].sum() == 0            # 也没有画到画面中心


def test_mark_crop_box_dims_outside() -> None:
  vis = np.full((100, 200, 3), 200, np.uint8)
  out = CameraDetector.mark_crop_box(vis, (50, 0, 100))
  assert out.shape == vis.shape
  assert out[50, 5].mean() < out[50, 100].mean()   # 框外更暗


class _FakeInferrerWithHeat:
  """带热力图的替身 Inferer：`infer` 返回原始三张图，`decode` 返回固定框。

    `predict` 故意抛错：开了热力图预览就不该再走 predict（会多跑一次前向）。
  """

  def __init__(self) -> None:
    self.hm = np.full((48, 48), 0.01, np.float32)
    self.hm[10, 20] = 0.9
    self.infer_calls = 0

  def infer(self, tensor: Any) -> tuple[Any, Any, Any]:
    import torch
    self.infer_calls += 1
    hm = torch.from_numpy(self.hm)[None, None]
    return hm, torch.zeros(1, 8, 48, 48), torch.zeros(1, 10, 48, 48)

  def decode(self, hm_p: Any, reg_p: Any, cls_p: Any, thr: float = 0.3,
             topk: int = 30) -> list[tuple[Any, ...]]:
    import torch
    pts = torch.tensor([[[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]]])
    return [(pts, torch.tensor([7]), torch.tensor([0.9]))]

  def predict(self, *_args: Any, **_kwargs: Any) -> Any:
    raise AssertionError("开关热力图时不应该再调 predict")


def _heat_detector(show_heatmap: bool) -> CameraDetector:
  frame = np.full((480, 640, 3), 128, np.uint8)
  return CameraDetector(_FakeInferrerWithHeat(), show=False,
                        show_heatmap=show_heatmap,
                        capture=_FakeCapture(frame))


def test_heatmap_disabled_keeps_old_path() -> None:
  """默认关闭：走 predict，不返回热力图（不增加任何开销）"""
  cam = _detector("crop")
  import torch
  result, hm = cam.infer(torch.zeros(1, 3, 384, 384))
  assert hm is None
  assert result[0].shape[-1] == 2


def test_heatmap_enabled_returns_raw_map() -> None:
  """开启后：一次前向同时拿到解码结果与 `(h, w)` 热力图"""
  cam = _heat_detector(show_heatmap=True)
  import torch
  result, hm = cam.infer(torch.zeros(1, 3, 384, 384))
  assert hm is not None and hm.shape == (48, 48)
  assert cam.inferrer.infer_calls == 1
  assert result[0].shape[0] == 1


def test_heatmap_image_shape_and_markers() -> None:
  """热力图预览图尺寸与主画面一致，且会标出检出的中心点"""
  cam = _heat_detector(show_heatmap=True)
  import torch
  result, hm = cam.infer(torch.zeros(1, 3, 384, 384))
  img = cam.heatmap_image(hm, (640, 480), result)
  assert img.shape == (480, 640, 3) and img.dtype == np.uint8
  # 中心点 (0.5, 0.5) 附近应该有白色十字
  assert img[236:244, 316:324].max() >= 200


def test_heatmap_uses_fixed_color_scale(monkeypatch) -> None:
  """色标固定：同一张图，改 vmax 会改颜色（否则会掩盖「峰值只有 0.72」）"""
  from configs.realtime_cfg import RealtimeConfig as rtcfg

  cam = _heat_detector(show_heatmap=True)
  hm = np.full((8, 8), 0.35, np.float32)
  monkeypatch.setattr(rtcfg, "HEATMAP_VMAX", 1.0)
  a = cam.heatmap_image(hm, (64, 64))
  monkeypatch.setattr(rtcfg, "HEATMAP_VMAX", 0.35)
  b = cam.heatmap_image(hm, (64, 64))
  assert not np.array_equal(a, b), "vmax 变化时颜色必须跟着变（说明没有自动归一化）"

