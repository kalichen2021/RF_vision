"""``src/inference/infer.py`` 的单元测试（不需要真实数据 / 已训练权重）。"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from src.inference.infer import Inferer, load_image
from src.training.model import CenterNetDigit, build_target

FH = FW = 48


def _batch():
  """两张图：第一张 1 个数字，第二张 2 个。四边形用归一化坐标 [TL,TR,BR,BL]"""
  quads_list = [
      torch.tensor([[[0.20, 0.30], [0.40, 0.30], [0.40, 0.50], [0.20, 0.50]]]),
      torch.tensor([[[0.60, 0.10], [0.70, 0.10], [0.70, 0.20], [0.60, 0.20]],
                    [[0.10, 0.75], [0.25, 0.80], [0.22, 0.95], [0.08, 0.90]]]),
  ]
  digits_list = [torch.tensor([3]), torch.tensor([7, 0])]

  return quads_list, digits_list


def _perfect_preds(quads_list, digits_list):
  """用 build_target 的标签造「完美预测」：网络输出恰好等于真值"""
  hm_t, reg_t, cls_t, _ = build_target(quads_list, digits_list, FH, FW)
  # 类别用 one-hot 当 logits（argmax 就是真类别）
  cls_p = F.one_hot(cls_t, num_classes=10).permute(0, 3, 1, 2).float()

  return hm_t, reg_t, cls_p


def test_decode_recovers_ground_truth() -> None:
  """decode 应该把「完美预测」还原成原始四边形与类别"""
  quads_list, digits_list = _batch()
  hm_p, reg_p, cls_p = _perfect_preds(quads_list, digits_list)

  results = Inferer.decode(hm_p, reg_p, cls_p, thr=0.5, topk=10)

  assert len(results) == 2
  for b, quads in enumerate(quads_list):
    pts, digits, scores = results[b]
    assert pts.shape == (quads.shape[0], 4, 2)
    assert scores.min().item() > 0.5
    # 同分时 topk 的顺序不定，按四边形整体排序后再比
    got = torch.tensor(sorted(pts.tolist()))
    want = torch.tensor(sorted(quads.tolist()))
    assert torch.allclose(got, want, atol=1e-4), f"{got} != {want}"
    assert sorted(digits.tolist()) == sorted(digits_list[b].tolist())


def test_decode_threshold_filters_everything() -> None:
  """热度全压到阈值以下时，返回 0 个目标但形状仍然合法"""
  quads_list, digits_list = _batch()
  hm_p, reg_p, cls_p = _perfect_preds(quads_list, digits_list)

  results = Inferer.decode(hm_p * 0.2, reg_p, cls_p, thr=0.5)

  for pts, digits, scores in results:
    assert pts.shape == (0, 4, 2)
    assert digits.numel() == 0 and scores.numel() == 0


def test_decode_nms_keeps_one_peak_per_digit() -> None:
  """高斯峰的邻格会被 3x3 局部极大值抑制掉，不会一个数字检出多个"""
  quads_list, digits_list = _batch()
  hm_p, reg_p, cls_p = _perfect_preds(quads_list, digits_list)

  for b, quads in enumerate(quads_list):
    pts, _, _ = Inferer.decode(hm_p, reg_p, cls_p, thr=0.05, topk=50)[b]
    # 阈值放得很低（高斯邻域 0.5 以上都算），仍然只应得到真实个数
    assert pts.shape[0] == quads.shape[0]


def test_init_loads_state_dict(tmp_path) -> None:
  """Inferer 应能从 model.state_dict() 还原权重，并自动 eval()"""
  src = CenterNetDigit(num_classes=10)
  ckpt = tmp_path / "ckpt.pt"
  torch.save(src.state_dict(), ckpt)

  inf = Inferer(CenterNetDigit(num_classes=10), ckpt, device="cpu")

  assert isinstance(inf.model, CenterNetDigit)
  assert not inf.model.training, "构造后应处于 eval 模式"
  for a, b in zip(src.state_dict().values(), inf.model.state_dict().values()):
    assert torch.equal(a.cpu(), b.cpu()), "权重没有被正确加载"


def test_init_loads_legacy_checkpoint_without_dropout(tmp_path) -> None:
  """旧权重（head 里没有 Dropout2d）应能自动剥掉 Dropout 后加载"""
  legacy = CenterNetDigit(num_classes=10)          # dropout=0，对应旧结构
  ckpt = tmp_path / "legacy.pt"
  torch.save(legacy.state_dict(), ckpt)

  inf = Inferer(CenterNetDigit(num_classes=10, dropout=0.3), ckpt, device="cpu")

  assert not any(isinstance(m, torch.nn.Dropout2d) for m in inf.model.hm_head), \
      "旧权重加载后 head 里不该还留着 Dropout"
  for a, b in zip(legacy.state_dict().values(), inf.model.state_dict().values()):
    assert torch.equal(a.cpu(), b.cpu()), "权重没有被正确加载"


def test_init_loads_dropout_checkpoint(tmp_path) -> None:
  """带 dropout 训练出的权重，结构与 dropout=0.3 的模型一致，走正常 strict 加载"""
  src = CenterNetDigit(num_classes=10, dropout=0.3)
  ckpt = tmp_path / "drop.pt"
  torch.save(src.state_dict(), ckpt)

  inf = Inferer(CenterNetDigit(num_classes=10, dropout=0.3), ckpt, device="cpu")

  assert any(isinstance(m, torch.nn.Dropout2d) for m in inf.model.hm_head)


def test_init_uses_checkpoint_metadata(tmp_path) -> None:
  """新版 checkpoint 自带 dropout 元信息：带回 dropout 的权重按元信息对齐"""
  src = CenterNetDigit(num_classes=10, dropout=0.3)
  ckpt = tmp_path / "meta.pt"
  torch.save({"state_dict": src.state_dict(),
              "num_classes": 10, "dropout": 0.3}, ckpt)

  inf = Inferer(CenterNetDigit(num_classes=10, dropout=0.3), ckpt, device="cpu")

  assert any(isinstance(m, torch.nn.Dropout2d) for m in inf.model.hm_head)


def test_init_dropout_mismatch_raises(tmp_path) -> None:
  """dropout 不一致又没法自动修正时，要报得清楚"""
  src = CenterNetDigit(num_classes=10, dropout=0.3)
  ckpt = tmp_path / "meta.pt"
  torch.save({"state_dict": src.state_dict(),
              "num_classes": 10, "dropout": 0.3}, ckpt)

  with pytest.raises(ValueError, match="dropout=0.3"):
    Inferer(CenterNetDigit(num_classes=10), ckpt, device="cpu")


def test_load_image_makes_three_channels(tmp_path) -> None:
  """灰度 PNG（mode='L'）经 load_image 后应是 (1,3,384,384)，r18 只吃 3 通道"""
  p = tmp_path / "gray.png"
  Image.new("L", (64, 32), 128).save(p)

  t = load_image(p, (384, 384))

  assert tuple(t.shape) == (1, 3, 384, 384)
  assert t.dtype == torch.float32
