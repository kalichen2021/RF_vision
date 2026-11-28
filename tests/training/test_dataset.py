"""``src/training/dataset.py`` 的单元测试。

数据集是「真实产物」：``dataset/processed/<uuid>/labels.json``。
没有可用数据集时整组跳过（先跑 ``src/cli/dataset_preprocess.py`` 生成）。
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from configs.path import PathConfig
from configs.training_cfg import TrainingConfig
from src.training.dataset import DigitDataset, batch_to_torch, collate_fn
from src.training.model import CenterNetDigit, build_target, total_loss

pytestmark = pytest.mark.skipif(
    not PathConfig.DATASET_LABEL_JSON.exists(),
    reason="没有可用的数据集（先跑 src/cli/dataset_preprocess.py）")


@pytest.fixture(scope="module")
def ds() -> DigitDataset:
  """整个模块复用一份数据集（只读）。"""
  return DigitDataset(PathConfig.DATASET_LABEL_JSON, PathConfig.SELECT_IMGSET)


@pytest.fixture(scope="module")
def batch(ds: DigitDataset) -> tuple:
  """取一个 batch：``(imgs, quads_list, digits_list)``。"""
  return batch_to_torch(next(iter(DataLoader(ds, batch_size=4,
                                             collate_fn=collate_fn))))


def test_single(ds: DigitDataset) -> None:
  """单样本：通道数 / 四边形形状 / 标签个数都要对得上。"""
  img, quads, digits = ds[0]
  assert img.ndim == 3 and img.shape[0] == 3
  assert quads.ndim == 3 and quads.shape[1:] == (4, 2)
  assert len(digits) == quads.shape[0]


def test_sanity_check(batch: tuple) -> None:
  """标签生成：中心格峰值为 1，且没有数字因为量化到同一格被覆盖。"""
  imgs, quads_list, digits_list = batch
  fh = fw = imgs.shape[2] // TrainingConfig.FEAT_STRIDE
  hm_t, _reg_t, _cls_t, mask = build_target(quads_list, digits_list, fh, fw)

  n_tgt = torch.tensor([q.shape[0] for q in quads_list])
  n_pos = mask.sum(dim=(1, 2)).long()      # 每张图落到不同格子的数字数

  # 两个数字中心如果量化到同一个格子，后来的会覆盖前面的
  assert torch.equal(
      n_tgt, n_pos), f"有数字被覆盖: {n_tgt.tolist()} vs {n_pos.tolist()}"
  # hm 用高斯渲染：中心格恒为 1，邻域 >0
  assert hm_t.max().item() == 1.0
  assert float(hm_t.sum()) >= float(n_pos.sum())


def test_loss(batch: tuple) -> None:
  """loss 能前向 + 反传，且反传不改参数、梯度确实流到 backbone。"""
  imgs, quads_list, digits_list = batch
  fh = fw = imgs.shape[2] // TrainingConfig.FEAT_STRIDE
  hm_t, reg_t, cls_t, mask = build_target(quads_list, digits_list, fh, fw)

  model = CenterNetDigit(num_classes=10)
  hm_p, reg_p, cls_p = model(imgs)
  loss, _parts = total_loss(hm_p, reg_p, cls_p, hm_t, reg_t, cls_t, mask)

  weight = model.backbone[0].weight
  before = weight.detach().clone()
  loss.backward()

  assert torch.equal(weight, before), "backward 不该改参数"
  assert weight.grad is not None and weight.grad.abs().sum() > 0, "梯度没流到 backbone"
