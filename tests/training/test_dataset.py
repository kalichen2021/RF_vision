"""`src/training/dataset.py` 的单元测试。"""
from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from src.training.model import CenterNetDigit, build_target, total_loss
from src.training.dataset import DigitDataset, collate_fn
from configs.path import PathConfig


# @pytest.fixture(scope="module")
# def ds() -> DigitDataset:
#   """数据集只读，模块内复用一份。  """
#   return DigitDataset(PathConfig.DATASET_LABEL_JSON, PathConfig.SELECT_IMGSET)

ds = DigitDataset(PathConfig.DATASET_LABEL_JSON, PathConfig.SELECT_IMGSET)
dl = DataLoader(ds, batch_size=4, collate_fn=collate_fn)
model = CenterNetDigit(num_classes=10)
imgs, quads_list, digits_list = next(iter(dl))
fh = fw = 384 // 8
hm_t, reg_t, cls_t, mask = build_target(quads_list, digits_list, fh, fw)


def test_single() -> None:
  img, quads, digits = ds[0]
  print(img.shape, quads.shape, digits)


def test_sanity_check():
  # 按样本统计，才能和 target count 直接对比（mask 是 (B, fh, fw)）
  n_tgt = torch.tensor([q.shape[0] for q in quads_list])
  n_pos = mask.sum(dim=(1, 2)).long()      # 每张图「落到不同格子」的数字数

  print("imgs:", imgs.shape, "| target count:", n_tgt.tolist())
  print("pos count:", n_pos.tolist(), "| 合计:", int(n_pos.sum()),
        "| hm_t:", hm_t.shape, "| reg_t:", reg_t.shape, "| cls_t:", cls_t.shape)

  # 两个数字中心如果量化到同一个格子，后来的会覆盖前面的，这个断言就是用来发现它的
  assert torch.equal(
      n_tgt, n_pos), f"有数字被覆盖: {n_tgt.tolist()} vs {n_pos.tolist()}"
  # hm 用高斯渲染：中心格为 1，邻域 >0，所以总和 >= 正样本数，峰值恒为 1
  assert hm_t.max().item() == 1.0, "heatmap 峰值应为 1（中心格）"
  assert float(hm_t.sum()) >= float(n_pos.sum()), "heatmap 高斯和应不小于正样本数"


def test_loss():
  hm_p, reg_p, cls_p = model.forward(imgs)
  loss, part = total_loss(hm_p, reg_p, cls_p, hm_t, reg_t, cls_t, mask)
  print(
      f"loss: {loss.item():.4f} | heatmap: {part[0]:.4f} | offset: {part[1]:.4f} | cls: {part[2]:.4f}")
  w = model.backbone[0].weight
  before = w.detach().clone()
  loss.backward()  # 反向传播，检查梯度计算是否正常
  print("loss backward ok")
  print(f"grad: {w.grad.sum().item():.4f}")
  assert torch.equal(w, before), "backward 不该改参数"
  assert w.grad is not None and w.grad.abs().sum() > 0, "梯度没流到这个参数"
  assert w.grad.data_ptr() != w.data_ptr(), "grad 与 data 是两块内存"
