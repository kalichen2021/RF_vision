"""`src/training/train.py` 的单元测试（不跑完整训练，只验冻结 / 加载逻辑）。"""

from __future__ import annotations

import pytest
import torch
from src.training.model import CenterNetDigit
from src.training.train import Trainer


def _grad_flags(blocks) -> list[bool]:
  return [p.requires_grad for block in blocks for p in block.parameters()]


def test_freeze_and_unfreeze_backbone() -> None:
  """冻结 = backbone 全部 requires_grad=False，head 保持可训练"""
  model = CenterNetDigit(num_classes=10)
  trainer = Trainer(epochs=1, device="cpu")

  trainer.freeze_backbone(model)

  assert not any(_grad_flags(model.backbone)), "backbone 应全部冻结"
  assert any(p.requires_grad for p in model.hm_head.parameters()), "head 不该被冻结"
  assert trainer._frozen_blocks == set(range(len(model.backbone)))

  trainer.unfreeze_backbone(model)

  assert all(_grad_flags(model.backbone)), "解冻后 backbone 应全部可训练"
  assert trainer._frozen_blocks == set()


def test_freeze_only_selected_blocks() -> None:
  """blocks 只冻结指定子模块（0=conv1, 1=bn1，其余保持可训练）"""
  model = CenterNetDigit(num_classes=10)
  trainer = Trainer(epochs=1, device="cpu")

  trainer.freeze_backbone(model, blocks=(0, 1))

  frozen = [block for i, block in enumerate(model.backbone) if i in (0, 1)]
  others = [block for i, block in enumerate(model.backbone) if i not in (0, 1)]
  assert not any(_grad_flags(frozen))
  assert all(_grad_flags(others))
  assert trainer._frozen_blocks == {0, 1}


def test_load_model_missing_checkpoint(tmp_path) -> None:
  """权重不存在时要给人话，而不是 torch.load 的原始报错"""
  model = CenterNetDigit(num_classes=10)
  trainer = Trainer(epochs=1, device="cpu", ckpt_path=tmp_path / "nope.pt")

  with pytest.raises(FileNotFoundError, match="找不到 checkpoint"):
    trainer._load_model(model)
    
def test_single(tmp_path) -> None:
  """只训练一次（空 dataloader：验证前向前的准备流程不报错）"""
  model = CenterNetDigit(num_classes=10)
  trainer = Trainer(epochs=1,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    ckpt_path=tmp_path / "ckpt.pt")
  opt = torch.optim.Adam(model.parameters(), lr=1e-3)
  sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1)
  trainer.train(model, [], opt, sched, freeze_backbone=False)
  assert all(_grad_flags(model.backbone))