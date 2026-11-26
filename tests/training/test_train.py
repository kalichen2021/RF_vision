"""`src/training/train.py` 的单元测试（不跑完整训练，只验冻结 / 加载逻辑）。"""

from __future__ import annotations

import json

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


def test_save_ckpt_meta(tmp_path) -> None:
  """保存权重时要在旁边写一份同名 JSON（含 extra_meta 与最优指标）"""
  model = CenterNetDigit(num_classes=10)
  ckpt = tmp_path / "ckpt_x.pt"
  trainer = Trainer(epochs=1, device="cpu", ckpt_path=ckpt,
                    extra_meta={"stage": "head",
                                "training_config": {"BATCH_SIZE": 48}})
  opt = torch.optim.Adam(model.parameters(), lr=1e-3)
  sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1)
  trainer.train(model, [], opt, sched, freeze_backbone=False)

  meta_path = ckpt.with_suffix(".json")
  assert meta_path.exists(), "权重旁边应该有参数快照 JSON"
  meta = json.loads(meta_path.read_text(encoding="utf-8"))
  assert meta["stage"] == "head"
  assert meta["training_config"]["BATCH_SIZE"] == 48
  assert meta["epochs_run"] == 1
  # 没给验证集 -> 没有 best，值留 None（不再拿训练 loss 凑数）
  assert meta["best_metric"]["name"] == "val loss"
  assert meta["best_metric"]["value"] is None


def test_early_stopping_uses_val_loss(tmp_path, monkeypatch) -> None:
  """早停只看验证 loss：验证 loss 连续 patience 轮没变好才停"""
  model = CenterNetDigit(num_classes=10)
  ckpt = tmp_path / "ckpt_es.pt"
  trainer = Trainer(epochs=10, device="cpu", ckpt_path=ckpt,
                    early_stopping_patience=2, val_interval=1)
  vals = iter([1.0, 0.9, 1.1, 1.2])        # 第 2 轮最优，之后不再变好
  monkeypatch.setattr(trainer, "_eval_losses",
                      lambda *a, **k: (next(vals), 0.0, 0.0, 0.0))
  opt = torch.optim.Adam(model.parameters(), lr=1e-3)
  sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1)

  best = trainer.train(model, [], opt, sched, freeze_backbone=False,
                       val_dataloader=[object()])

  assert len(trainer._records) == 4, "应在第 4 轮早停（连续 2 轮未改善）"
  assert best == pytest.approx(0.9)
  assert trainer._records[1]["is_best"] is True
  assert trainer._records[-1]["bad_epochs"] == 2
  meta = json.loads(ckpt.with_suffix(".json").read_text(encoding="utf-8"))
  assert meta["epochs_run"] == 4
  assert meta["best_metric"]["value"] == pytest.approx(0.9)