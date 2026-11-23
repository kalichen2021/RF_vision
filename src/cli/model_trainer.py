import torch
import os
import sys
from pathlib import Path
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.training_cfg import (EPOCHS_HEAD, EPOCHS_FINETUNE, BATCH_SIZE, TRAININGSET_RATIO,
                                  DROPOUT, EARLY_STOPPING_PATIENCE,
                                  LR_HEAD, LR_FINETUNE)
from configs.path import SELECT_IMGSET, DATASET_LABEL_JSON, MODELS_DIR, SELECT_MODEL
from configs.log import UUID as _uuid
from src.training.dataset import DigitDataset, collate_fn
from src.training.model import CenterNetDigit
from src.training.train import Trainer

# TODO: 训练时，按四个点的顺序排列，让模型分清上下左右。
# TODO: 神经网络对尺度缩放的兼容性不够敏感

device = "cuda" if torch.cuda.is_available() else "cpu"

ds = DigitDataset(DATASET_LABEL_JSON, SELECT_IMGSET, TRAININGSET_RATIO)
dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

ds_test = DigitDataset(DATASET_LABEL_JSON, SELECT_IMGSET,
                       TRAININGSET_RATIO, mode="testing")
dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE,
                     shuffle=False, collate_fn=collate_fn)
model = CenterNetDigit(num_classes=10, dropout=DROPOUT)
ckpt_path = MODELS_DIR / f"ckpt_{_uuid}.pt"


def make_trainer(epochs: int) -> Trainer:
  """每阶段一个 Trainer：轮数与 scheduler 的 T_max 一一对应"""
  return Trainer(epochs=epochs, device=device,
                 early_stopping_patience=EARLY_STOPPING_PATIENCE,
                 ckpt_path=ckpt_path)


def build_opt(lr=LR_HEAD):
  """按当前 requires_grad 重建优化器

  必须重建：Adam 只为「有梯度」的参数建 state，
  冻结/解冻后旧 optimizer 的 param_group 已经不合适了。
  """
  params = [p for p in model.parameters() if p.requires_grad]
  return torch.optim.Adam(params, lr=lr)


if __name__ == "__main__":
  # 阶段一：冻结 backbone，只训三个 head（学习率可以大一点，轮数少一些）
  opt = build_opt(LR_HEAD)
  sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_HEAD)
  make_trainer(EPOCHS_HEAD).train(model, dl, opt, sched, freeze_backbone=True)

  # 阶段二：解冻 backbone 整体微调（学习率降一个量级，轮数多一些）
  trainer = make_trainer(EPOCHS_FINETUNE)
  trainer.unfreeze_backbone(model)
  opt = build_opt(LR_FINETUNE)
  sched = torch.optim.lr_scheduler.CosineAnnealingLR(
      opt, T_max=EPOCHS_FINETUNE)
  trainer.train(model, dl, opt, sched, freeze_backbone=False)

  # 两阶段都训完后，ckpt_path 里是「阶段二中最好的一轮」
  trainer.evaluate(model, dl_test)
