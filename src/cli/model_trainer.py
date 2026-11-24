import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# 本模块会被 DataLoader 的 spawn worker 以 __main__ / __mp_main__ 方式 import。
# 顶层只保留标准库：torch / torchvision / configs / model 全部延迟到函数内部。
# 否则每个 worker 都会加载一整套 torchvision + ResNet，常驻内存接近 1GB/进程，
# NUM_WORKERS=10 时光 worker 就要 9~10GB，直接把 16G 内存吃爆（GPU 反而空转）。

# TODO: 训练时，按四个点的顺序排列，让模型分清上下左右。
# TODO: 神经网络对尺度缩放的兼容性不够敏感


def get_device() -> str:
  """只在主进程里调用。worker 里不要碰 CUDA（每个进程各建 context 会吃显存）。"""
  import torch
  return "cuda" if torch.cuda.is_available() else "cpu"


def _worker_init(_worker_id: int) -> None:
  """限制 worker 内部线程数，避免和主进程 / 其他 worker 抢 CPU 核"""
  import torch
  from configs.training_cfg import WORKER_TORCH_THREADS

  torch.set_num_threads(max(1, int(WORKER_TORCH_THREADS)))
  try:                       # OpenCV 也会自己开线程池
    import cv2
    cv2.setNumThreads(0)
  except Exception:
    pass


def build_loaders(batch_size=None, num_workers=None):
  """构造训练 / 验证 DataLoader

  关键在 `num_workers` / `pin_memory` / `prefetch_factor`：
  不配子进程时读图、Resize、Normalize 全在主进程里串行做，GPU 每个 batch
  都要等 CPU 喂数据（表现为 CPU 90%、GPU 20%、显存用不满）。
  """
  import torch
  from torch.utils.data import DataLoader

  from configs.path import PathConfig
  from configs.training_cfg import TrainingConfig as tc
  from src.training.dataset import DigitDataset, collate_fn

  batch_size = tc.BATCH_SIZE if batch_size is None else int(batch_size)
  num_workers = tc.NUM_WORKERS if num_workers is None else int(num_workers)
  num_workers = max(0, num_workers)

  pin = bool(tc.PIN_MEMORY and torch.cuda.is_available())
  prefetch = int(tc.PREFETCH_FACTOR)
  if num_workers > 0:
    # 排队上限保护：float32 张量 = batch*3*384*384*4 字节，batch=32 时约 57MB，
    # workers * prefetch 个队列很容易吃掉 ~1GB（再叠一份 pinned 拷贝就翻倍）
    per_batch = batch_size * 3 * 384 * 384 * 4
    budget = max(1, int(1.2e9 // max(per_batch, 1)))
    prefetch = max(1, min(prefetch, budget // num_workers))
  common = dict(
      num_workers=num_workers,
      pin_memory=pin,
      prefetch_factor=prefetch if num_workers > 0 else None,
      persistent_workers=num_workers > 0,   # 不每个 epoch 重新拉起子进程
      worker_init_fn=_worker_init if num_workers > 0 else None,
  )

  ds = DigitDataset(PathConfig.DATASET_LABEL_JSON, PathConfig.SELECT_IMGSET,
                    tc.TRAININGSET_RATIO)
  dl = DataLoader(ds, batch_size=batch_size, shuffle=True,
                  collate_fn=collate_fn, **common)

  ds_test = DigitDataset(PathConfig.DATASET_LABEL_JSON, PathConfig.SELECT_IMGSET,
                         tc.TRAININGSET_RATIO, mode="testing")
  dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False,
                       collate_fn=collate_fn, **common)
  return ds, dl, ds_test, dl_test


def build_opt(model, lr=None):
  """按当前 requires_grad 重建优化器

  必须重建：Adam 只为「有梯度」的参数建 state，
  冻结/解冻后旧 optimizer 的 param_group 已经不合适了。
  """
  import torch
  from configs.training_cfg import TrainingConfig as tc

  lr = tc.LR_HEAD if lr is None else float(lr)
  params = [p for p in model.parameters() if p.requires_grad]
  return torch.optim.Adam(params, lr=lr)


def make_trainer(epochs: int, ckpt_path, dev: str):
  """每阶段一个 Trainer（每轮会在测试集上验证，best / 早停看 val loss）"""
  from configs.training_cfg import TrainingConfig as tc
  from src.training.train import Trainer

  return Trainer(epochs=epochs, device=dev,
                 early_stopping_patience=tc.EARLY_STOPPING_PATIENCE,
                 ckpt_path=ckpt_path, use_amp=tc.USE_AMP,
                 cudnn_benchmark=tc.CUDNN_BENCHMARK)


def make_scheduler(opt):
  """ReduceLROnPlateau：val loss 停滞时按 factor 降 lr

    与 CosineAnnealingLR 的区别：不再按固定轮数把 lr 退到 0，
    而是“不见好才降”，所以训练中后期的 loss 平台不再被学习率结构性地造成。
  """
  import torch
  from configs.training_cfg import TrainingConfig as tc

  return torch.optim.lr_scheduler.ReduceLROnPlateau(
      opt, mode="min", factor=float(tc.LR_PLATEAU_FACTOR),
      patience=int(tc.LR_PLATEAU_PATIENCE), min_lr=float(tc.LR_MIN))


def main() -> None:
  import torch

  from configs.log import LogConfig
  from configs.path import PathConfig
  from configs.training_cfg import TrainingConfig as tc
  from src.training.model import CenterNetDigit
  from src.utils.logging_tools import setup_logging

  # 控制台输出的格式 / 级别统一在这里配置（src 模块内部只发 logger）
  setup_logging()

  # torch 默认按逻辑核数开线程（这台机器是 14），会和 DataLoader 抢 CPU
  if tc.TORCH_NUM_THREADS:
    torch.set_num_threads(int(tc.TORCH_NUM_THREADS))
  dev = get_device()
  # DataLoader 必须在 main 里构造（Windows spawn 会 import 本模块）
  ds, dl, ds_test, dl_test = build_loaders()
  model = CenterNetDigit(num_classes=10, dropout=tc.DROPOUT)
  ckpt_path = PathConfig.MODELS_DIR / f"ckpt_{LogConfig.UUID}.pt"

  print(f"使用设备: {dev} | DataLoader workers={tc.NUM_WORKERS} | "
        f"batch={tc.BATCH_SIZE} | amp={tc.USE_AMP} | "
        f"torch_threads={torch.get_num_threads()}")
  print(f"使用数据集：{PathConfig.SELECT_IMGSET}"
        f" | {len(ds)} + {len(ds_test)} = {len(ds) + len(ds_test)} 张")

  # 验证集：每个 epoch 都跑一次（best / 早停 / 降 lr 都盯它）
  val_dl = dl_test if tc.VALIDATE_EVERY_EPOCH else None

  # 阶段一：冻结 backbone，只训三个 head（学习率可以大一点，轮数少一些）
  opt = build_opt(model, tc.LR_HEAD)
  sched = make_scheduler(opt)
  make_trainer(tc.EPOCHS_HEAD, ckpt_path, dev).train(
      model, dl, opt, sched, freeze_backbone=True, val_dataloader=val_dl)

  # 阶段二：解冻 backbone 整体微调（学习率降一个量级，轮数多一些）
  trainer = make_trainer(tc.EPOCHS_FINETUNE, ckpt_path, dev)
  trainer.unfreeze_backbone(model)
  opt = build_opt(model, tc.LR_FINETUNE)
  sched = make_scheduler(opt)
  trainer.train(model, dl, opt, sched, freeze_backbone=False,
                val_dataloader=val_dl)

  # 两阶段都训完后，ckpt_path 里是「阶段二中 val loss 最好的一轮」
  trainer.evaluate(model, dl_test)


if __name__ == "__main__":
  from configs.path import PathConfig
  from configs.training_cfg import TrainingConfig as tc
  from src.training.model import CenterNetDigit

  # model = CenterNetDigit(num_classes=10, dropout=tc.DROPOUT)
  # ds, dl, ds_test, dl_test = build_loaders()
  # trainer = make_trainer(tc.EPOCHS_FINETUNE, SELECT_MODEL, get_device())
  # trainer.evaluate(model, dl_test)
  main()
