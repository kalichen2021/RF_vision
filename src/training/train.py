import os
import time
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader

from src.utils.logging_tools import get_logger

from .model import CenterNetDigit, build_target, total_loss

# 只发日志，不直接 print：输出格式 / 级别由 CLI 入口的 setup_logging() 决定
_logger = get_logger(__name__)


class Trainer:
  def __init__(self, epochs: int, device: str,
               early_stopping_patience: int = 3,
               ckpt_path: str | Path = "ckpt_last.pt",
               use_amp: bool = False, cudnn_benchmark: bool = False) -> None:
    self.epochs = epochs
    self.device = device
    self.ckpt_path = ckpt_path
    self.early_stopping_patience = early_stopping_patience
    self.use_amp = bool(use_amp)
    # cudnn.benchmark：输入尺寸固定时让 cudnn 自动挑最快的卷积算法
    if cudnn_benchmark and str(device).startswith("cuda"):
      torch.backends.cudnn.benchmark = True
    # 当前被冻结的 backbone 子模块下标（冻结块的 BN 要在每个 epoch 重新切回 eval）
    self._frozen_blocks: set[int] = set()

  def _load_model(self, model: CenterNetDigit,
                  ckpt_path: str | Path | None = None) -> CenterNetDigit:
    """把 checkpoint 还原成可用的模型（Trainer 存的是 `model.state_dict()`）

    Args:
        model (CenterNetDigit): 待载入权重的模型（结构需与权重一致）
        ckpt_path (str | Path | None): 权重路径，缺省用 `self.ckpt_path`
    """
    ckpt_path = Path(ckpt_path if ckpt_path is not None else self.ckpt_path)
    if not ckpt_path.exists():
      raise FileNotFoundError(
          f"找不到 checkpoint: {ckpt_path}\n"
          "  先跑完 train() 生成权重，或把 ckpt_path 指向已有权重"
          "（如 configs.path.SELECT_MODEL）")

    try:
      state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:        # 老式 checkpoint 里可能含非张量对象
      state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(state, torch.nn.Module):       # 直接存了整个模型
      return state
    if isinstance(state, dict) and "state_dict" in state:
      state = state["state_dict"]

    # strict=True：键名或形状不匹配就当场报错，避免带病推理
    model.load_state_dict(state)
    return model

  def freeze_backbone(self, model: CenterNetDigit,
                      blocks: Iterable[int] | None = None) -> None:
    """冻结 backbone 参数（默认整个 backbone，只训三个 head）

    `requires_grad=False` 只停梯度，Batchnorm 的 running_mean/var 在
    `model.train()` 下仍会更新；要连 BN 统计量一起冻住，本类在
    每个 epoch 开头对冻结块调 `.eval()`。

    Args:
        model (CenterNetDigit): 目标模型
        blocks (Iterable[int] | None): 要冻结的 backbone 子模块下标
            （0=conv1, 1=bn1, 4=layer1, 5=layer2；2/3 无参数）。
            None 表示全冻。
    """
    frozen = set(range(len(model.backbone))) if blocks is None else set(blocks)
    for i, block in enumerate(model.backbone):
      if i in frozen:
        for p in block.parameters():
          p.requires_grad = False

    self._frozen_blocks = frozen
    _logger.info("已冻结 backbone 子模块 %s", sorted(frozen))

  def unfreeze_backbone(self, model: CenterNetDigit) -> None:
    """解冻 backbone 全部参数（第二阶段微调用）

    解冻后 `requires_grad` 变了，必须**重新创建 optimizer**（旧的
    param_group 里没有这些参数，学习率也不会按新组生效），
    并把学习率调小（如 1e-4）避免打坏预训练特征。
    """
    for p in model.backbone.parameters():
      p.requires_grad = True

    self._frozen_blocks = set()
    _logger.info("已解冻 backbone（记得重建 optimizer / 调小学习率）")

  def _set_frozen_bn_eval(self, model: CenterNetDigit) -> None:
    """`model.train()` 会把所有子模块切回训练态，这里把冻结块再切回 eval"""
    for i, block in enumerate(model.backbone):
      if i in self._frozen_blocks:
        block.eval()

  @staticmethod
  def _scheduler_step(scheduler: Any, metric: float) -> None:
    """推进学习率调度器

    `ReduceLROnPlateau` 必须喂指标（`scheduler.step(metric)`），
    其余按 epoch 走的调度器（如 CosineAnnealingLR）不带参数。

    Args:
        scheduler: 学习率调度器实例
        metric (float): 用于判停 / 降 lr 的指标（有验证集时是 val loss）
    """
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
      scheduler.step(metric)
    else:
      scheduler.step()

  @torch.no_grad()
  def _eval_losses(self, model: CenterNetDigit, dataloader: DataLoader
                   ) -> tuple[float, float, float, float]:
    """在给定 dataloader 上跑一遍 eval 模式，返回四路平均 loss

    与训练阶段的区别：`model.eval()` 关掉 head 里的 dropout、backbone 的 BN
    改用 running stats，所以这个数才代表「推理时」的真实水平。

    Args:
        model (CenterNetDigit): 已在 `self.device` 上的模型
        dataloader (DataLoader): 验证 / 测试集

    Returns:
        tuple[float, float, float, float]: (loss, heatmap, offset, classification)
    """
    was_training = model.training
    model.eval()
    device_type = "cuda" if str(self.device).startswith("cuda") else "cpu"
    amp_enabled = bool(self.use_amp and device_type == "cuda")
    agg = [.0, .0, .0, .0]  # loss, heatmap, offset, cls
    for imgs, quads_list, digits_list in dataloader:
      imgs = imgs.to(self.device, non_blocking=True)
      hm_t, reg_t, cls_t, mask = build_target(
          quads_list, digits_list, imgs.shape[2] // 8, imgs.shape[3] // 8)
      hm_t, reg_t, cls_t, mask = (
          hm_t.to(self.device, non_blocking=True),
          reg_t.to(self.device, non_blocking=True),
          cls_t.to(self.device, non_blocking=True),
          mask.to(self.device, non_blocking=True))

      with torch.amp.autocast(device_type, enabled=amp_enabled):
        hm_p, reg_p, cls_p = model(imgs)
        loss, parts = total_loss(hm_p, reg_p, cls_p, hm_t, reg_t, cls_t, mask)
      for i, v in enumerate((loss.item(), *parts)):
        agg[i] += v

    if was_training:
      # 验证完必须切回训练态，否则 dropout 会被一直关掉；冻结块的 BN 也要复原
      model.train()
      self._set_frozen_bn_eval(model)
    n = max(len(dataloader), 1)
    return tuple(v / n for v in agg)

  def train(self,
            model: CenterNetDigit,
            dataloader: DataLoader,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler._LRScheduler,
            freeze_backbone: bool = True,
            val_dataloader: DataLoader | None = None,
            ) -> float:
    """训练模型，返回监控指标最好的那一轮的值

    「冻结 → 解冻」两阶段训练的用法::

        trainer.train(model, dl, opt, sched, freeze_backbone=True,
                      val_dataloader=dl_test)                    # 只训 head
        trainer.unfreeze_backbone(model)                          # 解冻
        opt, sched = 重新建（更小的 lr）
        trainer.train(model, dl, opt, sched, freeze_backbone=False,
                      val_dataloader=dl_test)                    # 整体微调

    Args:
        model (CenterNetDigit): 待训练模型
        dataloader (DataLoader): 训练集 dataloader
        optimizer (torch.optim.Optimizer): 优化器（解冻后需重建）
        scheduler (LRScheduler): 学习率调度器（支持 ReduceLROnPlateau）
        freeze_backbone (bool): True 时先冻结整个 backbone
        val_dataloader (DataLoader | None): 验证集；给定时每轮跑一次 eval，
            并**用它**决定 best 权重 / 早停 / 调度器（训练 loss 受 dropout 与
            BN batch 统计影响，不能代表泛化能力）。None 时退回训练 loss。
    """
    model.to(self.device)
    if freeze_backbone:
      self.freeze_backbone(model)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    lr = optimizer.param_groups[0]["lr"]
    device_type = "cuda" if str(self.device).startswith("cuda") else "cpu"
    amp_enabled = bool(self.use_amp and device_type == "cuda")
    scaler = torch.amp.GradScaler(device_type, enabled=amp_enabled)
    _logger.info("训练开始 | device=%s | epochs=%d | lr=%g | %s+%s | amp=%s | "
                 "可训练参数 %s/%s (%.1f%%)",
                 self.device, self.epochs, lr,
                 optimizer.__class__.__name__, scheduler.__class__.__name__,
                 "on" if amp_enabled else "off",
                 f"{n_train:,}", f"{n_total:,}", n_train / max(n_total, 1) * 100)

    # 监控指标：有验证集时是 val loss，否则退回训练 loss
    best_metric = float("inf")
    metric_name = "val loss" if val_dataloader is not None else "loss"
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    bad_epochs = 0

    for epoch in range(self.epochs):
      model.train()
      self._set_frozen_bn_eval(model)      # 冻结块的 BN 保持推理态
      agg = [0.0] * 4  # loss, heatmap, offset, cls
      epoch_start = time.perf_counter()
      data_wait = 0.0                      # 等 dataloader 出数据的累计时间
      last = epoch_start
      for imgs, quads_list, digits_list in dataloader:
        now = time.perf_counter()
        data_wait += now - last            # 上一轮 step 结束到收到这个 batch
        imgs = imgs.to(self.device, non_blocking=True)
        hm_t, reg_t, cls_t, mask = build_target(
            quads_list, digits_list, imgs.shape[2] // 8, imgs.shape[3] // 8)
        hm_t, reg_t, cls_t, mask = (
            hm_t.to(self.device, non_blocking=True),
            reg_t.to(self.device, non_blocking=True),
            cls_t.to(self.device, non_blocking=True),
            mask.to(self.device, non_blocking=True))

        with torch.amp.autocast(device_type, enabled=amp_enabled):
          hm_p, reg_p, cls_p = model(imgs)
          loss, parts = total_loss(hm_p, reg_p, cls_p, hm_t, reg_t, cls_t, mask)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        for i, v in enumerate((loss.item(), *parts)):
          agg[i] += v
        last = time.perf_counter()

      n = max(len(dataloader), 1)
      loss_avg, hm_avg, reg_avg, cls_avg = (v / n for v in agg)

      # 每轮在验证集上跑一次 eval：best / 早停 / 调度器全部盯 val loss
      if val_dataloader is not None:
        val_loss, val_hm, val_reg, val_cls = self._eval_losses(
            model, val_dataloader)
        metric = val_loss
        val_note = (f" | Val Loss: {val_loss:.4f}, Heatmap: {val_hm:.4f}, "
                    f"Offset: {val_reg:.4f}, Classification: {val_cls:.4f}")
      else:
        metric = loss_avg
        val_note = ""

      self._scheduler_step(scheduler, metric)
      epoch_time = time.perf_counter() - epoch_start
      # GPU 忙的比例 ≈ (总时间 - 等数据时间) / 总时间；持续偏低说明数据管线是瓶颈
      gpu_ratio = max(0.0, 1.0 - data_wait / max(epoch_time, 1e-6))
      lr_now = optimizer.param_groups[0]["lr"]

      # 早停：连续 early_stopping_patience 轮监控指标没有变好就停
      # （best_metric / bad_epochs 必须定义在 epoch 循环外，否则条件恒为 False）
      if metric < best_metric:
        best_metric, bad_epochs = metric, 0
        best_state = {k: v.detach().cpu().clone()
                      for k, v in model.state_dict().items()}
        note = "  (best)"
      else:
        bad_epochs += 1
        note = f"  (no improve {bad_epochs}/{self.early_stopping_patience})"

      _logger.info("Epoch %d/%d, Loss: %.4f, Heatmap: %.4f, Offset: %.4f, "
                   "Classification: %.4f%s | lr=%.2e | %.1fs, 等数据 %.1fs, "
                   "GPU≈%.0f%%%s",
                   epoch + 1, self.epochs, loss_avg, hm_avg, reg_avg, cls_avg,
                   val_note, lr_now, epoch_time, data_wait, gpu_ratio * 100, note)

      if bad_epochs >= self.early_stopping_patience:
        _logger.info("Early stopping at epoch %d：%s 已连续 %d 轮未下降",
                     epoch + 1, metric_name, bad_epochs)
        break

    # 保存「最好的一轮」而不是最后一轮：早停时最后一轮往往已经抖坏了
    model.load_state_dict(best_state)
    # 连结构元信息一起存：推理端据此选对 dropout / num_classes，
    # 避免“训练时不带 Dropout2d、推理时带了”导致的键名错位
    torch.save({
        "state_dict": model.state_dict(),
        "num_classes": getattr(model, "num_classes", None),
        "dropout": getattr(model, "dropout_p", None),
    }, self.ckpt_path)
    _logger.info("权重已保存: %s  (best %s %.4f)",
                 self.ckpt_path, metric_name, best_metric)
    return best_metric

  def evaluate(self, model: CenterNetDigit, dataloader: DataLoader,
               ckpt_path: str | Path | None = None) -> float:
    """在测试集上评估模型的泛化能力，返回平均总 loss

    Args:
        model (CenterNetDigit): 待载入权重的模型（会被载入后再评估）
        dataloader (DataLoader): 测试集 dataloader
        ckpt_path (str | Path | None): 权重路径，缺省用 `self.ckpt_path`
    """
    model = self._load_model(model, ckpt_path)
    model.to(self.device)
    loss_avg, hm_avg, reg_avg, cls_avg = self._eval_losses(model, dataloader)
    _logger.info("Evaluation Loss: %.4f, Heatmap: %.4f, Offset: %.4f, "
                 "Classification: %.4f", loss_avg, hm_avg, reg_avg, cls_avg)
    return loss_avg
