import torch
import os
from pathlib import Path
from torch.utils.data import DataLoader
from .model import CenterNetDigit, build_target, total_loss


class Trainer:
  def __init__(self, epochs, device, early_stopping_patience: int = 3, ckpt_path="ckpt_last.pt"):
    self.epochs = epochs
    self.device = device
    self.ckpt_path = ckpt_path
    self.early_stopping_patience = early_stopping_patience
    # 当前被冻结的 backbone 子模块下标（冻结块的 BN 要在每个 epoch 重新切回 eval）
    self._frozen_blocks: set[int] = set()

  def _load_model(self, model: CenterNetDigit, ckpt_path=None) -> CenterNetDigit:
    """把 checkpoint 还原成可用的模型（Trainer 存的是 ``model.state_dict()``）

    Args:
        model (CenterNetDigit): 待载入权重的模型（结构需与权重一致）
        ckpt_path (str | Path | None): 权重路径，缺省用 ``self.ckpt_path``
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

  def freeze_backbone(self, model: CenterNetDigit, blocks=None) -> None:
    """冻结 backbone 参数（默认整个 backbone，只训三个 head）

    ``requires_grad=False`` 只停梯度，Batchnorm 的 running_mean/var 在
    ``model.train()`` 下仍会更新；要连 BN 统计量一起冻住，本类在
    每个 epoch 开头对冻结块调 ``.eval()``。

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
    print(f"已冻结 backbone 子模块 {sorted(frozen)}")

  def unfreeze_backbone(self, model: CenterNetDigit) -> None:
    """解冻 backbone 全部参数（第二阶段微调用）

    解冻后 ``requires_grad`` 变了，必须**重新创建 optimizer**（旧的
    param_group 里没有这些参数，学习率也不会按新组生效），
    并把学习率调小（如 1e-4）避免打坏预训练特征。
    """
    for p in model.backbone.parameters():
      p.requires_grad = True

    self._frozen_blocks = set()
    print("已解冻 backbone（记得重建 optimizer / 调小学习率）")

  def _set_frozen_bn_eval(self, model: CenterNetDigit) -> None:
    """``model.train()`` 会把所有子模块切回训练态，这里把冻结块再切回 eval"""
    for i, block in enumerate(model.backbone):
      if i in self._frozen_blocks:
        block.eval()

  def train(self,
            model: CenterNetDigit,
            dataloader: DataLoader,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler._LRScheduler,
            freeze_backbone: bool = True,
            ) -> float:
    """训练模型，返回最好那一轮的平均总 loss

    「冻结 → 解冻」两阶段训练的用法::

        trainer.train(model, dl, opt, sched, freeze_backbone=True)   # 只训 head
        trainer.unfreeze_backbone(model)                            # 解冻
        opt, sched = 重新建（更小的 lr）
        trainer.train(model, dl, opt, sched, freeze_backbone=False)  # 整体微调

    Args:
        model (CenterNetDigit): 待训练模型
        dataloader (DataLoader): 训练集 dataloader
        optimizer (torch.optim.Optimizer): 优化器（解冻后需重建）
        scheduler (LRScheduler): 学习率调度器
        freeze_backbone (bool): True 时先冻结整个 backbone
    """
    model.to(self.device)
    if freeze_backbone:
      self.freeze_backbone(model)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    lr = optimizer.param_groups[0]["lr"]
    print(f"训练开始 | device={self.device} | epochs={self.epochs} | lr={lr:g} | "
          f"{optimizer.__class__.__name__}+{scheduler.__class__.__name__} | "
          f"可训练参数 {n_train:,}/{n_total:,} ({n_train / max(n_total, 1):.1%})")

    best_loss = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    bad_epochs = 0

    for epoch in range(self.epochs):
      model.train()
      self._set_frozen_bn_eval(model)      # 冻结块的 BN 保持推理态
      agg = [0.0] * 4  # loss, heatmap, offset, cls
      for imgs, quads_list, digits_list in dataloader:
        imgs = imgs.to(self.device)
        hm_t, reg_t, cls_t, mask = build_target(
            quads_list, digits_list, imgs.shape[2] // 8, imgs.shape[3] // 8)
        hm_t, reg_t, cls_t, mask = hm_t.to(self.device), reg_t.to(
            self.device), cls_t.to(self.device), mask.to(self.device)

        hm_p, reg_p, cls_p = model(imgs)
        loss, parts = total_loss(hm_p, reg_p, cls_p, hm_t, reg_t, cls_t, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for i, v in enumerate((loss.item(), *parts)):
          agg[i] += v

      scheduler.step()
      n = max(len(dataloader), 1)
      loss_avg, hm_avg, reg_avg, cls_avg = (v / n for v in agg)

      # 早停：连续 early_stopping_patience 轮平均 loss 没有变好就停
      # （best_loss / bad_epochs 必须定义在 epoch 循环外，否则条件恒为 False）
      if loss_avg < best_loss:
        best_loss, bad_epochs = loss_avg, 0
        best_state = {k: v.detach().cpu().clone()
                      for k, v in model.state_dict().items()}
        note = "  (best)"
      else:
        bad_epochs += 1
        note = f"  (no improve {bad_epochs}/{self.early_stopping_patience})"

      print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss_avg:.4f}, "
            f"Heatmap: {hm_avg:.4f}, Offset: {reg_avg:.4f}, "
            f"Classification: {cls_avg:.4f}{note}")

      if bad_epochs >= self.early_stopping_patience:
        print(f"Early stopping at epoch {epoch+1}: "
              f"loss 已连续 {bad_epochs} 轮未下降")
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
    print(f"权重已保存: {self.ckpt_path}  (best loss {best_loss:.4f})")
    return best_loss

  def evaluate(self, model: CenterNetDigit, dataloader: DataLoader,
               ckpt_path=None) -> float:
    """在测试集上评估模型的泛化能力，返回平均总 loss

    Args:
        model (CenterNetDigit): 待载入权重的模型（会被载入后再评估）
        dataloader (DataLoader): 测试集 dataloader
        ckpt_path (str | Path | None): 权重路径，缺省用 ``self.ckpt_path``
    """
    model = self._load_model(model, ckpt_path)
    model.to(self.device)
    model.eval()
    agg = [.0, .0, .0, .0]  # loss, heatmap, offset, cls
    with torch.no_grad():
      for imgs, quads_list, digits_list in dataloader:
        imgs = imgs.to(self.device)
        hm_t, reg_t, cls_t, mask = build_target(
            quads_list, digits_list, imgs.shape[2] // 8, imgs.shape[3] // 8)
        hm_t, reg_t, cls_t, mask = hm_t.to(self.device), reg_t.to(
            self.device), cls_t.to(self.device), mask.to(self.device)

        hm_p, reg_p, cls_p = model(imgs)
        loss, parts = total_loss(hm_p, reg_p, cls_p, hm_t, reg_t, cls_t, mask)
        for i, v in enumerate((loss.item(), *parts)):
          agg[i] += v

    n = max(len(dataloader), 1)
    loss_avg, hm_avg, reg_avg, cls_avg = (v / n for v in agg)
    print(f"Evaluation Loss: {loss_avg:.4f}, Heatmap: {hm_avg:.4f}, "
          f"Offset: {reg_avg:.4f}, Classification: {cls_avg:.4f}")
    return loss_avg
