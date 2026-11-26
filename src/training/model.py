import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
from collections.abc import Sequence

os.environ["TORCH_HOME"] = 'D:/Dev-Lab/.TORCH_HOME'
# 这里不要定义模块级 device：model.py 会被 DataLoader worker 通过 spawn import，
# worker 里一旦初始化 CUDA，每个进程都会建一份 context（几百 MB），
# 多个 worker 叠加就会把显存吃光，并以 “CUDA error: unknown error” 形式炸掉训练。
# 设备统一由调用方（主进程）决定。



'''
(h)eat (map) head: 每格"是数字中心"的概率
(reg)ression head: 中心到 4 点的偏移 (4点*x,y)
(cls)assification head: 每格 10 类 digits
'''

# 原始 (B, 1, 384, 384)
class CenterNetDigit(nn.Module):
  """CenterNet-style digit heatmap, offset, and classification model."""

  def __init__(self, num_classes: int = 10, dropout: float = 0.0) -> None:
    """Initialize the backbone and prediction heads.

    Args:
        num_classes (int): Number of digit classes predicted by the
            classification head.
        dropout (float): 三个 head 里的 Dropout2d 丢弃概率（0 = 不加）。
            全卷积网络没有 FC 层，所以用逐通道空间 dropout；
            只在 >0 时插入模块，因此 0 时结构与旧权重完全一致。
            训练时用 `configs.training_cfg.DROPOUT`，推理 / 评估必须用同一个值。
    """
    super().__init__()
    dropout = float(dropout)
    if not 0.0 <= dropout < 1.0:
      raise ValueError(f"dropout 应为 [0, 1) 内的概率，收到 {dropout}")
    r18 = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    self.backbone = nn.Sequential(
        r18.conv1,  # (B, 192, 192, 64)
        r18.bn1,  # (B, 96, 96, 64)
        r18.relu,  # (B, 96, 96, 64)
        r18.maxpool,  # (B, 48, 48, 64)
        r18.layer1,  # (B, 48, 48, 64)
        r18.layer2,  # (B, 24, 24, 128)
    )
    
    def _head(out_ch: int, *, sigmoid: bool = False) -> nn.Sequential:
      """Conv3x3 -> ReLU -> [Dropout2d] -> Conv1x1 [-> Sigmoid]"""
      layers: list[nn.Module] = [
          nn.Conv2d(128, 64, 3, padding=1),
          nn.LeakyReLU(0.1, inplace=True),
      ]
      if dropout > 0.0:
        layers.append(nn.Dropout2d(dropout))
      layers.append(nn.Conv2d(64, out_ch, 1))
      if sigmoid:
        layers.append(nn.Sigmoid())
      return nn.Sequential(*layers)

    # 每格"是数字中心"的概率 out: (B,1,h,w)
    self.hm_head = _head(1, sigmoid=True)
    # 最后一层 Conv1x1 的 bias 从 -2.19 起步（sigmoid≈0.1）；
    # 否则训练后它会被推成正数，让"死区"漏出 0.72 的假高分
    nn.init.constant_(
        [m for m in self.hm_head if isinstance(m, nn.Conv2d)][-1].bias, -2.19)
    # 中心到 4 点的偏移 (4点×x,y) out: (B,8,h,w)
    self.reg_head =_head(8)
    # 每格 10 类 digits (B,10,h,w)
    self.cls_head = _head(num_classes)
    # 结构元信息：保存 checkpoint 时一并写出去，推理端可据此对齐结构
    # （不是 Parameter/Buffer，不会进 state_dict）
    self.num_classes = int(num_classes)
    self.dropout_p = dropout
    # 不在这里自动 .to(device)：设备由调用方决定
    # （单元测试用 CPU，训练时写 model.to(device) 并把 batch 也搬到同一设备）。
    # 早期版本在这里搬过一次，结果是模型半 GPU 半 CPU，前向直接报
    # "Input type (FloatTensor) and weight type (cuda.FloatTensor)"。

  def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run a forward pass and produce the three prediction maps.

    Args:
        x (torch.Tensor): Input image batch with shape
            `(batch_size, 3, height, width)`.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Heatmap,
        offset, and classification tensors.
    """
    x = self.backbone(x)
    return self.hm_head(x), self.reg_head(x), self.cls_head(x)



def _draw_gaussian(hm_b: torch.Tensor, ix: int, iy: int, radius: int = 2) -> None:
  """在 (iy, ix) 这个格子上画一个高斯峰到 `hm_b[0]`（形状 (1, fh, fw)）。

  中心格严格为 1.0（`focal_loss` 用 `gt.eq(1)` 找正样本），
  邻域按高斯衰减，给量化误差留一点容忍；重叠时取 `maximum`，不会互相覆盖。

  Args:
      hm_b (torch.Tensor): 单张图的 heatmap，形状 `(1, fh, fw)`。
      ix (int): 中心格列索引。
      iy (int): 中心格行索引。
      radius (int): 高斯半径（格数）。
  """
  fh, fw = hm_b.shape[-2], hm_b.shape[-1]
  sigma = max(radius / 3.0, 1e-3)
  x0, x1 = max(ix - radius, 0), min(ix + radius + 1, fw)
  y0, y1 = max(iy - radius, 0), min(iy + radius + 1, fh)
  ys = torch.arange(y0, y1, dtype=torch.float32)[:, None]
  xs = torch.arange(x0, x1, dtype=torch.float32)[None, :]
  g = torch.exp(-((xs - ix) ** 2 + (ys - iy) ** 2) / (2 * sigma ** 2))
  patch = hm_b[0, y0:y1, x0:x1]
  hm_b[0, y0:y1, x0:x1] = torch.maximum(patch, g)


def build_target(
    quads_list: Sequence[torch.Tensor],
    digits_list: Sequence[torch.Tensor],
    fh: int,
    fw: int,
    radius: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Construct CenterNet targets from normalized quadrilateral coordinates.

  `hm` 用高斯渲染（中心 1.0、邻域衰减），`mask` 仍是 0/1 的正样本位置：
  前者是 focal loss 的软目标，后者用来从 reg/cls 里挑出正样本格。

  Args:
      quads_list (Sequence[torch.Tensor]): Quadrilateral tensors for each
          sample. Each tensor has shape `(N, 4, 2)`.
      digits_list (Sequence[torch.Tensor]): Digit-label tensors for each
          sample.
      fh (int): Target feature-map height.
      fw (int): Target feature-map width.
      radius (int): 高斯半径（格数），越大对定位误差越宽容。Defaults to 2.

  Returns:
      tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
      Heatmap `(B, 1, fh, fw)`, offset `(B, 8, fh, fw)`,
      classification `(B, fh, fw)` int64, and valid-position mask
      `(B, fh, fw)`.
  """
  B = len(quads_list)
  hm = torch.zeros(B, 1, fh, fw)
  reg = torch.zeros(B, 8, fh, fw)
  # cls 存「该格子上的数字类别索引」，形状与 mask 对齐 (B, fh, fw)，
  # 这样 total_loss 里 cls_t[mask] 得到 (N,) 的标签，能直接进 cross_entropy；
  # 预测端 cls_head 输出 (B, C, fh, fw)，permute 后在 mask 上取 (N, C) logits。
  cls = torch.zeros(B, fh, fw, dtype=torch.long)
  mask = torch.zeros(B, fh, fw)

  for b in range(B):
    for quads, digits in zip(quads_list[b], digits_list[b]):
      cx, cy = quads[:, 0].mean(), quads[:, 1].mean()  # 中心点（浮点，仅用于定位格子）
      # 出界处理（量化到格子上；亚像素部分交给 reg 头回归）
      ix = min(max(int(cx * fw), 0), fw - 1)
      iy = min(max(int(cy * fh), 0), fh - 1)
      # 只有正确位置才标记相应数据
      _draw_gaussian(hm[b], ix, iy, radius)
      # offset 定义为「4 个角点 - 所在格子」的坐标差（单位：feature 格）。
      # 必须相对格子而不是浮点中心：推理端只知道格子下标，
      # 用 (格子 + off) / (fw, fh) 才能精确还原角点。
      cell = torch.tensor([ix, iy], dtype=quads.dtype)
      offset = quads * torch.tensor([fw, fh], dtype=quads.dtype) - cell
      reg[b, :, iy, ix] = offset.reshape(-1)
      cls[b, iy, ix] = int(digits)
      mask[b, iy, ix] = 1.0
  return hm, reg, cls, mask


def focal_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
  """Calculate focal loss for the center heatmap.

  Args:
      pred (torch.Tensor): Predicted heatmap values.
      gt (torch.Tensor): Ground-truth heatmap values.

  Returns:
      torch.Tensor: Scalar focal-loss tensor.
  """
  pos = gt.eq(1).float()
  neg = gt.lt(1).float()
  # 先转 fp32 再 clamp：fp16 表示不了 1-1e-6（会 round 成 1.0），
  # 于是 log(1-p) = log(0) = -inf，训练 loss 直接变 nan
  pred = pred.float().clamp(1e-6, 1 - 1e-6)
  pos_loss = -torch.log(pred) * torch.pow(1 - pred, 2) * pos
  neg_loss = -torch.log(1 - pred) * torch.pow(pred,
                                              2) * torch.pow(1 - gt, 4)*neg
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()
  return (pos_loss.sum() + neg_loss.sum()) / pos.sum().clamp(min=1.0)


def total_loss(
    hm_p: torch.Tensor,
    reg_p: torch.Tensor,
    cls_p: torch.Tensor,
    hm_t: torch.Tensor,
    reg_t: torch.Tensor,
    cls_t: torch.Tensor,
    mask: torch.Tensor,
    l_hm: float = 1.0,
    l_reg: float = 1.0,
    l_cls: float = 1.0,
) -> tuple[torch.Tensor, tuple[float, float, float]]:
  """Calculate the combined heatmap, offset, and classification loss.

  Args:
      hm_p (torch.Tensor): Predicted center heatmap.
      reg_p (torch.Tensor): Predicted offset map.
      cls_p (torch.Tensor): Predicted classification logits.
      hm_t (torch.Tensor): Ground-truth center heatmap.
      reg_t (torch.Tensor): Ground-truth offset map.
      cls_t (torch.Tensor): Ground-truth class labels.
      mask (torch.Tensor): Valid-position mask.
      l_hm (float): Heatmap-loss weight.
      l_reg (float): Offset-loss weight.
      l_cls (float): Classification-loss weight.

  Returns:
      tuple[torch.Tensor, tuple[float, float, float]]: Total loss and
      a tuple containing the heatmap, offset, and classification losses.
  """
  L_hm = focal_loss(hm_p, hm_t)
  m = mask.bool()
  # (B, C ,H, W) -> (B, H, W, C) -> (N, C)
  reg_p_m = reg_p.permute(0, 2, 3, 1)[m]
  reg_t_m = reg_t.permute(0, 2, 3, 1)[m]
  cls_p_m = cls_p.permute(0, 2, 3, 1)[m]
  cls_t_m = cls_t[m]
  
  L_reg = F.smooth_l1_loss(reg_p_m, reg_t_m) if reg_p_m.numel() else reg_p.sum() * 0
  L_cls = F.cross_entropy(cls_p_m, cls_t_m) if cls_p_m.numel() else cls_p.sum() * 0
  total_loss = l_hm * L_hm + l_reg * L_reg + l_cls * L_cls
  return total_loss, (L_hm.item(), L_reg.item(), L_cls.item())
