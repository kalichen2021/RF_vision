"""CenterNet 数字检测推理：加载权重 → 前向 → 解码出数字框。

用法::

    inf = Inferer(MODELS_DIR / "ckpt_xxx.pt", device="cuda")
    results = inf.predict(imgs)      # imgs: (B, 3, H, W)，已按训练方式归一化
    # results[i] = (pts_norm (K,4,2), digits (K,), scores (K,))

命令行::

    python src/inference/infer.py -c models/ckpt_xxx.pt \\
        -i dataset/processed/xxx/00000_0019.png

（用脚本路径而不是 `python -m src.inference.infer`：包 `__init__` 已经
导入过本模块，`-m` 会把同一个文件跑两次并给出 RuntimeWarning。）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from configs.training_cfg import TrainingConfig
from src.training.model import CenterNetDigit
from src.utils.image import IMAGENET_MEAN, IMAGENET_STD, load_tensor
from src.utils.logging_tools import get_logger

# 与 DigitDataset / 摄像头共用同一套 ImageNet 归一化参数
_MEAN = list(IMAGENET_MEAN)
_STD = list(IMAGENET_STD)
_logger = get_logger(__name__)


def build_transform(img_size: tuple[int, int] | None = None) -> T.Compose:
  """训练时用的预处理（灰度 PNG 也要补成三通道，r18 只吃 3 通道）

  Args:
      img_size (tuple[int, int] | None): 目标输入尺寸 `(高, 宽)`；
          None 时读 `configs.training_cfg.IMG_SIZE`

  Returns:
      torchvision.transforms.Compose: 可直接作用于 PIL 图像
  """
  img_size = tuple(TrainingConfig.IMG_SIZE if img_size is None else img_size)
  return T.Compose([
      T.Resize(tuple(img_size)),
      T.ToTensor(),
      T.Normalize(mean=_MEAN, std=_STD),
  ])


def load_image(path: str | Path,
               img_size: tuple[int, int] | None = None) -> torch.Tensor:
  """读一张图并转成 `(1, 3, H, W)` 张量（还没搬到 device）。

  Args:
      path: 图片路径。
      img_size: 目标输入尺寸 `(高, 宽)`；None 时读 `TrainingConfig.IMG_SIZE`。

  Returns:
      `(1, 3, H, W)` float32 张量。
  """
  return load_tensor(path, img_size, mode="rgb", interp="linear")


def _strip_head_dropout(model: "torch.nn.Module") -> bool:
  """把三个 head 里的 Dropout2d 去掉（用于兼容「无 dropout 结构」的旧权重）

  Dropout 在 `eval()` 下是恒等映射，去掉它不改变推理结果，
  但能让 `nn.Sequential` 的层序号回到旧权重的编号。

  Returns:
      bool: 是否真的移除过任何层（False 表示模型本来就没有 Dropout）
  """
  changed = False
  for name in ("hm_head", "reg_head", "cls_head"):
    head = getattr(model, name, None)
    if head is None:
      continue
    layers = [m for m in head if not isinstance(m, torch.nn.Dropout2d)]
    if len(layers) != len(head):
      setattr(model, name, torch.nn.Sequential(*layers))
      changed = True
  return changed


class Inferer:
  """CenterNet 数字检测推理器。

  Args:
      model (CenterNetDigit): 待加载权重的模型实例（结构需与训练时一致）
      ckpt_path (str | Path): 权重文件；兼容 `state_dict`、
          `{"state_dict": ...}`、整个 `nn.Module` 三种存法
      device (str | torch.device): 推理设备，默认 CPU
      num_classes (int): 分类头类别数，必须与训练时一致
      img_size (tuple[int, int]): 网络输入尺寸，必须与训练时一致
  """

  def __init__(self, model: CenterNetDigit, ckpt_path: str | Path,
               device: str = "cpu", num_classes: int = 10,
               img_size: tuple[int, int] | None = None) -> None:
    self.device = torch.device(device)
    # 输入尺寸：None 时读配置（与训练 / 摄像头共用同一个值）
    self.img_size = tuple(TrainingConfig.IMG_SIZE if img_size is None
                          else img_size)
    self.model = self._load(model,Path(ckpt_path), num_classes)
    self.model.to(self.device).eval()
    self.tf = build_transform(self.img_size)

  @staticmethod
  def _load(model: CenterNetDigit, ckpt_path: Path,
            num_classes: int) -> CenterNetDigit:
    """把 checkpoint 还原成可用的模型

    兼容三种存法：
      1. `{"state_dict": ..., "num_classes": ..., "dropout": ...}`（新版 Trainer）
      2. 裸 `state_dict`（旧版权重，没有结构元信息）
      3. 整个 `nn.Module`

    元信息里的 dropout 会与传入的模型比对：不一致且权重不带 Dropout 时
    自动摘掉 head 里的 Dropout（eval 下等价），否则直接报错提示传对参数。
    """
    try:
      state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:        # 老式 checkpoint 里可能含非张量对象
      state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(state, torch.nn.Module):       # 直接存了整个模型
      return state

    meta = {}
    if isinstance(state, dict) and "state_dict" in state:
      meta = {k: v for k, v in state.items() if k != "state_dict"}
      state = state["state_dict"]

    # 用 checkpoint 自带的结构信息校验 / 对齐 dropout
    ckpt_dropout = meta.get("dropout")
    model_dropout = getattr(model, "dropout_p", None)
    ckpt_stride = meta.get("feat_stride")
    model_stride = getattr(model, "feat_stride", None)
    if (ckpt_stride is not None and model_stride is not None
            and int(ckpt_stride) != int(model_stride)):
      raise ValueError(
          f"权重是按 feat_stride={ckpt_stride} 训练的，但当前模型是 "
          f"feat_stride={model_stride}；请用 CenterNetDigit("
          f"feat_stride={ckpt_stride}) 构造，或重新训练")
    # 结构代差：带 neck（stride 4）的新模型无法加载 stride 8 的旧权重
    ckpt_keys = set(state.keys())
    model_keys = set(model.state_dict().keys())
    if (any(k.startswith("neck.") for k in model_keys)
            and not any(k.startswith("neck.") for k in ckpt_keys)):
      raise ValueError(
          "这份权重是 stride 8 的旧结构（没有 neck），而当前模型是 stride 4。\n"
          "  想继续用它：CenterNetDigit(..., feat_stride=8)\n"
          "  想用新结构：重新训练一次即可（旧权重不兼容）")
    if ckpt_dropout is not None and model_dropout is not None \
            and float(ckpt_dropout) != float(model_dropout):
      if float(ckpt_dropout) == 0.0 and _strip_head_dropout(model):
        _logger.info("权重是用 dropout=0 训练的，已移除 head 中的 Dropout 后加载")
      else:
        raise ValueError(
            f"权重是用 dropout={ckpt_dropout} 训练的，但传入的模型是 "
            f"dropout={model_dropout}；请用 CenterNetDigit(dropout={ckpt_dropout}) 构造")

    # strict=True：键名或形状不匹配就当场报错，避免带病推理
    try:
      model.load_state_dict(state)
    except RuntimeError:
      # 旧权重没有元信息：早期 head 里没有 Dropout2d，层号整体前移 1
      # （hm_head.2 是 conv 而不是 hm_head.3）。Dropout 在 eval 下是恒等映射，
      # 直接把它从 head 里摘掉再加载，推理结果完全一致。
      if not _strip_head_dropout(model):
        raise
      model.load_state_dict(state)
      _logger.info("该权重是在「不带 Dropout2d」的结构下训练的，"
                   "已自动移除 head 中的 Dropout 后加载")
    return model

  @staticmethod
  @torch.no_grad()
  def decode(hm_p: torch.Tensor, reg_p: torch.Tensor, cls_p: torch.Tensor,
             thr: float = 0.3, topk: int = 30
             ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """把三个 head 的输出解码成数字框。

    Args:
        hm_p (torch.Tensor): 中心热度 `(B, 1, h, w)`（已过 sigmoid）
        reg_p (torch.Tensor): 中心到 4 个角的偏移 `(B, 8, h, w)`，单位是 feature 格；
            与 `build_target` 的 `reg_t` 同一套定义
        cls_p (torch.Tensor): 分类 logits `(B, C, h, w)`
        thr (float): 中心分数阈值
        topk (int): 每张图最多保留多少个中心

    Returns:
        list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]: 每张图的
        `(pts_norm (K,4,2), digits (K,), scores (K,))`，按置信度降序，
        角点顺序为 [左上, 右上, 右下, 左下]、坐标归一化到 0~1；
        一个都没检出时返回 `(0,4,2)` / `(0,)` / `(0,)` 空张量
    """
    if hm_p.ndim != 4 or hm_p.shape[1] != 1:
      raise ValueError(f"hm_p 应为 (B,1,h,w)，收到 {tuple(hm_p.shape)}")

    hm = hm_p.squeeze(1)                                  # (B, h, w)
    B, h, w = hm.shape
    # 3x3 局部极大值才保留：CenterNet 自带的 nms，避免一个数字冒出多个峰
    pooled = F.max_pool2d(hm.unsqueeze(1), 3, 1, 1).squeeze(1)
    hm = hm * (pooled == hm).to(hm.dtype)

    k = max(1, min(int(topk), h * w))
    scale = torch.tensor([w, h], dtype=hm.dtype, device=hm.device)

    results = []
    for b in range(B):
      scores, idxs = hm[b].flatten().topk(k)
      sel = scores > thr
      scores, idxs = scores[sel], idxs[sel]

      iy = torch.div(idxs, w, rounding_mode="floor").to(scores.dtype)
      ix = (idxs % w).to(scores.dtype)

      # (8, h*w) -> (K, 8) -> (K, 4, 2)：中心到 4 个角的偏移
      off = reg_p[b].reshape(8, h * w)[:, idxs].t().reshape(-1, 4, 2)
      center = torch.stack([ix, iy], dim=1).unsqueeze(1)   # (K, 1, 2)
      pts = ((center + off) / scale).clamp(0.0, 1.0)       # 归一化到 0~1

      digits = cls_p[b].reshape(cls_p.shape[1], h * w)[:, idxs].argmax(0)
      results.append((pts.cpu(), digits.cpu(), scores.cpu()))

    return results

  def infer(self, imgs: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """只做前向，返回 `(hm_p, reg_p, cls_p)` 三张原始输出图"""
    self.model.eval()
    with torch.no_grad():
      return self.model(imgs.to(self.device))
    
  def predict_testing(self, dl: "torch.utils.data.DataLoader") -> list:
    """按 dataloader 批量推理，返回 `(pts_norm, digits, scores)` 三张列表"""
    self.model.eval()
    results = []
    with torch.no_grad():
      for batch in dl:
        imgs = batch[0]
        if not isinstance(imgs, torch.Tensor):     # collate_fn 现在返回 numpy
          imgs = torch.as_tensor(imgs, dtype=torch.float32)
        results.extend(self.predict(imgs))
    return results

  def predict(self, imgs: torch.Tensor, thr: float = 0.3, topk: int = 30
              ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """前向 + 解码，返回结构与 :meth:`decode` 相同

    Args:
        imgs (torch.Tensor): `(B, 3, H, W)`，需已按训练方式归一化
        thr (float): 中心分数阈值
        topk (int): 每张图最多保留多少个中心
    """
    hm_p, reg_p, cls_p = self.infer(imgs)
    return self.decode(hm_p, reg_p, cls_p, thr=thr, topk=topk)

  def predict_images(self, paths: list[Path | str], thr: float = 0.3,
                     topk: int = 30) -> list:
    """按训练同款预处理批量读图再推理"""
    imgs = torch.cat([self.tf(Image.open(p).convert("RGB")).unsqueeze(0)
                      for p in paths], dim=0)
    return self.predict(imgs, thr=thr, topk=topk)


def main(argv: list[str] | None = None) -> int:
  """命令行冒烟入口：给若干张图，打印检出的数字与角点"""
  parser = argparse.ArgumentParser(description="CenterNet 数字检测推理")
  parser.add_argument("-c", "--ckpt", required=True, help="权重文件路径")
  parser.add_argument("-i", "--image", nargs="+", required=True,
                      help="待推理图片（可给多张）")
  parser.add_argument("--thr", type=float, default=0.3, help="中心分数阈值")
  parser.add_argument("--topk", type=int, default=30, help="每张图最大检出数")
  parser.add_argument("--device",
                      default="cuda" if torch.cuda.is_available() else "cpu")
  args = parser.parse_args(argv)

  inf = Inferer(CenterNetDigit(num_classes=10), args.ckpt, device=args.device)
  for path, (pts, digits, scores) in zip(
      args.image, inf.predict_images(args.image, args.thr, args.topk)):
    _logger.info("%s: 检出 %d 个数字", Path(path).name, len(digits))
    for p, d, s in zip(pts.tolist(), digits.tolist(), scores.tolist()):
      corners = " ".join(f"({x:.3f},{y:.3f})" for x, y in p)
      print(f"  digit={d} score={s:.3f} {corners}")

  return 0


if __name__ == "__main__":
  sys.exit(main())
