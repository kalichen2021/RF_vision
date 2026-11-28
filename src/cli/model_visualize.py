"""模型可视化 CLI：卷积核 / 特征图 / 三头输出 / 逐层统计（含 ReLU 死区比例）。

用法::

    conda activate NNVN
    python src/cli/model_visualize.py --list                  # 列出所有可看的层
    python src/cli/model_visualize.py --filters               # 第一层 7x7x3 卷积核
    python src/cli/model_visualize.py --filters --layer backbone.5.0.conv1 --max 32
    python src/cli/model_visualize.py --feature --layer backbone.5.0.conv1 --channels 16
    python src/cli/model_visualize.py --heads                 # 三个 head 的输出
    python src/cli/model_visualize.py --stats                 # 逐层权重 + 死区统计
    python src/cli/model_visualize.py --filters --feature --heads --stats   # 全来一遍

产物默认存 `dataset/log/visualize/<uuid>/`；`--no-show` 只存不弹窗；
`-m` 指定权重，`-p` 指定输入图（默认 `models/` 最新权重 + 测试集第一张）。
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from configs.log import LogConfig
from configs.path import PathConfig
from configs.training_cfg import TrainingConfig
from src.inference.infer import Inferer
from src.training.dataset import DigitDataset, batch_to_torch, collate_fn
from src.training.model import CenterNetDigit
from src.utils.debug_tools import tensor2img
from src.utils.image import load_tensor
from src.utils.logging_tools import setup_logging
from src.utils.plotting import save_figure, setup_cjk_font

_CONV = torch.nn.Conv2d
_ACT = (torch.nn.ReLU, torch.nn.LeakyReLU, torch.nn.SiLU)
_NORM = (torch.nn.BatchNorm2d,)


# ---------------------------------------------------------------- 基础查询

def list_layers(model: torch.nn.Module,
                kinds: tuple[type, ...] = (_CONV, _NORM, _ACT)) -> list[str]:
  """列出可以可视化的层名。

  Args:
      model: 模型。
      kinds: 关心的模块类型（默认卷积 / BN / 激活）。

  Returns:
      层名列表，如 `backbone.0`、`backbone.5.0.conv1`、`hm_head.0`。
  """
  return [n for n, m in model.named_modules() if n and isinstance(m, kinds)]


def get_module(model: torch.nn.Module, name: str) -> torch.nn.Module:
  """按名字取子模块。

  Args:
      model: 模型。
      name: `named_modules()` 里的名字。

  Returns:
      子模块。

  Raises:
      KeyError: 名字不存在。
  """
  mods = dict(model.named_modules())
  if name not in mods:
    raise KeyError(f"没有名为 {name!r} 的层，用 --list 查看可用层")
  return mods[name]


def feed_model(model: torch.nn.Module, tensor: torch.Tensor,
               names: Sequence[str]) -> dict[str, torch.Tensor]:
  """跑一次前向，用 hook 抓取指定层的输出。

  Args:
      model: 模型（应为 eval 模式）。
      tensor: `(B, 3, H, W)` 输入。
      names: 要抓的层名。

  Returns:
      `{层名: 该层输出}`（detach + float + cpu）。
  """
  acts: dict[str, torch.Tensor] = {}
  handles: list[Any] = []

  def _hook(layer: str):
    def _inner(_m: torch.nn.Module, _i: Any, out: Any) -> None:
      if isinstance(out, torch.Tensor):
        acts[layer] = out.detach().float().cpu()
    return _inner

  for n in names:
    handles.append(get_module(model, n).register_forward_hook(_hook(n)))
  try:
    with torch.no_grad():
      model(tensor)
  finally:
    for h in handles:
      h.remove()
  return acts


# ---------------------------------------------------------------- 卷积核

def _kernel_image(w: np.ndarray) -> np.ndarray:
  """单个卷积核 `(in_ch, kh, kw)` → 可 imshow 的图（3 通道按 RGB，否则取均值）。"""
  w = np.asarray(w, np.float32)
  img = np.transpose(w, (1, 2, 0)) if w.shape[0] == 3 else w.mean(axis=0)
  img = img - img.min()
  peak = float(img.max())
  return img / peak if peak > 0 else img


def plot_kernels(model: torch.nn.Module, name: str,
                 max_filters: int = 64, cols: int = 8) -> Figure:
  """画一个卷积层的所有卷积核（每个输出通道一张小图）。

  Args:
      model: 模型。
      name: 卷积层名（`backbone.0` 是第一层 7x7x3）。
      max_filters: 最多显示多少个输出通道。
      cols: 每行几张。

  Returns:
      matplotlib Figure。
  """
  setup_cjk_font()
  mod = get_module(model, name)
  if not isinstance(mod, _CONV):
    raise TypeError(f"{name!r} 不是卷积层（是 {type(mod).__name__}）")
  w = mod.weight.detach().float().cpu().numpy()      # (out, in, kh, kw)
  n = min(int(max_filters), w.shape[0])
  cols = max(1, min(int(cols), n))
  rows = int(np.ceil(n / cols))
  fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
  axes = np.atleast_2d(axes)
  for i in range(rows * cols):
    ax = axes[i // cols, i % cols]
    ax.axis("off")
    if i < n:
      ax.imshow(_kernel_image(w[i]), interpolation="nearest")
  fig.suptitle(f"{name} 卷积核：{w.shape[0]} 个 "
               f"({w.shape[1]}x{w.shape[2]}x{w.shape[3]})，显示前 {n} 个",
               fontsize=12)
  fig.tight_layout(rect=(0, 0, 1, 0.95))
  return fig


# ---------------------------------------------------------------- 特征图

def plot_feature_maps(acts: dict[str, torch.Tensor], name: str,
                      max_channels: int = 16, cols: int = 8) -> Figure:
  """画某层前 N 个通道的激活（各自按自身最大值归一化）。

  Args:
      acts: `feed_model` 的返回值。
      name: 层名。
      max_channels: 最多画多少个通道。
      cols: 每行几个。

  Returns:
      matplotlib Figure。

  Raises:
      KeyError: 该层没有抓到激活。
  """
  if name not in acts:
    raise KeyError(f"没有抓到 {name!r} 的激活")
  setup_cjk_font()
  a = acts[name]
  a = a[0] if a.ndim == 4 else a.unsqueeze(0)       # (C, H, W)
  n = min(int(max_channels), int(a.shape[0]))
  cols = max(1, min(int(cols), n))
  rows = int(np.ceil(n / cols))
  fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.6))
  axes = np.atleast_2d(axes)
  for i in range(rows * cols):
    ax = axes[i // cols, i % cols]
    ax.axis("off")
    if i >= n:
      continue
    m = a[i].numpy()
    peak = float(m.max()) if m.size else 0.0
    ax.imshow(m / peak if peak > 0 else m, interpolation="nearest",
              cmap="viridis")
    ax.set_title(f"ch{i} 零值{float((m == 0).mean()) * 100:.0f}%", fontsize=7)
  fig.suptitle(f"{name} 激活 shape={tuple(a.shape)}，显示前 {n} 个通道",
               fontsize=12)
  fig.tight_layout(rect=(0, 0, 1, 0.94))
  return fig


# ---------------------------------------------------------------- 三头输出

def plot_heads(hm: np.ndarray, reg: np.ndarray, cls: np.ndarray,
               image_bgr: np.ndarray | None = None) -> Figure:
  """把三个 head 的输出并排画出来（输入图 / 中心热度 / 偏移强度 / 分类 argmax）。

  Args:
      hm: 中心热度 `(h, w)`，0~1。
      reg: 偏移 `(8, h, w)`。
      cls: 分类 logits `(C, h, w)`。
      image_bgr: 可选的输入图，用于对照。

  Returns:
      matplotlib Figure。
  """
  setup_cjk_font()
  reg = np.asarray(reg, np.float32)
  cls = np.asarray(cls, np.float32)
  reg_mag = np.sqrt((reg ** 2).sum(axis=0))
  cls_idx = cls.argmax(axis=0)

  n = 4 if image_bgr is not None else 3
  fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.4))
  axes = np.atleast_1d(axes)
  k = 0
  if image_bgr is not None:
    axes[k].imshow(image_bgr[:, :, ::-1])
    axes[k].set_title("模型输入")
    k += 1
  axes[k].imshow(hm, vmin=0.0, vmax=1.0, cmap="jet", interpolation="nearest")
  axes[k].set_title(f"中心热度 max={float(hm.max()):.3f}")
  axes[k + 1].imshow(reg_mag, cmap="magma", interpolation="nearest")
  axes[k + 1].set_title(f"偏移强度 max={float(reg_mag.max()):.2f} 格")
  axes[k + 2].imshow(cls_idx, cmap="tab10", vmin=-0.5, vmax=9.5,
                     interpolation="nearest")
  axes[k + 2].set_title("分类 argmax")
  for ax in axes:
    ax.axis("off")
  fig.tight_layout()
  return fig


# ---------------------------------------------------------------- 统计

def plot_weight_hist(model: torch.nn.Module, max_layers: int = 8,
                     cols: int = 4) -> Figure:
  """画前若干个卷积层的权重直方图。

  Args:
      model: 模型。
      max_layers: 最多几层。
      cols: 每行几个。

  Returns:
      matplotlib Figure。
  """
  setup_cjk_font()
  convs = [(n, m) for n, m in model.named_modules() if isinstance(m, _CONV)]
  convs = convs[:int(max_layers)]
  cols = max(1, min(int(cols), len(convs)))
  rows = int(np.ceil(len(convs) / cols))
  fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.6, rows * 2.6))
  axes = np.atleast_2d(axes)
  for i in range(rows * cols):
    ax = axes[i // cols, i % cols]
    ax.axis("off")
    if i >= len(convs):
      continue
    name, mod = convs[i]
    w = mod.weight.detach().float().cpu().numpy().ravel()
    ax.hist(w, bins=60, color="#4C78A8")
    ax.set_title(f"{name}\nmean={w.mean():.4f} std={w.std():.4f}", fontsize=8)
    ax.tick_params(labelsize=6)
    ax.axis("on")
  fig.suptitle("卷积层权重分布", fontsize=12)
  fig.tight_layout(rect=(0, 0, 1, 0.94))
  return fig


# ---------------------------------------------------------------- CLI

def print_layers(model: torch.nn.Module, console: Console) -> None:
  """打印所有层的名字 / 类型 / 形状（好奇心专用）。

  Args:
      model: 模型。
      console: rich 控制台。
  """
  table = Table(title="可查看的层", header_style="bold")
  for col in ("层名", "类型", "形状 / 通道", "参数量"):
    table.add_column(col)
  for name in list_layers(model):
    mod = get_module(model, name)
    if isinstance(mod, _CONV):
      shape = (f"{mod.out_channels}x{mod.in_channels}x"
               f"{mod.kernel_size[0]}x{mod.kernel_size[1]}")
      n_par = mod.weight.numel() + (mod.bias.numel() if mod.bias is not None else 0)
    elif isinstance(mod, _NORM):
      shape = f"{mod.num_features}"
      n_par = mod.weight.numel()
    else:
      shape = "-"
      n_par = 0
    table.add_row(name, type(mod).__name__, shape, f"{n_par:,}")
  console.print(table)


def print_stats(model: torch.nn.Module, acts: dict[str, torch.Tensor],
                console: Console) -> None:
  """打印卷积层权重统计 + 激活层零值比例（死区）。

  Args:
      model: 模型。
      acts: `feed_model` 抓到的激活。
      console: rich 控制台。
  """
  t1 = Table(title="卷积层权重", header_style="bold")
  for col in ("层名", "形状 (out x in x kh x kw)", "参数量", "|w|max", "|w|mean"):
    t1.add_column(col, justify="right" if col != "层名" else "left")
  for name, mod in model.named_modules():
    if not name or not isinstance(mod, _CONV):
      continue
    w = mod.weight.detach().float()
    t1.add_row(name,
               f"{mod.out_channels}x{mod.in_channels}x"
               f"{mod.kernel_size[0]}x{mod.kernel_size[1]}",
               f"{w.numel():,}", f"{float(w.abs().max()):.4f}",
               f"{float(w.abs().mean()):.4f}")
  console.print(t1)

  t2 = Table(title="激活层（零值占比 = 死区比例）", header_style="bold")
  for col in ("层名", "类型", "零值占比", "均值", "最大值"):
    t2.add_column(col, justify="right" if col != "层名" else "left")
  for name in list_layers(model, _ACT):
    a = acts.get(name)
    if a is None:
      continue
    t2.add_row(name, type(get_module(model, name)).__name__,
               f"{float((a == 0).float().mean()) * 100:.1f}%",
               f"{float(a.mean()):.3f}", f"{float(a.max()):.3f}")
  console.print(t2)


def main(argv: list[str] | None = None) -> int:
  """命令行入口，返回进程退出码。"""
  parser = argparse.ArgumentParser(description="模型可视化：卷积核 / 特征图 / 三头 / 统计")
  parser.add_argument("-m", "--model", default=None,
                      help="权重路径（默认 models/ 下最新的 ckpt_*.pt）")
  parser.add_argument("-p", "--image", default=None,
                      help="输入图片（默认取测试集第一张）")
  parser.add_argument("--list", action="store_true", help="只列出层，不画图")
  parser.add_argument("--filters", action="store_true", help="画卷积核")
  parser.add_argument("--feature", action="store_true", help="画特征图（激活）")
  parser.add_argument("--heads", action="store_true", help="画三个 head 的输出")
  parser.add_argument("--stats", action="store_true", help="逐层权重 / 死区统计")
  parser.add_argument("--layer", action="append", default=None,
                      help="层名，可重复给多次；filters 默认 backbone.0，"
                           "feature 默认 backbone.5.0.conv1")
  parser.add_argument("--max", type=int, default=64, help="卷积核最多画多少个")
  parser.add_argument("--channels", type=int, default=16, help="特征图最多画多少个通道")
  parser.add_argument("--cols", type=int, default=8, help="每行几个")
  parser.add_argument("--no-show", action="store_true", help="只保存不弹窗")
  args = parser.parse_args(argv)

  setup_logging()
  console = Console()
  if args.no_show:
    matplotlib.use("Agg")

  ckpt = Path(args.model) if args.model else PathConfig.SELECT_MODEL
  device = "cuda" if torch.cuda.is_available() else "cpu"
  inferrer = Inferer(CenterNetDigit(num_classes=10, dropout=TrainingConfig.DROPOUT),
                     ckpt, device=device)
  model = inferrer.model
  console.print(f"权重: [bold]{ckpt}[/]  |  device={device}")

  if args.list:
    print_layers(model, console)
    return 0

  if not (args.filters or args.feature or args.heads or args.stats):
    parser.print_help()
    return 0

  # 输入张量：指定图片 > 测试集第一张
  if args.image:
    tensor = load_tensor(args.image, (384, 384), mode="gray", interp="linear",
                         crop="center")
    img_bgr = tensor2img(tensor[0])
  else:
    ds = DigitDataset(PathConfig.DATASET_LABEL_JSON, PathConfig.SELECT_IMGSET,
                      TrainingConfig.TRAININGSET_RATIO, mode="testing")
    batch = collate_fn([ds[0]])
    imgs, _, _ = batch_to_torch(batch)
    tensor = imgs[:1]
    img_bgr = tensor2img(tensor[0])

  # 权重的 device 由 Inferer 决定（可能是 cuda），输入张量必须跟它同一设备
  tensor = tensor.to(device)

  out_dir = PathConfig.LOG_DIR / "visualize" / LogConfig.UUID
  out_dir.mkdir(parents=True, exist_ok=True)
  show = not args.no_show

  if args.stats:
    # _CONV 是单个类型，_ACT / _NORM 是元组，拼起来要先把 _CONV 包成元组
    names = list_layers(model, (_CONV,) + _ACT)
    feats = feed_model(model, tensor, names)
    print_stats(model, feats, console)
    p = save_figure(plot_weight_hist(model), out_dir / "weight_hist.png", show=show)
    console.print(f"权重直方图: {p}")

  if args.filters:
    layers = args.layer or ["backbone.0"]
    for name in layers:
      fig = plot_kernels(model, name, args.max, args.cols)
      p = save_figure(fig, out_dir / f"kernels_{name.replace('.', '_')}.png",
                      show=show)
      console.print(f"卷积核 [{name}]: {p}")

  if args.feature:
    layers = args.layer or ["backbone.5.0.conv1"]
    feats = feed_model(model, tensor, layers)
    for name in layers:
      fig = plot_feature_maps(feats, name, args.channels, args.cols)
      p = save_figure(fig, out_dir / f"feature_{name.replace('.', '_')}.png",
                      show=show)
      console.print(f"特征图 [{name}]: {p}")

  if args.heads:
    hm_p, reg_p, cls_p = model(tensor)
    fig = plot_heads(hm_p[0, 0].detach().cpu().numpy(),
                     reg_p[0].detach().cpu().numpy(),
                     cls_p[0].detach().cpu().numpy(),
                     image_bgr=img_bgr)
    p = save_figure(fig, out_dir / "heads.png", show=show)
    console.print(f"三头输出: {p}")

  console.print(f"产物目录: [bold]{out_dir}[/]")
  return 0


if __name__ == "__main__":
  sys.exit(main())