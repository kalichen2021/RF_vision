"""训练日志的读取、摘要与曲线绘制（数据来自 `dataset/log/train/*.json`）。

日志由 `src/training/train.py::Trainer` 逐轮写盘，结构为::

    {
      "meta":   {"ckpt": ..., "epochs": 15, "lr_init": 1e-3,
                 "optimizer": "Adam", "scheduler": "ReduceLROnPlateau",
                 "amp": true, "trainable_params": ..., "total_params": ...},
      "epochs": [{"epoch": 1, "train": {"loss", "heatmap", "offset", "cls"},
                  "val": {...} | None, "lr": 1e-3,
                  "epoch_time": 126.1, "data_wait": 31.6, "gpu_ratio": 0.749,
                  "is_best": true, "bad_epochs": 0}, ...]
    }

一次训练分两个阶段（`head` → `finetune`），所以绘图接口一律接收**列表**，
按阶段顺序拼成连续的 epoch 轴，并用竖线标出阶段分界。

本模块不打印任何东西：摘要以 dict 返回，控制台输出在 `src/cli/train_plot.py`。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from src.utils.plotting import setup_cjk_font

# 每个阶段的配色：训练实线 + 验证虚线都用这一套颜色
_STAGE_COLORS: tuple[str, ...] = ("#4C78A8", "#F58518", "#54A24B", "#E45756")
# 曲线里的四个 loss 分量：(字段名, 中文标题)
_PARTS: tuple[tuple[str, str], ...] = (
    ("loss", "总 Loss"),
    ("heatmap", "Heatmap"),
    ("offset", "Offset"),
    ("cls", "Classification"),
)


# ---------------------------------------------------------------- 读取

def load_history(path: str | Path) -> dict:
  """读取一份训练记录 JSON。

  Args:
      path: `dataset/log/train/train_<uuid>_<stage>.json`。

  Returns:
      解析后的字典（含 `meta` 与 `epochs` 两个键）。

  Raises:
      FileNotFoundError: 文件不存在。
      ValueError: 结构不对（缺少 `epochs` 列表）。
  """
  p = Path(path)
  if not p.exists():
    raise FileNotFoundError(f"训练记录不存在: {p}")
  with p.open("r", encoding="utf-8") as fp:
    hist = json.load(fp)
  if not isinstance(hist.get("epochs"), list):
    raise ValueError(f"{p} 缺少 epochs 列表，不是训练记录文件")
  return hist


def _num(rec: dict | None, *keys: str) -> float:
  """从嵌套字典里安全取一个 float（缺失 / None 记为 nan）。"""
  cur: Any = rec
  for key in keys:
    if not isinstance(cur, dict):
      return float("nan")
    cur = cur.get(key)
  return float("nan") if cur is None else float(cur)


def stage_series(hist: dict) -> dict[str, np.ndarray]:
  """把一份记录摊平成绘图用的等长数组（缺失值填 nan）。

  Args:
      hist: `load_history` 的返回值。

  Returns:
      键为 `epoch / train_* / val_* / lr / epoch_time / data_wait / gpu_ratio`，
      值均为 `np.ndarray`。
  """
  recs: list[dict] = list(hist.get("epochs") or [])
  names = ("loss", "heatmap", "offset", "cls")
  out: dict[str, np.ndarray] = {
      "epoch": np.array([_num(r, "epoch") for r in recs], dtype=float),
      "lr": np.array([_num(r, "lr") for r in recs], dtype=float),
      "epoch_time": np.array([_num(r, "epoch_time") for r in recs], dtype=float),
      "data_wait": np.array([_num(r, "data_wait") for r in recs], dtype=float),
      "gpu_ratio": np.array([_num(r, "gpu_ratio") for r in recs], dtype=float),
  }
  for n in names:
    out[f"train_{n}"] = np.array([_num(r, "train", n) for r in recs], dtype=float)
    out[f"val_{n}"] = np.array([_num(r, "val", n) for r in recs], dtype=float)
  return out


def history_summary(histories: Sequence[dict],
                    labels: Sequence[str] | None = None) -> list[dict]:
  """汇总每个阶段的训练情况（供 CLI 打印）。

  Args:
      histories: 按训练顺序排列的记录列表。
      labels: 阶段名；None 时用 `阶段1 / 阶段2 …`。

  Returns:
      每个阶段一个 dict，字段见实现（`best_epoch` 为 nan 表示没跑过验证）。
  """
  names = list(labels or [f"阶段{i + 1}" for i in range(len(histories))])
  rows: list[dict] = []
  for name, hist in zip(names, histories):
    s = stage_series(hist)
    meta = hist.get("meta") or {}
    val = s["val_loss"]
    best_idx = int(np.nanargmin(val)) if np.isfinite(val).any() else -1
    total_time = float(np.nansum(s["epoch_time"]))
    data_wait = float(np.nansum(s["data_wait"]))
    patience = int(meta.get("early_stopping_patience") or 0)
    last_bad = int((hist["epochs"][-1].get("bad_epochs") or 0)) if hist["epochs"] else 0
    rows.append({
        "stage": name,
        "ckpt": Path(str(meta.get("ckpt", ""))).name,
        "epochs_planned": int(meta.get("epochs") or 0),
        "epochs_run": len(hist["epochs"]),
        "params": int(meta.get("total_params") or 0),
        "best_epoch": int(s["epoch"][best_idx]) if best_idx >= 0 else float("nan"),
        "best_val": float(val[best_idx]) if best_idx >= 0 else float("nan"),
        "final_train": float(s["train_loss"][-1]) if len(s["train_loss"]) else float("nan"),
        "final_val": float(val[-1]) if len(val) else float("nan"),
        "final_lr": float(s["lr"][-1]) if len(s["lr"]) else float("nan"),
        "total_seconds": total_time,
        "data_wait_ratio": data_wait / total_time if total_time else 0.0,
        "early_stopped": bool(patience) and last_bad >= patience,
    })
  return rows


# ---------------------------------------------------------------- 绘图

def _maybe_log(ax: plt.Axes, values: Sequence[np.ndarray]) -> None:
  """跨度超过两个数量级时用对数纵轴（focal loss 首轮会高出几十倍）。"""
  finite = np.concatenate([v[np.isfinite(v)] for v in values]) if values else None
  if finite is None or finite.size == 0:
    return
  lo, hi = float(np.min(finite)), float(np.max(finite))
  if lo > 0 and hi / lo > 20:
    ax.set_yscale("log")


def plot_history(histories: Sequence[dict],
                 labels: Sequence[str] | None = None,
                 figsize: tuple[float, float] = (18.0, 9.0)) -> Figure:
  """画训练曲线总览（2×4 面板），返回 Figure（不保存、不显示）。

  面板：总 Loss / Heatmap / Offset / Classification / 学习率 /
  单轮耗时与等数据 / GPU 利用率 / 文字摘要。每个阶段用不同颜色，
  训练为实线、验证为虚线，最优验证轮用星标标出。

  Args:
      histories: 按训练顺序排列的记录列表（`load_history` 的返回值）。
      labels: 阶段名（图例用）；None 时用 `阶段1 / 阶段2 …`。
      figsize: 画布尺寸（英寸）。

  Returns:
      matplotlib Figure。
  """
  if not histories:
    raise ValueError("plot_history 至少需要一份训练记录")
  setup_cjk_font()
  names = list(labels or [f"阶段{i + 1}" for i in range(len(histories))])

  fig, axes = plt.subplots(2, 4, figsize=figsize)
  ax_loss, ax_hm, ax_reg, ax_cls = axes[0]
  ax_lr, ax_time, ax_gpu, ax_text = axes[1]

  offset = 0.0
  bounds: list[float] = []
  series: list[dict[str, np.ndarray]] = []
  xs: list[np.ndarray] = []
  for hist in histories:
    s = stage_series(hist)
    x = s["epoch"] + offset
    series.append(s)
    xs.append(x)
    offset = float(x[-1]) if x.size else offset

  for idx, (name, s, x) in enumerate(zip(names, series, xs)):
    color = _STAGE_COLORS[idx % len(_STAGE_COLORS)]
    for ax, key in ((ax_loss, "loss"), (ax_hm, "heatmap"),
                    (ax_reg, "offset"), (ax_cls, "cls")):
      ax.plot(x, s[f"train_{key}"], color=color, lw=1.6,
              label=f"{name} 训练")
      if np.isfinite(s[f"val_{key}"]).any():
        ax.plot(x, s[f"val_{key}"], color=color, lw=1.6, ls="--",
                label=f"{name} 验证")
    # 最优轮星标（只看验证总 loss）
    val = s["val_loss"]
    if np.isfinite(val).any():
      j = int(np.nanargmin(val))
      ax_loss.scatter([x[j]], [val[j]], marker="*", s=140, color=color,
                      edgecolor="black", zorder=5)
    ax_lr.plot(x, s["lr"], color=color, lw=1.6, marker="o", ms=3, label=name)

    # 单轮耗时：计算时间 + 等数据时间堆叠（等数据 = DataLoader 瓶颈）
    wait = np.nan_to_num(s["data_wait"])
    busy = np.maximum(np.nan_to_num(s["epoch_time"]) - wait, 0.0)
    ax_time.bar(x, busy, color=color, alpha=0.55, label=f"{name} 计算")
    ax_time.bar(x, wait, bottom=busy, color="#C44E52", alpha=0.75,
                label=f"{name} 等数据")
    ax_gpu.plot(x, s["gpu_ratio"] * 100.0, color=color, lw=1.6,
                marker="o", ms=3, label=name)

    if x.size:
      bounds.append(float(x[-1]))

  # 阶段分界竖线（画在「两个阶段之间」）
  for b in bounds[:-1]:
    for ax in (ax_loss, ax_hm, ax_reg, ax_cls, ax_lr, ax_time, ax_gpu):
      ax.axvline(b + 0.5, color="#888888", ls=":", lw=1.2)

  for ax, key in ((ax_loss, "loss"), (ax_hm, "heatmap"),
                  (ax_reg, "offset"), (ax_cls, "cls")):
    ax.set_title(dict(_PARTS)[key])
    ax.set_xlabel("epoch（两阶段连续编号）")
    ax.grid(alpha=0.3)
  _maybe_log(ax_loss, [v for s in series for v in (s["train_loss"], s["val_loss"])])
  ax_loss.legend(fontsize=8, frameon=False)

  ax_lr.set_title("学习率（ReduceLROnPlateau）")
  ax_lr.set_xlabel("epoch")
  ax_lr.grid(alpha=0.3)
  lr_all = np.concatenate([s["lr"] for s in series]) if series else np.array([])
  lr_all = lr_all[np.isfinite(lr_all)]
  if lr_all.size and lr_all.max() > 0:
    ax_lr.set_yscale("log")

  ax_time.set_title("单轮耗时（计算 / 等数据）")
  ax_time.set_xlabel("epoch")
  ax_time.set_ylabel("秒")
  ax_time.grid(alpha=0.3, axis="y")

  ax_gpu.set_title("GPU 忙的比例（≈ 1 - 等数据占比）")
  ax_gpu.set_xlabel("epoch")
  ax_gpu.set_ylim(0, 105)
  ax_gpu.axhline(90, color="#C44E52", ls="--", lw=1)
  ax_gpu.grid(alpha=0.3)

  # 文字摘要面板
  rows = history_summary(histories, names)
  lines: list[str] = []
  for r in rows:
    lines.append(f"[{r['stage']}] {r['ckpt'] or '—'}")
    lines.append(f"  轮数        : {r['epochs_run']}/{r['epochs_planned']}"
                 + ("  (早停)" if r["early_stopped"] else ""))
    lines.append(f"  参数量      : {r['params']:,}")
    if np.isfinite(r["best_val"]):
      lines.append(f"  最优轮      : {r['best_epoch']:.0f} "
                   f"(val loss {r['best_val']:.4f})")
    lines.append(f"  末轮        : train {r['final_train']:.4f} / "
                 f"val {r['final_val']:.4f}")
    lines.append(f"  末轮 lr     : {r['final_lr']:.2e}")
    lines.append(f"  总耗时      : {r['total_seconds'] / 60:.1f} min"
                 f"（等数据占 {r['data_wait_ratio'] * 100:.1f}%）")
    lines.append("")
  ax_text.axis("off")
  ax_text.set_title("摘要")
  # 不用 family="monospace"：DejaVu Sans Mono 没有汉字，会出现方块
  ax_text.text(0.0, 1.0, "\n".join(lines), va="top", ha="left",
               fontsize=9, transform=ax_text.transAxes)

  ckpts = {Path(str((h.get("meta") or {}).get("ckpt", ""))).name
           for h in histories}
  ckpts.discard("")
  fig.suptitle("训练过程： " + (" → ".join(names))
               + (f"    ckpt: {', '.join(sorted(ckpts))}" if ckpts else ""),
               fontsize=13)
  fig.tight_layout(rect=(0, 0, 1, 0.96))
  return fig
