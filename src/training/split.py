"""按「数字四边形面积」分层划分训练集 / 验证集。

直接按前 80% / 后 20% 切分会出问题：`labels.json` 的样本是按生成顺序排列的
（环境图轮转 + 随机切分），面积大小并不均匀，前后两段很容易一个偏大一个偏小。
这里先按每个样本的数字四边形平均面积做**分位数分层**，再在每层内部随机分配，
保证训练 / 验证两边的面积分布（p10 / p50 / p90、最大最小值）基本一致。

用法::

    conda activate NNVN
    python -m src.training.split                       # 默认 SELECT_IMGSET 的 labels.json
    python -m src.training.split path/to/labels.json --ratio 0.8 --bins 10 --seed 42
    python -m src.training.split --no-save             # 只打印分布，不写文件

产物（与 labels.json 同目录）::

    train.json   {"count": n, "samples": [...]}
    val.json     {"count": m, "samples": [...]}

`DigitDataset` 可以直接把 `json_path` 指到这两个文件上。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


# ---------------------------------------------------------------- 基础统计

def _quad_area(quad: Sequence[Sequence[float]]) -> float:
  """四边形面积（鞋带公式，取绝对值）"""
  p = np.asarray(quad, dtype=np.float64).reshape(-1, 2)
  x, y = p[:, 0], p[:, 1]
  return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def sample_area(sample: dict) -> float:
  """样本的数字四边形平均面积（没有数字时返回 0）

  用「平均」而不是「总和」：样本内数字个数差别很大，总和会被个数带偏，
  分层时会把「数字多的小样本」和「数字少的大样本」混在一起。
  """
  objects = sample.get("objects") or []
  areas = [_quad_area(o["quad"]) for o in objects if o.get("quad")]
  return float(np.mean(areas)) if areas else 0.0


def area_percentiles(samples: Iterable[dict],
                     qs=(0, 10, 50, 90, 100)) -> dict[str, float]:
  """一批样本的面积分位数（只统计有数字的样本）"""
  areas = np.asarray([sample_area(s) for s in samples if s.get("objects")],
                     dtype=np.float64)
  if areas.size == 0:
    return {}
  return {f"p{q}": float(np.percentile(areas, q)) for q in qs}


# ---------------------------------------------------------------- 分层划分

def stratified_split(samples: Sequence[dict], ratio: float = 0.8,
                     bins: int = 8, seed: int | None = None
                     ) -> tuple[list[int], list[int]]:
  """按数字四边形面积分层划分，返回 (训练集下标, 验证集下标)

  实现：把面积取 `log1p` 后按分位数切成 `bins` 层（面积跨几个数量级，
  直接等宽分箱会让绝大多数样本挤在一层）；每层内部洗牌后按 `ratio` 切分。

  Args:
      samples: 样本记录列表（`labels.json` 的 `samples`）
      ratio: 训练集比例，0 < ratio < 1
      bins: 分位层层数（样本很少时自动减少）
      seed: 随机种子；相同种子结果可复现

  Returns:
      tuple[list[int], list[int]]: (train 下标, val 下标)，两边的面积分布接近
  """
  n = len(samples)
  ratio = float(ratio)
  if not 0.0 < ratio < 1.0:
    raise ValueError(f"ratio 应在 (0, 1) 内，收到 {ratio}")
  if n < 2:
    raise ValueError(f"样本太少（{n} 条），无法划分训练 / 验证集")

  rng = random.Random(seed)
  areas = np.asarray([sample_area(s) for s in samples], dtype=np.float64)

  # 分层依据：log1p 面积（0 面积样本落进第一层）
  values = np.log1p(areas)
  n_bins = max(1, min(int(bins), n))
  edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1)))
  if edges.size <= 1:                       # 所有面积相同：退化成一次随机切分
    strata = [list(range(n))]
  else:
    labels = np.clip(np.digitize(values, edges[1:-1]), 0, edges.size - 2)
    groups: dict[int, list[int]] = {}
    for idx, lab in enumerate(labels.tolist()):
      groups.setdefault(int(lab), []).append(idx)
    strata = [groups[k] for k in sorted(groups)]

  train_idx: list[int] = []
  val_idx: list[int] = []
  for group in strata:
    rng.shuffle(group)
    n_train = int(round(len(group) * ratio))
    n_train = min(max(n_train, 0), len(group))
    train_idx.extend(group[:n_train])
    val_idx.extend(group[n_train:])

  # 某层只有 1 条时可能全进训练集：从最大的层里挪一条给验证集
  if not val_idx:
    moved = train_idx.pop(rng.randrange(len(train_idx))
                          if len(train_idx) > 1 else 0)
    val_idx.append(moved)
  if not train_idx:
    moved = val_idx.pop(0)
    train_idx.append(moved)

  rng.shuffle(train_idx)
  rng.shuffle(val_idx)
  return train_idx, val_idx


# ---------------------------------------------------------------- 落盘

def split_dataset(labels: str | Path | dict, ratio: float = 0.8, bins: int = 8,
                  seed: int | None = None) -> dict[str, list[dict]]:
  """读标签并返回 `{"train": [...], "val": [...]}`（样本字典列表）"""
  if isinstance(labels, dict):
    data = labels
  else:
    data = json.loads(Path(labels).read_text(encoding="utf-8"))
  samples = data.get("samples", []) if isinstance(data, dict) else list(data)

  train_idx, val_idx = stratified_split(samples, ratio=ratio, bins=bins,
                                        seed=seed)
  return {"train": [samples[i] for i in train_idx],
          "val": [samples[i] for i in val_idx]}


def save_splits(splits: dict[str, list[dict]], out_dir: str | Path,
                names: tuple[str, str] = ("train", "val")) -> dict[str, Path]:
  """把划分结果写成 `train.json` / `val.json`，返回路径"""
  out_dir = Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  paths: dict[str, Path] = {}
  for key, name in zip(("train", "val"), names):
    path = out_dir / f"{name}.json"
    path.write_text(
        json.dumps({"count": len(splits[key]), "samples": splits[key]},
                   ensure_ascii=False, indent=2),
        encoding="utf-8")
    paths[key] = path
  return paths


def _print_report(splits: dict[str, list[dict]]) -> None:
  from src.utils.logging_tools import get_logger

  _logger = get_logger(__name__)
  """打印两边的样本数 / 数字数 / 面积分位数，便于确认分布是否对齐"""
  _logger.info(f"\n{'split':<6}{'samples':>9}{'digits':>9}"
        f"{'area p10':>11}{'p50':>10}{'p90':>10}{'max':>10}")
  for key, samples in splits.items():
    digits = sum(len(s.get("objects") or []) for s in samples)
    pct = area_percentiles(samples)
    if pct:
      _logger.info(f"{key:<6}{len(samples):>9}{digits:>9}"
            f"{pct['p10']:>11.0f}{pct['p50']:>10.0f}"
            f"{pct['p90']:>10.0f}{pct['p100']:>10.0f}")
    else:
      _logger.info(f"{key:<6}{len(samples):>9}{digits:>9}" + f"{'-':>41}")


# ---------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description="按数字四边形面积分层划分训练/验证集")
  parser.add_argument("labels", nargs="?", default=None,
                      help="labels.json 路径；缺省取 configs.path.DATASET_LABEL_JSON")
  parser.add_argument("--ratio", type=float, default=0.8, help="训练集比例（默认 0.8）")
  parser.add_argument("--bins", type=int, default=8, help="面积分位层层数（默认 8）")
  parser.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
  parser.add_argument("-o", "--out-dir", default=None,
                      help="输出目录（默认与 labels.json 同目录）")
  parser.add_argument("--no-save", action="store_true", help="只打印分布，不写文件")
  args = parser.parse_args(argv)

  labels = args.labels
  if labels is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from configs.path import PathConfig
    labels = PathConfig.DATASET_LABEL_JSON
  labels = Path(labels)
  if not labels.exists():
    raise FileNotFoundError(f"找不到标签文件：{labels}")

  splits = split_dataset(labels, ratio=args.ratio, bins=args.bins,
                         seed=args.seed)
  print(f"标签文件: {labels}")
  print(f"分层: ratio={args.ratio} bins={args.bins} seed={args.seed}")
  _print_report(splits)

  if not args.no_save:
    out_dir = Path(args.out_dir) if args.out_dir else labels.parent
    paths = save_splits(splits, out_dir)
    print(f"\n已保存: {paths['train']}  ({len(splits['train'])} 条)")
    print(f"已保存: {paths['val']}  ({len(splits['val'])} 条)")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
