"""数据集统计 + 概览图：命令行入口。

``src/preprocess/analysis.py`` 只负责统计与画图，**所有输出都在这里**。

用法::

    conda activate NNVN
    python src/cli/dataset_analyze.py                    # 自动选最新的完整数据集
    python src/cli/dataset_analyze.py -d dataset/processed/20261124_091503_195
    python src/cli/dataset_analyze.py --no-show -o out.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
from rich.console import Console

from configs.path import PathConfig
from src.preprocess.analysis import Analysis
from src.utils.logging_tools import setup_logging


def default_dataset_dir() -> Path:
  """选默认数据集目录：优先 ``PathConfig.SELECT_IMGSET``，否则取最新的完整目录。

  Returns:
      含有 ``labels.json`` 的数据集目录。

  Raises:
      FileNotFoundError: 一个完整数据集都找不到。
  """
  if (PathConfig.SELECT_IMGSET / "labels.json").exists():
    return PathConfig.SELECT_IMGSET
  candidates = sorted(
      (p for p in PathConfig.PROCESSED_DIR.iterdir()
       if p.is_dir() and (p / "labels.json").exists()),
      key=lambda p: p.name)
  if not candidates:
    raise FileNotFoundError(
        f"{PathConfig.PROCESSED_DIR} 下没有含 labels.json 的数据集")
  return candidates[-1]


def print_summary(stats: dict, console: Console) -> None:
  """把 :meth:`Analysis.summary` 的结果打印成人类可读的报告。

  Args:
      stats: ``Analysis.summary()`` 返回的统计字典。
      console: rich 控制台。
  """
  console.print(f"\n[bold]数据集[/] {stats['dataset_dir']}")
  console.print(f"  样本数        : {stats['n_samples']}")
  console.print(f"  数字总数      : {stats['n_digits']}")
  console.print(f"  源图数        : {stats['n_sources']}")

  dps = stats["digits_per_sample"]
  console.print(f"  每样本数字数  : mean={dps['mean']:.2f} min={dps['min']} "
                f"max={dps['max']} median={dps['median']:.1f}")
  console.print(f"  数字类别分布  : {stats['digit_counts']}")
  console.print(f"  类别均衡度    : {stats['digit_balance']:.3f} (min/max)")

  def _pct_row(label: str, key: str, fmt: str,
               keys: tuple[str, ...] = ("p5", "p50", "p95")) -> None:
    """打印一行分位数（字段缺失时跳过）。"""
    row = stats.get(key) or {}
    if not row:
      return
    body = " ".join(f"{k}={format(row[k], fmt)}" for k in keys if k in row)
    console.print(f"  {label:<22}: {body}")

  _pct_row("数字四边形面积(pixel^2)", "quad_area", ".0f")
  _pct_row("数字宽度/高度(pixel)", "quad_width", ".1f")
  _pct_row("板面/数字面积比", "board_digit_ratio", ".2f")
  _pct_row("depth", "depth", ".3f")
  _pct_row("sigma", "sigma", ".2f")

  top_sources = stats.get("top_sources") or []
  if top_sources:
    console.print("  数字最多的源图:")
    for src, item in top_sources:
      console.print(f"    {src:<16} samples={item['samples']:<4} "
                    f"digits={item['digits']}")


def main(argv: list[str] | None = None) -> int:
  """命令行入口，返回进程退出码。

  Args:
      argv: 参数列表，None 表示取 ``sys.argv[1:]``。

  Returns:
      0 成功，1 数据集有问题。
  """
  parser = argparse.ArgumentParser(description="数据集统计 + 概览图")
  parser.add_argument("-d", "--dataset-dir", default=None,
                      help="数据集目录（默认自动选最新的完整数据集）")
  parser.add_argument("-o", "--out", default=None,
                      help="总览图保存路径（默认 dataset/log/analysis/<数据集名>_overview.png）")
  parser.add_argument("--no-show", action="store_true", help="不弹窗，只保存")
  parser.add_argument("--no-save", action="store_true", help="不保存图片")
  args = parser.parse_args(argv)

  setup_logging()      # src 模块只发 logger，格式 / 级别在这里定
  console = Console()
  if args.no_show:
    matplotlib.use("Agg")     # 无显示器 / 只想保存时用非交互后端

  try:
    dataset_dir = (Path(args.dataset_dir) if args.dataset_dir
                   else default_dataset_dir())
    out_path = args.out
    if out_path is None and not args.no_save:
      out_path = (PathConfig.LOG_DIR / "analysis"
                  / f"{Path(dataset_dir).name}_overview.png")
    analysis = Analysis(dataset_dir)
    stats = analysis.summary()          # 统计结果由 CLI 负责打印
    analysis.run(out_path=out_path, show=not args.no_show)
  except (FileNotFoundError, ValueError) as exc:
    console.print(f"[bold red]{exc}[/]")
    return 1

  print_summary(stats, console)
  if out_path is not None:
    console.print(f"\n总览图已保存: {out_path}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
