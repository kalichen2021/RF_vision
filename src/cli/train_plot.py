"""训练日志可视化入口：读 `dataset/log/train/*.json`，画曲线 + 打印摘要。

``src/training/plot.py`` 只负责读数据、算摘要、画图，**所有输出都在这里**。

用法::

    conda activate NNVN
    python src/cli/train_plot.py                 # 最新一次训练（head + finetune 自动配对）
    python src/cli/train_plot.py --list          # 列出所有训练记录
    python src/cli/train_plot.py --uuid 20261124_202633_404
    python src/cli/train_plot.py -f a_head.json -f a_finetune.json
    python src/cli/train_plot.py --no-show       # 只存图（默认存 dataset/log/train/plots/）
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
from rich.console import Console

from configs.path import PathConfig
from src.training.plot import history_summary, load_history, plot_history
from src.utils.logging_tools import setup_logging
from src.utils.plotting import save_figure

# 文件名格式：train_<uuid>_<stage>.json
_NAME_RE = re.compile(r"^train_(?P<uuid>.+)_(?P<stage>head|finetune)\.json$")
# 阶段绘制顺序（head 在前，finetune 在后）
_STAGE_ORDER: tuple[str, ...] = ("head", "finetune")


def train_log_dir() -> Path:
  """训练记录目录（`dataset/log/train`）。"""
  return PathConfig.LOG_DIR / "train"


def list_runs(log_dir: Path | None = None) -> dict[str, dict[str, Path]]:
  """按 UUID 归组训练记录。

  Args:
      log_dir: 记录目录，None 表示 ``dataset/log/train``。

  Returns:
      `{uuid: {"head": 路径, "finetune": 路径}}`；同一 UUID 缺哪个阶段就没有
      对应键，按最新修改时间**倒序**排列。
  """
  log_dir = train_log_dir() if log_dir is None else Path(log_dir)
  runs: dict[str, dict[str, Path]] = {}
  for p in log_dir.glob("train_*.json"):
    m = _NAME_RE.match(p.name)
    if not m:
      continue
    runs.setdefault(m.group("uuid"), {})[m.group("stage")] = p
  return dict(sorted(runs.items(),
                     key=lambda kv: max(f.stat().st_mtime for f in kv[1].values()),
                     reverse=True))


def resolve_files(args: argparse.Namespace, console: Console) -> tuple[list[Path], str]:
  """决定要画哪些记录文件。

  Args:
      args: 命令行参数（`files` / `uuid`）。
      console: rich 控制台（用于错误提示）。

  Returns:
      `(按阶段排好序的记录文件, 该组训练的 UUID / 标识)`。

  Raises:
      SystemExit: 找不到可画的记录。
  """
  if args.files:
    files = [Path(f) for f in args.files]
    tag = _NAME_RE.match(files[0].name)
    return files, (tag.group("uuid") if tag else files[0].stem)

  runs = list_runs()
  if not runs:
    console.print(f"[red]没找到训练记录[/] {train_log_dir()}\\train_*.json")
    raise SystemExit(1)
  if args.uuid:
    if args.uuid not in runs:
      console.print(f"[red]没有 UUID 为 {args.uuid} 的记录[/]"
                    f"，可用 --list 查看")
      raise SystemExit(1)
    uuid = args.uuid
  else:
    uuid = next(iter(runs))          # list_runs 已按时间倒序
  stages = runs[uuid]
  files = [stages[s] for s in _STAGE_ORDER if s in stages]
  return files, uuid


def print_rows(rows: list[dict], console: Console) -> None:
  """把 :func:`history_summary` 的结果逐行打印（比宽表格更适合窄终端）。

  Args:
      rows: 每个阶段一条摘要 dict。
      console: rich 控制台。
  """
  for r in rows:
    head = f"[bold]{r['stage']}[/]"
    if r["ckpt"]:
      head += f"  {r['ckpt']}"
    console.print(head)
    console.print(f"  轮数        : {r['epochs_run']}/{r['epochs_planned']}"
                  + ("  (早停)" if r["early_stopped"] else ""))
    console.print(f"  参数量      : {r['params']:,}")
    if r["best_epoch"] == r["best_epoch"]:        # nan != nan
      console.print(f"  最优轮      : {r['best_epoch']:.0f}"
                    f"  (val loss {r['best_val']:.4f})")
    console.print(f"  末轮 loss   : train {r['final_train']:.4f} / "
                  f"val {r['final_val']:.4f}")
    console.print(f"  末轮 lr     : {r['final_lr']:.2e}")
    console.print(f"  总耗时      : {r['total_seconds'] / 60:.1f} min "
                  f"(等数据占 {r['data_wait_ratio'] * 100:.1f}%)")
    console.print("")


def main(argv: list[str] | None = None) -> int:
  """命令行入口，返回进程退出码。

  Args:
      argv: 参数列表，None 表示取 ``sys.argv[1:]``。

  Returns:
      0 成功，1 记录缺失 / 结构不对。
  """
  parser = argparse.ArgumentParser(description="训练日志可视化（曲线 + 摘要）")
  parser.add_argument("-f", "--files", action="append", default=None,
                      help="训练记录 JSON，可重复给多次（head → finetune 顺序）")
  parser.add_argument("--uuid", default=None, help="按 UUID 选一组（默认最新一组）")
  parser.add_argument("--list", action="store_true", help="只列出已有记录，不画图")
  parser.add_argument("-o", "--out", default=None,
                      help="图片保存路径（默认 dataset/log/train/plots/train_<uuid>.png）")
  parser.add_argument("--no-show", action="store_true", help="不弹窗，只保存")
  parser.add_argument("--no-save", action="store_true", help="不保存，只弹窗")
  args = parser.parse_args(argv)

  setup_logging()      # src 模块只发 logger，格式 / 级别在这里定
  console = Console()
  if args.no_show:
    matplotlib.use("Agg")     # 无显示器 / 只保存时用非交互后端

  if args.list:
    runs = list_runs()
    if not runs:
      console.print(f"[yellow]暂无训练记录[/] {train_log_dir()}")
      return 0
    for uuid, stages in runs.items():
      names = " + ".join(s for s in _STAGE_ORDER if s in stages)
      console.print(f"{uuid}  [{names}]  "
                    f"{max(p.stat().st_mtime for p in stages.values()):.0f}")
    return 0

  files, tag = resolve_files(args, console)
  try:
    histories = [load_history(f) for f in files]
  except (FileNotFoundError, ValueError) as exc:
    console.print(f"[red]{exc}[/]")
    return 1

  labels = [f.stem.split("_")[-1] for f in files]     # head / finetune
  print_rows(history_summary(histories, labels), console)

  out_path = args.out
  if out_path is None and not args.no_save:
    out_path = train_log_dir() / "plots" / f"train_{tag}.png"
  fig = plot_history(histories, labels)
  saved = save_figure(fig, out_path, show=not args.no_show)
  if saved is not None:
    console.print(f"曲线已保存: [bold]{saved}[/]")
  return 0


if __name__ == "__main__":
  sys.exit(main())
