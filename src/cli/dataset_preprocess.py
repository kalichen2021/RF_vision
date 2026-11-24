"""批量生成「环境 + 数字」融合样本的 CLI。

流程：读环境图 → ``get_flat_quad`` 找平坦区域 → ``segment_quad`` 按 ``AREA_RANGE``
切出板位 → 数字做透视变换贴上去 → 按基准平面做景深失焦 → 多层灰度融合。

样本多样性 = 环境图 × 字体文件 / 字重 / 画布 × 板位切分 × 数字类别：
``SAMPLE_GEN_NUM`` 可以远多于环境图数量，同一张图会被反复使用，
每轮都重新随机切分板位、重新挑字体与数字。

产物::

    <out_dir>/00000_0072.png   灰度样本图
    <out_dir>/labels.json      整批样本的标签（所有样本合并为一个文件）

用法::

    conda activate NNVN
    python -m src.cli.dataset_preprocess            # 默认 SAMPLE_GEN_NUM 张
    python -m src.cli.dataset_preprocess -n 50 --show
"""

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.progress import (BarColumn, MofNCompleteColumn, Progress,
                           SpinnerColumn, TextColumn, TimeElapsedColumn)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from configs import path as pathcfg
from configs import preprocess_cfg as ppcfg
from configs.preprocess_cfg import THICKNESS_CHOICES, CANVAS_CHOICES, FONT_VARIANTS
from configs.log import UUID as _uuid
from src.preprocess import Preprocess
from src.preprocess.process import build_record, save_dataset, save_sample
from src.utils.debug_tools import read_image
from src.utils.font_processor import font2img, get_available_fonts, get_font_path

# 笔画粗细 / 画布尺寸的候选，用来增加字形多样性
# THICKNESS_CHOICES = (-1, 0, 0, 1, 1, 2)
# CANVAS_CHOICES = (48, 64, 64, 80)
# 每个数字渲染几个字形变体（不同字体 / 字重 / 画布）
# FONT_VARIANTS = 2


def build_pp() -> Preprocess:
  """按 ``configs/preprocess_cfg.py`` 组装 Preprocess（含融合 / 失焦参数）"""
  return Preprocess(
      ppcfg.DISTORTION, ppcfg.NORMALIZE, ppcfg.AREA_RANGE,
      ppcfg.QUADS_SELECT_RATE,
      ppcfg.CONTRAST_RANGE, ppcfg.FUSE_ALPHA, ppcfg.DEFOCUS_STRENGTH,
      ppcfg.DEFOCUS_MAX_SIGMA, ppcfg.DEFOCUS_PERSPECTIVE,
      ppcfg.BASE_DEPTH, ppcfg.PLANE_TOL,
      ppcfg.BOARD_TONE, ppcfg.BOARD_MARGIN, ppcfg.BOARD_SHADOW,
      ppcfg.DEFOCUS_SIZE_RATIO)


def make_font_imgs(count=None, fonts=None, variants=FONT_VARIANTS):
  """渲染一批白字黑底的数字贴图（字体 / 字重 / 画布尺寸随机）

  每个数字可以出多个字形变体，所以池子长度 ≠ 数字类别数：
  第 ``i`` 张渲染的是 ``i % count``，同时把类别一并返回，
  由调用方传给 ``Preprocess.process(..., digit_ids=...)`` 作为标签。

  Args:
      count (int | None): 数字类别数量，缺省取 ``ppcfg.SAMPLE_COUNT_RATE``（需覆盖 0~9）
      fonts (list[str] | None): 候选字体文件名；缺省读本机可用字体
      variants (int): 每个数字出几个字形变体（不同字体 / 字重 / 画布）

  Returns:
      tuple[list[NDArray[np.uint8]], list[int]]: (贴图列表, 与之一一对应的数字类别)
  """
  count = ppcfg.SAMPLE_COUNT_RATE if count is None else int(count)
  fonts = get_available_fonts() if fonts is None else list(fonts)
  if not fonts:
    raise FileNotFoundError("没有可用字体，请检查 FONTS_DIR")

  glyphs: list[np.ndarray] = []
  digit_ids: list[int] = []
  for i in range(count * max(1, int(variants))):
    digit = i % count           # 类别只在 0 ~ count-1 之间，不要用池下标
    glyphs.append(font2img(
        get_font_path(random.choice(fonts)),
        str(digit),
        thickness=random.choice(THICKNESS_CHOICES),
        canvas=random.choice(CANVAS_CHOICES)))
    digit_ids.append(digit)

  return glyphs, digit_ids


def pick_env_path(env_paths, usage, failures,
                  cap=ppcfg.ENV_USE_SOFT_CAP, decay=ppcfg.ENV_USE_DECAY):
  """按使用次数加权随机选一张环境图

  成功使用次数未达 ``cap`` 时权重为 1；达到后每多用一次权重乘以 ``decay``，
  让高频图渐渐降温。读取 / 切分失败的图额外乘 ``0.25^失败次数``，避免坏图被
  反复抽到。

  Args:
      env_paths (list[Path]): 候选环境图
      usage (dict[Path, int]): 已成功使用的次数
      failures (dict[Path, int]): 连续失败次数
      cap (int): 开始降温的使用次数阈值
      decay (float): 超阈值后的权重衰减系数

  Returns:
      Path: 选中的环境图
  """
  weights = []
  for path in env_paths:
    used = usage.get(path, 0)
    weight = 1.0 if used < cap else decay ** (used - cap + 1)
    weight *= 0.25 ** failures.get(path, 0)
    weights.append(max(weight, 1e-3))
  return random.choices(env_paths, weights=weights, k=1)[0]


def main(argv: list[str] | None = None) -> int:
  """批量生成融合样本，返回进程退出码"""
  console = Console()
  parser = argparse.ArgumentParser(description="批量生成环境 + 数字融合样本")
  parser.add_argument("-n", "--num", type=int, default=ppcfg.SAMPLE_GEN_NUM,
                      help="生成样本数（默认 SAMPLE_GEN_NUM）")
  parser.add_argument("--show", action="store_true",
                      help="把四边形 / 贴图 / 融合等中间结果写进日志目录")
  args = parser.parse_args(argv)

  total = max(0, int(args.num))
  env_root = pathcfg.RAW_DIR / "env"
  env_paths = sorted(env_root.glob("*.jpg"))
  if not env_paths:
    console.print(f"[bold red]未找到环境图：[/]{env_root}")
    return 1
  if total == 0:
    console.print("[yellow]生成数量为 0，无事可做[/]")
    return 0

  out_dir = pathcfg.PROCESSED_DIR / _uuid
  out_dir.mkdir(parents=True, exist_ok=True)
  pp = build_pp()
  fonts = get_available_fonts()
  console.print(f"[bold]环境图[/] {len(env_paths)} 张 · "
                f"[bold]字体[/] {len(fonts)} 个 · "
                f"[bold]目标[/] {total} 张 · "
                f"[bold]输出[/] {out_dir}")

  quads_cache: dict[Path, list] = {}
  records: list[dict] = []
  skipped = 0
  usage = {p: 0 for p in env_paths}     # 每张图成功使用次数（用于降权）
  failures = {p: 0 for p in env_paths}  # 每张图失败次数（用于避开坏图）
  attempts = 0
  max_attempts = max(total * 20, len(env_paths) * 50)
  started = time.perf_counter()

  columns = (SpinnerColumn(), TextColumn("[bold cyan]{task.description}"),
             BarColumn(), MofNCompleteColumn(),
             TextColumn("{task.fields[note]}"), TimeElapsedColumn())
  with Progress(*columns, console=console) as progress:
    task = progress.add_task("生成样本", total=total, note="")

    while len(records) < total and attempts < max_attempts:
      attempts += 1
      path = pick_env_path(env_paths, usage, failures)

      try:
        env_img = read_image(path)
      except ValueError as exc:      # 坏图只跳过，不中断整批
        skipped += 1
        failures[path] = failures.get(path, 0) + 1
        progress.update(task, note=f"[red]读取失败[/] {path.name}")
        console.print(f"[red]{exc}[/]")
        continue

      # 平坦区域对同一张图是确定的：缓存后，同一张图只换切分 / 字体 / 数字
      if path not in quads_cache:
        quads_cache[path] = pp.detect_quads(env_img)

      glyphs, digit_ids = make_font_imgs(ppcfg.SAMPLE_COUNT_RATE, fonts)
      fused, labels = pp.process(
          glyphs, env_img, show_cnt=args.show, quads=quads_cache[path],
          digit_ids=digit_ids)

      if fused is None:
        skipped += 1
        failures[path] = failures.get(path, 0) + 1
        progress.update(task, note=f"[yellow]跳过[/] {path.name} "
                                   f"无可用板位（累计 {skipped} 次）")
        continue

      usage[path] = usage.get(path, 0) + 1
      failures[path] = 0
      name = f"{len(records):05d}_{path.stem}"
      extra = {"source": path.name,
               "digit_count": len(labels),
               "area_range": list(ppcfg.AREA_RANGE)}
      save_sample(out_dir, name, fused, labels, extra, write_labels=False)
      records.append(build_record(f"{name}.png", fused, labels, extra))
      progress.update(task, advance=1,
                      note=f"[green]ok[/] {path.name} 数字 {len(labels)} 个")

  labels_path = save_dataset(out_dir, records)
  elapsed = time.perf_counter() - started

  console.print()
  console.print(f"[bold]输出目录[/]  {out_dir}", soft_wrap=True)
  console.print(f"[bold]标签文件[/]  {labels_path}"
                f"（{len(records)} 条记录）", soft_wrap=True)
  console.print(f"[bold]灰度样本[/]  {len(records)}/{total} 张    "
                f"[bold]跳过[/]  {skipped} 次    "
                f"[bold]耗时[/]  {elapsed:.1f} s")
  if len(records) < total:
    console.print(f"[yellow]只生成 {len(records)}/{total} 张[/]"
                  f"（尝试 {attempts} 次，跳过 {skipped} 次）")
    console.print("[yellow]提示：数字个数范围 / 面积区间可能超出环境图的承载能力，"
                  "可调小 SAMPLE_DIGIT_RANGE 或 AREA_RANGE[/]")

  return 0 if records else 1


if __name__ == "__main__":
  sys.exit(main())
  