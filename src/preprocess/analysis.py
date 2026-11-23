
"""数据集统计与可视化（matplotlib）。

读取 ``dataset/processed/<uuid>/labels.json``（或逐样本 ``*.json``），
统计并可视化：

- 来源数据：每个源图 ``source`` 贡献的样本数 / 数字数
- 数字：类别分布（0~9）、每张样本的数字个数
- 四边形大小：数字四边形面积 / 长宽 / 长宽比、板面面积、板/数字面积比
- 位置与景深：数字中心热力图、``depth`` / ``sigma`` 分布

用法::

    conda activate NNVN
    python -m src.preprocess.analysis                 # 自动选含 labels.json 的最新数据集
    python -m src.preprocess.analysis dataset/processed/20261121_091927_521
    python -m src.preprocess.analysis --no-show -o out.png

Sample dataset format.
'''
{
  "image": "00000_0038.png",
  "width": 1280,
  "height": 720,
  "source": "0038.jpg",
  "digit_count": 3,
  "area_range": [5000, 40000],
  "objects": [
    { 
      "digit": 5,
      "quad": [
        [54.38, 617.94],
        [152.37, 621.06],
        [142.38, 684.12],
        [43.01, 689.06]
      ],
      "quad_norm": [
        [0.0425, 0.8594],
        [0.1191, 0.8638],
        [0.1113, 0.9515],
        [0.0336, 0.9584]
      ],
      "board_quad": [
        [30.12, 596.42],
        [182.64, 604.83],
        [167.62, 699.83],
        [11.76, 711.09]
      ],
      "center": [98.04, 653.04],
      "depth": 1.0,
      "sigma": 0.0
    },...
    
}
'''
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


# ---------------------------------------------------------------- 读取

def load_dataset(dataset_dir: str | Path) -> list[dict]:
  """读取数据集标签。

  支持三种输入：

  - 指向 ``labels.json`` 文件
  - 目录下的 ``labels.json``（批量生成产物）
  - 目录下逐样本 ``<name>.json``（``save_sample(write_labels=True)`` 产物）

  Args:
      dataset_dir: 数据集目录或标签文件路径

  Returns:
      list[dict]: 样本记录列表（仅保留含 ``objects`` 的记录）
  """
  path = Path(dataset_dir)
  if path.is_file():
    data = json.loads(path.read_text(encoding="utf-8"))
  else:
    labels = path / "labels.json"
    if labels.exists():
      data = json.loads(labels.read_text(encoding="utf-8"))
    else:
      files = sorted(p for p in path.glob("*.json") if p.name != "labels.json")
      data = [json.loads(p.read_text(encoding="utf-8")) for p in files]
      if not data:
        raise FileNotFoundError(
            f"{path} 下没有 labels.json 或逐样本 *.json，无法分析")

  samples = data.get("samples", []) if isinstance(data, dict) else data
  return [s for s in samples if s.get("objects")]


def iter_objects(samples: Iterable[dict]):
  """遍历所有 ``(样本, object)`` 组合"""
  for sample in samples:
    for obj in sample.get("objects", []):
      yield sample, obj


# ---------------------------------------------------------------- 几何统计

def quad_area(quad: Sequence[Sequence[float]]) -> float:
  """四边形面积（鞋带公式，取绝对值）"""
  p = np.asarray(quad, dtype=float).reshape(-1, 2)
  x, y = p[:, 0], p[:, 1]
  return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def quad_size(quad: Sequence[Sequence[float]]) -> tuple[float, float]:
  """四边形的平均宽 / 高（上下边、左右边中点距离）"""
  p = np.asarray(quad, dtype=float).reshape(-1, 2)
  w = (np.linalg.norm(p[1] - p[0]) + np.linalg.norm(p[2] - p[3])) / 2.0
  h = (np.linalg.norm(p[3] - p[0]) + np.linalg.norm(p[2] - p[1])) / 2.0
  return float(w), float(h)


def quad_aspect(quad: Sequence[Sequence[float]]) -> float:
  """长宽比（>=1，长边 / 短边）"""
  w, h = quad_size(quad)
  lo = max(min(w, h), 1e-6)
  return float(max(w, h) / lo)


# ---------------------------------------------------------------- 统计类

class Analysis:
  """分析数据集的统计信息，并提供 matplotlib 可视化。"""

  def __init__(self, dataset_dir: str | Path):
    self.dataset_dir = Path(dataset_dir)
    self.samples = load_dataset(self.dataset_dir)
    if not self.samples:
      raise ValueError(f"{self.dataset_dir} 中没有可用样本")

    self.n_samples = len(self.samples)
    self.sources: list[str] = [
        str(s.get("source", "unknown")) for s in self.samples]

    # 逐 object 展开的数组，绘图时直接使用
    self.digits: list[int] = []
    self.areas: list[float] = []
    self.area_ratios: list[float] = []   # 数字面积 / 整图面积
    self.widths: list[float] = []
    self.heights: list[float] = []
    self.aspects: list[float] = []
    self.board_areas: list[float] = []
    self.board_ratios: list[float] = []  # 板面面积 / 数字面积
    self.centers_x: list[float] = []
    self.centers_y: list[float] = []     # 归一化坐标（0~1）
    self.depths: list[float] = []
    self.sigmas: list[float] = []
    self.digits_per_sample: list[int] = []

    for sample in self.samples:
      objects = sample.get("objects", [])
      self.digits_per_sample.append(len(objects))
      w_img = float(sample.get("width") or 0.0)
      h_img = float(sample.get("height") or 0.0)
      img_area = w_img * h_img

      for obj in objects:
        self.digits.append(int(obj.get("digit", -1)))
        area = 0.0
        quad = obj.get("quad")
        if quad:
          area = quad_area(quad)
          w, h = quad_size(quad)
          self.areas.append(area)
          self.widths.append(w)
          self.heights.append(h)
          self.aspects.append(quad_aspect(quad))
          if img_area > 0:
            self.area_ratios.append(area / img_area)

        board = obj.get("board_quad")
        if board:
          b_area = quad_area(board)
          self.board_areas.append(b_area)
          if area > 1e-6:
            self.board_ratios.append(b_area / area)

        center = obj.get("center")
        if center:
          cx, cy = float(center[0]), float(center[1])
          if w_img > 0 and h_img > 0:
            self.centers_x.append(cx / w_img)
            self.centers_y.append(cy / h_img)
          else:
            self.centers_x.append(cx)
            self.centers_y.append(cy)

        if "depth" in obj:
          self.depths.append(float(obj["depth"]))
        if "sigma" in obj:
          self.sigmas.append(float(obj["sigma"]))

    self.n_digits = len(self.digits)

  # ------------------------------------------------------------ 概览

  def digit_counts(self) -> dict[int, int]:
    """数字类别 -> 出现次数（覆盖 0~9，缺失补 0）"""
    counts = Counter(self.digits)
    return {d: counts.get(d, 0) for d in range(10)}

  def source_stats(self) -> dict[str, dict[str, int]]:
    """每个源图的样本数 / 数字数"""
    stats: dict[str, dict[str, int]] = {}
    for sample, _ in iter_objects(self.samples):
      src = str(sample.get("source", "unknown"))
      stats.setdefault(src, {"samples": 0, "digits": 0})["digits"] += 1
    for src in self.sources:
      stats.setdefault(src, {"samples": 0, "digits": 0})["samples"] += 1
    return stats

  def summary(self) -> dict:
    """汇总关键统计量（同时打印到控制台）"""
    def _pct(values, qs=(0, 5, 50, 95, 100)) -> dict[str, float]:
      if not values:
        return {}
      arr = np.asarray(values, dtype=float)
      return {f"p{q}": float(np.percentile(arr, q)) for q in qs}

    digit_counts = self.digit_counts()
    per_sample = np.asarray(self.digits_per_sample, dtype=float)
    src_stats = self.source_stats()
    top_sources = sorted(src_stats.items(), key=lambda kv: kv[1]["digits"],
                         reverse=True)[:5]
    max_count = max(digit_counts.values()) if digit_counts else 0

    stats = {
        "dataset_dir": str(self.dataset_dir),
        "n_samples": self.n_samples,
        "n_digits": self.n_digits,
        "n_sources": len(src_stats),
        "digits_per_sample": {
            "mean": float(per_sample.mean()) if len(per_sample) else 0.0,
            "min": int(per_sample.min()) if len(per_sample) else 0,
            "max": int(per_sample.max()) if len(per_sample) else 0,
            "median": float(np.median(per_sample)) if len(per_sample) else 0.0,
        },
        "digit_counts": digit_counts,
        "digit_balance": (min(digit_counts.values()) / max_count)
        if max_count else 0.0,
        "quad_area": _pct(self.areas),
        "quad_area_ratio": _pct(self.area_ratios),
        "quad_width": _pct(self.widths),
        "quad_height": _pct(self.heights),
        "quad_aspect": _pct(self.aspects),
        "board_area": _pct(self.board_areas),
        "board_digit_ratio": _pct(self.board_ratios),
        "depth": _pct(self.depths),
        "sigma": _pct(self.sigmas),
        "top_sources": top_sources,
    }

    print(f"\n数据集: {stats['dataset_dir']}")
    print(f"  样本数        : {stats['n_samples']}")
    print(f"  数字总数      : {stats['n_digits']}")
    print(f"  源图数        : {stats['n_sources']}")
    print(f"  每样本数字数  : "
          f"mean={stats['digits_per_sample']['mean']:.2f} "
          f"min={stats['digits_per_sample']['min']} "
          f"max={stats['digits_per_sample']['max']} "
          f"median={stats['digits_per_sample']['median']:.1f}")
    print(f"  数字类别分布  : {digit_counts}")
    print(f"  类别均衡度    : {stats['digit_balance']:.3f} (min/max)")
    if self.areas:
      print(f"  数字四边形面积(pixel^2): "
            f"min={stats['quad_area']['p0']:.0f} "
            f"p5={stats['quad_area']['p5']:.0f} "
            f"p50={stats['quad_area']['p50']:.0f} "
            f"p95={stats['quad_area']['p95']:.0f} "
            f"max={stats['quad_area']['p100']:.0f}")
      print(f"  数字宽度/高度(pixel)  : "
            f"w_p50={stats['quad_width']['p50']:.1f} "
            f"h_p50={stats['quad_height']['p50']:.1f} "
            f"aspect_p50={stats['quad_aspect']['p50']:.2f} "
            f"aspect_p95={stats['quad_aspect']['p95']:.2f}")
    if self.board_ratios:
      print(f"  板面/数字面积比       : "
            f"p5={stats['board_digit_ratio']['p5']:.2f} "
            f"p50={stats['board_digit_ratio']['p50']:.2f} "
            f"p95={stats['board_digit_ratio']['p95']:.2f}")
    if self.depths:
      print(f"  depth                 : "
            f"min={stats['depth']['p0']:.3f} p50={stats['depth']['p50']:.3f} "
            f"max={stats['depth']['p100']:.3f}")
    if self.sigmas:
      print(f"  sigma                 : "
            f"min={stats['sigma']['p0']:.2f} p50={stats['sigma']['p50']:.2f} "
            f"max={stats['sigma']['p100']:.2f}")
    print("  数字最多的源图:")
    for src, s in top_sources:
      print(f"    {src:<16} samples={s['samples']:<4} digits={s['digits']}")

    return stats

  # ------------------------------------------------------------ 单图绘制

  def plot_digits(self, ax) -> None:
    """数字类别分布柱状图"""
    counts = self.digit_counts()
    digits = list(counts)
    values = [counts[d] for d in digits]
    bars = ax.bar([str(d) for d in digits], values, color="#4C78A8")
    for bar, v in zip(bars, values):
      ax.text(bar.get_x() + bar.get_width() / 2, v, str(v),
              ha="center", va="bottom", fontsize=8)
    ax.set_title("数字类别分布")
    ax.set_xlabel("digit")
    ax.set_ylabel("数量")

  def plot_digits_per_sample(self, ax) -> None:
    """每张样本的数字个数分布"""
    values = self.digits_per_sample
    if not values:
      ax.set_visible(False)
      return
    hi = max(values)
    bins = np.arange(0, hi + 2) - 0.5
    ax.hist(values, bins=bins, color="#F58518", edgecolor="white")
    ax.set_title("每张样本的数字个数")
    ax.set_xlabel("digits / sample")
    ax.set_ylabel("样本数")
    ax.set_xticks(range(0, hi + 1, 2 if hi > 12 else 1))

  def plot_sources(self, ax, top: int = 15) -> None:
    """来源图贡献的数字数（Top N）"""
    stats = self.source_stats()
    items = sorted(stats.items(), key=lambda kv: kv[1]["digits"],
                   reverse=True)[:top]
    if not items:
      ax.set_visible(False)
      return
    names = [k for k, _ in items][::-1]
    values = [v["digits"] for _, v in items][::-1]
    ax.barh(range(len(names)), values, color="#54A24B")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_title(f"来源图片数字数 (Top {len(names)}/{len(stats)})")
    ax.set_xlabel("digits")

  def plot_quad_area(self, ax) -> None:
    """数字四边形面积分布"""
    if not self.areas:
      ax.set_visible(False)
      return
    ax.hist(self.areas, bins=40, color="#E45756", edgecolor="white")
    med = float(np.median(self.areas))
    ax.axvline(med, color="black", ls="--", lw=1, label=f"median={med:.0f}")
    ax.set_title("数字四边形面积")
    ax.set_xlabel("area (pixel²)")
    ax.set_ylabel("数量")
    ax.legend(fontsize=8)

  def plot_area_ratio(self, ax) -> None:
    """数字面积占整图面积比例"""
    if not self.area_ratios:
      ax.set_visible(False)
      return
    ax.hist(self.area_ratios, bins=40, color="#B279A2", edgecolor="white")
    ax.set_title("数字面积 / 整图面积")
    ax.set_xlabel("ratio")
    ax.set_ylabel("数量")

  def plot_width_height(self, ax) -> None:
    """数字宽高散点（等比例参考线）"""
    if not self.widths:
      ax.set_visible(False)
      return
    ax.scatter(self.widths, self.heights, s=6, alpha=0.35, color="#4C78A8")
    hi = max(max(self.widths), max(self.heights))
    ax.plot([0, hi], [0, hi], color="gray", ls="--", lw=1)
    ax.set_title("数字宽 × 高")
    ax.set_xlabel("width (pixel)")
    ax.set_ylabel("height (pixel)")

  def plot_aspect(self, ax) -> None:
    """数字四边形长宽比"""
    if not self.aspects:
      ax.set_visible(False)
      return
    ax.hist(self.aspects, bins=40, color="#72B7B2", edgecolor="white")
    ax.axvline(1.0, color="black", ls=":", lw=1, label="square")
    p95 = float(np.percentile(self.aspects, 95))
    ax.axvline(p95, color="red", ls="--", lw=1, label=f"p95={p95:.2f}")
    ax.set_title("数字四边形长宽比 (长/短)")
    ax.set_xlabel("aspect")
    ax.set_ylabel("数量")
    ax.legend(fontsize=8)

  def plot_board_area(self, ax) -> None:
    """板面四边形面积分布"""
    if not self.board_areas:
      ax.set_visible(False)
      return
    ax.hist(self.board_areas, bins=40, color="#EECA3B", edgecolor="white")
    med = float(np.median(self.board_areas))
    ax.axvline(med, color="black", ls="--", lw=1, label=f"median={med:.0f}")
    ax.set_title("板面四边形面积")
    ax.set_xlabel("area (pixel²)")
    ax.set_ylabel("数量")
    ax.legend(fontsize=8)

  def plot_centers(self, ax) -> None:
    """数字中心位置热力图（归一化坐标）"""
    if not self.centers_x:
      ax.set_visible(False)
      return
    h = ax.hist2d(self.centers_x, self.centers_y, bins=24,
                  range=[[0, 1], [0, 1]], cmap="viridis")
    ax.invert_yaxis()
    ax.set_title("数字中心分布 (归一化)")
    ax.set_xlabel("x / width")
    ax.set_ylabel("y / height")
    plt.colorbar(h[3], ax=ax, fraction=0.046, pad=0.03)

  def plot_depth(self, ax) -> None:
    """深度分布"""
    if not self.depths:
      ax.set_visible(False)
      return
    ax.hist(self.depths, bins=30, color="#9D755D", edgecolor="white")
    ax.set_title("depth 分布")
    ax.set_xlabel("depth")
    ax.set_ylabel("数量")

  def plot_sigma(self, ax) -> None:
    """失焦 sigma 分布"""
    if not self.sigmas:
      ax.set_visible(False)
      return
    ax.hist(self.sigmas, bins=30, color="#BAB0AC", edgecolor="white")
    ax.set_title("失焦 sigma 分布")
    ax.set_xlabel("sigma (pixel)")
    ax.set_ylabel("数量")

  # ------------------------------------------------------------ 组合图

  def plot_overview(self):
    """12 面板总览图，返回 matplotlib Figure"""
    _setup_cjk_font()
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    fig.suptitle(
        f"数据集统计总览 — {self.dataset_dir.name} "
        f"(samples={self.n_samples}, digits={self.n_digits})",
        fontsize=15)

    self.plot_digits(axes[0, 0])
    self.plot_digits_per_sample(axes[0, 1])
    self.plot_sources(axes[0, 2])
    self.plot_quad_area(axes[0, 3])

    self.plot_area_ratio(axes[1, 0])
    self.plot_width_height(axes[1, 1])
    self.plot_aspect(axes[1, 2])
    self.plot_board_area(axes[1, 3])

    self.plot_centers(axes[2, 0])
    self.plot_depth(axes[2, 1])
    self.plot_sigma(axes[2, 2])

    # 第 12 格：每个源图的「样本数 × 数字数」，直观看出复用程度
    ax = axes[2, 3]
    src_stats = self.source_stats()
    if src_stats:
      xs = [v["samples"] for v in src_stats.values()]
      ys = [v["digits"] for v in src_stats.values()]
      ax.scatter(xs, ys, s=12, alpha=0.6, color="#4C78A8")
      ax.set_title("源图复用情况")
      ax.set_xlabel("该源图生成的样本数")
      ax.set_ylabel("该源图贡献的数字数")
    else:
      ax.set_visible(False)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig

  def run(self, out_path: str | Path | None = None, show: bool = True):
    """打印统计 + 输出总览图

    Args:
        out_path: 图片保存路径；None 表示不保存
        show: 是否交互显示（阻塞）

    Returns:
        matplotlib.figure.Figure: 总览图
    """
    self.summary()
    fig = self.plot_overview()
    if out_path is not None:
      out_path = Path(out_path)
      out_path.parent.mkdir(parents=True, exist_ok=True)
      fig.savefig(out_path, dpi=130, bbox_inches="tight")
      print(f"\n总览图已保存: {out_path}")
    if show:
      plt.show()
    else:
      plt.close(fig)
    return fig


# ---------------------------------------------------------------- CLI

def _default_dataset_dir() -> Path:
  """优先 SELECT_IMGSET，否则取最新一个含 labels.json 的目录"""
  sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
  from configs import path as pathcfg

  if (pathcfg.SELECT_IMGSET / "labels.json").exists():
    return pathcfg.SELECT_IMGSET
  candidates = sorted(
      (p for p in pathcfg.PROCESSED_DIR.iterdir()
       if p.is_dir() and (p / "labels.json").exists()),
      key=lambda p: p.name)
  if not candidates:
    raise FileNotFoundError(
        f"{pathcfg.PROCESSED_DIR} 下没有含 labels.json 的数据集")
  return candidates[-1]


def _setup_cjk_font() -> None:
  """尽量启用中文字体，避免标题显示为方块"""
  prefer = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
            "PingFang SC", "WenQuanYi Zen Hei"]
  available = {f.name for f in font_manager.fontManager.ttflist}
  picked = [name for name in prefer if name in available]
  if picked:
    plt.rcParams["font.sans-serif"] = picked + ["DejaVu Sans"]
  plt.rcParams["axes.unicode_minus"] = False


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(
      description="数据集统计与 matplotlib 可视化")
  parser.add_argument("dataset_dir", nargs="?",
                      help="数据集目录或 labels.json（缺省自动选择）")
  parser.add_argument("-o", "--out", default=None,
                      help="总览图保存路径（缺省 dataset/log/analysis/<name>_overview.png）")
  parser.add_argument("--no-show", action="store_true",
                      help="不弹窗，仅保存图片")
  parser.add_argument("--no-save", action="store_true",
                      help="不保存图片")
  args = parser.parse_args(argv)

  if args.no_show:
    import matplotlib
    matplotlib.use("Agg")

  _setup_cjk_font()

  dataset_dir = (Path(args.dataset_dir) if args.dataset_dir
                 else _default_dataset_dir())

  out_path = args.out
  if out_path is None and not args.no_save:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from configs import path as pathcfg
    out_path = pathcfg.LOG_DIR / "analysis" / \
        f"{Path(dataset_dir).name}_overview.png"

  analysis = Analysis(dataset_dir)
  analysis.run(out_path=out_path, show=not args.no_show)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
