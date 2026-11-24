"""matplotlib 的公共绘制设置与保存工具（中文字体、保存 / 显示收尾）。

数据集分析（`src/preprocess/analysis.py`）与训练日志可视化
（`src/training/plot.py`）共用同一套设置，避免各自维护一份字体表。

注意：matplotlib 很重，本模块只允许在 CLI 侧导入（`src/cli/*`），
不要放进训练 / 数据管线（`DigitDataset` / DataLoader worker）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.figure import Figure

# 优先尝试的中文字体：Windows 一般有前两个，Linux 常见 Noto / 文泉驿
_CJK_CANDIDATES: tuple[str, ...] = (
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
    "PingFang SC", "WenQuanYi Zen Hei",
)


def setup_cjk_font(prefer: Sequence[str] | None = None) -> str | None:
  """尽量启用中文字体，避免标题 / 图例显示成方块。

  Args:
      prefer: 候选字体名（按优先级）；None 表示用内置候选表。

  Returns:
      实际选中的字体名；一个都没装时返回 None（仍会关闭负号乱码）。
  """
  candidates = tuple(_CJK_CANDIDATES if prefer is None else prefer)
  available = {f.name for f in font_manager.fontManager.ttflist}
  picked = [name for name in candidates if name in available]
  if picked:
    plt.rcParams["font.sans-serif"] = picked + ["DejaVu Sans"]
  plt.rcParams["axes.unicode_minus"] = False
  return picked[0] if picked else None


def save_figure(fig: Figure, out_path: str | Path | None,
                show: bool = False, dpi: int = 130,
                close: bool = True) -> Path | None:
  """保存 / 显示图像并收尾（本函数不打印任何东西）。

  Args:
      fig: matplotlib Figure。
      out_path: 保存路径；None 表示不保存（父目录会自动创建）。
      show: 是否阻塞显示窗口。
      dpi: 保存分辨率。
      close: 结束前是否 `plt.close(fig)` 释放内存；`show=True` 时不关闭
          （交给用户关窗）。

  Returns:
      实际保存的路径；没有保存时返回 None。
  """
  saved: Path | None = None
  if out_path is not None:
    saved = Path(out_path)
    saved.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(saved, dpi=dpi, bbox_inches="tight")
  if show:
    plt.show()
  elif close:
    plt.close(fig)
  return saved
