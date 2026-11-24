"""训练日志可视化的单元测试（不需要真的训练过）。"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")            # 测试环境没有显示器

from src.training.plot import (history_summary, load_history, plot_history,
                               stage_series)


def _fake_history(val: bool = True) -> dict:
  """构造一份两轮训练记录（val=False 时模拟没跑验证）。"""
  return {
      "meta": {"ckpt": "models/ckpt_x.pt", "epochs": 2, "lr_init": 1e-3,
               "optimizer": "Adam", "scheduler": "ReduceLROnPlateau",
               "amp": True, "trainable_params": 100, "total_params": 1000,
               "val_interval": 1, "early_stopping_patience": 4},
      "epochs": [
          {"epoch": 1, "train": {"loss": 7.7, "heatmap": 7.0, "offset": 0.43,
                                 "cls": 0.31},
           "val": {"loss": 0.72, "heatmap": 0.51, "offset": 0.16,
                   "cls": 0.046} if val else None,
           "lr": 0.001, "epoch_time": 126.0, "data_wait": 31.6,
           "gpu_ratio": 0.75, "is_best": True, "bad_epochs": 0},
          {"epoch": 2, "train": {"loss": 0.74, "heatmap": 0.48, "offset": 0.22,
                                 "cls": 0.041},
           "val": {"loss": 0.58, "heatmap": 0.43, "offset": 0.13,
                   "cls": 0.022} if val else None,
           "lr": 0.0005, "epoch_time": 57.5, "data_wait": 2.8,
           "gpu_ratio": 0.95, "is_best": True, "bad_epochs": 0},
      ],
  }


def test_load_history_roundtrip(tmp_path: Path) -> None:
  path = tmp_path / "train_u1_head.json"
  path.write_text(json.dumps(_fake_history()), encoding="utf-8")
  hist = load_history(path)
  assert len(hist["epochs"]) == 2
  assert hist["meta"]["scheduler"] == "ReduceLROnPlateau"


def test_load_history_rejects_bad_structure(tmp_path: Path) -> None:
  path = tmp_path / "bad.json"
  path.write_text('{"meta": {}}', encoding="utf-8")
  with pytest.raises(ValueError):
    load_history(path)


def test_stage_series_fills_nan_when_no_val() -> None:
  s = stage_series(_fake_history(val=False))
  assert s["train_loss"].tolist() == [7.7, 0.74]
  assert s["val_loss"].size == 2
  assert not any(v == v for v in s["val_loss"])      # 全是 nan


def test_history_summary_best_and_ratio() -> None:
  rows = history_summary([_fake_history()], ["head"])
  assert rows[0]["stage"] == "head"
  assert rows[0]["best_epoch"] == 2
  assert rows[0]["best_val"] == pytest.approx(0.58)
  assert rows[0]["epochs_run"] == 2
  # 等数据 34.4s / 总 183.5s
  assert rows[0]["data_wait_ratio"] == pytest.approx(34.4 / 183.5, rel=1e-6)
  assert rows[0]["early_stopped"] is False


def test_plot_history_returns_figure_with_8_panels() -> None:
  fig = plot_history([_fake_history(), _fake_history()], ["head", "finetune"])
  assert len(fig.axes) == 8
  labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
  assert "head 训练" in labels and "finetune 验证" in labels
