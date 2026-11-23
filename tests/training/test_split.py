"""``src/training/split.py`` 的单元测试。"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.training.split import (area_percentiles, sample_area, save_splits,
                                split_dataset, stratified_split)


def _sample(area_side: float, n_digits: int = 1) -> dict:
  """构造一个数字四边形边长约 area_side 的样本（正方形）"""
  half = area_side / 2.0
  quad = [[-half, -half], [half, -half], [half, half], [-half, half]]
  return {"image": f"{area_side}.png", "width": 1000, "height": 1000,
          "objects": [{"digit": i % 10, "quad": quad}
                      for i in range(n_digits)]}


@pytest.fixture()
def samples():
  """面积从 10 到 1000 边长（面积 100~1e6）的对数均匀样本"""
  rng = np.random.default_rng(0)
  sides = np.exp(rng.uniform(np.log(10), np.log(1000), 100))
  return [_sample(float(s), int(rng.integers(1, 6))) for s in sides]


def test_sample_area_average() -> None:
  s = _sample(10.0, n_digits=3)
  assert sample_area(s) == pytest.approx(100.0)
  assert sample_area({"objects": []}) == 0.0


def test_stratified_split_ratio_and_balance(samples) -> None:
  train, val = stratified_split(samples, ratio=0.8, bins=10, seed=0)
  assert len(train) == 80 and len(val) == 20
  assert set(train).isdisjoint(val)
  assert sorted(train + val) == list(range(len(samples)))

  tr = area_percentiles([samples[i] for i in train], qs=(25, 50, 75))
  va = area_percentiles([samples[i] for i in val], qs=(25, 50, 75))
  for key in ("p25", "p50", "p75"):
    # 两边的面积分位数应接近（相差不超过 35%；极端分位数样本太少不用来卡）
    assert abs(np.log(tr[key] / va[key])) < np.log(1.35), (key, tr[key], va[key])


def test_stratified_split_reproducible(samples) -> None:
  a = stratified_split(samples, ratio=0.7, bins=6, seed=123)
  b = stratified_split(samples, ratio=0.7, bins=6, seed=123)
  assert a == b


def test_stratified_split_guards(samples) -> None:
  with pytest.raises(ValueError):
    stratified_split(samples[:1], ratio=0.8)
  with pytest.raises(ValueError):
    stratified_split(samples, ratio=1.0)


def test_split_dataset_and_save(tmp_path, samples) -> None:
  labels = tmp_path / "labels.json"
  labels.write_text(json.dumps({"count": len(samples), "samples": samples},
                               ensure_ascii=False),
                    encoding="utf-8")

  splits = split_dataset(labels, ratio=0.8, bins=8, seed=1)
  assert len(splits["train"]) + len(splits["val"]) == len(samples)

  paths = save_splits(splits, tmp_path)
  train = json.loads(paths["train"].read_text(encoding="utf-8"))
  val = json.loads(paths["val"].read_text(encoding="utf-8"))
  assert train["count"] == len(splits["train"])
  assert val["count"] == len(splits["val"])
  assert train["samples"] and val["samples"]
