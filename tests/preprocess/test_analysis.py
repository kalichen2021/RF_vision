"""``src/preprocess/analysis.py`` 的单元测试。"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")  # 测试环境无交互后端

import pytest

from src.preprocess.analysis import (Analysis, load_dataset, quad_area,
                                     quad_aspect, quad_size)

SQUARE = [[0, 0], [10, 0], [10, 10], [0, 10]]
RECT = [[0, 0], [30, 0], [30, 10], [0, 10]]


def _sample(name: str, source: str, objects: list[dict]) -> dict:
  return {"image": name, "width": 100, "height": 50,
          "source": source, "objects": objects}


def _obj(digit: int, quad, board=None, center=(10, 20)) -> dict:
  return {"digit": digit, "quad": quad, "board_quad": board or quad,
          "center": list(center), "depth": 0.5, "sigma": 1.0}


@pytest.fixture()
def dataset_dir(tmp_path):
  records = [
      _sample("a.png", "src1.jpg",
              [_obj(1, SQUARE), _obj(2, RECT, center=(50, 25))]),
      _sample("b.png", "src2.jpg", [_obj(1, SQUARE)]),
  ]
  (tmp_path / "labels.json").write_text(
      json.dumps({"count": len(records), "samples": records},
                 ensure_ascii=False),
      encoding="utf-8")
  return tmp_path


def test_quad_geometry() -> None:
  assert quad_area(SQUARE) == pytest.approx(100.0)
  assert quad_area(RECT) == pytest.approx(300.0)
  assert quad_size(SQUARE) == pytest.approx((10.0, 10.0))
  assert quad_aspect(SQUARE) == pytest.approx(1.0)
  assert quad_aspect(RECT) == pytest.approx(3.0)


def test_load_dataset_merges_per_sample_json(tmp_path) -> None:
  for i, (name, digit) in enumerate([("a", 1), ("b", 2)]):
    (tmp_path / f"{name}.json").write_text(
        json.dumps(_sample(f"{name}.png", "s.jpg", [_obj(digit, SQUARE)])),
        encoding="utf-8")
  samples = load_dataset(tmp_path)
  assert [s["image"] for s in samples] == ["a.png", "b.png"]


def test_analysis_summary(dataset_dir) -> None:
  an = Analysis(dataset_dir)
  assert an.n_samples == 2
  assert an.n_digits == 3
  expected_counts = {d: 0 for d in range(10)}
  expected_counts[1], expected_counts[2] = 2, 1
  assert an.digit_counts() == expected_counts
  assert an.digits_per_sample == [2, 1]
  assert an.areas == pytest.approx([100.0, 300.0, 100.0])
  assert len(an.area_ratios) == 3  # 面积按 100×50 归一化
  assert an.area_ratios[0] == pytest.approx(100.0 / 5000)

  src = an.source_stats()
  assert src["src1.jpg"] == {"samples": 1, "digits": 2}
  assert src["src2.jpg"] == {"samples": 1, "digits": 1}

  stats = an.summary()
  assert stats["quad_area"]["p50"] == pytest.approx(100.0)
  assert stats["n_sources"] == 2


def test_analysis_plot_overview(dataset_dir, tmp_path) -> None:
  an = Analysis(dataset_dir)
  out = tmp_path / "overview.png"
  fig = an.run(out_path=out, show=False)
  assert out.exists() and out.stat().st_size > 0
  assert fig.axes  # 有子图
