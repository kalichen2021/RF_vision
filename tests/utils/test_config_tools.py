"""`src/utils/config_tools.py` 的单元测试。"""

from __future__ import annotations

import json

from src.utils.config_tools import class_attrs, configs_snapshot


class _Demo:
  """模仿 configs 里的参数类：全是类属性 + 一个方法。"""

  A: int = 1
  B: tuple[float, ...] = (0.1, 0.2)
  C: tuple[tuple[float, float], ...] = ((0.05, 0.1), (0.15, 0.2))
  D: None = None
  _PRIVATE = "x"

  @staticmethod
  def helper() -> int:
    return 1


def test_class_attrs_skips_private_and_methods() -> None:
  attrs = class_attrs(_Demo)
  assert attrs["A"] == 1
  assert attrs["B"] == [0.1, 0.2]            # tuple -> list（JSON 友好）
  assert attrs["C"] == [[0.05, 0.1], [0.15, 0.2]]
  assert attrs["D"] is None
  assert "_PRIVATE" not in attrs
  assert "helper" not in attrs


def test_class_attrs_is_json_serializable() -> None:
  text = json.dumps(class_attrs(_Demo), ensure_ascii=False)
  assert '"A": 1' in text


def test_configs_snapshot_names_each_class() -> None:
  snap = configs_snapshot(training_config=_Demo, preprocess_config=_Demo)
  assert set(snap) == {"training_config", "preprocess_config"}
  assert snap["training_config"]["A"] == 1
