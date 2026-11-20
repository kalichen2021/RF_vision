"""``src/utils/typing_tools.py`` 的单元测试。"""

from __future__ import annotations

import numpy as np
import pytest

from utils.typing_tools import is_number, random_choose


class TestIsNumber:
  """``is_number`` 的类型判定"""

  def test_numbers_are_numbers(self) -> None:
    for v in (42, 3.14, -7, np.float32(1.5), np.int64(3)):
      assert is_number(v) is True

  def test_bool_and_non_numbers_are_rejected(self) -> None:
    # bool 是 int 的子类，必须被排除
    for v in (True, False, "1", None, [1], {}, (1, 2)):
      assert is_number(v) is False


class TestRandomChoose:
  """``random_choose`` 的抽样行为"""

  def test_returns_requested_count(self) -> None:
    assert len(random_choose(list(range(10)), 4)) == 4

  def test_elements_come_from_list_without_replacement(self) -> None:
    src = list(range(10))
    picked = random_choose(src, 5)
    assert set(picked) <= set(src)
    assert len(set(picked)) == 5

  def test_is_actually_random(self) -> None:
    # 回归：曾经返回 [_list[i] for i in range(n)]，即永远取前 n 个
    src = list(range(20))
    np.random.seed(0)
    picks = [tuple(random_choose(src, 5)) for _ in range(20)]
    assert len(set(picks)) > 1
    assert any(p != tuple(src[:5]) for p in picks)

  def test_n_larger_than_list_wraps_and_repeats(self) -> None:
    picked = random_choose([1, 2, 3], 7)
    assert len(picked) == 7
    assert set(picked) == {1, 2, 3}

  def test_non_positive_n_returns_empty(self) -> None:
    assert random_choose([1, 2, 3], 0) == []
    assert random_choose([1, 2, 3], -2) == []

  def test_empty_list_raises(self) -> None:
    with pytest.raises(ValueError):
      random_choose([], 2)
