"""类型判断 / 随机取样等小工具。"""

from __future__ import annotations

import numbers
from typing import Any

import numpy as np


def is_number(v: Any) -> bool:
  """判断是否为数字（不含 bool）。

  ``bool`` 是 ``int`` 的子类，但语义上不是数字，这里显式排除。

  Args:
      v: 任意对象。

  Returns:
      是数值类型（int / float / numpy 数值等）返回 True。
  """
  if isinstance(v, bool):
    return False
  return isinstance(v, numbers.Number)


def random_choose(items: list[Any], n: int) -> list[Any]:
  """从 ``items`` 中不放回地随机抽 ``n`` 个元素。

  ``n`` 超过列表长度时先把列表循环平铺（允许重复抽取）。

  Args:
      items: 待选元素列表（不会被修改）。
      n: 要抽取的个数；<= 0 时返回空列表。

  Returns:
      抽中的**元素**列表（不是下标）。

  Raises:
      ValueError: ``items`` 为空且 ``n > 0``。
  """
  if n <= 0:
    return []
  if not items:
    raise ValueError("random_choose: 待选列表为空，无法选取元素")
  pool = list(items)
  if n > len(pool):
    pool = pool * (n // len(pool) + 1)
  idx = np.random.choice(len(pool), size=n, replace=False)
  return [pool[int(i)] for i in idx]


if __name__ == "__main__":
  print(random_choose([1, 2, 3], 10))