import numbers, numpy as np

def is_number(v):
  # 排除 bool（bool 是 int 的子类）
  if isinstance(v, bool):
    return False
  return isinstance(v, numbers.Number)


def random_choose(_list, n):
  """从 _list 中随机选择 n 个元素（不放回；n 超过长度时先循环平铺再选）

  Parameters
  ----------
  _list : list
    要从中选择的元素列表
  n : int
    要选择的元素数量

  Returns
  -------
  list
    随机选择的**元素**列表（不是索引）

  Raises
  ------
  ValueError
    _list 为空且 n > 0 时无法选元素
  """
  if n <= 0:
    return []
  if not _list:
    raise ValueError("random_choose: 待选列表为空，无法选取元素")
  if n > len(_list):
    _add = n // len(_list)
    _list = _list * (_add + 1)
  idx = np.random.choice(len(_list), size=n, replace=False)
  return [_list[int(i)] for i in idx]

if __name__ == "__main__":
  # test_values = [42, 3.14, -7, 0, 1e-5, True, False, "hello", None, [], {}, (1, 2), set()]
  # for value in test_values:
  #   print(f"{value}: {is_number(value)}")
  _l = random_choose([1,2,3], 10)
  print(_l)