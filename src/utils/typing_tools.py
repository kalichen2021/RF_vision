import numbers, numpy as np

def is_number(v):
  # 排除 bool（bool 是 int 的子类）
  if isinstance(v, bool):
    return False
  return isinstance(v, numbers.Number)


def random_choose(_list, n):
  """从 _list中随机选择 n 个元素的索引

  Parameters
  ----------
  _list : list
    要从中选择的元素列表
  n : int
    要选择的元素数量

  Returns
  -------
  list[int]
    随机选择的列表
  """
  if n > len(_list):
    _add = n // len(_list)
    _list = _list * (_add + 1)
  idx = np.random.choice(len(_list), size=n, replace=False).tolist()
  return [_list[i] for i in range(n)]

if __name__ == "__main__":
  # test_values = [42, 3.14, -7, 0, 1e-5, True, False, "hello", None, [], {}, (1, 2), set()]
  # for value in test_values:
  #   print(f"{value}: {is_number(value)}")
  _l = random_choose([1,2,3], 10)
  print(_l)