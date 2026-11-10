import numbers

def is_number(v):
  # 排除 bool（bool 是 int 的子类）
  if isinstance(v, bool):
    return False
  return isinstance(v, numbers.Number)


if __name__ == "__main__":
  test_values = [42, 3.14, -7, 0, 1e-5, True, False, "hello", None, [], {}, (1, 2), set()]
  for value in test_values:
    print(f"{value}: {is_number(value)}")