"""配置类的序列化小工具。

项目的所有参数都以「类 + 类属性」的形式写在 `configs/*.py` 里
（`TrainingConfig.BATCH_SIZE` 这类）。训练结束时需要把当时的参数
原样存档（`models/ckpt_<uuid>.json`），这个模块负责把类摊平成 JSON 可写的 dict。

用法::

    from configs.training_cfg import TrainingConfig
    from src.utils.config_tools import class_attrs

    class_attrs(TrainingConfig)     # {"BATCH_SIZE": 48, "LR_HEAD": 0.001, ...}
"""

from __future__ import annotations

import types
from typing import Any


def _jsonable(value: Any) -> Any:
  """把类属性值转成 JSON 能写的形态（tuple / set → list，其余原样）。"""
  if isinstance(value, (str, int, float, bool)) or value is None:
    return value
  if isinstance(value, (tuple, list)):
    return [_jsonable(v) for v in value]
  if isinstance(value, (set, frozenset)):
    return sorted(_jsonable(v) for v in value)
  if isinstance(value, dict):
    return {str(k): _jsonable(v) for k, v in value.items()}
  return repr(value)          # 其它对象（如 Path）统一存字符串形式


def class_attrs(cls: type) -> dict[str, Any]:
  """取出一个配置类的全部公开类属性（可直接 json.dumps）。

  跳过：下划线开头的私有名、方法 / 函数 / 描述符等可调用对象、类本身。

  Args:
      cls: 配置类（如 ``TrainingConfig``）。

  Returns:
      属性名 → JSON 友好值的字典，保持类里的定义顺序。
  """
  out: dict[str, Any] = {}
  for name, value in vars(cls).items():
    if name.startswith("_"):
      continue
    if isinstance(value, (types.FunctionType, types.MethodType,
                          classmethod, staticmethod, property, type)):
      continue
    if callable(value):
      continue
    out[name] = _jsonable(value)
  return out


def configs_snapshot(**classes: type) -> dict[str, dict[str, Any]]:
  """一次摊平多个配置类，用于写「本次运行用了什么参数」。

  Args:
      **classes: 名字 → 配置类，例如 ``training_config=TrainingConfig``。

  Returns:
      ``{名字: {属性名: 值}}``。
  """
  return {name: class_attrs(cls) for name, cls in classes.items()}
