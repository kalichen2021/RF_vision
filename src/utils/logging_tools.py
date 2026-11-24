"""日志工具：模块里只取 logger，控制台输出统一由 CLI 配置。

设计约定：

- `src/` 下的模块（`cli` 之外）**不允许 print**，只能 `get_logger(__name__)`
  后写 `logger.info/debug/warning`；
- 日志格式、级别、输出流全部在 CLI 入口调用 :func:`setup_logging` 决定；
- 没人调用 `setup_logging` 时走 `logging` 默认策略（warning 及以上才输出，
  相当于静默），这样库代码被单测 / 其它脚本导入时不会污染输出。
"""

from __future__ import annotations

import logging
import sys

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def get_logger(name: str) -> logging.Logger:
  """取一个模块级 logger。

  Args:
      name: 一般直接传 `__name__`。

  Returns:
      对应名字的 `logging.Logger`；没有 `setup_logging` 时不会输出 info。
  """
  return logging.getLogger(name)


def setup_logging(level: int | str = logging.INFO,
                  fmt: str = _DEFAULT_FORMAT) -> None:
  """配置根 logger（只能在 CLI 入口调用一次）。

  Args:
      level: 日志级别，可传 `logging.INFO` 或 `"INFO"` 这类字符串。
      fmt: 输出格式。
  """
  if isinstance(level, str):
    level = getattr(logging, level.upper(), logging.INFO)
  handler = logging.StreamHandler(stream=sys.stdout)
  handler.setFormatter(logging.Formatter(fmt, datefmt=_DATE_FORMAT))

  root = logging.getLogger()
  for old in list(root.handlers):      # 重复调用时避免叠加多份 handler
    root.removeHandler(old)
  root.addHandler(handler)
  root.setLevel(level)
