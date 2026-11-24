"""运行标识：一次进程一个时间戳 UUID。"""

from __future__ import annotations

import time


def _new_uuid() -> str:
  """生成形如 `20261124_091503_195` 的时间戳字符串。"""
  return time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"


class LogConfig:
  """日志 / 运行目录相关配置。

  `UUID` 在模块导入时生成一次，同一进程内所有产物目录（日志、调试图、
  快照）共用它，方便把同一次运行的产物归到一起。
  """

  # 本次运行的标识，用作 dataset/log/<模块>/<UUID>/ 之类的目录名
  UUID: str = _new_uuid()

  @staticmethod
  def new_uuid() -> str:
    """生成一个新的时间戳 UUID（需要“每次调用都不同”时用）。"""
    return _new_uuid()