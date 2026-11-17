"""pytest 会话级「原地刷新」进度条。

背景：pytest 默认（以及 pytest-sugar）的进度显示都是「每个用例往终端追加一行」，
没有真正的动画。这里用回车 ``\\r`` + ``ESC[K`` 清行，实现单行进度条原地重绘。

行为约定：
- 仅在真实终端启用；VS Code 测试面板等捕获输出的场景自动退回原生输出。
- 调试时可用环境变量 ``PYTEST_LIVE_BAR=1`` 强制启用。
- ``-v``/``-vv`` 时不接管，保留 pytest 逐用例输出。
"""

from __future__ import annotations

import os
import sys
import time

import pytest

_BAR_WIDTH = 30
# 仅在真实终端接管；否则 pytest 的捕获输出里塞满控制符反而更乱
_ENABLED = (
    sys.__stdout__ is not None
    and (sys.__stdout__.isatty() or os.environ.get("PYTEST_LIVE_BAR") == "1")
)

_GREEN = "\x1b[32m"
_YELLOW = "\x1b[33m"
_RED = "\x1b[31m"
_DIM = "\x1b[90m"
_RESET = "\x1b[0m"


def _pick(char: str, fallback: str) -> str:
    """终端编码（如 Windows 下的 GBK）无法表示该字符时退回 ASCII。"""
    encoding = getattr(sys.__stdout__, "encoding", None) or "utf-8"
    try:
        char.encode(encoding)
    except (LookupError, UnicodeEncodeError):
        return fallback
    return char


_FULL = _pick("█", "#")
_EMPTY = _pick("░", "-")
_OK = _pick("✓", "ok")
_FAIL = _pick("✗", "x")


class _LiveBar:
    """单行进度条：每次 tick 覆盖上一帧，而不是另起一行。"""

    def __init__(self, total: int) -> None:
        self.total = max(total, 1)
        self.done = 0
        self.passed = 0
        self.skipped = 0
        self.failed = 0
        self.started = time.perf_counter()

    def tick(self, outcome: str) -> None:
        self.done += 1
        if outcome == "passed":
            self.passed += 1
        elif outcome == "skipped":
            self.skipped += 1
        else:
            self.failed += 1
        self._draw()

    def _draw(self) -> None:
        ratio = min(self.done / self.total, 1.0)
        filled = int(_BAR_WIDTH * ratio)
        color = _RED if self.failed else _GREEN
        bar = (
            f"{color}{_FULL * filled}{_RESET}"
            f"{_DIM}{_EMPTY * (_BAR_WIDTH - filled)}{_RESET}"
        )
        elapsed = time.perf_counter() - self.started
        line = (
            f"\r[{bar}] {ratio * 100:5.1f}%  {self.done}/{self.total}  "
            f"{_GREEN}{_OK}{self.passed}{_RESET} "
            f"{_YELLOW}s{self.skipped}{_RESET} "
            f"{_RED}{_FAIL}{self.failed}{_RESET}  "
            f"{_DIM}{elapsed:4.1f}s{_RESET}"
        )
        # \r 回到行首 + ESC[K 清除上一次更长的残留，形成原地刷新
        sys.__stdout__.write(line + "\x1b[K")
        sys.__stdout__.flush()


_bar: _LiveBar | None = None


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(
    session: pytest.Session,
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    global _bar
    if not _ENABLED or config.option.collectonly or config.option.verbose > 0:
        return
    _bar = _LiveBar(len(items))
    cfg = config.pluginmanager.getplugin("terminalreporter")
    if cfg is not None:
        # 关掉原生百分比信息，避免与进度条抢同一行
        cfg._show_progress_info = False


@pytest.hookimpl(trylast=True)
def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    # trylast：等 TerminalReporter 写完本用例记号后再覆盖整行
    if _bar is None:
        return
    if report.when == "call":
        _bar.tick(report.outcome)
    elif report.when == "setup" and report.outcome in {"skipped", "failed"}:
        _bar.tick(report.outcome)
