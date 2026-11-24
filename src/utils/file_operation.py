"""文件 / 目录操作的通用工具。"""

from __future__ import annotations

from pathlib import Path


def ensure_dir(path: Path | str) -> Path:
  """确保目录存在（父目录递归创建）并返回它。

  Args:
      path: 目标目录。

  Returns:
      同一个目录对象。
  """
  p = Path(path)
  p.mkdir(parents=True, exist_ok=True)
  return p


def get_latest_dir(base: Path | str, required: tuple[str, ...] = ()) -> Path:
  """取 `base` 下「名字最大」的子目录（目录名本身就是时间戳）。

  数据集目录形如 `20261124_091503_195`，按名字排序即按生成时间排序；
  不用 `st_ctime`：拷贝 / 解压后创建时间会变，且秒级精度容易打平。

  Args:
      base: 父目录。
      required: 需要同时存在的文件名（如 `("labels.json",)`），用于跳过
          生成中断、只有半成品的目录；为空表示不做过滤。

  Returns:
      最新的子目录；`base` 不存在或没有子目录时返回 `base` 本身。
  """
  root = Path(base)
  if not root.is_dir():
    return root
  dirs = [p for p in root.iterdir() if p.is_dir()]
  if required:
    complete = [p for p in dirs
                if all((p / name).exists() for name in required)]
    if complete:
      dirs = complete
  if not dirs:
    return root
  return max(dirs, key=lambda p: p.name)


def get_latest_file(base: Path | str, pattern: str = "*",
                    required: tuple[str, ...] = ()) -> Path:
  """取 `base` 下匹配 `pattern` 的最新**文件**。

  项目里的产物文件名都带时间戳（`ckpt_20261124_091503_195.pt`），
  先按文件名排序、同名时再比修改时间，保证拿到的是最新那个。

  旧版 `get_latest` 只扫子目录，传 `models/` 时会返回目录本身，
  后面 `torch.load(目录)` 直接报 IsADirectoryError；这里改成扫文件。

  Args:
      base: 父目录。
      pattern: 文件名通配符，如 `"ckpt_*.pt"`。
      required: 需要同时存在的额外文件名（一般不用）。

  Returns:
      最新的文件路径。

  Raises:
      FileNotFoundError: 目录不存在或没有匹配文件（宁可早报错，也不带病继续）。
  """
  root = Path(base)
  candidates = ([p for p in root.glob(pattern) if p.is_file()]
                if root.is_dir() else [])
  if required:
    candidates = [p for p in candidates
                  if all((root / name).exists() for name in required)]
  if not candidates:
    raise FileNotFoundError(f"{root} 下找不到匹配 {pattern!r} 的文件")
  return max(candidates, key=lambda p: (p.name, p.stat().st_mtime))
