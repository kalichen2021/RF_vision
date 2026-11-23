from pathlib import Path


def get_latest(dir: Path, required: tuple[str, ...] = ()) -> Path:
  """取最新一次生成的目录（目录名即时间戳 UUID，取最大者）

  优先在「包含 ``required`` 里所有文件」的目录中挑选：数据集要求有
  ``labels.json``，否则生成中断的目录（只有图片、没写标签）会被误选。
  没有满足条件的目录时退回全部目录。

  ``dir`` 不存在或为空时返回 ``dir`` 本身（交由调用方按缺省路径处理）。
  """
  if not dir.is_dir():
    return dir
  dirs = [p for p in dir.iterdir() if p.is_dir()]
  if not dirs:
    return dir
  if required:
    complete = [p for p in dirs
                if all((p / name).exists() for name in required)]
    if complete:
      dirs = complete
  # 目录名就是生成时间（YYYYMMDD_HHMMSS_mmm），按名字排序即按时间排序；
  # 不用 st_ctime：拷贝 / 解压后创建时间会变，且秒级精度在同一批里会打平
  return max(dirs, key=lambda p: p.name)
