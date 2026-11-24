"""项目路径配置：所有目录 / 文件路径集中在这里。

用法::

    from configs.path import PathConfig

    PathConfig.SELECT_MODEL        # 推理用的权重文件（不是目录）
    PathConfig.DATASET_LABEL_JSON  # 训练用的 labels.json
"""

from __future__ import annotations

from pathlib import Path

from configs.infer_cfg import InferConfig
from configs.training_cfg import TrainingConfig
from src.utils.file_operation import get_latest_dir, get_latest_file


class PathConfig:
  """项目内所有路径（均为绝对路径，导入时计算一次）。"""

  ROOT: Path = Path(__file__).resolve().parent.parent

  DATASET_DIR: Path = ROOT / "dataset"
  TEST_DIR: Path = ROOT / "tests"
  MODELS_DIR: Path = ROOT / "models"
  SRC_DIR: Path = ROOT / "src"

  RAW_DIR: Path = DATASET_DIR / "raw"
  # 环境图目录（合成样本的背景图池）
  ENV_DIR: Path = RAW_DIR / "env"
  PROCESSED_DIR: Path = DATASET_DIR / "processed"
  LOG_DIR: Path = DATASET_DIR / "log"

  # 训练用数据集目录：SELECT_IMGSET_ID 为 "0" 时取 processed 下最新的完整目录
  SELECT_IMGSET: Path = (
      get_latest_dir(PROCESSED_DIR, required=("labels.json",))
      if TrainingConfig.SELECT_IMGSET_ID == "0"
      else PROCESSED_DIR / TrainingConfig.SELECT_IMGSET_ID)

  # 推理用权重**文件**：SELECT_MODEL_ID 为 "0" 时取 models 下最新的 ckpt_*.pt
  SELECT_MODEL: Path = (
      get_latest_file(MODELS_DIR, pattern="ckpt_*.pt")
      if InferConfig.SELECT_MODEL_ID == "0"
      else MODELS_DIR / InferConfig.SELECT_MODEL_ID)

  DATASET_LABEL_JSON: Path = SELECT_IMGSET / "labels.json"


if __name__ == "__main__":
  for _name in ("ROOT", "DATASET_DIR", "TEST_DIR", "SRC_DIR", "RAW_DIR",
                "ENV_DIR", "PROCESSED_DIR", "MODELS_DIR", "LOG_DIR",
                "SELECT_IMGSET", "SELECT_MODEL", "DATASET_LABEL_JSON"):
    print(f"{_name}: {getattr(PathConfig, _name)}")
