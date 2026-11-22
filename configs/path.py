from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = ROOT/ "dataset"
TEST_DIR = ROOT/ "tests"
SRC_DIR = ROOT/ "src"

RAW_DIR = DATASET_DIR / "raw"
PROCESSED_DIR = DATASET_DIR / "processed"
LOG_DIR = DATASET_DIR / "log"

SELECT_IMGSET = PROCESSED_DIR / "20261121_091927_521"
# 批量生成的合并标签文件名（cli/dataset_preprocess 通过 save_dataset 写成 labels.json）
DATASET_LABEL_JSON = SELECT_IMGSET / "labels.json"



if __name__ == "__main__":
  print(f"ROOT: {ROOT}")
  print(f"DATASET_DIR: {DATASET_DIR}")
  print(f"TEST_DIR: {TEST_DIR}")
  print(f"PROCESSED_DIR: {PROCESSED_DIR}")
  print(f"RAW_DIR: {RAW_DIR}")
  print(f"LOG_DIR: {LOG_DIR}")
  print(f"SRC_DIR: {SRC_DIR}")