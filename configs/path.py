from pathlib import Path
from .infer_cfg import SELECT_MODEL_ID
from .training_cfg import SELECT_IMGSET_ID
from src.utils.file_operation import get_latest

ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = ROOT/ "dataset"
TEST_DIR = ROOT/ "tests"
MODELS_DIR = ROOT/ "models"
SRC_DIR = ROOT/ "src"

RAW_DIR = DATASET_DIR / "raw"
PROCESSED_DIR = DATASET_DIR / "processed"
LOG_DIR = DATASET_DIR / "log"


SELECT_IMGSET = PROCESSED_DIR / (
    get_latest(PROCESSED_DIR, required=("labels.json",))
    if SELECT_IMGSET_ID == '0' else SELECT_IMGSET_ID)
SELECT_MODEL = MODELS_DIR / (get_latest(MODELS_DIR)
                             if SELECT_MODEL_ID == '0' else SELECT_MODEL_ID)
DATASET_LABEL_JSON = SELECT_IMGSET / "labels.json"



if __name__ == "__main__":
  print(f"ROOT: {ROOT}")
  print(f"DATASET_DIR: {DATASET_DIR}")
  print(f"TEST_DIR: {TEST_DIR}")
  print(f"PROCESSED_DIR: {PROCESSED_DIR}")
  print(f"RAW_DIR: {RAW_DIR}")
  print(f"LOG_DIR: {LOG_DIR}")
  print(f"SRC_DIR: {SRC_DIR}")