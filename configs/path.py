from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = ROOT/ "dataset"
RAW_DIR = DATASET_DIR / "raw"
PROCESSED_DIR = DATASET_DIR / "processed"
LOG_DIR = DATASET_DIR / "log"


SRC_DIR = ROOT/ "src"


if __name__ == "__main__":
  print(f"ROOT: {ROOT}")
  print(f"DATASET_DIR: {DATASET_DIR}")
  print(f"RAW_DIR: {RAW_DIR}")
  print(f"LOG_DIR: {LOG_DIR}")
  print(f"SRC_DIR: {SRC_DIR}")