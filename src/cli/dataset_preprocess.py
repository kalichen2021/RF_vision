import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs import path as pathcfg
from configs import preprocess_cfg as ppcfg
from src.preprocess import Preprocess


PP = Preprocess()
for i in range(ppcfg.SELECT_NUM):
  raw_img = 1
  