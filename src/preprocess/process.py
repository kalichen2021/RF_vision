import numpy as np
from numpy.typing import NDArray
import cv2, random, time
from utils.feature_extration import get_flat_quad, segment_quad
from utils.geometry import proj_tran, quad2M
from utils.font_processor import font2img, get_font_path

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs import path as pathcfg
from configs import preprocess_cfg as ppcfg


# for log
_uuid = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
_save_dir = Path(pathcfg.LOG_DIR) / "preprocess" / _uuid
_save_dir.mkdir(parents=True, exist_ok=True)


class Preprocess:
  def __init__(self,distortion,normalize,seg_count,area_range,select_num, quads_select_rate):
    self.config = {
      "distortion": distortion,
      "normalize": normalize,
      "count": seg_count,
      "area_range": area_range
    }
    self.select_num = select_num
    self.quads_select_rate = quads_select_rate
    
    

  def process(self, font_img, env_img):
    quads = get_flat_quad(env_img)
    if quads is None:
      print("未检测到符合条件的四边形区域")
      return None
    segments = [s for q in quads for s in segment_quad(q, **self.config)]
    if not segments:
      print("分割后没有符合 distortion/area 条件的四边形")
      return None

    n = min(len(segments), int(len(segments) * self.quads_select_rate + 1))
    idx = np.random.choice(len(segments), size=n, replace=False)
    quads = [segments[i] for i in idx]
    
    # print(f"q: {quads[-1]}")
    Ms = [quad2M(q, font_img.shape[1], font_img.shape[0]) for q in quads]
    traned_font_imgs = [np.array(proj_tran(font_img, M)) for M in Ms]
    # _img = traned_font_imgs[-1]
    
    

    cv2.imencode('.png', font_img)[1].tofile(f"{_save_dir}/{font_img.shape}_ori_0.png")
    for i, _img in enumerate(traned_font_imgs):
      cv2.imencode('.png', _img)[1].tofile(f"{_save_dir}/{_img.shape}_projed_{i}.png")
    
  
if __name__ == "__main__":
  PP = Preprocess(ppcfg.DISTORTION, ppcfg.NORMALIZE, ppcfg.SEG_COUNT, ppcfg.AREA_RANGE, ppcfg.SAMPLE_SELECT_NUM, ppcfg.QUADS_SELECT_RATE)
  env_img_path = pathcfg.RAW_DIR / "env/0036.jpg"
  font_img =  font2img(get_font_path(0), "1", thickness=1)
  env_img = cv2.imdecode(np.fromfile(str(env_img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
  assert font_img is not None, f"无法读取图片: {font_img.stem}"
  assert env_img is not None, f"无法读取图片: {env_img_path}"
  
  
  PP.process(font_img, env_img)