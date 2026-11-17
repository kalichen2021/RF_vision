import cv2, random, time,numpy as np
from numpy.typing import NDArray
from utils.feature_extration import get_flat_quad, segment_quad, _get_mask
from utils.geometry import proj_tran, quad2M
from utils.font_processor import font2img, get_font_path
from utils.typing_tools import is_number, random_choose
from utils.debug_tools import cv_show, hstack_images


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs import path as pathcfg
from configs import preprocess_cfg as ppcfg
from configs.log import UUID as _uuid


# for log
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
    
    

  def process(self, _font_imgs, env_img, show_cnt = False):
    quads = get_flat_quad(env_img)
    if quads is None:
      print("未检测到符合条件的四边形区域")
      return None
    
    segments = [s for q in quads for s in segment_quad(q, **self.config)]
    if not segments:
      print("分割后没有符合 distortion/area 条件的四边形")
      return None

    # 随机选择env_img中的若干个四边形进行投影变换
    boxed_text_count = min(len(segments), int(len(segments) * self.quads_select_rate + 1))
    # idx = np.random.choice(len(segments), size=boxed_text_count, replace=False)
    quads = random_choose(segments, boxed_text_count)
    font_imgs = random_choose(_font_imgs, boxed_text_count)
    
    traned_font_imgs = [
      np.array(proj_tran(
        _img,
        quad2M(quads[idx], *_img.shape)
      )) for idx, _img in enumerate(font_imgs)
    ]
    
    # DEBUG: 显示透视矩形（quads 是 env_img 坐标系下的 float64 点，drawContours 只接受 int32）
    if show_cnt:
      vis_env = env_img.copy()
      for q in quads:
        cv2.drawContours(vis_env, [np.round(q).astype(np.int32)], -1, (0, 0, 255), 2)
      cv2.imencode('.png', vis_env)[1].tofile(f"{_save_dir}/{vis_env.shape}_env_quads.png")
    
    print(f"quads[0].offset: {quads[0].offset}")
    for i, _ in enumerate(traned_font_imgs):
      # proj_tran 的输出画布 = 目标 quad 的外接矩形, 且已把坐标平移 (0,0),
      # 所以要先减去外接矩形左上角, 得到画布内的局部坐标才能画上框
      q = np.asarray(quads[i], dtype=np.float64).reshape(4, 2)
      q_local = np.round(q - q.min(axis=0)).astype(np.int32)
      # 灰度图上 drawContours 只取 color[0]=0(B) 会画成黑线, 黑底上不可见, 先转 BGR
      if traned_font_imgs[i].ndim == 2:
        traned_font_imgs[i] = cv2.cvtColor(traned_font_imgs[i], cv2.COLOR_GRAY2BGR)
      cv2.drawContours(traned_font_imgs[i], [q_local], -1, (0, 0, 255), 2)
      _img = hstack_images([font_imgs[i], traned_font_imgs[i]], pad=5)
      cv2.imencode('.png', _img)[1].tofile(f"{_save_dir}/{_img.shape}_projed_{i}.png")
      
      
      # test get mask
      font_mask = _get_mask(traned_font_imgs[i])
      cv2.imencode('.png', _img)[1].tofile(f"{_save_dir}/{_img.shape}_mask_{i}.png")
      break # Debug
    print(f"quads[0].offset: {quads[0].offset}")

if __name__ == "__main__":
  PP = Preprocess(ppcfg.DISTORTION, ppcfg.NORMALIZE, ppcfg.SEG_COUNT, ppcfg.AREA_RANGE, ppcfg.SAMPLE_SELECT_NUM, ppcfg.QUADS_SELECT_RATE)
  env_img_path = pathcfg.RAW_DIR / "env/0072.jpg"
  font_imgs =  [font2img(get_font_path(0), str(i), thickness=1) for i in range(10)]
  env_img = cv2.imdecode(np.fromfile(str(env_img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
  
  
  PP.process(font_imgs, env_img, True)