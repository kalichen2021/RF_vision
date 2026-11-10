import cv2
from pathlib import Path
import numpy as np


def cv_show(img, title='image'):
  if img is None:
      raise ValueError(f"cv_show 收到 None, 无法显示: {title}")
  cv2.imshow(title, img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  
  


def read_image(img_path: Path):
  data = np.fromfile(str(img_path), dtype=np.uint8)
  img = cv2.imdecode(data, cv2.IMREAD_COLOR)

  if img is None:
    raise ValueError(f"无法读取图片: {img_path}")

  return img