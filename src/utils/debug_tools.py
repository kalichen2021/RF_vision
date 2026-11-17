import cv2,numpy as np
from pathlib import Path
from functools import wraps 


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

def hstack_images(images, height=None, pad=10, bg=(255, 255, 255)):
  """横向拼接：统一高度后左右排列"""
  imgs = [cv2.cvtColor(i, cv2.COLOR_GRAY2BGR) if i.ndim == 2 else i for i in images]
  if height is None:
    height = min(im.shape[0] for im in imgs)
  resized = []
  for im in imgs:
    h, w = im.shape[:2]
    nw = int(round(w * height / h))
    resized.append(cv2.resize(im, (nw, height), interpolation=cv2.INTER_AREA))
  total_w = sum(im.shape[1] for im in resized) + pad * (len(resized) - 1)
  canvas = np.full((height, total_w, 3), bg, dtype=np.uint8)
  x = 0
  for im in resized:
    canvas[:, x:x + im.shape[1]] = im
    x += im.shape[1] + pad
  return canvas


def vstack_images(images, width=None, pad=10, bg=(255, 255, 255)):
  """纵向拼接：统一宽度后上下排列"""
  imgs = [cv2.cvtColor(i, cv2.COLOR_GRAY2BGR) if i.ndim == 2 else i for i in images]
  if width is None:
    width = min(im.shape[1] for im in imgs)
  resized = []
  for im in imgs:
    h, w = im.shape[:2]
    nh = int(round(h * width / w))
    resized.append(cv2.resize(im, (width, nh), interpolation=cv2.INTER_AREA))
  total_h = sum(im.shape[0] for im in resized) + pad * (len(resized) - 1)
  canvas = np.full((total_h, width, 3), bg, dtype=np.uint8)
  y = 0
  for im in resized:
    canvas[y:y + im.shape[0], :] = im
    y += im.shape[0] + pad
  return canvas


def loop_stop(times):
  def _decorator(func):
    @wraps(func)
    def _wrapper(*arg, **kwargs):
      for i in range(times):
        result = func(*arg, **kwargs)
      return result
    return _wrapper
  return _decorator