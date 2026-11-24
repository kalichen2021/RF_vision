from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
from pathlib import Path
import cv2
from .typing_tools import is_number
from .debug_tools import cv_show, hstack_images
from .logging_tools import get_logger

_logger = get_logger(__name__)

FONTS_DIR = Path("C:/Windows/Fonts")
# 覆盖无衬线 / 衬线 / 等宽 / 手写 / 粗体等常见字形
FONT_NAMES = [
  "arial.ttf",
  "calibri.ttf",
  "consola.ttf",
  "times.ttf",
  "verdana.ttf",
  "segoeui.ttf",
  "cambria.ttc",
]

def get_available_fonts() -> list[str]:
  """返回本机实际存在的字体文件名（`FONT_NAMES` 的子集）"""
  return [name for name in FONT_NAMES if (FONTS_DIR / name).exists()]


def get_font_path(font_name: str | int) -> str:
  if is_number(font_name):
    _logger.debug("font_name is a number: %s", font_name)
    font_name = FONT_NAMES[int(font_name) % len(FONT_NAMES)]
  font_path = FONTS_DIR / str(font_name)
  if not Path(font_path).exists():
    raise FileNotFoundError(f"Font file not found: {font_path}")
  return str(font_path)



def font2img(font_path: str, char: str, thickness: int = 1,
             canvas: int = 64, frontcolor: int = 0) -> np.ndarray:
  """把单个字符渲染成固定尺寸的灰度图。

  流程：按较大字号渲染 → 按 ``thickness`` 膨胀（加粗）/ 腐蚀（变细）→
  **裁到墨迹包围盒**（去掉字体自带的留白）→ 等比缩放铺满画布并居中。

  关键在“裁到墨迹包围盒”：字体本身的 bbox 带着左右边距 / 行距，直接用会让
  数字只占画布中间一小块；去掉这些留白后，字形在较长的那一维上真正贴近画布，
  贴到四边形上（``proj_tran``）才会“顶格”。

  Args:
      font_path: 字体文件路径（.ttf / .ttc），可用 :func:`get_font_path` 取。
      char: 要渲染的字符，一般只有一个字符。
      thickness: 笔画粗细调整量，正数加粗、负数变细、0 不处理。
      canvas: 输出画布边长（像素）。
      frontcolor: 0 = 返回白字黑底（默认，下游按“白字黑底”处理）；1 = 黑字白底。

  Returns:
      ``(canvas, canvas)`` 的 uint8 灰度图；空格等空字形返回全 255 图。
  """
  canvas = int(canvas)
  thickness = int(thickness)
  # 先按比画布更大的字号渲染，再缩回来：避免放大造成锯齿/发虚
  font_size = max(1, int(round(canvas * 1.4)))
  padding = max(2, int(canvas // 10))     # 临时画布留白，防止加粗后贴边

  font = ImageFont.truetype(font_path, font_size)
  bbox = font.getbbox(char)  # (l, t, r, b)
  w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
  if w <= 0 or h <= 0:  # 空格等空字形
    return np.full((canvas, canvas), 255, np.uint8)

  # 先画到刚好大小的临时图（黑字白底）
  tmp = Image.new("L", (w + 2 * padding, h + 2 * padding), 255)
  ImageDraw.Draw(tmp).text(
      (padding - bbox[0], padding - bbox[1]), char, font=font, fill=0
  )

  k = abs(int(thickness))
  if k > 0:
    kernel = np.ones((k, k), np.uint8)
    # cv2 只接受 numpy 数组：翻转为白字黑底后再膨胀/腐蚀，最后翻转回黑字白底
    arr = 255 - np.array(tmp)
    if thickness > 0:
      arr = cv2.dilate(arr, kernel, iterations=1)
    else:
      arr = cv2.erode(arr, kernel, iterations=1)
    tmp = Image.fromarray(255 - arr)

  # -- 去掉原字体自带的留白：裁到墨迹包围盒 --
  ink = 255 - np.array(tmp)              # 白字黑底，>0 即为墨迹
  ys, xs = np.nonzero(ink > 0)
  if ys.size == 0:                        # 腐蚀过度把笔画吃没了
    return np.full((canvas, canvas), 255, np.uint8)
  tmp = tmp.crop((int(xs.min()), int(ys.min()),
                  int(xs.max()) + 1, int(ys.max()) + 1))

  # -- 等比缩放铺满画布（保留 1px 边，避免抗锯齿被切）--
  target = max(1, canvas - 2)
  scale = min(target / tmp.width, target / tmp.height)
  new_size = (max(1, int(round(tmp.width * scale))),
              max(1, int(round(tmp.height * scale))))
  tmp = tmp.resize(new_size, Image.LANCZOS)

  # 居中贴到固定画布
  out = Image.new("L", (canvas, canvas), 255)
  out.paste(tmp, ((canvas - new_size[0]) // 2, (canvas - new_size[1]) // 2))

  out = np.array(out, np.uint8)
  if frontcolor == 0:
    out = 255 - out
  return np.array(out)


if __name__ == "__main__":
  # font_img = font2img(get_font_path(0), "H", 5)
  # cv_show(font_img, "Font Image")
  font_img = []
  for i in range(len(FONT_NAMES)):
    font_img .append(font2img(get_font_path(i), str(i) + FONT_NAMES[i][:3], 4))
  cv_show(hstack_images(font_img))