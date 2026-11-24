from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
from pathlib import Path
import cv2
from .typing_tools import is_number
from .debug_tools import cv_show
from .logging_tools import get_logger

_logger = get_logger(__name__)

FONTS_DIR = Path("C:/Windows/Fonts")
# 覆盖无衬线 / 衬线 / 等宽 / 手写 / 粗体等常见字形
FONT_NAMES = [
  "arial.ttf",
  "arialbd.ttf",
  "calibri.ttf",
  "calibrib.ttf",
  "consola.ttf",
  "consolab.ttf",
  "times.ttf",
  "timesbd.ttf",
  "verdana.ttf",
  "verdanab.ttf",
  "tahoma.ttf",
  "tahomab.ttf",
  "segoeui.ttf",
  "segoeuib.ttf",
  "comic.ttf",
  "comicbd.ttf",
  "georgia.ttf",
  "georgiab.ttf",
  "cour.ttf",
  "courbd.ttf",
  "impact.ttf",
  "cambria.ttc",
  "calibrii.ttf",
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

  先用指定字体把字符画到紧凑的临时画布上（黑字白底），再按 thickness
  做膨胀（加粗）/ 腐蚀（变细）；字符超出画布时等比缩小，最后居中贴到
  ``canvas × canvas`` 的白底画布上。

  Args:
      font_path: 字体文件路径（.ttf / .ttc），可用 :func:`get_font_path` 取。
      char: 要渲染的字符，一般只有一个字符。
      thickness: 笔画粗细调整量，正数加粗、负数变细、0 不处理。
      canvas: 输出画布边长（像素），同时决定字号。
      frontcolor: 前景色参数，0 = 白色前景，1 = 黑色前景。

  Returns:
      ``(canvas, canvas)`` 的 uint8 灰度图，白底 (255)、黑字 (0)；
      空格等空字形返回全白图。
  """
  canvas = int(canvas)
  thickness = int(thickness)
  font_size = max(1, int(canvas // 1.5 - thickness * 0.2))
  padding = max(0, int(canvas // 10 - thickness*0.1))  # 画布边缘留白，避免膨胀/腐蚀后贴边

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

  # 超画布时等比缩小
  scale = min((canvas - 2) / tmp.width, (canvas - 2) / tmp.height, 1.0)
  if scale < 1.0:
      tmp = tmp.resize(
        (max(1, int(tmp.width * scale)), max(1, int(tmp.height * scale))),
        Image.BILINEAR,
      )

  # 居中贴到固定画布
  out = Image.new("L", (canvas, canvas), 255)
  out.paste(tmp, ((canvas - tmp.width) // 2, (canvas - tmp.height) // 2))

  out = np.array(out, np.uint8)
  if frontcolor == 0:
    out = 255 - out
  return np.array(out)


if __name__ == "__main__":
  font_img = font2img(get_font_path(0), "H", 5)
  cv_show(font_img, "Font Image")