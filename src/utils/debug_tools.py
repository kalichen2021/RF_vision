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


# ---------------- 可视化（训练 / 推理结果检查）----------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def tensor2img(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
  """把归一化后的图像张量还原成可显示 / 可保存的 BGR uint8 图

  Args:
    tensor (torch.Tensor): ``(3, H, W)``；也接受 ``(1, 3, H, W)``（取第 0 个）、
      ``(H, W)`` / ``(1, H, W)`` 灰度
    mean (tuple): 归一化时用的均值
    std (tuple): 归一化时用的标准差

  Returns:
    np.ndarray: ``(H, W, 3)`` BGR 图（单通道输入返回 ``(H, W)`` 灰度图）
  """
  t = tensor.detach().float().cpu()
  if t.ndim == 4:
    t = t[0]
  if t.ndim == 2:
    return (t.numpy() * 255).round().clip(0, 255).astype(np.uint8)
  if t.shape[0] == 1:
    arr = (t[0].numpy() * std[0] + mean[0]).clip(0, 1)
    return (arr * 255).round().astype(np.uint8)

  m = np.asarray(mean, np.float32).reshape(3, 1, 1)
  s = np.asarray(std, np.float32).reshape(3, 1, 1)
  arr = (t.numpy() * s + m).clip(0, 1)
  arr = (arr * 255).round().astype(np.uint8).transpose(1, 2, 0)  # CHW -> HWC(RGB)
  return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def draw_quads(img, quads, labels=None, scores=None, color=(0, 255, 0),
               thickness=2, normalized=True, tag="", font_scale=0.5,
               inplace=False, label_pos="top", skip_edges=None):
  """在图上画出若干四边形，可带「数字 + 置信度」标签（推理 / 训练结果可视化用）

  Args:
    img (np.ndarray): BGR 或灰度图
    quads (np.ndarray | list): ``(K, 4, 2)``，顺序 [左上, 右上, 右下, 左下]
    labels (list[int] | None): 每个框对应的数字类别
    scores (list[float] | None): 每个框的置信度，显示两位小数
    color (tuple): BGR 颜色；建议 GT 用蓝 ``(255,0,0)``、预测用绿 ``(0,255,0)``
    thickness (int): 线宽
    normalized (bool): True 时把 0~1 坐标按图像尺寸缩放，False 按像素坐标
    tag (str): 文字前缀（如 ``"gt"`` / ``"pred"``）
    font_scale (float): 字号
    inplace (bool): True 直接画在原图上（灰度图会被转成三通道新图）
    label_pos (str): ``"top"`` 文字在框上方，``"bottom"`` 在框下方；
        同时画 GT 与预测时把其中一个放下面才能不互相遮挡
    skip_edges (Iterable | None): 要跳过的边；既可写边序号 ``2``（0~3，第 k 条边
        连接点 k 与点 (k+1)%4），也可写点对 ``(2, 3)``（仅相邻两点有效）。
        例如 ``{(2, 3)}`` 表示不在第 3、4 个点之间连线

  Returns:
    np.ndarray: 画好的三通道 BGR 图
  """
  if img is None:
    raise ValueError("draw_quads 收到 None")
  vis = img if inplace else img.copy()
  if vis.ndim == 2:
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

  # 四条边：第 k 条边连接点 k 与点 (k+1)%4
  edges = ((0, 1), (1, 2), (2, 3), (3, 0))
  skip = set()
  for e in (skip_edges or ()):
    if not isinstance(e, (tuple, list, set)):
      skip.add(int(e) % 4)                       # 直接给边序号
      continue
    a, b = int(e[0]) % 4, int(e[1]) % 4          # 给点对 (2, 3) 也支持
    if (a + 1) % 4 == b:
      skip.add(a)
    elif (b + 1) % 4 == a:
      skip.add(b)
    else:
      raise ValueError(f"skip_edges 中的点对 {e!r} 不相邻")

  h, w = vis.shape[:2]
  pts_all = np.asarray(quads, dtype=np.float64).reshape(-1, 4, 2)
  if normalized:
    pts_all = pts_all * np.array([max(w - 1, 1), max(h - 1, 1)], np.float64)

  for i, pts in enumerate(pts_all):
    poly = np.round(pts).astype(np.int32)
    # 逐条画边，跳过的边不画（用 polylines 无法省略某一条）
    for k, (a, b) in enumerate(edges):
      if k in skip:
        continue
      cv2.line(vis, tuple(poly[a]), tuple(poly[b]), color,
               max(1, int(thickness)), cv2.LINE_AA)

    text = ""
    if labels is not None:
      text = str(int(labels[i]))
    if scores is not None:
      text = f"{text} {float(scores[i]):.2f}".strip()
    if tag:
      text = f"{tag} {text}".strip()
    if not text:
      continue

    x = int(poly[:, 0].min())
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                     font_scale, 1)
    if label_pos == "bottom":
      y = min(int(poly[:, 1].max()) + th + 6, h - 2)
    else:
      y = max(int(poly[:, 1].min()) - 4, th + 6)
    # 贴在框角上，但不允许超出画布（否则边缘的框会只剩半截文字）
    x = min(max(x, 0), max(w - tw - 6, 0))
    cv2.rectangle(vis, (x, y - th - 4), (x + tw + 4, y + base), color, -1)
    # 底色偏亮用黑字，偏暗用白字
    text_color = (0, 0, 0) if sum(color) > 380 else (255, 255, 255)
    cv2.putText(vis, text, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, text_color, 1, cv2.LINE_AA)

  return vis


def save_image(img, path):
  """保存图片（imencode + tofile，避开 Windows 中文路径下 cv2.imwrite 失败）

  Args:
    img (np.ndarray): 要保存的图
    path (str | Path): 目标路径，后缀决定编码格式（缺省 .png）

  Returns:
    Path: 实际写入的路径
  """
  path = Path(path)
  if path.suffix == "":
    path = path.with_suffix(".png")
  path.parent.mkdir(parents=True, exist_ok=True)
  ok, buf = cv2.imencode(path.suffix, img)
  if not ok:
    raise IOError(f"图片编码失败: {path}")
  buf.tofile(str(path))
  return path
