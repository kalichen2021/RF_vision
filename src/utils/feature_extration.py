import cv2,sys,numpy as np, random, math
from numpy.typing import NDArray
from .debug_tools import cv_show, read_image
from .geometry import order_quad
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.path import RAW_DIR, LOG_DIR


def _smooth_mask(gray, win=15, var_thresh=8.0, iter = 1):
  """基于局部方差的平滑区域掩膜

  在 win×win 窗口内计算灰度方差，方差小于 var_thresh 的像素视为平滑区域，
  再通过形态学开 / 闭运算去除噪点并填补空洞。

  Args:
      gray (NDArray): 单通道灰度图像
      win (int, optional): 计算局部方差的窗口大小（px）。 Defaults to 15.
      var_thresh (float, optional): 判定为平滑区域的方差阈值，
        方差越小越平滑。 Defaults to 8.0.

  Returns:
      NDArray[np.uint8]: 二值掩膜，平滑区域为 255，其余为 0
  """
  g = gray.astype(np.float32)
  mu = cv2.blur(g, (win, win))
  var = cv2.blur(g * g, (win, win)) - mu * mu
  mask = (var < var_thresh).astype(np.uint8) * 255
  k = np.ones((win, win), np.uint8)
  for _ in range(iter):
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.dilate(mask, k, iterations=1)
  return mask


def _get_maxquad(cnt):
  """获取轮廓的最大内接四边形

  Args:
      cnt (NDArray): 输入轮廓点，通常来自 cv2.findContours

  Returns:
      NDArray[np.float64] | None: 最大内接四边形顶点坐标，形状为 (4, 2)；
        无法拟合时返回 None
  """
  peri = cv2.arcLength(cnt, True)
  for k in np.arange(0.01, 0.11, 0.005):
    approx = cv2.approxPolyDP(cnt, k * peri, True)
    if len(approx) == 4 and cv2.isContourConvex(approx):
      return approx.reshape(4, 2).astype(np.float64)
  return None


def _is_acceptable(sub_quad, distortion_limit):
  """判断切出来的小四边形是否可接受（畸变是否在容忍范围内）"""
  # 计算四条边
  edges = [
    np.linalg.norm(sub_quad[0] - sub_quad[1]),
    np.linalg.norm(sub_quad[1] - sub_quad[2]),
    np.linalg.norm(sub_quad[2] - sub_quad[3]),
    np.linalg.norm(sub_quad[3] - sub_quad[0])
  ]
  
  min_edge = min(edges)
  # 如果最短边太小，说明退化成了三角形或极小碎块，直接舍弃
  if min_edge < 1e-4:
    return False
    
  max_edge = max(edges)
  aspect_ratio = max_edge / min_edge
  
  # 将 distortion_limit (0~1) 映射为可接受的最大长宽比阈值
  # distortion_limit=0 时，只接受接近正方形的 (比例 <= 1.5)
  # distortion_limit=1 时，接受较扁的 (比例 <= 5.0)
  allowed_ratio = 1.5 + distortion_limit * 3.5
  
  if aspect_ratio > allowed_ratio:
    return False
    
  return True


def segment_quad(quad, distortion, normalize, count, gap=0.12, area_range=(.3, 3000)):
  """将四边形分割成多个四边形，通过畸变过滤舍弃无法挽救的碎片。
  
  Args:
    quad (NDArray[np.float64]): 输入的四边形，(4, 2) 的顶点坐标 [左上, 右上, 右下, 左下]
    distortion (float | tuple): 畸变容忍度（0~1），越小要求越严格（丢弃越多的畸变块）。
    normalize (float | tuple): 归一化程度（0~1），值越大内部网格越接近菱形分布。
    count (tuple): 分割后的四边形数量范围（这是网格划分的目标数，实际返回可能会因为舍弃而变少）。
    area_range (tuple): 每个分割出的四边形的面积范围。
    gap (float): 切割间隙 (0~1，0为无缝)。
  """
  # 1. 处理范围参数
  if isinstance(distortion, tuple):
    distortion = random.uniform(distortion[0], distortion[1])
  if isinstance(normalize, tuple):
    normalize = random.uniform(normalize[0], normalize[1])
    
  # 2. 计算物理空间的平均宽高 (用于自适应分配行列数，防止长条)
  width_top = np.linalg.norm(quad[1] - quad[0])
  width_bottom = np.linalg.norm(quad[2] - quad[3])
  height_left = np.linalg.norm(quad[3] - quad[0])
  height_right = np.linalg.norm(quad[2] - quad[1])
  
  avg_w = (width_top + width_bottom) / 2.0
  avg_h = (height_left + height_right) / 2.0
  
  target_num = random.randint(count[0], count[1])
  
  # 根据物理长宽比分配行列数
  aspect_ratio = max(avg_w, 1e-6) / max(avg_h, 1e-6)
  cols = max(1, int(round(math.sqrt(target_num * aspect_ratio))))
  rows = max(1, int(round(target_num / cols)))
  
  # 确保网格数量在目标范围内
  while cols * rows > count[1]:
    if cols > rows: cols -= 1
    else: rows -= 1
  while cols * rows < count[0]:
    if avg_w > avg_h: cols += 1
    else: rows += 1

  # 3. 生成内部网格点
  u_vals = np.linspace(0, 1, cols + 1)
  v_vals = np.linspace(0, 1, rows + 1)
  
  grid_points = np.zeros((rows + 1, cols + 1, 2), dtype=np.float64)
  
  for j in range(rows + 1):
    for i in range(cols + 1):
      u = u_vals[i]
      v = v_vals[j]
      
      p_top = (1 - u) * quad[0] + u * quad[1]
      p_bot = (1 - u) * quad[3] + u * quad[2]
      p_base = (1 - v) * p_top + v * p_bot
      
      # 内部网格正则化处理 (normalize 控制菱形分布)
      if 0 < i < cols and 0 < j < rows:
        u_center = u - 0.5
        v_center = v - 0.5
        
        if normalize > 0.5:
          w_diamond = (normalize - 0.5) * 2.0
          p_base[0] += u_center * v_center * w_diamond * avg_w * 0.5
          p_base[1] += (u_center**2 - v_center**2) * w_diamond * avg_h * 0.5

      grid_points[j, i] = p_base

  # 4. 切片提取、过滤与 gap 处理
  segments = []
  for j in range(rows):
    for i in range(cols):
      p_tl = grid_points[j, i]
      p_tr = grid_points[j, i + 1]
      p_br = grid_points[j + 1, i + 1]
      p_bl = grid_points[j + 1, i]

      center = (p_tl + p_tr + p_br + p_bl) / 4.0
      
      g = max(0.0, min(0.9, gap))
      p_tl = p_tl + (center - p_tl) * g
      p_tr = p_tr + (center - p_tr) * g
      p_br = p_br + (center - p_br) * g
      p_bl = p_bl + (center - p_bl) * g

      sub_quad = np.array([p_tl, p_tr, p_br, p_bl], dtype=np.float64)

      _area_range = []      
      for i in range(len(area_range)):
        if area_range[i] < 1:
          _area_range.append(area_range[i] * (avg_w * avg_h))
        else:
          _area_range.append(area_range[i])
      _area_range = sorted(_area_range)
      print(f"sub_quad: {sub_quad}, area: {cv2.contourArea(np.array(sub_quad, dtype=np.int32))}, area_range: {_area_range}, distortion: {distortion}")
      # 如果该碎片的长宽比超过了 distortion 允许的阈值，则直接丢弃，不加入结果
      if _is_acceptable(sub_quad, distortion):
        # 如果该碎片的面积在指定范围内，则加入结果
        if _area_range[0] <= cv2.contourArea(np.array(sub_quad, dtype=np.int32)) <= _area_range[1]:
          segments.append(sub_quad)
      
  return segments
  
  
def get_flat_quad(img: NDArray[np.uint8]):
  """获得图片平滑区域，返回四边形

  Args:
    img (NDArray[np.uint8]): 输入的图像，可以bgr或gray，数据类型为 uint8
  Returns:
    NDArray[np.float64]: 四边形顶点坐标，形状为 (4, 2)
  """
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
  mask = cv2.blur(gray, (5, 5))
  mask = _smooth_mask(gray)
  cv2.imencode('.png', mask)[1].tofile(f"{LOG_DIR}/{img.shape}_debug_0_mask.png")
  
  contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  if not contours:
    print("未检测到平滑区域")
    return None
  
  _img = cv2.drawContours(img.copy(), contours, -1, (0, 255, 0), 2)
  cv2.imencode('.png', _img)[1].tofile(f"{LOG_DIR}/{img.shape}_debug_1_contours.png")

  
  quads = [_get_maxquad(cnt) for cnt in contours]  
  quads = [np.array(q, dtype=np.int32) for q in quads if q is not None]
  _img = cv2.drawContours(img.copy(), np.array(quads, dtype=np.int32), -1, (0, 255, 0), 2)
  cv2.imencode('.png', _img)[1].tofile(f"{LOG_DIR}/{img.shape}_debug_1_quads.png")
  # quads = [q for q in quads if cv2.contourArea(q) >= max(_area_range[0], _area_range[1])]
  
  quads = [order_quad(q) for q in quads]
  _img = cv2.drawContours(img.copy(), np.array(quads, dtype=np.int32), -1, (0, 255, 0), 2)
  cv2.imencode('.png', _img)[1].tofile(f"{LOG_DIR}/{img.shape}_debug_2_contours.png")

  return quads





if __name__ == "__main__":

  img_path = RAW_DIR / "env/0036.jpg"
  img = read_image(img_path)
  assert img is not None, f"无法读取图片: {img_path}"

  quads = get_flat_quad(img)
  
  if quads is None:
    print("未检测到符合条件的四边形区域")
  else:
    vis = img.copy()
    for i, quad in enumerate(quads):
      quad = segment_quad(quad, distortion=(0.1, 0.3), normalize= (0.5, 0.8),count=(1, 5), area_range=(1000, 30000))
      print(f"Quad {i+1}: {quad} segments")
      cv2.polylines(vis, np.array(quad, dtype=np.int32), True, (0, 255, 0), 2)
    cv_show(vis, f"Quad {i+1}")
