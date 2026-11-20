import cv2
import numpy as np
from dataclasses import dataclass
from numpy.typing import NDArray
from pathlib import Path
from .debug_tools import cv_show

def order_quad(q, reset = True) -> NDArray[np.floating]:
  """将任意顺序的四个点排序为 [左上, 右上, 右下, 左下]（图像坐标，y 向下）。
    「绕质心极角排序 → 旋转到左上 → 方向校准」，对任意
    凸四边形都只会得到 4 个互不相同的点。

  Args:
      q ([]): 点的列表(4,2)
      reset (bool, optional): 是否将图像移动到(0, 0)点. Defaults to True.

  Raises:
      ValueError: _description_

  Returns:
      NDArray[np.floating]: 处理后的点列表(4,2)
  """
  q = np.asarray(q, dtype=np.float64)
  if q.shape != (4, 2):
      raise ValueError(f"expect 4 points, got shape {q.shape}")

  # 1) 绕质心按极角排序 —— 得到一致的环形顺序，绝不重复选点
  c = q.mean(axis=0)
  angles = np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0])
  q = q[np.argsort(angles)]

  # 2) 旋转使最“左上”的点（x+y 最小）位于首位
  start = int(np.argmin(q.sum(axis=1)))
  q = np.roll(q, -start, axis=0)

  # 3) 图像坐标下 [TL, TR, BR, BL] 是顺时针：TL→TR 应使 x 增大。
  #    若方向反了（解出的是 TL→BL→BR→TR），翻转余下三点。
  if q[1, 0] < q[0, 0]:
      q = np.concatenate([q[:1], q[:0:-1]])

  if reset:
      q -= q.min(axis=0)
  return q


def _as_3x3(M: NDArray[np.floating]) -> NDArray[np.float64]:
  """将变换矩阵统一转换为 (3, 3) 的射影变换矩阵

  Parameters
  ----------
  M : NDArray[np.floating]
    变换矩阵, 形状为 (2, 3) 或 (3, 3)

  Returns
  -------
  NDArray[np.float64]
      形状为 (3, 3) 的射影变换矩阵
  """
  M = np.array(M, dtype=np.float64)
  if M.shape == (2, 3):
    M = np.vstack([M, [0.0, 0.0, 1.0]])
  return M[:3]


class QuadMatrix(np.ndarray):
  """携带目标四边形信息的射影变换矩阵（3, 3）。

  普通 ndarray 无法附带元数据，因此用子类记录：
  - ``dst_quad``：目标四边形。``proj_tran`` 据此把输出画布裁剪为其外接矩形，
    输出尺寸因此与四边形尺寸一致（不再铺满整幅变换后图像的包围盒）；
  - ``src_size``：生成矩阵时用的源矩形尺寸；为 ``None`` 表示「源尺寸待定」，
    ``proj_tran`` 会用输入图像的宽高补全，从而把整幅图像拉伸铺满 ``dst_quad``。
  """
  dst_quad: NDArray[np.float64] | None = None
  src_size: tuple[int, int] | None = None

  def __new__(cls, M, dst_quad=None, src_size=None):
    obj = np.asarray(M, dtype=np.float64).view(cls)
    obj.dst_quad = None if dst_quad is None else np.asarray(dst_quad, dtype=np.float64).reshape(4, 2)
    obj.src_size = None if src_size is None else (int(src_size[0]), int(src_size[1]))
    return obj

  def __array_finalize__(self, obj):
    self.dst_quad = getattr(obj, "dst_quad", None)
    self.src_size = getattr(obj, "src_size", None)
    

class OffsetImg(np.ndarray):
  """携带偏移量信息的图片。

  普通 ndarray 无法附带元数据，因此用子类记录：
  - ``offset``：该图片在环境图中的左上角坐标 ``(x_min, y_min)``，
    ``_get_mask`` 据此把贴图贴回整幅画布。
  """
  offset: NDArray[np.uint32] | None = None

  def __new__(cls, img, offset=None):
    obj = np.asarray(img, dtype=np.uint8).view(cls)
    obj.offset = None if offset is None else np.asarray(offset, dtype=np.uint32)
    return obj

  def __array_finalize__(self, obj):
    self.offset = getattr(obj, "offset", None)


@dataclass
class QuadSegment:
  """环境图中的一块四边形切块（坐标 + 定位信息）。

  这不是图片，不应用 ``OffsetImg`` 承载 —— 后者会把坐标强转成 ``uint8``，
  大于 255 的坐标按 256 取模后被折叠到画布左上角。

  Attributes:
      quad (NDArray[np.float64]): 四边形顶点，形状 (4, 2)，顺序 [左上, 右上, 右下, 左下]，
        使用环境图的绝对坐标。
  """
  quad: NDArray[np.float64]

  def __post_init__(self):
    self.quad = np.asarray(self.quad, dtype=np.float64).reshape(4, 2)

  @property
  def offset(self) -> tuple[float, float]:
    """外接矩形左上角 ``(x_min, y_min)``，与 ``proj_tran`` 输出的 offset 语义一致。"""
    return float(self.quad[:, 0].min()), float(self.quad[:, 1].min())

  @property
  def size(self) -> tuple[float, float]:
    """外接矩形宽高 ``(w, h)``。"""
    (x0, y0), (x1, y1) = self.offset, (
        float(self.quad[:, 0].max()), float(self.quad[:, 1].max()))
    return x1 - x0, y1 - y0


def _get_quad_wh(w: int, h: int, M: NDArray[np.floating]) -> tuple[int, int]:
  """根据射影变换矩阵 M 计算变换后图像的宽高

  Parameters
  ----------
  w : int
    原图像宽度
  h : int
    原图像高度
  M : NDArray[np.floating]
    射影变换矩阵, 形状为 (2, 3) 或 (3, 3)

  Returns
  -------
  tuple[int, int]
      新图像的宽高 (new_w, new_h)
  """
  # 获取原图像四个角点坐标
  corners = np.array([[[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]], dtype=np.float64)
  # 对角点进行射影变换
  new_corners = cv2.perspectiveTransform(corners, _as_3x3(M))[0]
  # 获取新图像的边界框
  x_min, y_min = np.min(new_corners, axis=0)
  x_max, y_max = np.max(new_corners, axis=0)
  new_w = int(np.ceil(x_max - x_min))
  new_h = int(np.ceil(y_max - y_min))
  return new_w, new_h


def get_rand_M(w: int, h: int, scale_range=(0.8, 1.2), shear_range=(-0.2, 0.2), rotation_range=(-15, 15), translation_range=(-0.2, 0.2), perspective_range=(-0.001, 0.001)) -> NDArray[np.float64]:
  """生成随机的射影变换矩阵

  Parameters
  ----------
  w : int
    原图像宽度
  h : int
    原图像高度
  scale_range : tuple[float, float], optional
    缩放范围, 默认为 (0.8, 1.2)
  shear_range : tuple[float, float], optional
    剪切范围, 默认为 (-0.2, 0.2)
  rotation_range : tuple[float, float], optional
    旋转范围（角度）, 默认为 (-15, 15)
  translation_range : tuple[float, float], optional
    平移范围（相对原图像尺寸的比例）, 默认为 (-0.2, 0.2)
  perspective_range : tuple[float, float], optional
    透视变换范围, 默认为 (-0.001, 0.001)  

  Returns
  -------
  NDArray[np.float64]
      随机生成的射影变换矩阵，形状为 (3, 3)
  """
  # 随机缩放
  scale_x = np.random.uniform(*scale_range)
  scale_y = np.random.uniform(*scale_range)
  S = np.array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]], dtype=np.float64)

  # 随机剪切
  shear_x = np.random.uniform(*shear_range)
  shear_y = np.random.uniform(*shear_range)
  Sh = np.array([[1, shear_x, 0], [shear_y, 1, 0], [0, 0, 1]], dtype=np.float64)

  # 随机旋转
  angle = np.random.uniform(*rotation_range)
  rad = np.deg2rad(angle)
  R = np.array([[np.cos(rad), -np.sin(rad), 0], [np.sin(rad), np.cos(rad), 0], [0, 0, 1]], dtype=np.float64)

  # 随机平移
  trans_x = np.random.uniform(*translation_range) * w
  trans_y = np.random.uniform(*translation_range) * h
  T = np.array([[1, 0, trans_x], [0, 1, trans_y], [0, 0, 1]], dtype=np.float64)
  
  # 随机透视变换
  p1 = np.random.uniform(*perspective_range)
  p2 = np.random.uniform(*perspective_range)
  P = np.array([[1, 0, 0], [0, 1, 0], [p1, p2, 1]], dtype=np.float64) 

  # 综合变换矩阵
  M = T @ R @ Sh @ S @ P
  return M


def proj_tran(img: NDArray[np.uint8], M: NDArray[np.floating], overflow = False, dst_quad = None) -> NDArray[np.uint8]:
  """将图片进行射影变换

  Parameters
  ----------
  img : NDArray[np.uint8]
    图片数据, 形状为 (N, M, 3) 的 BGR 图或 (N, M) 的灰度图
  M : NDArray[np.floating]
    变换矩阵, 形状为 (2, 3) 或 (3, 3)
  overflow : bool, optional
    是否允许溢出, 默认为 False, 若为 True, 则输出图像的尺寸与输入图像一致, 否则输出图像的尺寸会根据变换矩阵自动调整, 以保证所有像素都在画布内
  dst_quad : array_like, optional
    目标四边形（图像坐标系, 形状 (4, 2)）。给定时, 输出画布被裁剪为该四边形的外接矩形,
    输出尺寸因此与四边形尺寸一致; 内容仍按四边形的真实形状变换（不做二次拉伸）,
    所以四边形之外、外接矩形之内的地方会是 0。缺省时自动取 ``M`` 上携带的目标四边形
    （即 ``quad2M`` 的返回值）, 若该四边形还标记了「源尺寸待定」, 则把**输入图像的四个角**
    对准四边形的四个角（整幅 img 拉伸铺满 quad）。既没有 dst_quad 也没有元数据时,
    按 ``M`` 原样变换, 画布取「整幅变换后图像的包围盒」。

  Returns
  -------
  NDArray[np.uint8]
      变换后的图片
  """
  w, h = img.shape[1], img.shape[0]
  auto_fill = False
  if dst_quad is None:
    # quad2M 返回的 QuadMatrix 自带目标四边形, 直接用它决定画布
    dst_quad = getattr(M, "dst_quad", None)
    # src_size 为 None => quad2M(quad) 未指定源尺寸, 用 img 尺寸补全
    auto_fill = getattr(M, "src_size", False) is None
  M3 = _as_3x3(M)
  if overflow:
    return cv2.warpPerspective(img, M3, (w, h), flags=cv2.INTER_LINEAR)

  if dst_quad is not None:
    # 画布 = 目标四边形的外接矩形, 输出尺寸因此与 quad 尺寸一致
    quad = np.asarray(dst_quad, dtype=np.float64).reshape(4, 2)
    if auto_fill:
      # 源尺寸取整幅 img: img 四角对准 quad 四角, 图像被拉伸铺满 quad
      M3 = _as_3x3(quad2M(quad, w, h))
    x_min, y_min = quad.min(axis=0)
    x_max, y_max = quad.max(axis=0)
    new_w = max(int(np.ceil(x_max - x_min)), 1)
    new_h = max(int(np.ceil(y_max - y_min)), 1)
  else:
    new_w, new_h = _get_quad_wh(w, h, M3)
    corners = np.array([[[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]], dtype=np.float64)
    new_corners = cv2.perspectiveTransform(corners, M3)[0]
    x_min, y_min = np.min(new_corners, axis=0)
  if new_w < 1 or new_h < 1:
    raise ValueError(f"degenerate output size: {new_w}x{new_h}")

  # 只做平移（左乘）把结果挪到画布坐标, 保留 quad 的真实形状, 不把四边形硬压成矩形。
  # 注意透视除法会等比例缩放平移量, 因此不能像仿射那样直接改 M[0, 2] / M[1, 2]。
  T = np.array([[1.0, 0.0, -x_min], [0.0, 1.0, -y_min], [0.0, 0.0, 1.0]], dtype=np.float64)
  return OffsetImg(cv2.warpPerspective(img, T @ M3, (new_w, new_h), flags=cv2.INTER_LINEAR),
                 offset=(x_min, y_min))


def quad2M(quad, w: int = 0, h: int = 0) -> NDArray[np.float64]:
  """生成把 (0, 0)-(w-1, h-1) 矩形映射到 quad 的射影变换矩阵

  Parameters
  ----------
  quad : array_like
    目标四边形, 形状 (4, 2), 顺序为 [左上, 右上, 右下, 左下]
  w, h : int, optional
    源矩形宽高。缺省 (0, 0) 表示「源尺寸待定」: 返回的矩阵会标记 ``src_size=None``,
    ``proj_tran`` 会用输入图像的宽高补全, 于是**整幅图像拉伸铺满 quad**（四角对准四角）。

  Returns
  -------
  NDArray[np.float64]
    形状为 (3, 3) 的 ``QuadMatrix``, 携带 ``dst_quad``/``src_size``, ``proj_tran``
    会据此把输出画布裁剪到该四边形的外接矩形, 使输出尺寸与 quad 一致。
  """
  quad = np.asarray(quad, dtype=np.float64).reshape(4, 2)
  auto_src = w <= 1 or h <= 1
  if auto_src:
    # 未指定源尺寸时先用 quad 外接矩形占位（保证矩阵本身可用、不退化）
    x_min, y_min = quad.min(axis=0)
    x_max, y_max = quad.max(axis=0)
    w = max(int(np.ceil(x_max - x_min)), 2)
    h = max(int(np.ceil(y_max - y_min)), 2)
  src_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
             dtype=np.float32)
  dst_pts = quad.astype(np.float32)
  M = cv2.getPerspectiveTransform(src_pts, dst_pts)
  return QuadMatrix(M, quad, src_size=None if auto_src else (w, h))

if __name__ == "__main__":
  # 需要先用 np.fromfile 读入字节再 imdecode
  img_path = Path(__file__).with_name("lena.png")
  img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
  assert img is not None, f"无法读取图片: {img_path}"
  quad1 = np.array([[1167.3775, 8.48  ],
        [1179.9175, 8.48],
        [1180.1375,15.52],
        [1167.8175,15.52]], dtype=np.float64)
  quad = np.array([[50, 60],
          [100, 0],
          [200,100],
          [0,250]], dtype=np.float64)
  print(f"M: {quad2M(quad)}")
  # for i in range(5):
  # quad2M(quad) 未给源尺寸 => proj_tran 用 img 尺寸补全: img 四角对准 quad1 四角,
  # 整幅图被拉伸铺满 quad1, 输出画布 = quad1 的外接矩形
  out = proj_tran(img, quad2M(order_quad(quad)))
  out = cv2.drawContours(out, [quad.astype(np.int32)], -1, (0, 255, 0), 2)
  print(f"out.shape = {out.shape}")  # == quad1 外接矩形尺寸
  cv_show(out, "射影变换")
