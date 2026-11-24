"""`src/utils/geometry.py` 的单元测试（样例）。

覆盖对象：
- 纯 numpy 逻辑：`order_quad` / `_as_3x3` / `_get_quad_wh`
- 携带元数据的子类：`QuadMatrix` / `OffsetImg`
- 依赖 OpenCV 的变换：`quad2M` / `proj_tran` / `get_rand_M`
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from utils.geometry import (
    OffsetImg,
    QuadMatrix,
    _as_3x3,
    _get_quad_wh,
    get_rand_M,
    order_quad,
    proj_tran,
    quad2M,
)

# 图像坐标（y 向下）下的标准顺序：[左上, 右上, 右下, 左下]
SQUARE = np.array([[10.0, 10.0], [30.0, 10.0], [30.0, 30.0], [10.0, 30.0]])

# 用于 quad2M / proj_tran 的目标四边形及其外接矩形（80 x 30）
QUAD = np.array([[10.0, 10.0], [90.0, 10.0], [90.0, 40.0], [10.0, 40.0]])


def _signed_area(quad: np.ndarray) -> float:
  """鞋带公式求有向面积；图像坐标（y 向下）中顺时针为正。"""
  x, y = quad[:, 0], quad[:, 1]
  return 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


class TestOrderQuad:
  """`order_quad`：任意顺序的四个点 → [左上, 右上, 右下, 左下]。"""

  def test_axis_aligned_square(self):
    shuffled = SQUARE[[2, 0, 3, 1]]  # 打乱输入顺序
    ordered = order_quad(shuffled, reset=False)

    assert ordered.shape == (4, 2)
    assert ordered.dtype == np.float64  # 内部统一转成 float64
    np.testing.assert_allclose(ordered, SQUARE)

  def test_reset_moves_points_to_origin(self):
    ordered = order_quad(SQUARE[::-1])  # reset 默认为 True

    np.testing.assert_allclose(ordered.min(axis=0), [0.0, 0.0])
    np.testing.assert_allclose(ordered, SQUARE - SQUARE.min(axis=0))

  def test_rotated_square_keeps_all_points_and_clockwise_order(self):
    rotated = np.array([[5.0, 0.0], [0.0, 5.0], [-5.0, 0.0], [0.0, -5.0]])
    ordered = order_quad(rotated, reset=False)

    # 不丢点、不重复
    assert {tuple(p) for p in ordered.tolist()} == {tuple(p) for p in rotated.tolist()}
    # 图像坐标下顺序应为顺时针
    assert _signed_area(ordered) > 0

  def test_rejects_invalid_shape(self):
    with pytest.raises(ValueError, match="expect 4 points"):
      order_quad(np.zeros((3, 2)))

    with pytest.raises(ValueError, match="expect 4 points"):
      order_quad(np.zeros((4, 3)))


class TestMatrixHelpers:
  """`_as_3x3` 与 `_get_quad_wh`。"""

  def test_as_3x3_pads_affine_matrix(self):
    affine = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 7.0]])
    M3 = _as_3x3(affine)

    assert M3.shape == (3, 3)
    np.testing.assert_allclose(M3[:2], affine)
    np.testing.assert_allclose(M3[2], [0.0, 0.0, 1.0])

  def test_as_3x3_keeps_existing_3x3(self):
    np.testing.assert_allclose(_as_3x3(np.eye(3)), np.eye(3))

  def test_get_quad_wh_for_scale_matrix(self):
    # 源码角点取 (0,0)-(w-1,h-1)，放大 2 倍后宽高为 2*(w-1), 2*(h-1)
    M = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
    assert _get_quad_wh(10, 20, M) == (18, 38)


class TestMetadataSubclasses:
  """`QuadMatrix` / `OffsetImg` 的元数据携带能力。"""

  def test_quad_matrix_stores_metadata(self):
    M = QuadMatrix(np.eye(3), dst_quad=QUAD, src_size=(100, 50))

    assert isinstance(M, np.ndarray)
    np.testing.assert_allclose(M.dst_quad, QUAD)
    assert M.src_size == (100, 50)

  def test_quad_matrix_metadata_survives_slicing(self):
    M = QuadMatrix(np.eye(3), dst_quad=QUAD, src_size=(100, 50))

    # __array_finalize__ 会把元数据带到视图上
    assert M[0].dst_quad is not None
    assert M[0].src_size == (100, 50)

  def test_offset_img_stores_offset(self):
    img = OffsetImg(np.zeros((4, 5), dtype=np.uint8), offset=(3, 7))

    assert isinstance(img, np.ndarray)
    assert img.shape == (4, 5)
    assert img.offset.dtype == np.uint32
    assert img.offset.tolist() == [3, 7]


class TestQuad2M:
  """`quad2M`：源矩形 → 目标四边形的射影矩阵。"""

  def test_maps_source_corners_onto_quad(self):
    M = quad2M(QUAD, 100, 50)

    assert isinstance(M, QuadMatrix)
    np.testing.assert_allclose(M.dst_quad, QUAD)
    assert M.src_size == (100, 50)  # 显式给尺寸时记录源尺寸

    src_corners = np.array(
        [[[0, 0], [99, 0], [99, 49], [0, 49]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(src_corners, M.astype(np.float64))[0]
    np.testing.assert_allclose(mapped, QUAD, atol=1e-4)

  def test_without_size_marks_auto_src(self):
    M = quad2M(QUAD)  # 不传 w/h ⇒ 源尺寸待定

    assert M.src_size is None
    np.testing.assert_allclose(M.dst_quad, QUAD)


class TestProjTran:
  """`proj_tran`：画布尺寸、偏移元数据与 overflow 行为。"""

  def test_overflow_keeps_input_canvas(self, capsys):
    img = np.zeros((50, 100, 3), dtype=np.uint8)
    out = proj_tran(img, np.eye(3), overflow=True)

    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert capsys.readouterr().out == ""  # overflow 分支不打印

  def test_canvas_is_quad_bounding_box(self):
    img = np.zeros((50, 100, 3), dtype=np.uint8)
    out = proj_tran(img, quad2M(QUAD, 100, 50))

    assert isinstance(out, OffsetImg)  # 普通分支返回带偏移的图片
    assert out.shape == (30, 80, 3)  # QUAD 外接矩形 80 x 30
    assert out.offset.tolist() == [10, 10]

  def test_auto_fill_uses_image_size(self):
    img = np.full((50, 100, 3), 255, dtype=np.uint8)
    out = proj_tran(img, quad2M(QUAD))  # 源尺寸待定 ⇒ 整幅图铺满 QUAD

    assert out.shape == (30, 80, 3)
    # QUAD 是轴对齐矩形，铺满后画布内应几乎全是原图内容
    assert out.mean() > 250


class TestGetRandM:
  """`get_rand_M`：随机矩阵的形状与数值有效性。"""

  def test_shape_and_finiteness(self):
    np.random.seed(0)
    M = get_rand_M(100, 50)

    assert M.shape == (3, 3)
    assert np.isfinite(M).all()
    assert M[2, 2] != 0  # 齐次坐标项非零，矩阵可正常归一化

  def test_seed_is_deterministic(self):
    np.random.seed(42)
    first = get_rand_M(100, 50)
    np.random.seed(42)
    second = get_rand_M(100, 50)

    np.testing.assert_allclose(first, second)
