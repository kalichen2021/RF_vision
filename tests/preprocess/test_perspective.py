"""`src/preprocess/perspective.py` 的单元测试（板面姿态重采样）。"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.preprocess.perspective import resample_pose
from src.utils.geometry import order_quad

SQUARE = np.array([[100.0, 100.0], [200.0, 100.0],
                   [200.0, 200.0], [100.0, 200.0]])


def _area(q: np.ndarray) -> float:
  return abs(cv2.contourArea(np.asarray(q, np.float32)))


def _persp(q: np.ndarray) -> float:
  """对边长度差 / 平均（0 = 无透视）。"""
  top = np.linalg.norm(q[1] - q[0])
  bot = np.linalg.norm(q[2] - q[3])
  left = np.linalg.norm(q[3] - q[0])
  right = np.linalg.norm(q[2] - q[1])
  return 0.5 * (abs(top - bot) / max((top + bot) / 2, 1e-6)
                + abs(left - right) / max((left + right) / 2, 1e-6))


def test_keep_mode_preserves_area_and_center() -> None:
  """keep 模式：面积拉回原值，中心不动。"""
  q = resample_pose(SQUARE, distance=(2.0, 2.0), rot_x=(-20, 20),
                    rot_y=(60.0, 60.0), rot_z=(-10, 10),
                    area_jitter=(1.0, 1.0), area_mode="keep",
                    rng=np.random.default_rng(0))
  assert _area(q) == pytest.approx(_area(SQUARE), rel=0.02)
  assert np.allclose(q.mean(axis=0), SQUARE.mean(axis=0), atol=1e-6)


def test_natural_mode_keeps_foreshortening() -> None:
  """natural 模式：60° 偏航会让面积真的变小（约 cos60 = 0.5）。"""
  q = resample_pose(SQUARE, distance=(2.0, 2.0), rot_x=(0.0, 0.0),
                    rot_y=(60.0, 60.0), rot_z=(0.0, 0.0),
                    area_jitter=(1.0, 1.0), area_mode="natural",
                    rng=np.random.default_rng(0))
  ratio = _area(q) / _area(SQUARE)
  assert 0.30 < ratio < 0.75, f"natural 模式应保留透视压缩，实际面积比 {ratio:.2f}"
  assert np.allclose(q.mean(axis=0), SQUARE.mean(axis=0), atol=1e-6)


def test_distance_controls_perspective_strength() -> None:
  """距离越远、透视越弱（对边长度差越小）。"""
  near = resample_pose(SQUARE, distance=(2.0, 2.0), rot_x=(20.0, 20.0),
                       area_mode="natural", rng=np.random.default_rng(1))
  far = resample_pose(SQUARE, distance=(20.0, 20.0), rot_x=(20.0, 20.0),
                      area_mode="natural", rng=np.random.default_rng(1))
  assert _persp(far) < _persp(near)


def test_blend_zero_returns_original_shape() -> None:
  """blend=0 应原样返回（只做点序归一）。"""
  q = resample_pose(SQUARE, distance=(3.0, 3.0), rot_y=(30.0, 30.0),
                    blend=0.0, rng=np.random.default_rng(0))
  assert np.allclose(q, order_quad(SQUARE, reset=False))


def test_output_is_convex_and_inside_bounds() -> None:
  """大角度旋转 + 画布约束：结果必须是凸的且落在画布内。"""
  for seed in range(5):
    q = resample_pose(SQUARE, distance=(1.5, 2.5), rot_x=(-40, 40),
                      rot_y=(-70, 70), rot_z=(-40, 40), blend=1.0,
                      bounds=(300, 300),
                      rng=np.random.default_rng(seed))
    p = np.asarray(q, np.float64)
    assert p.min() >= -0.01 and p.max() <= 300.01
    # 凸性：相邻边叉积同号
    crosses = []
    for i in range(4):
      a = p[(i + 1) % 4] - p[i]
      b = p[(i + 2) % 4] - p[(i + 1) % 4]
      crosses.append(a[0] * b[1] - a[1] * b[0])
    assert all(c > 0 for c in crosses) or all(c < 0 for c in crosses)


def test_invalid_area_mode_raises() -> None:
  with pytest.raises(ValueError):
    resample_pose(SQUARE, area_mode="stretch")
