"""``src/utils/feature_extration.py`` 的单元测试。"""

from __future__ import annotations

import pytest
import numpy as np
import cv2
from utils.feature_extration import (
    _get_mask,
    _get_threhold_edge,
    contrast_gray,
    defocus_glyph,
    defocus_sigma,
    draw_board,
    estimate_depths,
    fuse_img,
    fuse_scene,
    get_flat_quad,
    inset_quad,
    is_planar_quad,
    pick_base_plane,
    pick_contrast,
    quad_perspective_ramp,
    segment_quad

)
from utils.font_processor import font2img, get_font_path
from utils.geometry import OffsetImg, QuadSegment, quad2M, proj_tran
from utils.debug_tools import read_image
from utils.typing_tools import random_choose
from configs.path import RAW_DIR, TEST_DIR, LOG_DIR
import configs.preprocess_cfg as ppcfg

# pytestmark = pytest.mark.skip(reason="待实现：补充该模块的单元测试")

env_img = read_image(RAW_DIR / "env/0072.jpg")
env_img = cv2.cvtColor(env_img, cv2.COLOR_BGR2GRAY)
font_imgs = [font2img(get_font_path(0), str(i), thickness=1)
             for i in range(10)]

_config = {
    "distortion": ppcfg.DISTORTION,
    "normalize": ppcfg.NORMALIZE,
    "area_range": ppcfg.AREA_RANGE
}

quads = get_flat_quad(env_img)
if quads is None:
  print("未检测到符合条件的四边形区域")

segments = [s for q in quads for s in segment_quad(q, **_config)]
if not segments:
  print("分割后没有符合 distortion/area 条件的四边形")

# 随机选择env_img中的若干个四边形进行投影变换
boxed_text_count = min(len(segments), int(
    len(segments) * .5 + 1))
# idx = np.random.choice(len(segments), size=boxed_text_count, replace=False)
quads = random_choose(segments, boxed_text_count)
picked_fonts = random_choose(font_imgs, boxed_text_count)

traned_font_imgs = [
    np.array(proj_tran(
        _img,
        quad2M(quads[idx].quad, *_img.shape)
    )) for idx, _img in enumerate(picked_fonts)
]


class TestImgPreProcessor:
  """图像预处理测试"""

  def test_get_correct_mask(self) -> np.ndarray:
    cv2.imencode('.png', traned_font_imgs[-1])[1].tofile(
        f"{TEST_DIR}/log/{env_img.shape}_debug_traned.png")
    print(f"quads[-1] offset: {quads[-1].offset}")
    _img = _get_mask(traned_font_imgs[-1], quads[-1].offset)
    cv2.imencode('.png', _img)[1].tofile(
        f"{TEST_DIR}/log/{env_img.shape}_debug_mask.png")

    assert _img is not None, "掩码生成失败"

  def test_get_whole_mask(self) -> None:
    h, w = env_img.shape[:2]                      # 注意是 (高, 宽)，别写成 (w, h)
    font_masks = np.zeros((h, w), dtype=np.uint8)  # 用 uint8，别用默认 float64

    for _img, _quad in zip(traned_font_imgs, quads):
      m = _get_mask(_img, _quad.offset)
      mh, mw = m.shape
      region = font_masks[:mh, :mw]
      np.maximum(region, m, out=region)          # 用最大值合并，避免叠加溢出

    font_edge_mask = _get_threhold_edge(font_masks)
    assert font_masks.any(), "整图掩码全空"
    
    env_gray_img = env_img
    overlay_img = np.maximum(font_masks, env_gray_img)  # 用最大值叠加，避免溢出
    
    contours = [q.quad.round().astype(np.int32) for q in quads]
    cv2.drawContours(env_img, contours, -1, (0, 0, 255), 2)
    cv2.imencode(".png", font_masks)[1].tofile(
        str(TEST_DIR / "log" / f"{w}x{h}_debug_env_whole.png"))
    cv2. imencode(".png", font_edge_mask)[1].tofile(
        str(TEST_DIR / "log" / f"{w}x{h}_debug_font_edge.png"))
    cv2.imencode(".png", env_img)[1].tofile(
        str(TEST_DIR / "log" / f"{w}x{h}_debug_mask_whole.png"))
    cv2.imencode(".png", overlay_img)[1].tofile(
        str(TEST_DIR / "log" / f"{w}x{h}_debug_overlay.png"))


def _block_glyph(size: int = 16) -> np.ndarray:
  """白字黑底的方块字形，便于断言覆盖度"""
  g = np.zeros((size, size), np.uint8)
  g[size // 4: 3 * size // 4, size // 4: 3 * size // 4] = 255
  return g


class TestFuseWithEnv:
  """第一步：数字与环境融合（对比度定色 + 可调透明度）"""

  def test_pick_contrast_modes(self) -> None:
    assert pick_contrast(0.5) == 0.5
    assert all(0.3 <= pick_contrast((0.3, 0.6)) <= 0.6 for _ in range(20))
    candidates = [0.2, 0.4]
    assert all(pick_contrast(candidates) in candidates for _ in range(20))
    with pytest.raises(ValueError):
      pick_contrast([])

  def test_contrast_gray_hits_target(self) -> None:
    for bg in (30.0, 100.0, 200.0):
      assert abs(contrast_gray(bg, 0.5) - bg) == pytest.approx(127.5, abs=1.0)

  def test_contrast_gray_picks_roomier_direction(self) -> None:
    assert contrast_gray(240.0, 0.5) < 240     # 亮背景压暗
    assert contrast_gray(10.0, 0.5) > 10       # 暗背景提亮
    assert 0 <= contrast_gray(250.0, 1.0) <= 255

  def test_fuse_img_uses_absolute_offset(self) -> None:
    env = np.full((100, 200, 3), 10, np.uint8)
    out = fuse_img(env, _block_glyph(16), offset=(150, 80), gray=200, alpha=1.0)
    assert out.shape == env.shape
    assert out[88, 158].tolist() == [200, 200, 200]   # 字形中心被换成灰度 200
    assert out[0, 0].tolist() == [10, 10, 10]         # 字形之外保持原样
    assert env[88, 158].tolist() == [10, 10, 10]      # 不修改入参

  def test_fuse_img_auto_colors_by_contrast(self) -> None:
    env = np.full((40, 40, 3), 30, np.uint8)
    out = fuse_img(env, _block_glyph(16), offset=(0, 0), contrast=0.5, alpha=1.0)
    assert int(out[8, 8, 0]) > 30                     # 暗背景 → 数字提亮
    assert int(out[8, 8, 0]) - 30 == pytest.approx(127.5, abs=1.0)

  def test_fuse_img_alpha_controls_strength(self) -> None:
    env = np.full((16, 16, 3), 100, np.uint8)
    out = fuse_img(env, _block_glyph(16), offset=(0, 0), gray=255, alpha=0.5)
    assert int(out[8, 8, 0]) == 177                   # 100*(1-0.5) + 255*0.5

  def test_fuse_img_needs_offset(self) -> None:
    with pytest.raises(ValueError):
      fuse_img(np.zeros((8, 8, 3), np.uint8), _block_glyph(8))

  def test_fuse_img_works_on_grayscale_env(self) -> None:
    env = np.full((40, 40), 30, np.uint8)          # 单通道环境图
    out = fuse_img(env, _block_glyph(16), offset=(0, 0), contrast=0.5, alpha=1.0)
    assert out.shape == env.shape
    assert int(out[8, 8]) == contrast_gray(30.0, 0.5)


class TestBoard:
  """板面渲染：板 + 数字化"""

  def test_inset_quad_keeps_shape_and_shrinks(self) -> None:
    quad = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], np.float64)
    inner = inset_quad(quad, margin=0.2)
    assert np.allclose(inner, [[20, 20], [80, 20], [80, 80], [20, 80]])

  def test_draw_board_recolors_inside_only(self) -> None:
    env = np.full((60, 60), 100, np.uint8)
    quad = np.array([[20, 20], [40, 20], [40, 40], [20, 40]], np.float64)
    out = draw_board(env, quad, tone_delta=20, feather=0, shadow=False)
    assert out.shape == env.shape
    assert int(out[30, 30]) == 120          # 板面内部 = 背景 + 亮度差
    assert int(out[0, 0]) == 100            # 板外保持原样
    assert int(env[30, 30]) == 100          # 不修改入参

  def test_draw_board_blurs_edges_with_sigma(self) -> None:
    env = np.full((80, 80), 100, np.uint8)
    quad = np.array([[30, 30], [50, 30], [50, 50], [30, 50]], np.float64)
    sharp = draw_board(env, quad, tone_delta=40, feather=0, shadow=False, sigma=0)
    blurry = draw_board(env, quad, tone_delta=40, feather=0, shadow=False, sigma=3)
    # sigma=0 时板外完全不受影响；sigma>0 时失焦让板边向外扩散
    assert int(sharp[30, 52]) == 100
    assert int(blurry[30, 52]) > int(sharp[30, 52])
    assert int(blurry[40, 40]) < int(sharp[40, 40])   # 模糊把板心的不透明度也摊薄了

  def test_fuse_img_reads_offset_metadata(self) -> None:
    env = np.full((40, 40, 3), 0, np.uint8)
    out = fuse_img(env, OffsetImg(_block_glyph(16), offset=(10, 12)),
                   gray=90, alpha=1.0)
    assert out[20, 18].tolist() == [90, 90, 90]


class TestDefocus:
  """第二步：按基准平面做景深失焦（程度可调）"""

  def test_defocus_glyph_zero_sigma_is_identity(self) -> None:
    g = _block_glyph(16)
    assert defocus_glyph(g, 0) is g

  def test_defocus_glyph_spreads_and_dims(self) -> None:
    g = _block_glyph(16)
    b = defocus_glyph(g, 2.0)
    assert b.shape == g.shape
    assert b[3, 3] > 0        # 模糊后边缘被点亮
    assert b[8, 8] < 255      # 峰值被摊平

  def test_defocus_glyph_uses_perspective_ramp(self) -> None:
    g = _block_glyph(16)
    # 左近右远的渐变：远端应该比近端更糊（同一行上峰值更低）
    ramp = np.tile(np.linspace(0.0, 1.0, 16, dtype=np.float32), (16, 1))
    b = defocus_glyph(g, 2.0, ramp, spread=0.8)
    assert b.shape == g.shape
    assert b[8, 11] < b[8, 4]

  def test_perspective_ramp_points_to_far_side(self) -> None:
    # 上窄下宽：顶边更远
    quad = np.array([[0, 0], [80, 0], [100, 100], [0, 100]], np.float64)
    ramp = quad_perspective_ramp(quad, (100, 100))
    assert ramp.shape == (100, 100)
    assert ramp[0].mean() > 0.9
    assert ramp[-1].mean() < 0.1

  def test_perspective_ramp_flat_quad_is_uniform(self) -> None:
    quad = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], np.float64)
    ramp = quad_perspective_ramp(quad, (50, 50))
    assert np.allclose(ramp, 0.5)


class TestPlanarity:
  """平面性校验：平整面通过，折痕 / 跨物体拒绝"""

  def test_flat_patch_passes(self) -> None:
    gray = np.full((200, 200), 128, np.uint8)
    quad = np.array([[20, 20], [180, 20], [180, 180], [20, 180]], np.float64)
    mask = np.full((200, 200), 255, np.uint8)
    assert is_planar_quad(gray, quad, mask) is True

  def test_patch_with_strong_edge_is_rejected(self) -> None:
    gray = np.full((200, 200), 128, np.uint8)
    cv2.line(gray, (0, 100), (200, 100), 0, 7)
    quad = np.array([[20, 20], [180, 20], [180, 180], [20, 180]], np.float64)
    mask = np.full((200, 200), 255, np.uint8)
    assert is_planar_quad(gray, quad, mask) is False

  def test_low_flat_coverage_is_rejected(self) -> None:
    gray = np.full((200, 200), 128, np.uint8)
    quad = np.array([[20, 20], [180, 20], [180, 180], [20, 180]], np.float64)
    mask = np.zeros((200, 200), np.uint8)
    assert is_planar_quad(gray, quad, mask) is False

  def test_estimate_depths_near_is_bigger_and_lower(self) -> None:
    near = np.array([[0, 300], [100, 300], [100, 400], [0, 400]], np.float64)
    far = np.array([[0, 20], [20, 20], [20, 40], [0, 40]], np.float64)
    d = estimate_depths([near, far], (480, 640))
    assert d.shape == (2,)
    assert d[0] == 0.0 and d[1] == 1.0       # min-max 归一化：0 最近、1 最远

  def test_pick_base_plane_picks_majority_cluster(self) -> None:
    base, mask = pick_base_plane([0.0, 0.03, 0.06, 0.95], tol=0.12)
    assert mask.tolist() == [True, True, True, False]
    assert 0.0 <= base <= 0.07

  def test_defocus_sigma_is_adjustable(self) -> None:
    assert defocus_sigma(0.5, 0.5) == 0.0
    weak = defocus_sigma(0.75, 0.5, strength=1.0, max_sigma=10.0)
    strong = defocus_sigma(0.75, 0.5, strength=2.0, max_sigma=10.0)
    assert 0 < weak < strong
    assert defocus_sigma(1.0, 0.0, strength=10.0, max_sigma=6.0) == 6.0

  def test_fuse_scene_blurs_only_off_plane(self) -> None:
    env = np.full((480, 640, 3), 20, np.uint8)
    base = QuadSegment(
        np.array([[100, 300], [200, 300], [200, 400], [100, 400]], np.float64))
    far = QuadSegment(
        np.array([[400, 40], [440, 40], [440, 80], [400, 80]], np.float64))
    out, infos = fuse_scene(env, [_block_glyph(16), _block_glyph(16)],
                            [base, far], contrast=0.5, alpha=1.0,
                            defocus_strength=1.0, max_sigma=6.0)

    assert out.shape == env.shape
    assert infos[0]["on_base"] is True and infos[1]["on_base"] is False
    assert infos[0]["sigma"] == 0.0 and infos[1]["sigma"] > 0.0
    # 基准平面上的数字保持清晰，且与该处背景保持目标对比度
    assert int(out[308, 108, 0]) == infos[0]["gray"]
    assert abs(infos[0]["gray"] - infos[0]["bg"]) == pytest.approx(127.5, abs=1.0)

