from configs.log import UUID as _uuid
from configs import preprocess_cfg as ppcfg
from configs import path as pathcfg
import cv2
import json
import random
import time
import numpy as np
from numpy.typing import NDArray
from utils.feature_extration import get_flat_quad, segment_quad, fuse_scene, inset_quad
from utils.geometry import proj_tran, quad2M
from utils.font_processor import font2img, get_font_path
from utils.typing_tools import is_number, random_choose
from utils.debug_tools import cv_show, hstack_images


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# for log
_save_dir = Path(pathcfg.LOG_DIR) / "preprocess" / _uuid
_save_dir.mkdir(parents=True, exist_ok=True)


def build_record(name, img, labels, extra=None):
  """构造一条样本标签记录（``image`` / ``width`` / ``height`` / extra / ``objects``）

  Args:
      name (str): 图片文件名（含扩展名）
      img (NDArray[np.uint8]): 样本图（用它的尺寸填 width / height）
      labels (list[dict]): ``Preprocess.process`` 返回的标签
      extra (dict | None): 附加元信息（如源图名、参数快照）

  Returns:
      dict: 可直接写进合并标签文件的记录
  """
  h, w = img.shape[:2]
  record = {"image": str(name), "width": int(w), "height": int(h)}
  if extra:
    record.update(extra)
  record["objects"] = labels

  return record


def save_sample(out_dir, name, img, labels, extra=None, write_labels=True):
  """保存一个训练样本：灰度 PNG（可选：单个 JSON 标签）

  Args:
      out_dir (Path | str): 输出目录
      name (str): 文件名（不含扩展名）
      img (NDArray[np.uint8]): 合成图（三通道会自动转灰度）
      labels (list[dict]): ``Preprocess.process`` 返回的标签（digit / quad / ...）
      extra (dict | None): 附加元信息（如源图名、参数快照）
      write_labels (bool): 是否额外写一个同名 JSON；批量生成时置 False，
          改用 ``save_dataset`` 把所有标签合成一个文件

  Returns:
      tuple[Path, Path | None]: (图片路径, 标签路径)；``write_labels=False`` 时标签路径为 None
  """
  out_dir = Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img_path = out_dir / f"{name}.png"
  cv2.imencode(".png", gray)[1].tofile(str(img_path))

  lbl_path = None
  if write_labels:
    lbl_path = out_dir / f"{name}.json"
    lbl_path.write_text(
        json.dumps(build_record(img_path.name, gray, labels, extra),
                   ensure_ascii=False, indent=2),
        encoding="utf-8")

  return img_path, lbl_path


def save_dataset(out_dir, records, filename="labels.json"):
  """把整批样本的标签合并写成一个 JSON（``{"count": n, "samples": [...]}``）

  Args:
      out_dir (Path | str): 输出目录（与图片同目录）
      records (list[dict]): ``build_record`` 产出的记录列表
      filename (str): 合并标签文件名

  Returns:
      Path: 合并标签文件路径
  """
  out_dir = Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  lbl_path = out_dir / filename
  lbl_path.write_text(
      json.dumps({"count": len(records), "samples": list(records)},
                 ensure_ascii=False, indent=2),
      encoding="utf-8")

  return lbl_path


class Preprocess:
  def __init__(self, distortion, normalize, area_range, quads_select_rate,
               contrast_range=ppcfg.CONTRAST_RANGE, fuse_alpha=ppcfg.FUSE_ALPHA,
               defocus_strength=ppcfg.DEFOCUS_STRENGTH,
               defocus_max_sigma=ppcfg.DEFOCUS_MAX_SIGMA,
               persp_spread=ppcfg.DEFOCUS_PERSPECTIVE,
               base_depth=None, plane_tol=ppcfg.PLANE_TOL,
               board_tone=ppcfg.BOARD_TONE, board_margin=ppcfg.BOARD_MARGIN,
               board_shadow=ppcfg.BOARD_SHADOW,
               defocus_size_ratio=ppcfg.DEFOCUS_SIZE_RATIO):
    self.config = {
        "distortion": distortion,
        "normalize": normalize,
        "area_range": area_range
    }
    self.quads_select_rate = quads_select_rate
    # 数字融合 / 景深失焦 配置
    self.contrast_range = contrast_range
    self.fuse_alpha = fuse_alpha
    self.defocus_strength = defocus_strength
    self.defocus_max_sigma = defocus_max_sigma
    self.persp_spread = persp_spread
    self.base_depth = base_depth
    self.plane_tol = plane_tol
    # 板面渲染配置
    self.board_tone = board_tone
    self.board_margin = board_margin
    self.board_shadow = board_shadow
    self.defocus_size_ratio = defocus_size_ratio
    # AREA_RANGE 描述的是「数字四边形」面积；板位要比它大 (1-2*margin)^-2 倍，
    # 这样 inset_quad 内缩之后，数字面积正好落在 AREA_RANGE 里
    self.area_scale = (1.0 - 2.0 * float(board_margin)) ** 2

  def detect_quads(self, env_img):
    """检测环境图上的平坦区域（同一张图结果确定，批量生成时可缓存复用）

    Args:
        env_img (NDArray[np.uint8]): 环境图（BGR 或灰度）

    Returns:
        list[NDArray[np.float64]]: 平坦区域四边形列表，可能为空
    """
    gray = env_img if env_img.ndim == 2 else cv2.cvtColor(
        env_img, cv2.COLOR_BGR2GRAY)
    return get_flat_quad(gray)

  def process(self, _font_imgs, env_img, show_cnt=False, quads=None,
              digit_ids=None):
    """把数字贴到环境图的平面上，返回 (灰度合成图, 标签列表)

    标签按“板面 → 数字内缩四边形”的顺序给出，每个元素包含：
    ``digit``（数字类别，取自 ``digit_ids``）、``quad``（数字的透视四边形，
    [左上, 右上, 右下, 左下]，绝对像素坐标）、``quad_norm``（归一化到 0~1）、
    ``board_quad``（板面四边形）、``depth`` / ``sigma``（合成时用的深度与失焦量）。

    Args:
        _font_imgs (list[NDArray[np.uint8]]): 白字黑底的数字贴图池
        env_img (NDArray[np.uint8]): 环境图（BGR 或灰度）
        show_cnt (bool, optional): 是否把中间结果写进日志目录。Defaults to False.
        quads (list[NDArray[np.float64]] | None, optional): 预先算好的平坦区域（可缓存复用）；
            缺省时在本次调用里检测。Defaults to None.
        digit_ids (list[int] | None, optional): ``_font_imgs`` 每张图对应的数字类别
            （一个数字可以有多个字形变体）；缺省时直接用池下标，即「池下标 = 类别」。
            Defaults to None.

    Returns:
        tuple[NDArray[np.uint8] | None, list[dict]]: (灰度合成图, 标签)；
        找不到可用平面时返回 ``(None, [])``
    """
    gray = env_img if env_img.ndim == 2 else cv2.cvtColor(
        env_img, cv2.COLOR_BGR2GRAY)

    # 缺省契约：池下标就是数字类别；池子里有同数字的多个字形时必须显式传 digit_ids
    if digit_ids is None:
      digit_ids = list(range(len(_font_imgs)))
    elif len(digit_ids) != len(_font_imgs):
      raise ValueError(
          f"digit_ids 长度({len(digit_ids)})与贴图池({len(_font_imgs)})不一致")

    if quads is None:
      quads = get_flat_quad(gray)
    if not quads:
      print("未检测到符合条件的平面区域")
      return None, []

    segments = [s for q in quads
                for s in segment_quad(q, **self.config,
                                      area_scale=self.area_scale)]
    if not segments:
      print("分割后没有符合 distortion/area 条件的四边形")
      return None, []

    # 随机选择若干个板位；数字也从贴图池里随机取（类别由 digit_ids 给出）
    boxed_text_count = min(len(segments), int(
        len(segments) * self.quads_select_rate + 1))
    seg_idx = random_choose(list(range(len(segments))), boxed_text_count)
    segments = [segments[i] for i in seg_idx]
    font_idx = random_choose(list(range(len(_font_imgs))), boxed_text_count)
    font_imgs = [_font_imgs[i] for i in font_idx]

    # 数字贴在板面内部（留边），并按内缩四边形做透视变换
    digit_quads = [inset_quad(seg.quad, self.board_margin) for seg in segments]
    traned_font_imgs = [
        np.array(proj_tran(_img, quad2M(dq, *_img.shape)))
        for _img, dq in zip(font_imgs, digit_quads)
    ]

    # 画板 + 融合 + 按基准平面做景深失焦
    fused, infos = fuse_scene(
        gray, traned_font_imgs, segments,
        contrast=self.contrast_range, alpha=self.fuse_alpha,
        base_depth=self.base_depth, plane_tol=self.plane_tol,
        defocus_strength=self.defocus_strength,
        max_sigma=self.defocus_max_sigma,
        persp_spread=self.persp_spread,
        digit_quads=digit_quads,
        board_tone=self.board_tone,
        board_shadow=self.board_shadow,
        size_ratio=self.defocus_size_ratio)

    h, w = fused.shape[:2]
    labels = []
    for fidx, info in zip(font_idx, infos):
      q = np.asarray(info["digit_quad"], dtype=np.float64)
      labels.append({
          "digit": int(digit_ids[fidx]),
          "quad": [[round(float(x), 2), round(float(y), 2)] for x, y in q],
          "quad_norm": [[round(float(x) / max(w - 1, 1), 4),
                         round(float(y) / max(h - 1, 1), 4)] for x, y in q],
          "board_quad": [[round(float(x), 2), round(float(y), 2)]
                         for x, y in np.asarray(info["quad"], dtype=np.float64)],
          "center": [round(float(q[:, 0].mean()), 2),
                     round(float(q[:, 1].mean()), 2)],
          "depth": round(float(info["depth"]), 4),
          "sigma": round(float(info["sigma"]), 3),
      })

    # DEBUG: 显示板面 / 数字四边形与失焦强度
    if show_cnt:
      vis_env = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
      for info in infos:
        # 绿 = 基准平面（清晰），红 = 非基准平面（按 sigma 失焦）
        color = (0, 255, 0) if info["on_base"] else (0, 0, 255)
        board = np.round(np.asarray(info["quad"])).astype(np.int32)
        digit = np.round(np.asarray(info["digit_quad"])).astype(np.int32)
        cv2.polylines(vis_env, [board], True, color, 1, cv2.LINE_AA)
        cv2.polylines(vis_env, [digit], True, (255, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(vis_env, f"s={info['sigma']:.1f}",
                    (int(info["offset"][0]), max(
                        int(info["offset"][1]) - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
      cv2.imencode('.png', vis_env)[1].tofile(
          f"{_save_dir}/{vis_env.shape}_env_quads.png")
      cv2.imencode('.png', fused)[1].tofile(
          f"{_save_dir}/{fused.shape}_env_fused.png")

      for i, _ in enumerate(traned_font_imgs):
        # proj_tran 的输出画布 = 目标 quad 的外接矩形, 且已把坐标平移 (0,0),
        # 所以要先减去外接矩形左上角, 得到画布内的局部坐标才能画上框
        q = digit_quads[i]
        q_local = np.round(q - q.min(axis=0)).astype(np.int32)
        det = np.array(traned_font_imgs[i], copy=True)
        if det.ndim == 2:
          det = cv2.cvtColor(det, cv2.COLOR_GRAY2BGR)
        cv2.polylines(det, [q_local], True, (0, 0, 255), 1, cv2.LINE_AA)
        _img = hstack_images([font_imgs[i], det], pad=5)
        cv2.imencode('.png', _img)[1].tofile(
            f"{_save_dir}/{_img.shape}_projed_{i}.png")

    return fused, labels


if __name__ == "__main__":
  PP = Preprocess(ppcfg.DISTORTION, ppcfg.NORMALIZE, ppcfg.AREA_RANGE,
                  ppcfg.QUADS_SELECT_RATE,
                  ppcfg.CONTRAST_RANGE, ppcfg.FUSE_ALPHA, ppcfg.DEFOCUS_STRENGTH,
                  ppcfg.DEFOCUS_MAX_SIGMA, ppcfg.DEFOCUS_PERSPECTIVE,
                  ppcfg.BASE_DEPTH, ppcfg.PLANE_TOL,
                  ppcfg.BOARD_TONE, ppcfg.BOARD_MARGIN, ppcfg.BOARD_SHADOW,
                  ppcfg.DEFOCUS_SIZE_RATIO)
  env_img_path = pathcfg.RAW_DIR / "env/0072.jpg"
  font_imgs = [font2img(get_font_path(0), str(i), thickness=1)
               for i in range(10)]
  env_img = cv2.imdecode(np.fromfile(
      str(env_img_path), dtype=np.uint8), cv2.IMREAD_COLOR)

  fused, labels = PP.process(font_imgs, env_img, True)
  if fused is not None:
    img_path, lbl_path = save_sample(
        pathcfg.PROCESSED_DIR / "fused", env_img_path.stem, fused, labels,
        extra={"source": env_img_path.name})
    print(f"样本: {img_path}\n标签: {lbl_path} ({len(labels)} 个数字)")
