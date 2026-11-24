"""测试集推理 + 可视化：把 GT（蓝）与预测（绿）画在同一张图上，输出到 ``dataset/log/infer``。

用法::

    conda activate NNVN
    python src/cli/target_inference.py                     # 全部测试样本
    python src/cli/target_inference.py --limit 8 --thr .3  # 只画前 8 张
    python src/cli/target_inference.py -i 3                # 只跑测试集第 4 张
    python src/cli/target_inference.py --name 0032         # 只跑文件名包含 0032 的那张
    python src/cli/target_inference.py -p D:/a/b.jpg       # 任意图片（无 GT，只画预测）
    python src/cli/target_inference.py -p a.jpg b.png      # 多张
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from configs import path as pathcfg
from configs.log import UUID as _uuid
from configs.path import SELECT_MODEL, SELECT_IMGSET, DATASET_LABEL_JSON
from configs.training_cfg import TRAININGSET_RATIO, DROPOUT
# _MEAN / _STD 与 DigitDataset 共用，保证自定义图片的归一化完全一致
from src.training.dataset import (DigitDataset, collate_fn,
                                  _MEAN as _DS_MEAN, _STD as _DS_STD)
from src.training.model import CenterNetDigit
from src.inference.infer import Inferer
from src.utils.debug_tools import draw_quads, save_image, tensor2img

# 可视化产物目录
OUT_DIR = pathcfg.LOG_DIR / "infer" / _uuid
# 画框时跳过的边：第 3、4 个点（0-based 2、3）之间不连线
SKIP_EDGES = {(2, 3)}

device = "cuda" if torch.cuda.is_available() else "cpu"
# dropout 必须与训练时一致：结构不同（有没有 Dropout2d 层）会导致
# load_state_dict(strict=True) 键名对不上
model = CenterNetDigit(num_classes=10, dropout=DROPOUT)
inferrer = Inferer(model, SELECT_MODEL, device=device)
# 必须传 TRAININGSET_RATIO：默认 1.0 会把全部样本划给训练集，测试集为空
# （数据集里已加校验，这种情况会直接报错而不是静默返回 0 条）
ds = DigitDataset(DATASET_LABEL_JSON, SELECT_IMGSET,
                  TRAININGSET_RATIO, mode="testing")
dl = DataLoader(ds, batch_size=4, collate_fn=collate_fn)


def load_image_tensor(path, img_size=(384, 384)) -> torch.Tensor:
  """读任意图片文件 → 与 ``DigitDataset`` 完全一致的 ``(1, 3, H, W)`` 张量

    先转灰度再复制成 3 通道（训练集样本就是灰度 PNG），Resize / Normalize
    的参数也从 ``dataset.py`` 拿来，避免自定义图片与训练分布不一致。

    Args:
        path (Path | str): 图片路径（支持中文路径）
        img_size (tuple[int, int]): 送进网络的 ``(H, W)``，必须与训练一致

    Returns:
        torch.Tensor: ``(1, 3, H, W)`` float32

    Raises:
        ValueError: 图片读不出来
    """
  buf = np.fromfile(str(path), dtype=np.uint8)
  gray = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
  if gray is None:
    raise ValueError(f"图片读取失败: {path}")

  h, w = int(img_size[0]), int(img_size[1])
  gray = cv2.resize(gray, (w, h), interpolation=cv2.INTER_LINEAR)
  img = gray.astype(np.float32) / 255.0
  img = np.repeat(img[None], 3, axis=0)
  img = (img - _DS_MEAN) / _DS_STD
  return torch.from_numpy(np.ascontiguousarray(img)).unsqueeze(0)


def visualize_files(paths, thr=0.3, topk=30, out_dir=OUT_DIR):
  """对任意图片文件跑推理（没有 GT，只画预测框）

  Args:
      paths (Sequence[Path | str]): 图片路径列表
      thr (float): 中心分数阈值
      topk (int): 每张图最多保留多少个中心
      out_dir (Path): 输出目录，默认 ``dataset/log/infer``

  Returns:
      tuple[Path, int, int]: (输出目录, 保存张数, 检出框总数)
  """
  out_dir = Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  saved, n_det = 0, 0
  for p in paths:
    p = Path(p)
    if not p.exists():
      raise ValueError(f"图片不存在: {p}")
    tensor = load_image_tensor(p)
    pts, digits, scores = inferrer.predict(tensor, thr=thr, topk=topk)[0]

    vis = tensor2img(tensor[0])          # 反归一化回 384x384 的 BGR 图
    vis = draw_quads(vis, pts.numpy(), digits.tolist(), scores.tolist(),
                     color=(0, 255, 0), tag="pred",
                     skip_edges=SKIP_EDGES)   # 绿：预测（无 GT）
    save_image(vis, out_dir / f"{_uuid}_{saved:03d}_{p.stem}.png")
    print(f"[{saved + 1}] {p} -> 预测 {len(digits)} 个")
    saved += 1
    n_det += len(digits)

  return out_dir, saved, n_det


def resolve_index(ds, index=None, name=None) -> int:
  """定位测试集里的某一张：优先按文件名关键字，其次按序号

  Args:
      ds (DigitDataset): 测试集
      index (int | None): 序号（0 起，支持负数）
      name (str | None): 文件名（或其中一段）

  Returns:
      int: 可直接索引 ``ds`` 的下标

  Raises:
      ValueError: 序号越界，或没有文件名匹配
  """
  total = len(ds)
  if name:
    hits = [i for i, s in enumerate(ds.samples) if str(name) in s["image"]]
    if not hits:
      raise ValueError(
          f"测试集里没有文件名包含 {name!r} 的样本（共 {total} 张）")
    if len(hits) > 1:
      print(f"匹配到 {len(hits)} 张，使用第一张：{ds.samples[hits[0]]['image']}")
    return hits[0]

  idx = 0 if index is None else int(index)
  if not -total <= idx < total:
    raise ValueError(f"--index 应在 {-total}~{total - 1} 之间，收到 {idx}")
  return idx % total


def visualize(limit=None, thr=0.3, topk=30, out_dir=OUT_DIR,
              index=None, name=None):
  """跑测试集（或其中一张），把 GT 与预测画到同一张图并存盘

  Args:
      limit (int | None): 最多处理多少张（None 表示全部）
      thr (float): 中心分数阈值
      topk (int): 每张图最多保留多少个中心
      out_dir (Path): 输出目录，默认 ``dataset/log/infer``
      index (int | None): 只跑这一张（测试集内的序号，0 起）
      name (str | None): 只跑文件名包含该字符串的那一张

  Returns:
      tuple[Path, int, int]: (输出目录, 保存张数, 检出框总数)
  """
  out_dir = Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  # 指定了单张时，直接取那一条样本组成一个 batch，不用跑完整个测试集
  single = index is not None or name is not None
  if single:
    start = resolve_index(ds, index, name)
    batches = [collate_fn([ds[start]])]
    offsets = [start]
  else:
    batches = dl
    offsets = None

  saved, n_det = 0, 0
  for bi, (imgs, gt_quads, gt_digits) in enumerate(batches):
    base = offsets[bi] if offsets is not None else bi * dl.batch_size
    preds = inferrer.predict(imgs, thr=thr, topk=topk)   # predict 内部已 no_grad()
    for b, (pts, digits, scores) in enumerate(preds):
      idx = base + b
      if limit is not None and saved >= limit:
        return out_dir, saved, n_det
      if idx >= len(ds):
        return out_dir, saved, n_det

      vis = tensor2img(imgs[b])                        # 归一化张量 -> BGR 图
      vis = draw_quads(vis, gt_quads[b].numpy(), gt_digits[b].tolist(),
                       color=(255, 0, 0), tag="gt", label_pos="bottom",
                       skip_edges=SKIP_EDGES)  # 蓝：真值（不连第 3、4 点）
      vis = draw_quads(vis, pts.numpy(), digits.tolist(), scores.tolist(),
                       color=(0, 255, 0), tag="pred",
                       skip_edges=SKIP_EDGES)  # 绿：预测（不连第 3、4 点）

      stem = Path(ds.samples[idx]["image"]).stem
      save_image(vis, out_dir / f"{_uuid}_{idx:03d}_{stem}.png")
      print(f"[{saved + 1}] {ds.samples[idx]['image']} -> "
            f"GT {len(gt_digits[b])} 个 / 预测 {len(digits)} 个")
      saved += 1
      n_det += len(digits)

  return out_dir, saved, n_det

def main(argv=None):
  """命令行入口，返回进程退出码"""
  parser = argparse.ArgumentParser(description="测试集推理 + 可视化")
  parser.add_argument("-l", "--limit", type=int, default=None,
                      help="最多处理多少张（默认全部）")
  parser.add_argument("-i", "--index", type=int, default=None,
                      help="只测测试集里的第几张（0 起，支持负数）")
  parser.add_argument("--name", default=None,
                      help="只测文件名包含该字符串的那一张（如 0032）")
  parser.add_argument("--thr", type=float, default=0.3, help="中心分数阈值")
  parser.add_argument("--topk", type=int, default=30, help="每张图最大检出数")
  parser.add_argument("-p", "--image", nargs="+", default=None,
                      help="直接指定图片文件路径（可多张；只画预测，没有 GT）")
  args = parser.parse_args(argv)

  try:
    if args.image:
      # 自定义图片优先：忽略 --index / --name / --limit
      out_dir, saved, n_det = visualize_files(args.image, args.thr, args.topk)
    else:
      out_dir, saved, n_det = visualize(args.limit, args.thr, args.topk,
                                        index=args.index, name=args.name)
  except ValueError as exc:
    print(exc)
    return 1
  print(f"已保存 {saved} 张可视化（共 {n_det} 个检测框）-> {out_dir}")

  return 0


if __name__ == "__main__":
  sys.exit(main())