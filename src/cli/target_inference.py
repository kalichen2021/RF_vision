"""测试集推理 + 可视化：把 GT（蓝）与预测（绿）画在同一张图上，输出到 ``dataset/log/infer``。

用法::

    conda activate NNVN
    python src/cli/target_inference.py                     # 全部测试样本
    python src/cli/target_inference.py --limit 8 --thr .3  # 只画前 8 张
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from configs import path as pathcfg
from configs.log import UUID as _uuid
from configs.path import SELECT_MODEL, SELECT_IMGSET, DATASET_LABEL_JSON
from configs.training_cfg import TRAININGSET_RATIO, DROPOUT
from src.training.dataset import DigitDataset, collate_fn
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


def visualize(limit=None, thr=0.3, topk=30, out_dir=OUT_DIR):
  """跑一遍测试集，把 GT 与预测画到同一张图并存盘

  Args:
      limit (int | None): 最多处理多少张（None 表示全部）
      thr (float): 中心分数阈值
      topk (int): 每张图最多保留多少个中心
      out_dir (Path): 输出目录，默认 ``dataset/log/infer``

  Returns:
      tuple[Path, int, int]: (输出目录, 保存张数, 检出框总数)
  """
  out_dir = Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  saved, n_det, idx = 0, 0, 0
  for imgs, gt_quads, gt_digits in dl:
    preds = inferrer.predict(imgs, thr=thr, topk=topk)   # predict 内部已 no_grad()
    for b, (pts, digits, scores) in enumerate(preds):
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
      saved += 1
      n_det += len(digits)
      idx += 1

  return out_dir, saved, n_det

def main(argv=None):
  """命令行入口，返回进程退出码"""
  parser = argparse.ArgumentParser(description="测试集推理 + 可视化")
  parser.add_argument("-l", "--limit", type=int, default=None,
                      help="最多处理多少张（默认全部）")
  parser.add_argument("--thr", type=float, default=0.3, help="中心分数阈值")
  parser.add_argument("--topk", type=int, default=30, help="每张图最大检出数")
  args = parser.parse_args(argv)

  out_dir, saved, n_det = visualize(args.limit, args.thr, args.topk)
  print(f"已保存 {saved} 张可视化（共 {n_det} 个检测框）-> {out_dir}")

  return 0


if __name__ == "__main__":
  sys.exit(main())