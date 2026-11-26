"""摄像头实时检测入口：用 cv2 采集画面，CenterNet 实时识别数字并叠加显示。

用法::

    conda activate NNVN
    python src/cli/realtime_infer.py                    # 默认摄像头 + configs/camera_cfg.py
    python src/cli/realtime_infer.py -c 1 --thr 0.4     # 换摄像头 / 调阈值
    python src/cli/realtime_infer.py --no-show -n 100   # 不弹窗，跑 100 帧（无显示器环境）
    python src/cli/realtime_infer.py --save             # 按 s 把画面存到 dataset/log/camera/<uuid>

按键：`q` / `ESC` 退出，`s` 保存当前帧快照，`r` / `e` 顺时针 / 逆时针旋转 90°
（旋转同时作用于显示画面和送进模型的图像），`h` 开 / 关热力图预览窗口。

预览默认 `--view crop`：窗口里是**居中裁切后的正方形**（和模型真正的输入一样），
不再是 16:9 的长方形；`--view model` 看缩放到 384 后模型实际看到的画面，
`--view full` 看完整画面（裁切区域以外压暗）。

`--heatmap` 额外弹一个窗口显示中心热度图：**固定色标 0~1**，并在左上角给出
`max / mean / 超阈值格子数`——用来判断热力图是不是「恒等于一个常数」（塔缩）。
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from configs.camera_cfg import CameraConfig as ccfg
from configs.log import LogConfig
from configs.path import PathConfig
from configs.realtime_cfg import RealtimeConfig as rtcfg
from configs.training_cfg import TrainingConfig
from src.inference.infer import Inferer
from src.serving.camera import CameraDetector
from src.training.model import CenterNetDigit
from src.utils.logging_tools import setup_logging


def build_detector(args: argparse.Namespace) -> CameraDetector:
  """按命令行参数组装 Inferer + CameraDetector"""
  device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
  # dropout 必须与训练时一致，否则 head 结构不同、权重加载会键名错位
  model = CenterNetDigit(num_classes=10, dropout=TrainingConfig.DROPOUT)
  inferrer = Inferer(model, PathConfig.SELECT_MODEL, device=device)

  save_dir = ((PathConfig.LOG_DIR / "camera" / LogConfig.UUID)
              if args.save else None)
  return CameraDetector(
      inferrer,
      index=args.camera,
      width=ccfg.FRAME_WIDTH, height=ccfg.FRAME_HEIGHT,
      thr=args.thr, topk=args.topk,
      show=not args.no_show,
      window_name=ccfg.WINDOW_NAME,
      max_fps=args.max_fps,
      save_dir=save_dir,
      thickness=ccfg.DRAW_THICKNESS,
      font_scale=ccfg.FONT_SCALE,
      rotate=args.rotate,
      input_mode=args.input_mode,
      view=args.view,
      show_heatmap=args.heatmap)


def main(argv: list[str] | None = None) -> int:
  """命令行入口，返回进程退出码"""
  parser = argparse.ArgumentParser(description="摄像头实时数字检测")
  parser.add_argument("-c", "--camera", type=int, default=ccfg.CAMERA_INDEX,
                      help="摄像头序号（默认取 camera_cfg.CAMERA_INDEX）")
  parser.add_argument("--thr", type=float, default=ccfg.THR, help="中心分数阈值")
  parser.add_argument("--topk", type=int, default=ccfg.TOPK, help="每帧最大检出数")
  parser.add_argument("--max-fps", type=float, default=ccfg.MAX_FPS,
                      help="限制处理帧率（0 表示不限）")
  parser.add_argument("-n", "--frames", type=int, default=None,
                      help="处理多少帧后自动退出（默认一直跑）")
  parser.add_argument("--no-show", action="store_true", help="不弹窗（服务器 / 无显示器）")
  parser.add_argument("--save", action="store_true",
                      help="开启快照：按 s 存到 dataset/log/camera/<uuid>")
  parser.add_argument("--device", default=None, help="cuda / cpu，缺省自动选择")
  parser.add_argument("--rotate", type=int, default=ccfg.ROTATE, choices=(0, 90, 180, 270),
                      help="初始顺时针旋转角度（运行中按 r/e 调整），显示与推理共用")
  parser.add_argument("--input-mode", default=rtcfg.INPUT_MODE,
                      choices=("gray", "rgb"),
                      help="送进模型的通道模式：gray=灰度（同训练样本，默认） / rgb=彩色")
  parser.add_argument("--view", default=rtcfg.VIEW_MODE,
                      choices=("crop", "model", "full"),
                      help="预览显示什么：crop=裁切后的正方形（默认，同模型输入）"
                           " / model=模型真正看到的 384 画面 / full=整幅画面（框外压暗）")
  parser.add_argument("--heatmap", action=argparse.BooleanOptionalAction,
                      default=rtcfg.SHOW_HEATMAP,
                      help="是否额外显示中心热力图窗口（默认取 realtime_cfg.SHOW_HEATMAP）；"
                           "运行时按 h 也能切换")
  args = parser.parse_args(argv)

  setup_logging()      # src 模块只发 logger，格式 / 级别在这里定
  if args.heatmap and args.no_show:
    print("提示：--no-show 下不会弹窗，热力图预览自动失效")
  detector = build_detector(args)
  with detector:
    if not detector.is_open:
      print(f"打开摄像头失败（index={args.camera}）："
            "可能被其它程序占用，或用 -c 换一个序号（0/1/2…）")
      return 1
    n = detector.run(max_frames=args.frames)

  print(f"共处理 {n} 帧，平均处理帧率 {detector.fps:.1f} FPS")
  return 0


if __name__ == "__main__":
  sys.exit(main())
