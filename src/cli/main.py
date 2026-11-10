# src/cli/main.py
import argparse
import sys

from src.inference import run        # ✅ 顶层包绝对导入
from src.utils import transform       # 需要的话直接 import，也可以用


def build_parser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(prog="mypkg", description="xxx 推理入口")
  p.add_argument("--input", "-i", required=True, help="输入路径")
  p.add_argument("--output", "-o", required=True, help="输出路径")
  p.add_argument("--verbose", "-v", action="store_true")
  return p


def main(argv: list[str] | None = None) -> int:
  args = build_parser().parse_args(argv)
  try:
    run(args.input, args.output, verbose=args.verbose)
  except Exception as e:
    print(f"[digits_tracker] 失败: {e}", file=sys.stderr)
    return 1
  return 0


if __name__ == "__main__":
  sys.exit(main())