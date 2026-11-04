"""輸出所有預設濾波器的 PSNR 比較表（Markdown 格式），方便貼進報告。

用法（在專案根目錄）：
    python tools/psnr_report.py samples/coffee.png
    python tools/psnr_report.py 你的照片.jpg --max-side 2048 --border replicate
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from imgtool import filtering, io_utils  # noqa: E402

NORMALIZE = {"none": "—", "sum": "÷ 總和", "custom": "÷ 自訂"}
OUTPUT = {"clip": "截斷", "abs": "絕對值", "minmax": "線性拉伸"}


def main() -> None:
    parser = argparse.ArgumentParser(description="批次計算所有預設濾波器的 PSNR")
    parser.add_argument("image", help="影像路徑（PNG / JPG / BMP）")
    parser.add_argument("--max-side", type=int, default=2048, help="處理解析度上限（長邊像素），預設 2048")
    parser.add_argument("--corr", action="store_true", help="改用相關運算（不翻轉核），預設為卷積")
    parser.add_argument("--border", default="reflect101", choices=list(filtering.BORDER_MODES), help="邊界處理方式")
    args = parser.parse_args()

    path = Path(args.image)
    rgb, info = io_utils.load_image(path.read_bytes(), path.name)
    rgb, _ = io_utils.limit_size(rgb, args.max_side)
    height, width = rgb.shape[:2]
    print(f"影像：{info.file_name}（原始 {info.width}×{info.height}，處理尺寸 {width}×{height}）")
    print(f"設定：{'相關運算' if args.corr else '卷積'}、邊界 {args.border}\n")
    print("| 濾波器 | 尺寸 | 正規化 | 偏移 | 輸出映射 | PSNR (dB) | MSE |")
    print("|---|:---:|:---:|:---:|:---:|---:|---:|")
    for preset in filtering.PRESETS:
        kernel = np.array(preset.kernel, dtype=np.float64)
        result = filtering.apply_filter(
            rgb, kernel, filtering.params_for(preset, flip=not args.corr, border=args.border)
        )
        value = filtering.psnr(rgb, result.image)
        shown = "∞" if math.isinf(value) else f"{value:.2f}"
        bias = f"+{preset.bias:g}" if preset.bias else "—"
        print(
            f"| {preset.name} | {kernel.shape[0]}×{kernel.shape[1]} | {NORMALIZE[preset.normalize]} | {bias} | "
            f"{OUTPUT[preset.output]} | {shown} | {filtering.mse(rgb, result.image):.2f} |"
        )


if __name__ == "__main__":
    main()
