"""產生 samples/ 內的範例影像與邊界條件測試檔。

用法（在專案根目錄）：python tools/make_samples.py
產生的檔案已經放在儲存庫裡，一般不需要重新執行。
咖啡與貓的照片取自 scikit-image 內建資料集（CC0 授權），需先 pip install scikit-image。
"""

from __future__ import annotations

import io
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
EDGE = SAMPLES / "edge_cases"


def color_chart(width: int = 640, height: int = 480) -> np.ndarray:
    """色彩測試圖：飽和色條、灰階漸層、色相漸層、RGB 單色漸層與棋盤格／幾何圖形。"""
    img = np.zeros((height, width, 3), np.uint8)
    bars = [(255, 255, 255), (255, 255, 0), (0, 255, 255), (0, 255, 0),
            (255, 0, 255), (255, 0, 0), (0, 0, 255), (0, 0, 0)]
    bar_w = width // len(bars)
    for i, c in enumerate(bars):
        img[0:150, i * bar_w:(i + 1) * bar_w] = c
    ramp = np.linspace(0, 255, width).round().astype(np.uint8)
    img[150:200] = ramp[None, :, None]
    hsv = np.zeros((1, width, 3), np.uint8)
    hsv[..., 0] = np.linspace(0, 179, width).round().astype(np.uint8)
    hsv[..., 1:] = 255
    img[200:250] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    half = width // 2
    band = (height - 250) // 3
    for i in range(3):
        y0 = 250 + i * band
        y1 = height if i == 2 else y0 + band
        img[y0:y1, :half, i] = np.linspace(0, 255, half).round().astype(np.uint8)
    yy, xx = np.indices((height - 250, width - half))
    checker = (((yy // 20) + (xx // 20)) % 2 * 200 + 30).astype(np.uint8)
    img[250:, half:] = checker[..., None]
    sub = np.ascontiguousarray(img[250:, half:])
    cv2.circle(sub, (160, 115), 70, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.line(sub, (10, 220), (310, 10), (255, 80, 0), 3, cv2.LINE_AA)
    cv2.putText(sub, "YUV", (95, 135), cv2.FONT_HERSHEY_DUPLEX, 1.8, (255, 220, 0), 3, cv2.LINE_AA)
    img[250:, half:] = sub
    return img


def save(img: Image.Image, path: Path, **kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, **kwargs)
    print(f"  {path.relative_to(ROOT)}  ({path.stat().st_size:,} bytes, mode={img.mode}, {img.size[0]}x{img.size[1]})")


def main() -> None:
    SAMPLES.mkdir(exist_ok=True)
    print("範例影像：")
    try:
        import skimage

        data_dir = Path(skimage.__file__).parent / "data"
        for name in ("coffee.png", "chelsea.png"):
            shutil.copyfile(data_dir / name, SAMPLES / name)
            print(f"  samples/{name}（複製自 scikit-image，CC0）")
    except ImportError:
        print("  未安裝 scikit-image，略過 coffee.png、chelsea.png")
    chart = color_chart()
    save(Image.fromarray(chart), SAMPLES / "color_chart.png")

    print("邊界條件測試檔：")
    rng = np.random.default_rng(2024)
    coffee = Image.open(SAMPLES / "coffee.png").convert("RGB") if (SAMPLES / "coffee.png").exists() \
        else Image.fromarray(chart)

    save(Image.new("RGB", (1, 1), (200, 30, 30)), EDGE / "01_tiny_1x1.png")
    save(Image.fromarray(rng.integers(0, 256, (2, 3, 3), dtype=np.uint8)), EDGE / "02_tiny_3x2.bmp")
    g16 = np.tile(np.linspace(0, 65535, 512).round().astype(np.uint16), (128, 1))
    save(Image.fromarray(g16), EDGE / "03_gray_16bit.png")

    rgba = np.zeros((240, 240, 4), np.uint8)
    yy, xx = np.indices((240, 240))
    inside = (yy - 120) ** 2 + (xx - 120) ** 2 < 90 ** 2
    rgba[inside] = (230, 60, 60, 255)
    ring = ((yy - 120) ** 2 + (xx - 120) ** 2 >= 90 ** 2) & ((yy - 120) ** 2 + (xx - 120) ** 2 < 110 ** 2)
    rgba[ring] = (60, 60, 230, 128)  # 半透明外圈
    save(Image.fromarray(rgba, "RGBA"), EDGE / "04_rgba_transparent.png")

    save(coffee.quantize(64), EDGE / "05_palette_8bit.bmp")
    save(coffee.convert("L").point(lambda v: 255 if v > 110 else 0).convert("1"), EDGE / "06_bilevel_1bit.bmp")
    save(coffee.convert("CMYK"), EDGE / "07_cmyk.jpg", quality=92)

    # 像素資料以「逆時針轉 90°」存檔，EXIF Orientation = 6 告訴讀圖程式要順時針轉 90° 才是正的
    upright = np.full((300, 480, 3), 245, np.uint8)
    cv2.arrowedLine(upright, (240, 260), (240, 50), (220, 40, 40), 12, cv2.LINE_AA, tipLength=0.25)
    cv2.putText(upright, "TOP", (20, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (30, 30, 30), 3, cv2.LINE_AA)
    stored = Image.fromarray(np.ascontiguousarray(np.rot90(upright, 1)))
    exif = stored.getexif()
    exif[0x0112] = 6
    save(stored, EDGE / "08_exif_rotate90.jpg", exif=exif.tobytes(), quality=92)

    save(Image.new("RGB", (320, 240), (128, 128, 128)), EDGE / "09_flat_gray.png")

    big_w, big_h = 6000, 4000
    gx = np.linspace(0, 1, big_w, dtype=np.float32)[None, :]
    gy = np.linspace(0, 1, big_h, dtype=np.float32)[:, None]
    large = np.stack([255 * gx * np.ones_like(gy), 255 * gy * np.ones_like(gx), 255 * (1 - gx) * gy], axis=-1)
    large = large.astype(np.uint8)
    cv2.putText(large, "6000 x 4000", (900, 2200), cv2.FONT_HERSHEY_DUPLEX, 18, (255, 255, 255), 40, cv2.LINE_AA)
    save(Image.fromarray(large), EDGE / "10_large_6000x4000.jpg", quality=80)

    buffer = io.BytesIO()
    coffee.save(buffer, format="JPEG", quality=90)
    (EDGE / "11_corrupted.jpg").write_bytes(buffer.getvalue()[: len(buffer.getvalue()) * 2 // 5])
    print("  samples/edge_cases/11_corrupted.jpg（只保留前 40% 位元組）")
    (EDGE / "12_not_an_image.png").write_text("這不是圖片，只是副檔名取名為 .png 的文字檔。\n", encoding="utf-8")
    print("  samples/edge_cases/12_not_an_image.png（文字檔）")


if __name__ == "__main__":
    main()
