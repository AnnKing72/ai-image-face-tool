"""功能 1：影像載入與基礎解析。

以 Pillow 解碼 PNG / JPEG / BMP，統一轉成 RGB uint8 陣列 (H, W, 3) 供後續模組使用，
並回傳原始檔案的中繼資訊（解析度、色彩維度、檔案大小等）。
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

SUPPORTED_EXTENSIONS = ("png", "jpg", "jpeg", "bmp")

# Pillow 回報的格式名稱 → 顯示名稱（MPO 是部分相機輸出的多圖 JPEG，DIB 是無檔頭 BMP）
_ACCEPTED_FORMATS = {"PNG": "PNG", "JPEG": "JPEG", "MPO": "JPEG", "BMP": "BMP", "DIB": "BMP"}

# 約 6000 萬像素：防止解壓縮炸彈，也避免雲端主機記憶體不足
MAX_PIXELS = 60_000_000

# Pillow 色彩模式 → (通道數, 每通道位元數, 中文說明)
_MODE_TABLE: dict[str, tuple[int, int, str]] = {
    "1": (1, 1, "黑白二值"),
    "L": (1, 8, "灰階"),
    "LA": (2, 8, "灰階 + Alpha 透明"),
    "La": (2, 8, "灰階 + 預乘 Alpha"),
    "P": (1, 8, "調色盤索引色"),
    "PA": (2, 8, "調色盤索引色 + Alpha"),
    "RGB": (3, 8, "RGB 全彩"),
    "RGBA": (4, 8, "RGB + Alpha 透明"),
    "RGBa": (4, 8, "RGB + 預乘 Alpha"),
    "RGBX": (4, 8, "RGB + 填充位元"),
    "CMYK": (4, 8, "CMYK 印刷四色"),
    "YCbCr": (3, 8, "YCbCr"),
    "I": (1, 32, "32-bit 整數灰階"),
    "I;16": (1, 16, "16-bit 灰階"),
    "I;16L": (1, 16, "16-bit 灰階"),
    "I;16B": (1, 16, "16-bit 灰階"),
    "I;16N": (1, 16, "16-bit 灰階"),
    "F": (1, 32, "32-bit 浮點灰階"),
}

_ALPHA_MODES = {"LA", "La", "PA", "RGBA", "RGBa"}
_HIGH_DEPTH_MODES = {"I", "I;16", "I;16L", "I;16B", "I;16N", "F"}


class ImageLoadError(ValueError):
    """讀檔失敗；訊息可以直接顯示給使用者。"""


@dataclass(frozen=True)
class ImageInfo:
    """原始影像檔的中繼資訊（尺寸為套用 EXIF 方向校正之後的結果）。"""

    file_name: str
    file_size: int
    format: str
    width: int
    height: int
    mode: str
    mode_description: str
    channels: int
    bits_per_channel: int
    has_alpha: bool
    exif_transposed: bool
    dpi: tuple[float, float] | None
    notes: tuple[str, ...] = ()

    @property
    def pixel_count(self) -> int:
        return self.width * self.height

    @property
    def bits_per_pixel(self) -> int:
        return self.channels * self.bits_per_channel

    @property
    def raw_size(self) -> int:
        """未壓縮時的資料量（bytes）= 寬 × 高 × 每像素位元數 ÷ 8。"""
        return (self.pixel_count * self.bits_per_pixel + 7) // 8

    @property
    def compression_ratio(self) -> float:
        return self.raw_size / self.file_size if self.file_size else float("nan")


def human_size(num_bytes: int) -> str:
    """把位元組數轉成易讀字串（1 KB = 1024 B）。"""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} GB"


def load_image(data: bytes, file_name: str = "image") -> tuple[np.ndarray, ImageInfo]:
    """解碼影像檔位元組，回傳 (RGB uint8 陣列, ImageInfo)。

    任何解碼失敗都會轉成 ImageLoadError，呼叫端只需要處理這一種例外。
    """
    if not data:
        raise ImageLoadError("檔案是空的（0 bytes），請重新選擇影像。")
    try:
        with Image.open(io.BytesIO(data)) as img:
            fmt = _ACCEPTED_FORMATS.get(img.format or "")
            if fmt is None:
                raise ImageLoadError(
                    f"不支援的影像格式「{img.format}」，請上傳 PNG、JPG/JPEG 或 BMP。"
                )
            width, height = img.size
            if width * height > MAX_PIXELS:
                raise ImageLoadError(
                    f"影像解析度 {width}×{height}（約 {width * height / 1e6:.0f} 百萬像素）"
                    f"超過上限 {MAX_PIXELS / 1e6:.0f} 百萬像素，請先縮小後再上傳。"
                )
            mode = img.mode
            dpi = img.info.get("dpi")
            has_alpha = mode in _ALPHA_MODES or (mode == "P" and "transparency" in img.info)
            orientation = img.getexif().get(0x0112, 1)
            img.load()  # 立即完整解碼，截斷或損毀的檔案會在這裡丟出例外
            oriented = ImageOps.exif_transpose(img)
            rgb, notes = _to_rgb_uint8(oriented)
    except ImageLoadError:
        raise
    except Image.DecompressionBombError as exc:
        raise ImageLoadError("影像像素數量異常龐大（疑似解壓縮炸彈），已拒絕讀取。") from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, EOFError) as exc:
        raise ImageLoadError(
            "無法解析影像內容：檔案可能已損毀、未完整上傳，或副檔名與實際格式不符。"
        ) from exc

    exif_transposed = orientation in (2, 3, 4, 5, 6, 7, 8)
    if exif_transposed:
        notes.insert(0, f"已依 EXIF 方向標記（Orientation = {orientation}）自動轉正。")

    channels, bits, description = _MODE_TABLE.get(mode, (rgb.shape[2], 8, mode))
    info = ImageInfo(
        file_name=file_name,
        file_size=len(data),
        format=fmt,
        width=int(rgb.shape[1]),
        height=int(rgb.shape[0]),
        mode=mode,
        mode_description=description,
        channels=channels,
        bits_per_channel=bits,
        has_alpha=has_alpha,
        exif_transposed=exif_transposed,
        dpi=tuple(float(v) for v in dpi) if isinstance(dpi, tuple) and len(dpi) == 2 else None,
        notes=tuple(notes),
    )
    return rgb, info


def _to_rgb_uint8(img: Image.Image) -> tuple[np.ndarray, list[str]]:
    """把任意 Pillow 色彩模式轉成 RGB uint8，並記錄做了哪些轉換。"""
    notes: list[str] = []
    mode = img.mode

    if mode in _HIGH_DEPTH_MODES:
        arr = np.asarray(img).astype(np.float64)
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
        if mode == "F" or lo < 0 or hi > 65535:
            arr = (arr - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(arr)
            notes.append("高位元深度灰階已依最小值–最大值線性拉伸至 0–255。")
        elif hi > 255 or mode != "I":
            arr = arr / 257.0  # 65535 ÷ 257 = 255，線性對應到 8-bit
            notes.append("16-bit 灰階已線性縮放成 8-bit（數值 ÷ 257）。")
        gray = np.clip(np.rint(np.nan_to_num(arr)), 0, 255).astype(np.uint8)
        return np.repeat(gray[:, :, None], 3, axis=2), notes

    if mode in _ALPHA_MODES or (mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        if np.asarray(rgba)[:, :, 3].min() < 255:
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            rgb_img = Image.alpha_composite(background, rgba).convert("RGB")
            notes.append("影像含透明區域（Alpha 通道），已以白色背景合成後再處理。")
        else:
            rgb_img = rgba.convert("RGB")
            notes.append("影像含 Alpha 通道但完全不透明，已直接移除 Alpha。")
    else:
        rgb_img = img.convert("RGB")
        if mode == "CMYK":
            notes.append("CMYK 色彩已轉換為 RGB。")
        elif mode in ("L", "1"):
            notes.append("灰階影像已複製成 R = G = B 三通道，方便統一處理。")
        elif mode == "P":
            notes.append("調色盤（索引色）影像已展開成 RGB。")
        elif mode != "RGB":
            notes.append(f"{mode} 色彩模式已轉換為 RGB。")
    return np.array(rgb_img, dtype=np.uint8), notes


def limit_size(rgb: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """長邊超過 max_side 時等比例縮小（INTER_AREA 抗鋸齒），回傳 (影像, 縮放比例)。"""
    height, width = rgb.shape[:2]
    longest = max(height, width)
    if max_side <= 0 or longest <= max_side:
        return rgb, 1.0
    scale = max_side / longest
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(rgb, size, interpolation=cv2.INTER_AREA), scale


def to_png_bytes(rgb: np.ndarray) -> bytes:
    """把 RGB / 灰階 uint8 陣列編碼成 PNG（供下載按鈕使用）。"""
    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb)).save(buffer, format="PNG")
    return buffer.getvalue()
