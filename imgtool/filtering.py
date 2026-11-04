"""功能 3：自訂卷積核空間濾波與 PSNR 量化評估（核心運算全部以 NumPy 實作）。

流程：文字輸入 → parse_kernel（防呆）→ 正規化 → filter2d（補邊 + 卷積）→
      加偏移量 → 輸出映射（截斷 / 絕對值 / 線性拉伸）→ uint8 → PSNR。

卷積定義（數學定義，核旋轉 180°）：
    g(y, x) = Σ_i Σ_j K(i, j) · f(y − (i − c_y), x − (j − c_x))
相關運算（不翻轉，與 OpenCV filter2D 相同）：
    g(y, x) = Σ_i Σ_j K(i, j) · f(y + (i − c_y), x + (j − c_x))
其中 (c_y, c_x) = (⌊k_h / 2⌋, ⌊k_w / 2⌋) 為核的中心（錨點）。
"""

from __future__ import annotations

import math
import re
import time
import unicodedata
from dataclasses import dataclass

import numpy as np

MAX_KERNEL_SIDE = 31  # 單邊上限，避免運算過久
MAX_ABS_WEIGHT = 1e6  # 係數絕對值上限，避免數值溢位
FFT_MIN_NONZERO = 50  # 非零係數達此數量（約 7×7 以上的稠密核）就改用 FFT


class KernelError(ValueError):
    """卷積核或濾波參數不合法；訊息可以直接顯示給使用者。"""


# ───────────────────────────── 輸入解析與防呆 ─────────────────────────────

_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_TOKEN = re.compile(rf"({_NUMBER})(?:/({_NUMBER}))?")


def parse_kernel(text: str) -> np.ndarray:
    """把使用者輸入的文字解析成二維 float64 矩陣。

    支援：空白 / 逗號 / Tab 分隔、換行或分號分列、[[1,2],[3,4]] 與 [1 2; 3 4] 寫法、
    小數、科學記號（1e-3）、分數（1/16），以及中文輸入法的全形數字與符號。
    """
    s = unicodedata.normalize("NFKC", text or "")  # 全形 → 半形
    for dash in "−‒–—":  # 各種 Unicode 減號
        s = s.replace(dash, "-")
    s = s.replace("、", ",")  # 頓號「、」
    s = re.sub(r"\s*/\s*", "/", s)  # 「1 / 16」→「1/16」
    s = re.sub(r"\]\s*[,;]?\s*\[", "\n", s)  # [[1,2],[3,4]] 的列分隔
    s = re.sub(r"[\[\]()]", " ", s)

    rows: list[list[float]] = []
    for line in re.split(r"[;\r\n]+", s):
        tokens = [t for t in re.split(r"[,\s]+", line.strip()) if t]
        if not tokens:
            continue
        row_no = len(rows) + 1
        values: list[float] = []
        for token in tokens:
            match = _TOKEN.fullmatch(token)
            if match is None:
                raise KernelError(
                    f"第 {row_no} 列的「{token}」不是有效數字"
                    "（可用整數、小數、科學記號如 1e-3，或分數如 1/16）。"
                )
            value = float(match.group(1))
            if match.group(2) is not None:
                denominator = float(match.group(2))
                if denominator == 0:
                    raise KernelError(f"第 {row_no} 列的「{token}」分母為 0，無法計算（除以零）。")
                value /= denominator
            if not math.isfinite(value) or abs(value) > MAX_ABS_WEIGHT:
                raise KernelError(
                    f"第 {row_no} 列的「{token}」數值過大，係數絕對值上限為 {MAX_ABS_WEIGHT:g}。"
                )
            values.append(value)
        rows.append(values)

    if not rows:
        raise KernelError("卷積核是空的，請至少輸入一個數字。")
    lengths = [len(row) for row in rows]
    if len(set(lengths)) > 1:
        detail = "、".join(f"第 {i} 列 {n} 個" for i, n in enumerate(lengths, 1))
        raise KernelError(f"每一列的元素個數必須相同才能組成矩陣，目前為：{detail}。")
    kernel = np.array(rows, dtype=np.float64)
    if max(kernel.shape) > MAX_KERNEL_SIDE:
        raise KernelError(
            f"卷積核最大支援 {MAX_KERNEL_SIDE}×{MAX_KERNEL_SIDE}，"
            f"目前為 {kernel.shape[0]}×{kernel.shape[1]}。"
        )
    return kernel


def format_kernel(kernel: np.ndarray, precision: int = 6) -> str:
    """把矩陣轉回文字（整數不顯示小數點），用於填入輸入框。"""

    def fmt(value: float) -> str:
        value = float(value)
        if value.is_integer():
            return str(int(value))
        return f"{value:.{precision}g}"

    cells = [[fmt(v) for v in row] for row in np.atleast_2d(kernel)]
    width = max(len(c) for row in cells for c in row)
    return "\n".join(" ".join(c.rjust(width) for c in row) for row in cells)


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """產生 size×size、標準差 σ 的二維高斯核（總和為 1）。"""
    if size < 1 or size % 2 == 0:
        raise KernelError("高斯核尺寸必須是正奇數。")
    if not sigma > 0:
        raise KernelError("σ 必須大於 0。")
    radius = size // 2
    axis = np.arange(-radius, radius + 1, dtype=np.float64)
    g = np.exp(-(axis**2) / (2.0 * sigma**2))
    kernel = np.outer(g, g)
    return kernel / kernel.sum()


# ───────────────────────────── 預設範本 ─────────────────────────────


@dataclass(frozen=True)
class Preset:
    name: str
    kernel: tuple[tuple[float, ...], ...]
    normalize: str = "none"  # none / sum / custom
    divisor: float = 1.0
    bias: float = 0.0
    output: str = "clip"  # clip / abs / minmax
    description: str = ""


_G5 = ((1, 4, 6, 4, 1), (4, 16, 24, 16, 4), (6, 24, 36, 24, 6), (4, 16, 24, 16, 4), (1, 4, 6, 4, 1))

PRESETS: tuple[Preset, ...] = (
    Preset("恆等（Identity）", ((0, 0, 0), (0, 1, 0), (0, 0, 0)),
           description="輸出等於輸入，PSNR 為 ∞，可用來驗證流程與除以零防範。"),
    Preset("均值模糊 3×3", ((1, 1, 1), (1, 1, 1), (1, 1, 1)), normalize="sum",
           description="9 個像素取平均（除以總和 9），去除雜訊但邊緣變模糊。"),
    Preset("均值模糊 5×5", tuple((1,) * 5 for _ in range(5)), normalize="sum",
           description="範圍更大的平均，模糊更明顯。"),
    Preset("高斯平滑 3×3", ((1, 2, 1), (2, 4, 2), (1, 2, 1)), normalize="sum",
           description="中心權重較高的平滑（÷16），比均值模糊更能保留結構。"),
    Preset("高斯平滑 5×5", _G5, normalize="sum",
           description="二項式近似的 5×5 高斯核（÷256）。"),
    Preset("銳化（4 鄰域）", ((0, -1, 0), (-1, 5, -1), (0, -1, 0)),
           description="原圖加上拉普拉斯邊緣，係數總和為 1，亮度不變。"),
    Preset("強銳化（8 鄰域）", ((-1, -1, -1), (-1, 9, -1), (-1, -1, -1)),
           description="考慮對角方向的銳化，效果更強。"),
    Preset("反銳化遮罩 5×5（Unsharp）",
           tuple(tuple(-476 if (r, c) == (2, 2) else v for c, v in enumerate(row)) for r, row in enumerate(_G5)),
           normalize="sum",
           description="2 × 原圖 − 高斯模糊，總和 −256，除以總和後變成溫和的銳化。"),
    Preset("拉普拉斯邊緣（4 鄰域）", ((0, 1, 0), (1, -4, 1), (0, 1, 0)), output="abs",
           description="二階微分，總和為 0；輸出取絕對值顯示邊緣強度。"),
    Preset("拉普拉斯邊緣（8 鄰域）", ((1, 1, 1), (1, -8, 1), (1, 1, 1)), output="abs",
           description="含對角方向的二階微分邊緣偵測。"),
    Preset("Sobel 垂直邊緣（x 方向梯度）", ((-1, 0, 1), (-2, 0, 2), (-1, 0, 1)), output="abs",
           description="對水平方向變化敏感，突顯垂直邊緣。"),
    Preset("Sobel 水平邊緣（y 方向梯度）", ((-1, -2, -1), (0, 0, 0), (1, 2, 1)), output="abs",
           description="對垂直方向變化敏感，突顯水平邊緣。"),
    Preset("浮雕（灰色浮凸）", ((-2, -1, 0), (-1, 0, 1), (0, 1, 2)), bias=128,
           description="方向性差分，總和為 0，加上偏移量 128 形成灰底浮雕。"),
    Preset("浮雕（保留色彩）", ((-2, -1, 0), (-1, 1, 1), (0, 1, 2)),
           description="總和為 1，在原圖上疊加立體感。"),
    Preset("動態模糊（水平 1×9）", ((1,) * 9,), normalize="sum",
           description="非方陣卷積核範例：只在水平方向平均，模擬左右晃動。"),
    Preset("動態模糊（對角 5×5）", tuple(tuple(1 if r == c else 0 for c in range(5)) for r in range(5)),
           normalize="sum",
           description="沿對角線平均，模擬斜向晃動。"),
)

PRESET_BY_NAME = {p.name: p for p in PRESETS}


# ───────────────────────────── 卷積運算 ─────────────────────────────

# 鍵 → (說明, numpy.pad 的 mode)
BORDER_MODES: dict[str, tuple[str, str]] = {
    "reflect101": ("鏡射，不重複邊界像素（BORDER_REFLECT_101，OpenCV 預設）", "reflect"),
    "reflect": ("鏡射，重複邊界像素（BORDER_REFLECT）", "symmetric"),
    "replicate": ("延伸最外圈像素（BORDER_REPLICATE）", "edge"),
    "constant": ("外圍補 0（BORDER_CONSTANT / zero padding）", "constant"),
    "wrap": ("環繞，週期延伸（BORDER_WRAP）", "wrap"),
}


def filter2d(
    image: np.ndarray,
    kernel: np.ndarray,
    *,
    flip: bool = True,
    border: str = "reflect101",
    method: str = "auto",
) -> tuple[np.ndarray, str]:
    """對影像做二維卷積（flip=True）或相關運算（flip=False），回傳 (float32 結果, 實際使用的方法)。

    method：direct（逐係數平移相加）、fft（卷積定理）或 auto（依非零係數數量自動選擇）。
    """
    img = np.asarray(image)
    if img.ndim not in (2, 3) or img.size == 0:
        raise ValueError("影像必須是非空的 2D（灰階）或 3D（彩色）陣列")
    k = np.asarray(kernel, dtype=np.float64)
    if k.ndim != 2 or k.size == 0:
        raise KernelError("卷積核必須是非空的二維矩陣。")
    if not np.all(np.isfinite(k)):
        raise KernelError("卷積核含有無限大或非數值（NaN）。")
    if border not in BORDER_MODES:
        raise KernelError(f"未知的邊界處理方式：{border}")

    kh, kw = k.shape
    if flip:  # 數學卷積：核旋轉 180°，錨點跟著鏡射，使原核的中心 (⌊kh/2⌋, ⌊kw/2⌋) 不變
        k = k[::-1, ::-1]
        anchor_y, anchor_x = kh - 1 - kh // 2, kw - 1 - kw // 2
    else:
        anchor_y, anchor_x = kh // 2, kw // 2
    pad = ((anchor_y, kh - 1 - anchor_y), (anchor_x, kw - 1 - anchor_x))

    if method == "auto":
        method = "fft" if np.count_nonzero(k) >= FFT_MIN_NONZERO else "direct"
    if method not in ("direct", "fft"):
        raise ValueError(f"未知的運算方法：{method}")

    planes = img[..., None] if img.ndim == 2 else img
    height, width, channels = planes.shape
    out = np.empty((height, width, channels), dtype=np.float32)
    np_mode = BORDER_MODES[border][1]
    for c in range(channels):
        padded = np.pad(planes[..., c].astype(np.float32), pad, mode=np_mode)
        if method == "direct":
            out[..., c] = _correlate_direct(padded, k, height, width)
        else:
            out[..., c] = _correlate_fft(padded, k, height, width)
    return (out[..., 0] if img.ndim == 2 else out), method


def _correlate_direct(padded: np.ndarray, k: np.ndarray, height: int, width: int) -> np.ndarray:
    """out(y, x) = Σ K(i, j) · P(y + i, x + j)：每個非零係數做一次整張影像的平移相加。"""
    acc = np.zeros((height, width), dtype=np.float32)
    tmp = np.empty((height, width), dtype=np.float32)
    for (i, j), weight in np.ndenumerate(k):
        if weight == 0.0:
            continue  # 跳過 0 係數（拉普拉斯、Sobel 等稀疏核可省下大量運算）
        np.multiply(padded[i : i + height, j : j + width], np.float32(weight), out=tmp)
        acc += tmp
    return acc


def _correlate_fft(padded: np.ndarray, k: np.ndarray, height: int, width: int) -> np.ndarray:
    """卷積定理：空間域卷積 = 頻率域相乘。相關運算 = 與翻轉核做卷積，再取有效區域。"""
    kh, kw = k.shape
    fh, fw = _next_fast_len(padded.shape[0]), _next_fast_len(padded.shape[1])
    spectrum = np.fft.rfft2(padded, s=(fh, fw)) * np.fft.rfft2(k[::-1, ::-1], s=(fh, fw))
    full = np.fft.irfft2(spectrum, s=(fh, fw))
    return full[kh - 1 : kh - 1 + height, kw - 1 : kw - 1 + width].astype(np.float32)


def _next_fast_len(n: int) -> int:
    """不小於 n 的最小 2^a·3^b·5^c，讓 FFT 跑得快。"""
    m = max(1, n)
    while True:
        x = m
        for p in (2, 3, 5):
            while x % p == 0:
                x //= p
        if x == 1:
            return m
        m += 1


# ───────────────────────────── 正規化與輸出 ─────────────────────────────

NORMALIZE_MODES = {
    "none": "不正規化（直接使用輸入係數）",
    "sum": "除以係數總和（總和為 0 時自動停用）",
    "custom": "除以自訂除數",
}

OUTPUT_MODES = {
    "clip": "截斷：小於 0 設為 0、大於 255 設為 255",
    "abs": "取絕對值後截斷（適合邊緣偵測）",
    "minmax": "線性拉伸：最小值 → 0、最大值 → 255",
}


@dataclass(frozen=True)
class FilterParams:
    normalize: str = "none"
    divisor: float = 1.0
    bias: float = 0.0
    output: str = "clip"
    flip: bool = True
    border: str = "reflect101"


@dataclass(frozen=True)
class FilterResult:
    image: np.ndarray  # uint8 輸出影像
    effective_kernel: np.ndarray  # 正規化後實際使用的係數
    factor: float  # 正規化係數（實際核 = 輸入核 × factor）
    response_min: float  # 映射前（含偏移量）的最小值
    response_max: float  # 映射前（含偏移量）的最大值
    clipped_low: float  # 輸出時被截斷成 0 的比例
    clipped_high: float  # 輸出時被截斷成 255 的比例
    method: str  # direct / fft
    elapsed_ms: float
    warnings: tuple[str, ...]


def normalization_factor(kernel: np.ndarray, params: FilterParams) -> tuple[float, list[str]]:
    """計算正規化係數，含除以零防範。"""
    warnings: list[str] = []
    if params.normalize == "none":
        return 1.0, warnings
    if params.normalize == "sum":
        total = float(np.sum(kernel))
        if abs(total) < 1e-12:
            warnings.append(
                "這個卷積核的係數總和為 0（邊緣偵測類常見），無法「除以總和」；"
                "已自動改為不正規化，避免除以零。"
            )
            return 1.0, warnings
        return 1.0 / total, warnings
    if params.normalize == "custom":
        divisor = float(params.divisor)
        if not math.isfinite(divisor) or divisor == 0:
            raise KernelError("自訂除數不可為 0（除以零），請輸入非零的數值。")
        return 1.0 / divisor, warnings
    raise KernelError(f"未知的正規化方式：{params.normalize}")


def apply_filter(image: np.ndarray, kernel: np.ndarray, params: FilterParams) -> FilterResult:
    """完整濾波流程：正規化 → 卷積 → 偏移量 → 輸出映射 → uint8。"""
    start = time.perf_counter()
    k = np.asarray(kernel, dtype=np.float64)
    if params.output not in OUTPUT_MODES:
        raise KernelError(f"未知的輸出映射方式：{params.output}")
    if not math.isfinite(float(params.bias)):
        raise KernelError("偏移量必須是有限的數值。")
    factor, warnings = normalization_factor(k, params)
    if k.ndim == 2 and (k.shape[0] % 2 == 0 or k.shape[1] % 2 == 0):
        warnings.append(
            f"{k.shape[0]}×{k.shape[1]} 的核有偶數邊長，沒有正中心；以第 {k.shape[0] // 2 + 1} 列、"
            f"第 {k.shape[1] // 2 + 1} 行為錨點（與 OpenCV 相同），結果會有半個像素的位移。"
        )
    effective = k * factor
    response, method = filter2d(image, effective, flip=params.flip, border=params.border)
    if params.bias:
        response += np.float32(params.bias)
    response_min, response_max = float(response.min()), float(response.max())

    # 以下皆為原地運算（out=），大圖時可省下數倍記憶體
    if params.output == "abs":
        np.abs(response, out=response)
    elif params.output == "minmax":
        if response_max - response_min < 1e-6:
            warnings.append("濾波結果每個像素都相同，無法線性拉伸（分母為 0）；已改用直接截斷。")
        else:
            response -= np.float32(response_min)
            response *= np.float32(255.0 / (response_max - response_min))

    np.rint(response, out=response)
    clipped_low = float(np.mean(response < 0))
    clipped_high = float(np.mean(response > 255))
    np.clip(response, 0, 255, out=response)
    output = response.astype(np.uint8)
    return FilterResult(
        image=output,
        effective_kernel=effective,
        factor=factor,
        response_min=response_min,
        response_max=response_max,
        clipped_low=clipped_low,
        clipped_high=clipped_high,
        method=method,
        elapsed_ms=(time.perf_counter() - start) * 1000,
        warnings=tuple(warnings),
    )


def params_for(preset: Preset, *, flip: bool = True, border: str = "reflect101") -> FilterParams:
    return FilterParams(
        normalize=preset.normalize,
        divisor=preset.divisor,
        bias=preset.bias,
        output=preset.output,
        flip=flip,
        border=border,
    )


# ───────────────────────────── PSNR 與比對 ─────────────────────────────


_SSE_CHUNK = 4_000_000  # 每次處理的元素數（int32 暫存約 16 MB）


def _sum_squared_error(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    """計算平方誤差總和，回傳 (SSE, 元素個數)。uint8 影像以整數分塊精確計算，不受浮點誤差影響。"""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f"兩張影像尺寸不同：{a.shape} vs {b.shape}")
    if a.size == 0:
        raise ValueError("影像是空的")
    if a.dtype == np.uint8 and b.dtype == np.uint8:
        flat_a, flat_b = a.reshape(-1), b.reshape(-1)
        sse = 0
        for start in range(0, flat_a.size, _SSE_CHUNK):
            diff = flat_a[start : start + _SSE_CHUNK].astype(np.int32)
            diff -= flat_b[start : start + _SSE_CHUNK]
            np.square(diff, out=diff)  # 最大 255² = 65025，int32 不會溢位
            sse += int(diff.sum(dtype=np.int64))
        return sse, int(a.size)
    diff = a.astype(np.float64) - b.astype(np.float64)
    return float(np.sum(diff * diff)), int(a.size)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    """MSE = 1/(M·N·C) · ΣΣΣ [I(y,x,c) − K(y,x,c)]²"""
    sse, n = _sum_squared_error(a, b)
    return sse / n


def psnr(a: np.ndarray, b: np.ndarray, max_value: float = 255.0) -> float:
    """PSNR = 10·log10(MAX² / MSE)（dB）。MSE = 0（兩圖完全相同）時回傳 ∞，避免除以零。"""
    error = mse(a, b)
    if error == 0:
        return math.inf
    return 10.0 * math.log10(max_value**2 / error)


def psnr_per_channel(a: np.ndarray, b: np.ndarray, max_value: float = 255.0) -> list[float]:
    a, b = np.asarray(a), np.asarray(b)
    if a.ndim == 2:
        return [psnr(a, b, max_value)]
    return [psnr(a[..., c], b[..., c], max_value) for c in range(a.shape[2])]


def split_view(before: np.ndarray, after: np.ndarray, position: float) -> np.ndarray:
    """左邊顯示原圖、右邊顯示濾波結果，position ∈ [0, 1] 為分割線位置。"""
    if before.shape != after.shape:
        raise ValueError("兩張影像尺寸不同")
    width = before.shape[1]
    x = int(round(min(max(position, 0.0), 1.0) * width))
    out = after.copy()
    out[:, :x] = before[:, :x]
    if 0 < x < width:
        half = max(1, width // 400)
        out[:, max(0, x - half - 1) : x + half + 1] = 0
        out[:, max(0, x - half) : x + half] = 255
    return out


def difference_image(a: np.ndarray, b: np.ndarray, gain: float = 1.0) -> np.ndarray:
    """|原圖 − 結果| × gain，越亮代表差異越大。"""
    if a.shape != b.shape:
        raise ValueError("兩張影像尺寸不同")
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16)).astype(np.float32) * np.float32(gain)
    return np.clip(np.rint(diff), 0, 255).astype(np.uint8)
