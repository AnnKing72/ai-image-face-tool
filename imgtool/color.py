"""功能 2：多色彩空間轉換與直方圖分析（核心運算全部以 NumPy 實作）。

YUV 採 ITU-R BT.601 亮度權重：
    Y = 0.299 R + 0.587 G + 0.114 B
    U = k_u (B − Y) + 128,   V = k_v (R − Y) + 128
    類比 YUV：k_u = 0.492、k_v = 0.877（與 OpenCV COLOR_RGB2YUV 相同）
    YCbCr  ：k_u = 0.564、k_v = 0.713（JPEG 全幅，與 OpenCV COLOR_RGB2YCrCb 相同）
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114])  # ITU-R BT.601


@dataclass(frozen=True)
class YUVStandard:
    key: str
    label: str
    u_scale: float
    v_scale: float
    description: str


YUV_STANDARDS: dict[str, YUVStandard] = {
    "yuv": YUVStandard(
        key="yuv",
        label="BT.601 YUV（類比定義）",
        u_scale=0.492,
        v_scale=0.877,
        description=(
            "U = 0.492(B − Y) + 128、V = 0.877(R − Y) + 128（同 OpenCV COLOR_RGB2YUV）。"
            "V 的理論範圍約 −29～285，飽和的紅色、青色會被截斷在 0 或 255。"
        ),
    ),
    "ycbcr": YUVStandard(
        key="ycbcr",
        label="BT.601 YCbCr（JPEG 全幅）",
        u_scale=0.564,
        v_scale=0.713,
        description=(
            "Cb = 0.564(B − Y) + 128、Cr = 0.713(R − Y) + 128（同 OpenCV COLOR_RGB2YCrCb），"
            "色度範圍剛好落在 0–255，不會截斷。畫面上的 U、V 分別對應 Cb、Cr。"
        ),
    ),
}


def _check_rgb(rgb: np.ndarray) -> None:
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"需要 (H, W, 3) 的 uint8 RGB 影像，收到 shape={rgb.shape}、dtype={rgb.dtype}")


def yuv_matrix(standard: str = "yuv") -> np.ndarray:
    """回傳 3×3 轉換矩陣 M，使 [Y, U−128, V−128]ᵀ = M · [R, G, B]ᵀ。"""
    std = YUV_STANDARDS[standard]
    y = LUMA_WEIGHTS
    u = std.u_scale * (np.array([0.0, 0.0, 1.0]) - y)
    v = std.v_scale * (np.array([1.0, 0.0, 0.0]) - y)
    return np.vstack([y, u, v])


@dataclass(frozen=True)
class YUVResult:
    image: np.ndarray  # (H, W, 3) uint8，通道順序 Y, U, V
    clipped: tuple[float, float, float]  # 各分量超出 0–255 而被截斷的像素比例


def rgb_to_yuv(rgb: np.ndarray, standard: str = "yuv") -> YUVResult:
    """RGB → YUV（或 YCbCr），以矩陣乘法一次算完所有像素。"""
    _check_rgb(rgb)
    matrix = yuv_matrix(standard).astype(np.float32)
    flat = rgb.reshape(-1, 3).astype(np.float32)
    yuv = flat @ matrix.T
    yuv[:, 1:] += 128.0
    rounded = np.rint(yuv)
    clipped = tuple(
        float(np.mean((rounded[:, i] < 0) | (rounded[:, i] > 255))) for i in range(3)
    )
    out = np.clip(rounded, 0, 255).astype(np.uint8).reshape(rgb.shape)
    return YUVResult(image=out, clipped=clipped)


def yuv_to_rgb(yuv: np.ndarray, standard: str = "yuv") -> np.ndarray:
    """YUV → RGB（反矩陣），用於色度分量的偽彩色顯示。"""
    _check_rgb(yuv)
    inverse = np.linalg.inv(yuv_matrix(standard)).astype(np.float32)
    flat = yuv.reshape(-1, 3).astype(np.float32)
    flat[:, 1:] -= 128.0
    rgb = flat @ inverse.T
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8).reshape(yuv.shape)


def chroma_preview(yuv: np.ndarray, index: int, standard: str = "yuv") -> np.ndarray:
    """把單一色度分量畫成彩色：亮度固定 128、另一個色度設為 128（無色），只保留這個分量。"""
    if index not in (1, 2):
        raise ValueError("index 只能是 1（U）或 2（V）")
    fixed = np.full_like(yuv, 128)
    fixed[..., index] = yuv[..., index]
    return yuv_to_rgb(fixed, standard)


def tint_channel(rgb: np.ndarray, index: int) -> np.ndarray:
    """只保留 R、G 或 B 其中一個通道，其餘歸零（以該原色顯示分量）。"""
    _check_rgb(rgb)
    out = np.zeros_like(rgb)
    out[..., index] = rgb[..., index]
    return out


def histogram(channel: np.ndarray) -> np.ndarray:
    """計算 256 階直方圖：索引 = 像素強度 0–255，值 = 該強度的像素數量。"""
    channel = np.asarray(channel)
    if channel.dtype != np.uint8:
        raise TypeError("直方圖只接受 uint8（0–255）的影像。")
    return np.bincount(channel.ravel(), minlength=256).astype(np.int64)


@dataclass(frozen=True)
class ChannelStats:
    pixels: int
    mean: float
    std: float
    minimum: int
    maximum: int
    median: float
    mode: int
    entropy: float  # 資訊熵（bits），衡量強度分布的分散程度


def channel_stats(hist: np.ndarray) -> ChannelStats:
    """直接由直方圖計算統計量，不必再掃描一次影像。"""
    hist = np.asarray(hist, dtype=np.int64)
    total = int(hist.sum())
    if hist.shape != (256,) or total <= 0:
        raise ValueError("直方圖必須是長度 256 且總數大於 0 的陣列")
    levels = np.arange(256, dtype=np.float64)
    mean = float((hist * levels).sum() / total)
    std = float(np.sqrt((hist * (levels - mean) ** 2).sum() / total))
    nonzero = np.flatnonzero(hist)
    cdf = np.cumsum(hist)
    lower = int(np.searchsorted(cdf, (total - 1) // 2 + 1))  # 排序後第 ⌊(n−1)/2⌋ 個值
    upper = int(np.searchsorted(cdf, total // 2 + 1))  # 排序後第 ⌊n/2⌋ 個值
    probs = hist[nonzero] / total
    entropy = float(-(probs * np.log2(probs)).sum()) + 0.0  # + 0.0 避免出現 −0.0
    return ChannelStats(
        pixels=total,
        mean=mean,
        std=std,
        minimum=int(nonzero[0]),
        maximum=int(nonzero[-1]),
        median=(lower + upper) / 2,
        mode=int(np.argmax(hist)),
        entropy=entropy,
    )
