"""功能 2：色彩空間轉換與直方圖（與 OpenCV / NumPy 交叉驗證）。"""

import cv2
import numpy as np
import pytest

from imgtool import color


@pytest.fixture(scope="module")
def img():
    return np.random.default_rng(1).integers(0, 256, (64, 96, 3), dtype=np.uint8)


def rgb_cube_corners() -> np.ndarray:
    values = (0, 255)
    return np.array([[r, g, b] for r in values for g in values for b in values], np.uint8).reshape(1, -1, 3)


@pytest.mark.parametrize(
    "standard, code, order",
    [("yuv", cv2.COLOR_RGB2YUV, [0, 1, 2]), ("ycbcr", cv2.COLOR_RGB2YCrCb, [0, 2, 1])],
)
def test_matches_opencv(img, standard, code, order):
    ours = color.rgb_to_yuv(img, standard).image.astype(int)
    reference = cv2.cvtColor(img, code)[..., order].astype(int)  # OpenCV 是 Y, Cr, Cb 順序
    diff = np.abs(ours - reference)
    assert diff.max() <= 1
    assert (diff[..., 0] == 0).mean() > 0.99  # Y 幾乎完全一致


@pytest.mark.parametrize(
    "standard, code, order",
    [("yuv", cv2.COLOR_RGB2YUV, [0, 1, 2]), ("ycbcr", cv2.COLOR_RGB2YCrCb, [0, 2, 1])],
)
def test_opencv_difference_comes_from_integer_y(img, standard, code, order):
    """OpenCV 先把 Y 四捨五入成整數，再用整數 Y 算色度；本專案依公式用精確的 Y。
    用同樣的整數 Y 流程模擬後，U、V 幾乎完全一致，證明差異只來自中間值的捨入。"""
    std = color.YUV_STANDARDS[standard]
    f = img.astype(np.float64)
    y = np.floor(f @ color.LUMA_WEIGHTS + 0.5)
    u = np.floor(std.u_scale * (f[..., 2] - y) + 128.5)
    v = np.floor(std.v_scale * (f[..., 0] - y) + 128.5)
    emulated = np.clip(np.stack([y, u, v], axis=-1), 0, 255)
    reference = cv2.cvtColor(img, code)[..., order]
    assert (emulated[..., 1:] == reference[..., 1:]).mean() > 0.99


def test_known_values():
    px = np.array([[[255, 255, 255], [0, 0, 0], [255, 0, 0], [0, 0, 255]]], np.uint8)
    yuv = color.rgb_to_yuv(px, "yuv").image[0]
    assert tuple(yuv[0]) == (255, 128, 128)  # 白：無色度
    assert tuple(yuv[1]) == (0, 128, 128)  # 黑
    # 紅：Y = 0.299×255 = 76.2；U = 0.492×(0 − 76.2) + 128 = 90.5；V = 0.877×178.8 + 128 = 284.8 → 截斷 255
    assert tuple(yuv[2]) == (76, 90, 255)
    # 藍：Y = 29.1；U = 0.492×225.9 + 128 = 239.3；V = 0.877×(−29.1) + 128 = 102.5
    assert tuple(yuv[3]) == (29, 239, 103)


def test_clipping_reported():
    red = np.zeros((2, 2, 3), np.uint8)
    red[..., 0] = 255
    assert color.rgb_to_yuv(red, "yuv").clipped == (0.0, 0.0, 1.0)
    assert color.rgb_to_yuv(red, "ycbcr").clipped == (0.0, 0.0, 0.0)
    assert max(color.rgb_to_yuv(rgb_cube_corners(), "ycbcr").clipped) == 0


@pytest.mark.parametrize("standard", ["yuv", "ycbcr"])
def test_inverse_roundtrip(img, standard):
    src = (img // 2 + 64).astype(np.uint8) if standard == "yuv" else img  # 避開 V 截斷的飽和色
    back = color.yuv_to_rgb(color.rgb_to_yuv(src, standard).image, standard)
    assert np.abs(back.astype(int) - src).max() <= 2


def test_histogram_matches_numpy_and_opencv(img):
    channel = img[..., 1]
    hist = color.histogram(channel)
    assert hist.shape == (256,) and hist.sum() == channel.size
    assert np.array_equal(hist, np.histogram(channel, bins=256, range=(0, 256))[0])
    cv_hist = cv2.calcHist([channel], [0], None, [256], [0, 256]).ravel().astype(np.int64)
    assert np.array_equal(hist, cv_hist)


def test_histogram_rejects_non_uint8():
    with pytest.raises(TypeError):
        color.histogram(np.zeros((2, 2), np.float32))


def test_channel_stats(img):
    channel = img[..., 0]
    s = color.channel_stats(color.histogram(channel))
    assert s.mean == pytest.approx(channel.mean())
    assert s.std == pytest.approx(channel.std())
    assert (s.minimum, s.maximum) == (channel.min(), channel.max())
    assert s.median == pytest.approx(np.median(channel))
    values, counts = np.unique(channel, return_counts=True)
    assert s.mode == values[np.argmax(counts)]
    for arr in ([1, 2, 3, 4], [5, 1, 9], [7]):  # 偶數、奇數、單一像素的中位數
        a = np.array(arr, np.uint8)
        assert color.channel_stats(color.histogram(a)).median == pytest.approx(np.median(a))


def test_entropy_bounds():
    flat = color.channel_stats(color.histogram(np.full((10, 10), 77, np.uint8)))
    assert flat.entropy == 0.0 and flat.std == 0.0 and flat.median == 77
    uniform = color.channel_stats(color.histogram(np.arange(256, dtype=np.uint8)))
    assert uniform.entropy == pytest.approx(8.0)


def test_chroma_preview_and_tint(img):
    yuv = color.rgb_to_yuv(img).image
    assert color.chroma_preview(yuv, 1).shape == img.shape
    neutral = np.full_like(yuv, 128)
    assert np.all(color.chroma_preview(neutral, 2) == 128)  # Y = 128、無色度 → 中灰
    tinted = color.tint_channel(img, 0)
    assert np.array_equal(tinted[..., 0], img[..., 0]) and not tinted[..., 1:].any()
    with pytest.raises(ValueError):
        color.chroma_preview(yuv, 0)


def test_rejects_non_rgb():
    with pytest.raises(ValueError):
        color.rgb_to_yuv(np.zeros((4, 4), np.uint8))
