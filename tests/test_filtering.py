"""功能 3：卷積濾波、輸入防呆與 PSNR（與 OpenCV / SciPy / scikit-image 交叉驗證）。"""

import math

import cv2
import numpy as np
import pytest

from imgtool import filtering as F

CV_BORDERS = {
    "reflect101": cv2.BORDER_REFLECT_101,
    "reflect": cv2.BORDER_REFLECT,
    "replicate": cv2.BORDER_REPLICATE,
    "constant": cv2.BORDER_CONSTANT,
}
SCIPY_MODES = {"reflect101": "mirror", "reflect": "reflect", "replicate": "nearest", "constant": "constant", "wrap": "wrap"}

KERNELS = {
    "sharpen3x3": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], float),
    "sobel3x3": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float),
    "random3x5": np.random.default_rng(3).normal(size=(3, 5)),
    "even2x2": np.array([[1, 2], [3, 4]], float),
    "even4x3": np.random.default_rng(4).normal(size=(4, 3)),
    "row1x9": np.ones((1, 9)),
    "dense9x9": np.random.default_rng(5).normal(size=(9, 9)),
}


@pytest.fixture(scope="module")
def img():
    return np.random.default_rng(2).integers(0, 256, (31, 45, 3), dtype=np.uint8)


def opencv_correlate(img, k, border):
    return cv2.filter2D(img.astype(np.float32), cv2.CV_32F, k.astype(np.float32), borderType=CV_BORDERS[border])


def tolerance(k):
    return 1e-5 * 255 * np.abs(k).sum() + 1e-3


# ───────────── 卷積運算正確性 ─────────────


@pytest.mark.parametrize("method", ["direct", "fft"])
@pytest.mark.parametrize("name", list(KERNELS))
@pytest.mark.parametrize("border", list(CV_BORDERS))
def test_correlation_matches_opencv_filter2d(img, name, border, method):
    k = KERNELS[name]
    ours, used = F.filter2d(img, k, flip=False, border=border, method=method)
    assert used == method
    np.testing.assert_allclose(ours, opencv_correlate(img, k, border), rtol=1e-5, atol=tolerance(k))


@pytest.mark.parametrize("name", list(KERNELS))
@pytest.mark.parametrize("border", list(SCIPY_MODES))
def test_convolution_matches_scipy(img, name, border):
    ndimage = pytest.importorskip("scipy.ndimage")
    k = KERNELS[name]
    ours, _ = F.filter2d(img, k, flip=True, border=border)
    reference = np.stack(
        [ndimage.convolve(img[..., c].astype(np.float64), k, mode=SCIPY_MODES[border], cval=0.0) for c in range(3)],
        axis=-1,
    )
    np.testing.assert_allclose(ours, reference, rtol=1e-5, atol=tolerance(k))


def test_wrap_border_matches_numpy_roll(img):
    k = KERNELS["random3x5"]
    ours, _ = F.filter2d(img[..., 0], k, flip=False, border="wrap")
    cy, cx = k.shape[0] // 2, k.shape[1] // 2
    reference = np.zeros(img.shape[:2])
    for (i, j), weight in np.ndenumerate(k):
        reference += weight * np.roll(img[..., 0].astype(float), shift=(cy - i, cx - j), axis=(0, 1))
    np.testing.assert_allclose(ours, reference, rtol=1e-5, atol=tolerance(k))


def test_convolution_is_correlation_with_rotated_kernel(img):
    k = KERNELS["random3x5"]  # 奇數尺寸：卷積 = 用旋轉 180° 的核做相關運算
    a, _ = F.filter2d(img, k, flip=True)
    b, _ = F.filter2d(img, k[::-1, ::-1], flip=False)
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-3)


def test_direct_and_fft_agree(img):
    k = np.random.default_rng(9).normal(size=(15, 11))
    a, _ = F.filter2d(img, k, method="direct")
    b, _ = F.filter2d(img, k, method="fft")
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=tolerance(k))


def test_auto_method_selection(img):
    assert F.filter2d(img, np.ones((3, 3)))[1] == "direct"
    assert F.filter2d(img, np.ones((9, 9)))[1] == "fft"
    sparse = np.zeros((15, 15))
    sparse[7, :] = 1  # 稀疏的大核仍用直接法
    assert F.filter2d(img, sparse)[1] == "direct"


def test_grayscale_input(img):
    out, _ = F.filter2d(img[..., 0], KERNELS["sharpen3x3"])
    assert out.shape == img.shape[:2]


@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (2, 3), (3, 1)])
@pytest.mark.parametrize("border", list(CV_BORDERS))
def test_image_smaller_than_kernel(shape, border):
    tiny = np.random.default_rng(6).integers(0, 256, shape + (3,), dtype=np.uint8)
    k = np.random.default_rng(7).normal(size=(5, 5))
    ours, _ = F.filter2d(tiny, k, flip=False, border=border)
    np.testing.assert_allclose(ours, opencv_correlate(tiny, k, border), rtol=1e-5, atol=tolerance(k))


def test_rejects_invalid_kernel_arrays(img):
    with pytest.raises(F.KernelError):
        F.filter2d(img, np.array([[np.nan]]))
    with pytest.raises(F.KernelError):
        F.filter2d(img, np.zeros((0, 3)))
    with pytest.raises(F.KernelError):
        F.filter2d(img, np.ones((3, 3)), border="mirror-ish")


# ───────────── 文字輸入解析與防呆 ─────────────


@pytest.mark.parametrize(
    "text, expected",
    [
        ("1 2 3\n4 5 6", [[1, 2, 3], [4, 5, 6]]),
        ("1,2,3;4,5,6", [[1, 2, 3], [4, 5, 6]]),
        ("[[1, 2], [3, 4]]", [[1, 2], [3, 4]]),
        ("[1 2; 3 4]", [[1, 2], [3, 4]]),
        ("1/16 2/16\n-0.5 1e-1", [[0.0625, 0.125], [-0.5, 0.1]]),
        ("１，－２\n３　４", [[1, -2], [3, 4]]),
        ("  1\t2  \n\n 3 4 \n", [[1, 2], [3, 4]]),
        ("1 / 4", [[0.25]]),
        ("7", [[7]]),
        ("1 1 1 1 1", [[1, 1, 1, 1, 1]]),
    ],
    ids=["spaces", "comma-semicolon", "python-list", "matlab", "fraction-sci", "fullwidth",
         "messy-whitespace", "spaced-fraction", "scalar", "non-square-1x5"],
)
def test_parse_kernel_formats(text, expected):
    np.testing.assert_allclose(F.parse_kernel(text), np.array(expected, float))


@pytest.mark.parametrize(
    "text, message",
    [
        ("", "空的"),
        ("  \n ", "空的"),
        ("1 2 3\n4 5", "元素個數必須相同"),
        ("1 a 3", "不是有效數字"),
        ("nan 1", "不是有效數字"),
        ("1/0 1", "分母為 0"),
        ("1e7", "數值過大"),
        ("1e999", "數值過大"),
        ("\n".join(" ".join(["1"] * 33) for _ in range(33)), "最大支援"),
    ],
    ids=["empty", "blank", "ragged", "letter", "nan", "divide-by-zero", "too-large", "overflow", "33x33"],
)
def test_parse_kernel_rejects_bad_input(text, message):
    with pytest.raises(F.KernelError, match=message):
        F.parse_kernel(text)


def test_format_kernel_roundtrip():
    k = np.array([[1, -2.5], [0.125, 16]])
    np.testing.assert_allclose(F.parse_kernel(F.format_kernel(k)), k)
    assert F.format_kernel(np.array([[1.0, 2.0]])) == "1 2"


def test_gaussian_kernel_matches_opencv():
    g = F.gaussian_kernel(5, 1.0)
    assert g.sum() == pytest.approx(1.0)
    ref = cv2.getGaussianKernel(5, 1.0)
    np.testing.assert_allclose(g, ref @ ref.T, rtol=1e-6)
    with pytest.raises(F.KernelError):
        F.gaussian_kernel(4, 1.0)
    with pytest.raises(F.KernelError):
        F.gaussian_kernel(5, 0)


# ───────────── 正規化、偏移量與輸出映射 ─────────────


def test_identity_kernel_gives_infinite_psnr(img):
    res = F.apply_filter(img, np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]), F.FilterParams())
    assert np.array_equal(res.image, img)
    assert F.psnr(img, res.image) == math.inf


def test_normalize_by_sum_equals_opencv_box_blur(img):
    res = F.apply_filter(img, np.ones((3, 3)), F.FilterParams(normalize="sum"))
    assert res.factor == pytest.approx(1 / 9)
    reference = cv2.blur(img, (3, 3), borderType=cv2.BORDER_REFLECT_101)
    assert np.abs(res.image.astype(int) - reference).max() <= 1


def test_zero_sum_kernel_falls_back_without_division_by_zero(img):
    res = F.apply_filter(img, np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]), F.FilterParams(normalize="sum"))
    assert res.factor == 1.0
    assert any("總和為 0" in w for w in res.warnings)


def test_custom_divisor(img):
    gauss = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    a = F.apply_filter(img, gauss, F.FilterParams(normalize="custom", divisor=16))
    b = F.apply_filter(img, gauss, F.FilterParams(normalize="sum"))
    assert np.array_equal(a.image, b.image)
    with pytest.raises(F.KernelError, match="不可為 0"):
        F.apply_filter(img, gauss, F.FilterParams(normalize="custom", divisor=0))


def test_output_modes_on_ramp_with_non_square_kernel():
    ramp = np.tile(np.arange(0, 250, 10, dtype=np.uint8), (5, 1))[..., None].repeat(3, axis=2)
    k = np.array([[-1, 0, 1]])  # 1×3：卷積（翻轉）後內部響應為 f(x−1) − f(x+1) = −20
    inner = (slice(None), slice(1, -1))
    clip = F.apply_filter(ramp, k, F.FilterParams(output="clip"))
    absolute = F.apply_filter(ramp, k, F.FilterParams(output="abs"))
    stretch = F.apply_filter(ramp, k, F.FilterParams(output="minmax"))
    assert np.all(clip.image[inner] == 0) and clip.clipped_low > 0
    assert clip.response_min == pytest.approx(-20)
    assert np.all(absolute.image[inner] == 20)
    assert np.all(stretch.image[inner] == 0) and np.all(stretch.image[:, 0] == 255)


def test_bias_and_flat_result_minmax_guard():
    flat = np.full((6, 6, 3), 100, np.uint8)
    res = F.apply_filter(flat, np.zeros((3, 3)), F.FilterParams(bias=128, output="minmax"))
    assert np.all(res.image == 128)
    assert any("無法線性拉伸" in w for w in res.warnings)


def test_extreme_weights_saturate(img):
    res = F.apply_filter(img, np.array([[0, 0, 0], [0, 1e6, 0], [0, 0, -1e6]]), F.FilterParams())
    assert res.image.dtype == np.uint8
    assert set(np.unique(res.image)) <= {0, 255}


def test_even_kernel_warning(img):
    res = F.apply_filter(img, np.ones((2, 2)), F.FilterParams(normalize="sum"))
    assert any("偶數" in w for w in res.warnings)


@pytest.mark.parametrize("preset", F.PRESETS, ids=lambda p: p.name)
def test_every_preset_runs(img, preset):
    res = F.apply_filter(img, np.array(preset.kernel, float), F.params_for(preset))
    assert res.image.shape == img.shape and res.image.dtype == np.uint8


def test_preset_definitions():
    assert np.sum(F.PRESET_BY_NAME["反銳化遮罩 5×5（Unsharp）"].kernel) == -256
    assert np.sum(F.PRESET_BY_NAME["浮雕（灰色浮凸）"].kernel) == 0
    assert np.array(F.PRESET_BY_NAME["動態模糊（水平 1×9）"].kernel).shape == (1, 9)


# ───────────── PSNR ─────────────


def test_psnr_known_value():
    a = np.zeros((10, 10), np.uint8)
    b = a.copy()
    b[0, :] = 10  # 10 個像素各差 10 → SSE = 1000，MSE = 10
    assert F.mse(a, b) == pytest.approx(10.0)
    assert F.psnr(a, b) == pytest.approx(10 * math.log10(255**2 / 10))


def test_psnr_matches_opencv_and_skimage(img):
    noise = np.random.default_rng(8).normal(0, 12, img.shape)
    noisy = np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)
    ours = F.psnr(img, noisy)
    assert ours == pytest.approx(cv2.PSNR(img, noisy), abs=1e-9)
    metrics = pytest.importorskip("skimage.metrics")
    assert ours == pytest.approx(metrics.peak_signal_noise_ratio(img, noisy, data_range=255), rel=1e-9)


def test_psnr_identical_and_mismatched(img):
    assert F.psnr(img, img.copy()) == math.inf
    with pytest.raises(ValueError):
        F.psnr(img, img[:-1])


def test_psnr_per_channel(img):
    other = img.copy()
    other[..., 0] = 0
    per = F.psnr_per_channel(img, other)
    assert math.isfinite(per[0]) and per[1] == math.inf and per[2] == math.inf


def test_mse_large_image_no_overflow():
    a = np.zeros((3000, 3000), np.uint8)
    b = np.full((3000, 3000), 255, np.uint8)
    assert F.mse(a, b) == 255**2
    assert F.psnr(a, b) == pytest.approx(0.0)


def test_split_view_and_difference(img):
    other = 255 - img
    split = F.split_view(img, other, 0.5)
    assert np.array_equal(split[:, :10], img[:, :10]) and np.array_equal(split[:, -10:], other[:, -10:])
    diff = F.difference_image(img, other)
    assert np.array_equal(diff, np.abs(img.astype(int) - other).astype(np.uint8))
    assert F.difference_image(img, img).max() == 0
