"""功能 1：影像載入與中繼資訊。"""

import io

import numpy as np
import pytest
from PIL import Image

from imgtool import io_utils
from imgtool.io_utils import ImageLoadError, human_size, limit_size, load_image, to_png_bytes


def encode(img: Image.Image, fmt: str, **kwargs) -> bytes:
    buffer = io.BytesIO()
    img.save(buffer, format=fmt, **kwargs)
    return buffer.getvalue()


@pytest.fixture
def rgb_array():
    return np.random.default_rng(0).integers(0, 256, size=(37, 53, 3), dtype=np.uint8)


@pytest.mark.parametrize("fmt", ["PNG", "BMP"])
def test_lossless_formats_roundtrip(rgb_array, fmt):
    data = encode(Image.fromarray(rgb_array), fmt)
    rgb, info = load_image(data, f"x.{fmt.lower()}")
    assert np.array_equal(rgb, rgb_array)
    assert (info.width, info.height) == (53, 37)
    assert (info.channels, info.bits_per_channel, info.bits_per_pixel) == (3, 8, 24)
    assert info.format == fmt
    assert info.file_size == len(data)
    assert info.raw_size == 53 * 37 * 3


def test_jpeg_metadata(rgb_array):
    data = encode(Image.fromarray(rgb_array), "JPEG", quality=95, dpi=(300, 300))
    rgb, info = load_image(data, "x.jpg")
    assert info.format == "JPEG"
    assert rgb.shape == (37, 53, 3) and rgb.dtype == np.uint8
    assert info.dpi == (300.0, 300.0)
    assert info.compression_ratio == pytest.approx(info.raw_size / len(data))


def test_grayscale_png_expands_to_three_channels():
    gray = np.arange(256, dtype=np.uint8).reshape(16, 16)
    rgb, info = load_image(encode(Image.fromarray(gray, "L"), "PNG"))
    assert info.mode == "L" and info.channels == 1
    assert np.array_equal(rgb[..., 0], gray) and np.array_equal(rgb[..., 0], rgb[..., 2])


def test_16bit_png_scaled_to_8bit():
    g16 = (np.arange(64, dtype=np.uint16) * 1040).reshape(8, 8)  # 0 … 65520
    rgb, info = load_image(encode(Image.fromarray(g16), "PNG"))
    assert info.bits_per_channel == 16
    assert np.array_equal(rgb[..., 0], np.rint(g16 / 257).astype(np.uint8))
    assert any("16-bit" in note for note in info.notes)


def test_rgba_composited_on_white():
    rgba = np.zeros((4, 4, 4), np.uint8)
    rgba[..., 0] = 255  # 完全透明的紅色
    rgba[0, 0] = (0, 0, 255, 255)  # 一個不透明的藍色像素
    rgb, info = load_image(encode(Image.fromarray(rgba, "RGBA"), "PNG"))
    assert info.has_alpha and info.channels == 4
    assert tuple(rgb[1, 1]) == (255, 255, 255)
    assert tuple(rgb[0, 0]) == (0, 0, 255)


def test_palette_and_bilevel_bmp(rgb_array):
    palette = Image.fromarray(rgb_array).quantize(16)  # 彩色調色盤（灰階調色盤會被 Pillow 讀成 L 模式）
    rgb, info = load_image(encode(palette, "BMP"))
    assert info.format == "BMP" and info.mode == "P" and rgb.shape == (37, 53, 3)
    assert np.array_equal(rgb, np.asarray(palette.convert("RGB")))

    checker = (np.indices((6, 6)).sum(0) % 2 * 255).astype(np.uint8)
    rgb, info = load_image(encode(Image.fromarray(checker).convert("1"), "BMP"))
    assert info.mode == "1" and info.bits_per_channel == 1
    assert set(np.unique(rgb)) <= {0, 255}


def test_cmyk_jpeg_converted():
    red = Image.new("CMYK", (8, 8), (0, 255, 255, 0))
    rgb, info = load_image(encode(red, "JPEG", quality=95))
    assert info.mode == "CMYK" and info.channels == 4
    r, g, b = rgb[4, 4].astype(int)
    assert r > 200 and g < 60 and b < 60


def test_exif_orientation_applied():
    arr = np.zeros((20, 40, 3), np.uint8)
    arr[:, :10] = 255  # 橫向影像，左側有白色帶
    img = Image.fromarray(arr)
    exif = img.getexif()
    exif[0x0112] = 6  # 顯示時需順時針旋轉 90°
    rgb, info = load_image(encode(img, "JPEG", exif=exif.tobytes(), quality=95))
    assert (info.width, info.height) == (20, 40)
    assert info.exif_transposed
    assert rgb[:8].mean() > 200 and rgb[-8:].mean() < 60  # 左側白帶轉到上方


@pytest.mark.parametrize(
    "data",
    [b"", b"not an image at all", b"\x89PNG\r\n\x1a\n" + b"\x00" * 20],
    ids=["empty", "text", "broken-png-header"],
)
def test_invalid_bytes_raise_friendly_error(data):
    with pytest.raises(ImageLoadError):
        load_image(data)


def test_truncated_jpeg(rgb_array):
    data = encode(Image.fromarray(rgb_array), "JPEG")
    with pytest.raises(ImageLoadError):
        load_image(data[: len(data) // 3])


def test_unsupported_format_rejected():
    with pytest.raises(ImageLoadError, match="不支援"):
        load_image(encode(Image.new("RGB", (4, 4)), "GIF"))


def test_pixel_limit(monkeypatch):
    monkeypatch.setattr(io_utils, "MAX_PIXELS", 100)
    with pytest.raises(ImageLoadError, match="超過上限"):
        io_utils.load_image(encode(Image.new("RGB", (20, 20)), "PNG"))


def test_one_pixel_image():
    rgb, info = load_image(encode(Image.new("RGB", (1, 1), (10, 20, 30)), "PNG"))
    assert rgb.shape == (1, 1, 3) and tuple(rgb[0, 0]) == (10, 20, 30)


def test_limit_size():
    img = np.zeros((1000, 3000, 3), np.uint8)
    out, scale = limit_size(img, 1500)
    assert out.shape == (500, 1500, 3) and scale == pytest.approx(0.5)
    same, scale = limit_size(img, 4000)
    assert same is img and scale == 1.0


def test_human_size():
    assert human_size(512) == "512 B"
    assert human_size(2048) == "2.00 KB"
    assert human_size(5 * 1024**2) == "5.00 MB"


def test_png_bytes_roundtrip(rgb_array):
    rgb, _ = load_image(to_png_bytes(rgb_array))
    assert np.array_equal(rgb, rgb_array)


@pytest.mark.parametrize(
    "name, ok",
    [
        ("01_tiny_1x1.png", True),
        ("02_tiny_3x2.bmp", True),
        ("03_gray_16bit.png", True),
        ("04_rgba_transparent.png", True),
        ("05_palette_8bit.bmp", True),
        ("06_bilevel_1bit.bmp", True),
        ("07_cmyk.jpg", True),
        ("08_exif_rotate90.jpg", True),
        ("09_flat_gray.png", True),
        ("10_large_6000x4000.jpg", True),
        ("11_corrupted.jpg", False),
        ("12_not_an_image.png", False),
    ],
)
def test_bundled_edge_case_files(name, ok):
    path = io_utils.__file__
    from pathlib import Path

    data = (Path(path).resolve().parents[1] / "samples" / "edge_cases" / name).read_bytes()
    if ok:
        rgb, info = load_image(data, name)
        assert rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3
    else:
        with pytest.raises(ImageLoadError):
            load_image(data, name)
