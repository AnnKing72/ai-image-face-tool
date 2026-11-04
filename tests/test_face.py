"""功能 4：人臉偵測。

有臉的測試使用 scikit-image 內建、公有領域的 astronaut 測試影像（未安裝時自動略過）。
"""

import cv2
import numpy as np
import pytest

from imgtool import face as fd


@pytest.fixture(scope="module")
def yunet():
    return fd.YuNetDetector()


@pytest.fixture(scope="module")
def haar():
    return fd.HaarDetector()


@pytest.fixture(scope="module")
def portrait():
    data = pytest.importorskip("skimage.data")
    return data.astronaut()  # 512×512，一張正臉


def make_group(portrait):
    """把同一張臉以 6 種大小貼到畫布上，模擬多人臉場景。"""
    crop = portrait[30:250, 120:340]
    canvas = np.full((900, 1400, 3), 200, np.uint8)
    for x, y, s in [(30, 40, 0.5), (200, 40, 0.8), (450, 60, 1.0), (750, 40, 1.3), (60, 450, 0.35), (300, 450, 1.6)]:
        patch = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        canvas[y : y + patch.shape[0], x : x + patch.shape[1]] = patch
    return canvas


def test_model_file_is_the_official_yunet():
    assert fd.YUNET_MODEL_PATH.is_file()
    assert fd.YUNET_MODEL_PATH.stat().st_size == 232589


def test_missing_model_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        fd.YuNetDetector(tmp_path / "missing.onnx")


@pytest.mark.parametrize("shape", [(1, 1), (10, 10), (240, 320), (480, 2000)])
def test_blank_images_have_no_faces(yunet, haar, shape):
    blank = np.full(shape + (3,), 127, np.uint8)
    assert yunet.detect(blank).count == 0
    assert haar.detect(blank).count == 0


@pytest.mark.parametrize("scale", [0.5, 1.0, 3.0])
def test_yunet_single_face(yunet, portrait, scale):
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    image = cv2.resize(portrait, None, fx=scale, fy=scale, interpolation=interp)
    result = yunet.detect(image)
    assert result.count == 1
    f = result.faces[0]
    assert f.score > 0.85
    height, width = image.shape[:2]
    assert 0 <= f.x and 0 <= f.y and f.x + f.w <= width and f.y + f.h <= height
    assert len(f.landmarks) == 5
    assert f.x / scale == pytest.approx(180, abs=25) and f.w / scale == pytest.approx(90, abs=25)


def test_multiscale_removes_false_positives_on_large_face(yunet, portrait):
    big = cv2.resize(portrait, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)  # 2048 px，臉約 360 px
    multi = yunet.detect(big, multiscale=True)
    single = yunet.detect(big, multiscale=False)
    assert multi.count == 1 and multi.faces[0].score > 0.85
    assert single.count >= multi.count
    assert len(multi.scales) > 1


@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
def test_multiple_faces(yunet, portrait, scale):
    group = make_group(portrait)
    group = cv2.resize(group, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    result = yunet.detect(group)
    assert result.count == 6
    centers = [f.x + f.w / 2 for f in result.faces]
    assert centers == sorted(centers)  # 編號由左到右


def test_threshold_is_monotonic(yunet, portrait):
    group = make_group(portrait)
    assert yunet.detect(group, score_threshold=0.99).count <= yunet.detect(group, score_threshold=0.3).count


def test_haar_detects_face_with_level_weight(haar, portrait):
    result = haar.detect(portrait)
    assert result.count >= 1
    assert all(f.score > 0 for f in result.faces)


def test_invalid_parameters(yunet, haar, portrait):
    with pytest.raises(ValueError):
        yunet.detect(portrait, score_threshold=1.5)
    with pytest.raises(ValueError):
        haar.detect(portrait, scale_factor=1.0)
    with pytest.raises(ValueError):
        yunet.detect(portrait[..., 0])


def test_draw_faces_does_not_modify_input(portrait):
    original = portrait.copy()
    faces = (fd.Face(-10, -10, 100, 100, 0.95, ((5.0, 5.0),) * 5), fd.Face(400, 450, 200, 200, 0.5))
    out = fd.draw_faces(portrait, faces)
    assert out.shape == portrait.shape and not np.array_equal(out, portrait)
    assert np.array_equal(portrait, original)
