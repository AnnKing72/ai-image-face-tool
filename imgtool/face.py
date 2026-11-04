"""功能 4：人臉偵測（YuNet 輕量深度學習模型 / Haar Cascade 經典特徵）。

YuNet 只對約 10～300 像素大小的人臉最準（官方說明），高解析度特寫照片的大臉
容易漏偵測或產生誤判。因此預設使用「影像金字塔 + 尺寸分帶」的多尺度偵測：
每次把影像縮小一半，各層只採用落在可靠尺寸範圍內的偵測框，
最後把各層結果換算回原圖座標，再用 NMS（非極大值抑制）合併重複的框。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

YUNET_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_detection_yunet_2023mar.onnx"
HAAR_CASCADE_FILE = "haarcascade_frontalface_default.xml"

# YuNet 輸出的 5 個關鍵點（左右以照片中人物自身為準，與 OpenCV 文件一致）
LANDMARK_NAMES = ("右眼", "左眼", "鼻尖", "右嘴角", "左嘴角")
LANDMARK_COLORS = ((255, 64, 64), (64, 128, 255), (64, 220, 64), (255, 64, 255), (255, 220, 0))
BOX_COLOR = (0, 230, 118)


@dataclass(frozen=True)
class Face:
    x: float
    y: float
    w: float
    h: float
    score: float
    landmarks: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True)
class DetectionResult:
    faces: tuple[Face, ...]
    method: str  # "yunet" / "haar"
    elapsed_ms: float
    image_size: tuple[int, int]  # (寬, 高)
    scales: tuple[float, ...] = (1.0,)

    @property
    def count(self) -> int:
        return len(self.faces)

    @property
    def mean_score(self) -> float:
        return float(np.mean([f.score for f in self.faces])) if self.faces else float("nan")


def _check_rgb(rgb: np.ndarray) -> None:
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"需要 (H, W, 3) 的 uint8 RGB 影像，收到 shape={rgb.shape}、dtype={rgb.dtype}")


def _clip_face(face: Face, width: int, height: int) -> Face | None:
    """把框裁到影像範圍內；完全落在影像外的框丟棄。"""
    x1, y1 = max(0.0, face.x), max(0.0, face.y)
    x2, y2 = min(float(width), face.x + face.w), min(float(height), face.y + face.h)
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return Face(x1, y1, x2 - x1, y2 - y1, face.score, face.landmarks)


def _sort_reading_order(faces: list[Face]) -> tuple[Face, ...]:
    """由左到右編號，方便對照畫面上的 #1、#2…"""
    return tuple(sorted(faces, key=lambda f: (f.x + f.w / 2, f.y)))


class YuNetDetector:
    """OpenCV FaceDetectorYN（YuNet 2023mar, ONNX, 約 230 KB）。"""

    BAND_MAX = 200  # 非最粗層：只採用邊長 ≤ 200 px 的框（在模型可靠範圍內）
    BAND_MIN = 40  # 非最細層：只採用邊長 ≥ 40 px 的框（小臉交給更細的層）
    MIN_LEVEL_SIDE = 160  # 金字塔最小一層的長邊下限

    def __init__(self, model_path: str | Path = YUNET_MODEL_PATH) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"找不到 YuNet 模型檔：{path}")
        self._net = cv2.FaceDetectorYN.create(str(path), "", (320, 320), 0.7, 0.3, 5000)
        self._lock = threading.Lock()  # 偵測器物件有狀態（輸入尺寸），多人同時使用時要上鎖

    def _infer(self, bgr: np.ndarray, score_threshold: float, nms_threshold: float, top_k: int) -> np.ndarray:
        height, width = bgr.shape[:2]
        with self._lock:
            self._net.setInputSize((width, height))
            self._net.setScoreThreshold(float(score_threshold))
            self._net.setNMSThreshold(float(nms_threshold))
            self._net.setTopK(int(top_k))
            _, faces = self._net.detect(bgr)
        return np.empty((0, 15), np.float32) if faces is None else faces

    def detect(
        self,
        rgb: np.ndarray,
        score_threshold: float = 0.7,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
        multiscale: bool = True,
    ) -> DetectionResult:
        _check_rgb(rgb)
        if not 0.0 < score_threshold < 1.0:
            raise ValueError("信心度門檻必須介於 0 與 1 之間")
        if not 0.0 < nms_threshold < 1.0:
            raise ValueError("NMS 門檻必須介於 0 與 1 之間")
        start = time.perf_counter()
        bgr = np.ascontiguousarray(rgb[..., ::-1])  # OpenCV 模型吃 BGR
        height, width = bgr.shape[:2]

        scales = [1.0]
        if multiscale:
            while max(height, width) * scales[-1] / 2 >= self.MIN_LEVEL_SIDE:
                scales.append(scales[-1] / 2)

        candidates: list[np.ndarray] = []
        for level, scale in enumerate(scales):
            if scale == 1.0:
                level_img = bgr
            else:
                size = (max(1, round(width * scale)), max(1, round(height * scale)))
                level_img = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
            rx = width / level_img.shape[1]
            ry = height / level_img.shape[0]
            for row in self._infer(level_img, score_threshold, nms_threshold, top_k):
                side = max(row[2], row[3])
                if level < len(scales) - 1 and side > self.BAND_MAX:
                    continue
                if level > 0 and side < self.BAND_MIN:
                    continue
                mapped = row.astype(np.float64)
                mapped[0:14:2] *= rx  # x、w 與關鍵點 x 座標
                mapped[1:14:2] *= ry  # y、h 與關鍵點 y 座標
                candidates.append(mapped)

        if len(scales) > 1 and candidates:
            stacked = np.array(candidates)
            keep = cv2.dnn.NMSBoxes(
                stacked[:, :4].tolist(), stacked[:, 14].tolist(), float(score_threshold), float(nms_threshold)
            )
            candidates = [stacked[i] for i in np.asarray(keep, dtype=int).ravel()]

        faces: list[Face] = []
        for row in candidates:
            landmarks = tuple((float(row[4 + 2 * i]), float(row[5 + 2 * i])) for i in range(5))
            clipped = _clip_face(Face(*map(float, row[:4]), float(row[14]), landmarks), width, height)
            if clipped is not None:
                faces.append(clipped)
        return DetectionResult(
            faces=_sort_reading_order(faces),
            method="yunet",
            elapsed_ms=(time.perf_counter() - start) * 1000,
            image_size=(width, height),
            scales=tuple(scales),
        )


class HaarDetector:
    """Viola–Jones Haar Cascade（OpenCV 內建 haarcascade_frontalface_default.xml）。

    信心分數使用 detectMultiScale3 的 levelWeight：最後一個 stage 的加權分數，
    數值越大代表越像人臉，但它不是 0～1 的機率。
    """

    def __init__(self, cascade_file: str = HAAR_CASCADE_FILE) -> None:
        path = Path(cv2.data.haarcascades) / cascade_file
        self._classifier = cv2.CascadeClassifier(str(path))
        if self._classifier.empty():
            raise RuntimeError(f"無法載入 Haar 分類器：{path}")
        self._lock = threading.Lock()

    def detect(
        self,
        rgb: np.ndarray,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: int = 30,
    ) -> DetectionResult:
        _check_rgb(rgb)
        if not scale_factor > 1.0:
            raise ValueError("scaleFactor 必須大於 1")
        start = time.perf_counter()
        gray = cv2.equalizeHist(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))  # 直方圖等化，降低光線影響
        with self._lock:
            rects, _levels, weights = self._classifier.detectMultiScale3(
                gray,
                scaleFactor=float(scale_factor),
                minNeighbors=int(min_neighbors),
                minSize=(int(min_size), int(min_size)),
                outputRejectLevels=True,
            )
        height, width = gray.shape
        faces = []
        for (x, y, w, h), score in zip(np.asarray(rects).reshape(-1, 4), np.ravel(weights)):
            clipped = _clip_face(Face(float(x), float(y), float(w), float(h), float(score)), width, height)
            if clipped is not None:
                faces.append(clipped)
        return DetectionResult(
            faces=_sort_reading_order(faces),
            method="haar",
            elapsed_ms=(time.perf_counter() - start) * 1000,
            image_size=(width, height),
        )


def draw_faces(
    rgb: np.ndarray,
    faces: tuple[Face, ...] | list[Face],
    *,
    show_landmarks: bool = True,
    score_decimals: int = 2,
) -> np.ndarray:
    """在影像上畫出邊界框、編號與信心分數（以及 YuNet 的五官關鍵點）。"""
    _check_rgb(rgb)
    out = rgb.copy()
    height, width = out.shape[:2]
    longest = max(height, width)
    thickness = max(2, round(longest / 400))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, longest / 1600)
    text_thickness = max(1, thickness // 2)

    for index, face in enumerate(faces, 1):
        x1, y1 = int(round(face.x)), int(round(face.y))
        x2 = min(width - 1, int(round(face.x + face.w)))
        y2 = min(height - 1, int(round(face.y + face.h)))
        cv2.rectangle(out, (x1, y1), (x2, y2), BOX_COLOR, thickness, cv2.LINE_AA)

        label = f"#{index} {face.score:.{score_decimals}f}"
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
        box_h = th + baseline + 6
        top = y1 - box_h if y1 - box_h >= 0 else y1  # 上方放不下就畫在框內
        cv2.rectangle(out, (x1, top), (min(width - 1, x1 + tw + 6), top + box_h), BOX_COLOR, -1)
        cv2.putText(out, label, (x1 + 3, top + th + 3), font, font_scale, (0, 0, 0), text_thickness, cv2.LINE_AA)

        if show_landmarks:
            radius = max(2, thickness)
            for (px, py), color in zip(face.landmarks, LANDMARK_COLORS):
                cv2.circle(out, (int(round(px)), int(round(py))), radius, color, -1, cv2.LINE_AA)
    return out
