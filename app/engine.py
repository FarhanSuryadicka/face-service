"""
Engine wajah berbasis OpenCV DNN (tanpa PyTorch/ONNX Runtime):
  - YuNet   : deteksi wajah + 5 landmark  (MIT)
  - SFace   : embedding 128-d             (Apache-2.0)
  - MiniFASNet V2 + V1SE : anti-spoofing  (Apache-2.0, Silent-Face-Anti-Spoofing)

Objek OpenCV DNN tidak thread-safe, jadi semua inferensi dijaga satu lock.
Satu permintaan hanya butuh puluhan milidetik, sehingga antrean ini tetap cepat.
"""
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import Settings

YUNET_FILE = "face_detection_yunet_2023mar.onnx"
SFACE_FILE = "face_recognition_sface_2021dec.onnx"
# (nama file, skala crop di sekitar bbox) sesuai model asli Silent-Face
ANTISPOOF_FILES = [
    ("antispoof_minifasnet_v2_2.7_80x80.onnx", 2.7),
    ("antispoof_minifasnet_v1se_4.0_80x80.onnx", 4.0),
]
ANTISPOOF_INPUT = 80
EMBEDDING_DIM = 128


class FaceError(Exception):
    def __init__(self, code: str, message: str, status: int = 422):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


@dataclass
class EmbedResult:
    faces_count: int
    bbox: List[int]
    det_score: float
    landmarks: List[List[float]]
    quality: Dict
    spoof: Optional[Dict]
    embedding: np.ndarray
    timing_ms: Dict[str, float]

    def to_dict(self) -> Dict:
        return {
            "faces_count": self.faces_count,
            "face": {
                "bbox": self.bbox,
                "score": round(self.det_score, 4),
                "landmarks": self.landmarks,
            },
            "quality": self.quality,
            "spoof": self.spoof,
            "embedding": [round(float(v), 6) for v in self.embedding],
            "timing_ms": self.timing_ms,
        }


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


def _antispoof_box(src_w: int, src_h: int, bbox: Tuple[float, float, float, float], scale: float):
    """Perluasan bbox persis seperti CropImage._get_new_box di repo Silent-Face."""
    x, y, box_w, box_h = bbox
    scale = min((src_h - 1) / box_h, min((src_w - 1) / box_w, scale))
    new_w, new_h = box_w * scale, box_h * scale
    cx, cy = box_w / 2 + x, box_h / 2 + y
    x1, y1 = cx - new_w / 2, cy - new_h / 2
    x2, y2 = cx + new_w / 2, cy + new_h / 2
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > src_w - 1:
        x1 -= x2 - src_w + 1
        x2 = src_w - 1
    if y2 > src_h - 1:
        y1 -= y2 - src_h + 1
        y2 = src_h - 1
    return int(x1), int(y1), int(x2), int(y2)


class FaceEngine:
    def __init__(self, settings: Settings):
        self.s = settings
        cv2.setNumThreads(max(1, settings.threads))
        d = settings.models_dir
        for name in [YUNET_FILE, SFACE_FILE]:
            if not (d / name).is_file():
                raise RuntimeError(f"Model tidak ditemukan: {d / name} (jalankan scripts/download_models.py)")

        self._lock = threading.Lock()
        self._detector = cv2.FaceDetectorYN.create(
            str(d / YUNET_FILE), "", (320, 320), settings.det_score, 0.3, 5000
        )
        self._recognizer = cv2.FaceRecognizerSF.create(str(d / SFACE_FILE), "")

        self._spoof_nets: List[Tuple[cv2.dnn.Net, float]] = []
        if settings.antispoof:
            for name, scale in ANTISPOOF_FILES:
                path = d / name
                if not path.is_file():
                    raise RuntimeError(f"Model anti-spoof tidak ditemukan: {path} (atau set FACE_ANTISPOOF=false)")
                self._spoof_nets.append((cv2.dnn.readNetFromONNX(str(path)), scale))

    @property
    def antispoof_enabled(self) -> bool:
        return bool(self._spoof_nets)

    # ------------------------------------------------------------------ input

    def decode(self, data: bytes) -> np.ndarray:
        if not data:
            raise FaceError("empty_image", "File gambar kosong", 400)
        if len(data) > self.s.max_image_bytes:
            raise FaceError("image_too_large", f"Ukuran gambar melebihi {self.s.max_image_bytes} byte", 413)
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise FaceError("invalid_image", "File bukan gambar yang valid (gunakan JPEG/PNG)", 400)
        h, w = img.shape[:2]
        longest = max(h, w)
        if longest > self.s.max_side:
            f = self.s.max_side / longest
            img = cv2.resize(img, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
        return img

    # --------------------------------------------------------------- pipeline

    def embed(self, img: np.ndarray) -> EmbedResult:
        timing: Dict[str, float] = {}
        h, w = img.shape[:2]

        with self._lock:
            t = time.perf_counter()
            self._detector.setInputSize((w, h))
            _, faces = self._detector.detect(img)
            timing["detect"] = _ms(t)

            if faces is None or len(faces) == 0:
                raise FaceError("no_face", "Wajah tidak terdeteksi", 422)

            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            face = faces[0]

            t = time.perf_counter()
            aligned = self._recognizer.alignCrop(img, face)
            feat = self._recognizer.feature(aligned).flatten().astype(np.float32)
            timing["embed"] = _ms(t)

            spoof = None
            if self._spoof_nets:
                t = time.perf_counter()
                spoof = self._antispoof(img, face)
                timing["antispoof"] = _ms(t)

        norm = float(np.linalg.norm(feat))
        if norm == 0:
            raise FaceError("embedding_failed", "Gagal menghitung embedding", 500)
        feat /= norm

        quality = self._quality(face, faces, aligned)
        timing["total"] = round(sum(timing.values()), 1)

        x, y, bw, bh = (int(round(v)) for v in face[:4])
        landmarks = [[round(float(face[4 + i * 2]), 1), round(float(face[5 + i * 2]), 1)] for i in range(5)]
        return EmbedResult(
            faces_count=len(faces),
            bbox=[x, y, bw, bh],
            det_score=float(face[14]),
            landmarks=landmarks,
            quality=quality,
            spoof=spoof,
            embedding=feat,
            timing_ms=timing,
        )

    # ---------------------------------------------------------------- helpers

    def _antispoof(self, img: np.ndarray, face: np.ndarray) -> Dict:
        h, w = img.shape[:2]
        bbox = (float(face[0]), float(face[1]), float(face[2]), float(face[3]))
        probs = np.zeros(3, dtype=np.float64)
        for net, scale in self._spoof_nets:
            x1, y1, x2, y2 = _antispoof_box(w, h, bbox, scale)
            crop = cv2.resize(img[y1:y2 + 1, x1:x2 + 1], (ANTISPOOF_INPUT, ANTISPOOF_INPUT))
            # model dilatih dengan BGR 0..255 tanpa normalisasi
            blob = crop.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
            net.setInput(blob)
            logits = net.forward().flatten().astype(np.float64)
            e = np.exp(logits - logits.max())
            probs += e / e.sum()
        probs /= len(self._spoof_nets)
        real = float(probs[1])  # kelas 1 = wajah asli
        return {
            "real_score": round(real, 4),
            "is_real": real >= self.s.spoof_threshold,
            "threshold": self.s.spoof_threshold,
        }

    def _quality(self, face: np.ndarray, faces: List[np.ndarray], aligned: np.ndarray) -> Dict:
        issues: List[str] = []
        face_w = float(face[2])

        # wajah lain yang cukup besar (bukan sekadar orang jauh di latar)
        main_area = face[2] * face[3]
        others = [f for f in faces[1:] if f[2] * f[3] >= 0.25 * main_area]
        if others:
            issues.append("multiple_faces")

        if face_w < self.s.min_face_px:
            issues.append("face_too_small")

        gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())
        if blur < self.s.min_blur:
            issues.append("blurry")
        if brightness < self.s.min_brightness:
            issues.append("too_dark")
        elif brightness > self.s.max_brightness:
            issues.append("too_bright")

        # perkiraan yaw: posisi hidung terhadap titik tengah kedua mata,
        # diproyeksikan ke garis antar-mata (tahan terhadap kepala miring/roll)
        re = np.array([face[4], face[5]])
        le = np.array([face[6], face[7]])
        nose = np.array([face[8], face[9]])
        eye_vec = le - re
        eye_dist = float(np.linalg.norm(eye_vec))
        yaw = 0.0
        if eye_dist > 0:
            yaw = float(np.dot(nose - (re + le) / 2, eye_vec / eye_dist) / eye_dist)
        if abs(yaw) > self.s.max_yaw:
            issues.append("not_frontal")

        return {
            "passed": not issues,
            "issues": issues,
            "face_width_px": round(face_w, 1),
            "blur": round(blur, 1),
            "brightness": round(brightness, 1),
            "yaw": round(yaw, 3),
        }


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Embedding sudah dinormalisasi L2, jadi cosine = dot product."""
    return float(np.dot(a, b))
