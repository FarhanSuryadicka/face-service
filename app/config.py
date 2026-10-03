import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _bool(name: str, default: bool) -> bool:
    return _env(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    api_key: str = field(default_factory=lambda: _env("FACE_API_KEY", ""))
    models_dir: Path = field(
        default_factory=lambda: Path(_env("FACE_MODELS_DIR", str(Path(__file__).resolve().parent.parent / "models")))
    )
    threads: int = field(default_factory=lambda: int(_env("FACE_THREADS", "2")))

    # input
    max_image_bytes: int = field(default_factory=lambda: int(_env("FACE_MAX_IMAGE_BYTES", str(3 * 1024 * 1024))))
    max_side: int = field(default_factory=lambda: int(_env("FACE_MAX_SIDE", "1280")))

    # deteksi & kualitas
    det_score: float = field(default_factory=lambda: float(_env("FACE_DET_SCORE", "0.8")))
    min_face_px: int = field(default_factory=lambda: int(_env("FACE_MIN_FACE_PX", "100")))
    min_blur: float = field(default_factory=lambda: float(_env("FACE_MIN_BLUR", "40")))
    min_brightness: float = field(default_factory=lambda: float(_env("FACE_MIN_BRIGHTNESS", "50")))
    max_brightness: float = field(default_factory=lambda: float(_env("FACE_MAX_BRIGHTNESS", "215")))
    max_yaw: float = field(default_factory=lambda: float(_env("FACE_MAX_YAW", "0.35")))

    # anti-spoofing
    antispoof: bool = field(default_factory=lambda: _bool("FACE_ANTISPOOF", True))
    spoof_threshold: float = field(default_factory=lambda: float(_env("FACE_SPOOF_THRESHOLD", "0.6")))

    # threshold cosine SFace (rekomendasi OpenCV: 0.363). Dipakai /compare; Laravel punya threshold sendiri.
    match_threshold: float = field(default_factory=lambda: float(_env("FACE_MATCH_THRESHOLD", "0.363")))


settings = Settings()
