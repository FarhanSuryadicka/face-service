import hmac
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, File, Header, Request, UploadFile
from fastapi.responses import JSONResponse

from .config import settings
from .engine import EMBEDDING_DIM, FaceEngine, FaceError, cosine

VERSION = "1.0.0"
log = logging.getLogger("face-service")
engine: Optional[FaceEngine] = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global engine
    if not settings.api_key:
        raise RuntimeError("FACE_API_KEY wajib diisi")
    engine = FaceEngine(settings)
    log.info("Model siap (anti-spoof: %s)", engine.antispoof_enabled)
    yield


app = FastAPI(title="Face Service", version=VERSION, lifespan=lifespan, docs_url=None, redoc_url=None)


@app.exception_handler(FaceError)
async def face_error_handler(_: Request, exc: FaceError):
    return JSONResponse(status_code=exc.status, content={"error": exc.code, "message": exc.message})


def require_api_key(x_api_key: str = Header(default="")) -> None:
    if not hmac.compare_digest(x_api_key.encode(), settings.api_key.encode()):
        raise FaceError("unauthorized", "API key tidak valid", 401)


def _read(upload: UploadFile) -> bytes:
    # baca maksimal batas + 1 byte supaya file raksasa tidak dimuat penuh ke RAM
    return upload.file.read(settings.max_image_bytes + 1)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": VERSION,
        "models": {
            "detector": "yunet_2023mar",
            "recognizer": "sface_2021dec",
            "embedding_dim": EMBEDDING_DIM,
            "antispoof": "minifasnet_v2+v1se" if engine and engine.antispoof_enabled else None,
        },
    }


@app.post("/embed", dependencies=[Depends(require_api_key)])
def embed(image: UploadFile = File(...)):
    """Selfie -> embedding + kualitas + skor anti-spoof. Keputusan akhir di Laravel."""
    result = engine.embed(engine.decode(_read(image)))
    return result.to_dict()


@app.post("/compare", dependencies=[Depends(require_api_key)])
def compare(image_a: UploadFile = File(...), image_b: UploadFile = File(...)):
    """Bandingkan dua foto langsung. Untuk pengujian & kalibrasi threshold."""
    a = engine.embed(engine.decode(_read(image_a)))
    b = engine.embed(engine.decode(_read(image_b)))
    similarity = cosine(a.embedding, b.embedding)

    def summary(r):
        d = r.to_dict()
        d.pop("embedding")
        return d

    return {
        "similarity": round(similarity, 4),
        "match": similarity >= settings.match_threshold,
        "threshold": settings.match_threshold,
        "a": summary(a),
        "b": summary(b),
    }
