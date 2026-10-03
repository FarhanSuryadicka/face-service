import os
from pathlib import Path

os.environ.setdefault("FACE_API_KEY", "test-key")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
AUTH = {"X-API-Key": "test-key"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _jpg(img: np.ndarray) -> bytes:
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()


def _file(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def test_health_tanpa_auth(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["models"]["antispoof"] is not None


def test_embed_tanpa_api_key_ditolak(client):
    r = client.post("/embed", files={"image": ("a.jpg", _file("real_face.jpg"))})
    assert r.status_code == 401
    assert r.json()["error"] == "unauthorized"


def test_embed_api_key_salah_ditolak(client):
    r = client.post("/embed", headers={"X-API-Key": "salah"}, files={"image": ("a.jpg", _file("real_face.jpg"))})
    assert r.status_code == 401


def test_embed_bukan_gambar(client):
    r = client.post("/embed", headers=AUTH, files={"image": ("a.jpg", b"bukan gambar")})
    assert r.status_code == 400
    assert r.json()["error"] == "invalid_image"


def test_embed_gambar_terlalu_besar(client):
    big = b"\xff" * (3 * 1024 * 1024 + 10)
    r = client.post("/embed", headers=AUTH, files={"image": ("a.jpg", big)})
    assert r.status_code == 413


def test_embed_tanpa_wajah(client):
    blank = np.full((480, 640, 3), 127, np.uint8)
    r = client.post("/embed", headers=AUTH, files={"image": ("a.jpg", _jpg(blank))})
    assert r.status_code == 422
    assert r.json()["error"] == "no_face"


def test_embed_wajah_asli(client):
    r = client.post("/embed", headers=AUTH, files={"image": ("a.jpg", _file("real_face.jpg"))})
    assert r.status_code == 200
    body = r.json()
    emb = np.array(body["embedding"])
    assert emb.shape == (128,)
    assert abs(np.linalg.norm(emb) - 1.0) < 1e-3
    assert body["faces_count"] == 1
    assert body["quality"]["passed"] is True
    assert body["spoof"]["is_real"] is True


def test_embed_foto_palsu_terdeteksi_spoof(client):
    r = client.post("/embed", headers=AUTH, files={"image": ("a.jpg", _file("spoof_face.jpg"))})
    assert r.status_code == 200
    assert r.json()["spoof"]["is_real"] is False


def test_compare_orang_sama_beda_kondisi(client):
    """Foto yang sama dengan skala & pencahayaan berbeda harus tetap cocok."""
    img = cv2.imdecode(np.frombuffer(_file("real_face.jpg"), np.uint8), cv2.IMREAD_COLOR)
    variant = cv2.convertScaleAbs(cv2.resize(img, None, fx=0.8, fy=0.8), alpha=1.0, beta=-25)
    r = client.post(
        "/compare",
        headers=AUTH,
        files={"image_a": ("a.jpg", _jpg(img)), "image_b": ("b.jpg", _jpg(variant))},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["match"] is True
    assert body["similarity"] > 0.8
    assert "embedding" not in body["a"]


def test_quality_wajah_kecil_dan_gelap(client):
    img = cv2.imdecode(np.frombuffer(_file("real_face.jpg"), np.uint8), cv2.IMREAD_COLOR)
    small_dark = cv2.convertScaleAbs(cv2.resize(img, None, fx=0.45, fy=0.45), alpha=0.35, beta=0)
    r = client.post("/embed", headers=AUTH, files={"image": ("a.jpg", _jpg(small_dark))})
    assert r.status_code == 200
    issues = r.json()["quality"]["issues"]
    assert "face_too_small" in issues
    assert "too_dark" in issues


def test_demo_mati_secara_default(client):
    assert client.get("/demo").status_code == 404
