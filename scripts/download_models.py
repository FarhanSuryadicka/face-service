"""
Unduh model YuNet & SFace dari opencv_zoo dan verifikasi checksum SHA-256.
Model anti-spoof (hasil tools/convert_antispoof.py) sudah ikut di repo, hanya diverifikasi.

    python scripts/download_models.py [folder_models]
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"
MODELS = {
    "face_detection_yunet_2023mar.onnx": (
        f"{ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    ),
    "face_recognition_sface_2021dec.onnx": (
        f"{ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    ),
    "antispoof_minifasnet_v2_2.7_80x80.onnx": (
        None,
        "12b4e0402bd3113b17b08cd1e9da2ceed594c5896f9e066485efab333c480ac8",
    ),
    "antispoof_minifasnet_v1se_4.0_80x80.onnx": (
        None,
        "8f67117591ca23212e36b91559c39c07de09f57ce552d1534bf53832055502cd",
    ),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "models")
    out.mkdir(parents=True, exist_ok=True)
    ok = True
    for name, (url, expected) in MODELS.items():
        path = out / name
        if path.is_file() and sha256(path) == expected:
            print(f"[ok]   {name}")
            continue
        if url is None:
            print(f"[FAIL] {name} tidak ada/rusak. Buat ulang dengan tools/convert_antispoof.py")
            ok = False
            continue
        print(f"[get]  {name}")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        if sha256(tmp) != expected:
            tmp.unlink()
            print(f"[FAIL] checksum {name} tidak cocok")
            ok = False
            continue
        tmp.replace(path)
        print(f"[ok]   {name}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
