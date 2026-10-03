# Face Service

Service internal untuk absensi HRIS: menerima selfie, lalu mengembalikan **embedding wajah**, **cek kualitas foto**, dan **skor anti-spoofing**.
Keputusan absensi (cocok/tidak, jam shift, geofence) tetap dibuat oleh **Laravel**.

| Komponen | Model | Lisensi |
|---|---|---|
| Deteksi wajah + 5 landmark | YuNet (2023mar) | MIT |
| Embedding wajah 128-d | SFace (2021dec) | Apache-2.0 |
| Anti-spoofing (foto/layar) | MiniFASNet V2 + V1SE (Silent-Face) | Apache-2.0 |

Semua dijalankan dengan OpenCV DNN di CPU, tanpa PyTorch dan tanpa GPU. Ketiga model aman untuk penggunaan komersial.

Hasil ukur di laptop developer: sekitar **55 ms/foto**, sekitar **18 req/detik** per proses, RAM sekitar **180 MB** per proses.

---

## API

Semua endpoint selain `/health` wajib header `X-API-Key`.

### `GET /health`
```json
{"status":"ok","version":"1.0.0","models":{"detector":"yunet_2023mar","recognizer":"sface_2021dec","embedding_dim":128,"antispoof":"minifasnet_v2+v1se"}}
```

### `POST /embed` (multipart, field `image`)
```json
{
  "faces_count": 1,
  "face": {"bbox": [119,113,170,240], "score": 0.93, "landmarks": [[188.7,207.4], "..."]},
  "quality": {"passed": true, "issues": [], "face_width_px": 169.6, "blur": 1169.3, "brightness": 118.3, "yaw": 0.166},
  "spoof": {"real_score": 0.9999, "is_real": true, "threshold": 0.6},
  "embedding": [0.0123, "... 128 angka, sudah dinormalisasi L2"],
  "timing_ms": {"detect": 22.1, "embed": 24.8, "antispoof": 5.2, "total": 52.1}
}
```

Kode `quality.issues`:

| Kode | Arti | Pesan untuk pengguna |
|---|---|---|
| `multiple_faces` | Ada lebih dari satu wajah besar di foto | "Pastikan hanya Anda di kamera" |
| `face_too_small` | Wajah terlalu jauh | "Dekatkan wajah ke kamera" |
| `blurry` | Foto buram | "Tahan HP dengan stabil" |
| `too_dark` / `too_bright` | Pencahayaan buruk | "Cari tempat yang lebih terang / hindari cahaya dari belakang" |
| `not_frontal` | Wajah menoleh | "Hadap lurus ke kamera" |

Error (format `{"error": "...", "message": "..."}`):

| HTTP | `error` |
|---|---|
| 400 | `empty_image`, `invalid_image` |
| 401 | `unauthorized` |
| 413 | `image_too_large` |
| 422 | `no_face` |

### `POST /compare` (multipart, field `image_a` & `image_b`)
Membandingkan dua foto secara langsung. Dipakai untuk **pengujian dan kalibrasi threshold**, bukan untuk alur absensi.

---

## Cara Laravel memakai service ini

1. **Pendaftaran wajah:** kirim 3–5 selfie ke `/embed`. Tolak foto yang `quality.passed = false` atau `spoof.is_real = false`. Simpan `embedding` (JSON, 128 angka) di tabel `employee_faces`.
2. **Absen:** kirim selfie ke `/embed`, lalu Laravel menghitung:
   ```php
   // embedding sudah dinormalisasi L2 → cosine similarity = dot product
   $similarity = max(array_map(fn ($ref) => array_sum(array_map(fn ($a, $b) => $a * $b, $ref, $selfie)), $references));
   $cocok = $similarity >= 0.45;   // kalibrasi! lihat bagian Threshold
   ```
3. Absensi sah kalau `quality.passed`, `spoof.is_real`, dan `$cocok` semuanya bernilai true, ditambah validasi shift, geofence, dan device di Laravel.

## Threshold

- **Kecocokan wajah (cosine SFace):** rekomendasi OpenCV adalah `0.363`, diukur di dataset LFW. Untuk absensi, mulai dari **0.40–0.45** supaya lebih ketat. Setelah itu kalibrasi dengan data karyawan sendiri: catat skor untuk pasangan "orang yang sama" dan "orang berbeda", lalu pilih nilai di antaranya.
- **Anti-spoof:** default `0.6`. Model ini dilatih dengan detektor wajah lain, jadi skornya bisa sedikit berbeda di HP dan kondisi cahaya kantor Anda. Uji dengan foto asli serta foto dari layar HP/kertas, lalu sesuaikan `FACE_SPOOF_THRESHOLD`.
- **Liveness aktif di HP** (kedip/toleh acak) tetap wajib. Anti-spoof di server adalah lapisan kedua, bukan pengganti.

---

## Menjalankan di lokal

```bash
python -m venv venv
venv/Scripts/activate          # Linux/Mac: source venv/bin/activate
pip install -r requirements.txt pytest httpx
python scripts/download_models.py

# tes
python -m pytest -q

# jalankan service
FACE_API_KEY=rahasia uvicorn app.main:app --port 8000
```

Contoh panggilan:
```bash
curl -H "X-API-Key: rahasia" -F image=@tests/fixtures/real_face.jpg http://127.0.0.1:8000/embed
```

## Benchmark

```bash
# engine langsung
python scripts/benchmark.py --image tests/fixtures/real_face.jpg -n 50

# lewat HTTP, mensimulasikan 8 orang absen bersamaan
python scripts/benchmark.py --image tests/fixtures/real_face.jpg -n 200 -c 8 --url http://127.0.0.1:8000 --key rahasia
```

> Di Windows, pakai `127.0.0.1`, bukan `localhost`. Resolusi `localhost` di Windows menambah jeda sekitar 200 ms per request.

---

## Deploy ke Coolify

1. Push folder ini ke repository Git (private), misalnya `hris-face-service`.
2. Di Coolify, buka **Projects → (project HRIS) → + New → Private Repository**, lalu pilih repo tersebut.
   - **Build Pack:** `Dockerfile`
   - **Base Directory:** `/` (atau `/face-service` kalau digabung dalam satu repo)
   - **Ports Exposes:** `8000`
3. **Domain: kosongkan.** Service ini hanya untuk jaringan internal dan tidak boleh dibuka ke publik.
4. **Environment Variables:** isi `FACE_API_KEY` dengan kunci acak yang panjang. Variabel lain opsional, lihat `.env.example`.
5. **Resource Limits:** Memory `900m`, CPU `2`.
6. **Healthcheck:** path `/health`, port `8000`. Dockerfile juga sudah punya `HEALTHCHECK` sendiri.
7. Deploy. Laravel (di server dan network Coolify yang sama) memanggil service ini lewat nama container atau host internal yang ditampilkan Coolify, misalnya `http://<nama-container>:8000`.

Kalau nanti butuh throughput lebih besar, naikkan `WEB_CONCURRENCY=2` (RAM bertambah sekitar 180 MB per proses).

## Mengganti / membuat ulang model anti-spoof

File `models/antispoof_*.onnx` adalah hasil konversi dari [Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) (Apache-2.0, lisensi di `models/LICENSE-Silent-Face-Anti-Spoofing`). Untuk membuat ulang:

```bash
pip install torch onnx opencv-python-headless numpy
python tools/convert_antispoof.py --repo path/ke/Silent-Face-Anti-Spoofing --out models
```

Skrip ini juga memverifikasi bahwa output OpenCV sama dengan PyTorch. Kalau file model diganti, perbarui checksum di `scripts/download_models.py`.

Gambar di `tests/fixtures/` berasal dari repo yang sama (Apache-2.0).
