# AGENTS.md: face-service (HRIS Face Recognition)

Konteks untuk AI agent (Claude Code, dll.) dan developer yang melanjutkan proyek ini dari komputer lain.
Bahasa komunikasi dengan pemilik proyek: **Bahasa Indonesia**.

## Gambaran proyek

Fitur absensi wajah untuk sistem **HRIS** perusahaan:

| Bagian | Teknologi | Status |
|---|---|---|
| **face-service** (repo ini) | Python FastAPI + OpenCV DNN (ONNX), CPU saja | ✅ Selesai, 11 tes lulus. Belum di-deploy |
| **face-demo-app** (`D:\face-demo-app`, folder terpisah) | Expo / React Native, prototipe Android | ✅ APK jadi. **Belum dites di HP sungguhan** |
| Backend HR | Laravel (web HR + API untuk mobile) | ❌ Belum dikerjakan |
| Mobile HRIS | React Native (dikerjakan tim mobile) | ❌ Belum. Logika liveness bisa diambil dari face-demo-app |

Pemakaian awal: internal perusahaan sendiri (ratusan karyawan), tapi arsitekturnya disiapkan supaya bisa dijual ke klien.

### Arsitektur target (produksi)

```
HP (React Native)                 Laravel (HR)                        face-service (repo ini)
- liveness aktif (ML Kit:  ──►    - simpan embedding (employee_faces) ──►  POST /embed
  kedip/toleh/senyum acak)        - cosine similarity + threshold          -> embedding 128-d,
- kirim 1 selfie + GPS            - aturan shift, geofence, device         kualitas foto, skor anti-spoof
                                  - keputusan absen sah/tidak              (stateless, tidak simpan data)
```

- **Verifikasi 1:1**: selfie hanya dibandingkan dengan wajah karyawan itu sendiri, jadi beban tidak bertambah walau karyawan banyak.
- face-service **stateless** dan **tidak boleh dibuka ke publik**. Hanya Laravel yang memanggilnya, lewat jaringan internal Docker/Coolify, dengan header `X-API-Key`.
- Keputusan absen ada di Laravel. HP tidak boleh dipercaya untuk memutuskan.

### Target deploy

VPS Linux milik perusahaan (4 vCPU / 4 GB RAM, tanpa GPU) yang dikelola **Coolify**. Laravel HRIS nanti juga di VPS yang sama.
- Deploy sebagai aplikasi **Dockerfile**, **tanpa domain**, **Ports Mappings dikosongkan** (port 8000 di host sudah dipakai dashboard Coolify), Ports Exposes `8000`.
- Resource limit memory `900m`. Env wajib: `FACE_API_KEY` (acak dan panjang).
- RAM VPS terbatas (sudah ada aplikasi POS, MySQL, Redis). Jangan menambah dependensi berat (PyTorch, TensorFlow, ONNX Runtime GPU).

## Struktur repo

| Path | Isi |
|---|---|
| `app/main.py` | Endpoint FastAPI: `GET /health`, `POST /embed`, `POST /compare`, `GET /demo` (hanya kalau `FACE_DEMO=true`) |
| `app/engine.py` | Pipeline: decode → YuNet (deteksi + 5 landmark) → cek kualitas → SFace (embedding) → MiniFASNet (anti-spoof) |
| `app/config.py` | Semua pengaturan lewat env `FACE_*` (lihat `.env.example`) |
| `app/static/demo.html` | Halaman uji webcam di browser (MediaPipe untuk liveness) |
| `models/` | Model ONNX. YuNet & SFace diunduh `scripts/download_models.py` (gitignored), anti-spoof ikut di repo |
| `tools/convert_antispoof.py` | Konversi Silent-Face PyTorch → ONNX (sudah dilakukan, hanya perlu kalau model diganti) |
| `scripts/benchmark.py` | Ukur kecepatan, lokal atau lewat HTTP |
| `tests/test_api.py` | Tes pytest. Fixture dari repo Silent-Face (Apache-2.0) |

Model: YuNet (MIT), SFace (Apache-2.0), MiniFASNet V2 + V1SE (Apache-2.0). Semuanya aman untuk komersial. **Jangan** ganti ke InsightFace/buffalo atau model lain yang lisensinya non-komersial.

## Menyiapkan di komputer baru (Windows)

Folder ini **belum berupa repo git**. Salin seluruh folder `face-service` (boleh tanpa `venv/`), lalu:

```powershell
cd D:\face-service
python -m venv venv
venv\Scripts\python -m pip install -r requirements.txt pytest httpx
venv\Scripts\python scripts\download_models.py
venv\Scripts\python -m pytest -q          # harus 11 passed
```

Python 3.9+ (sudah dites dengan 3.9 di Windows, Docker memakai 3.11).

## Menjalankan

```powershell
# hanya untuk laptop sendiri + halaman demo webcam di http://127.0.0.1:8000/demo
$env:FACE_API_KEY="rahasia"; $env:FACE_DEMO="true"
venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# supaya bisa diakses HP di Wi-Fi yang sama (untuk face-demo-app)
$env:FACE_API_KEY="rahasia"
venv\Scripts\python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

- `rahasia` hanya API key untuk uji lokal. Di server pakai kunci acak.
- `--host 0.0.0.0`: izinkan Python di pop-up firewall Windows (jaringan **Private**).
- Cari IP laptop dengan `ipconfig` (IPv4 di adapter Wi-Fi), lalu isi `http://<IP>:8000` di **Pengaturan** app. **IP berubah di setiap jaringan (kantor/rumah), dan tidak perlu build ulang APK**, cukup ubah di Pengaturan.
- Di Windows, panggil `127.0.0.1`, bukan `localhost` (jeda sekitar 200 ms karena IPv6).
- `FACE_DEMO` **tidak boleh** diaktifkan di server produksi.

## Aturan saat mengubah kode

- Jalankan `venv\Scripts\python -m pytest -q` setelah setiap perubahan.
- `cv2.imdecode` di OpenCV 4.10 **sudah menerapkan rotasi EXIF** (fixture tes memang berisi piksel miring + EXIF Orientation=6). Jangan tambahkan rotasi EXIF manual, nanti foto terputar dua kali.
- Embedding dikembalikan dalam bentuk ternormalisasi L2, jadi cosine similarity = dot product. Laravel bergantung pada ini.
- Format error tetap `{"error": "<kode>", "message": "..."}`. Kode error dan `quality.issues` adalah kontrak dengan Laravel dan app (lihat README).
- Inferensi OpenCV DNN tidak thread-safe, jadi dijaga satu lock di `FaceEngine`. Untuk skala, tambah proses (`WEB_CONCURRENCY`), bukan thread.

## Threshold (belum dikalibrasi)

- Kecocokan wajah (cosine SFace): mulai **0.45** untuk absensi (rekomendasi OpenCV 0.363). Kalibrasi dari skor "orang sama" vs "orang lain" di menu **Riwayat & skor** face-demo-app.
- Anti-spoof: `FACE_SPOOF_THRESHOLD=0.6`. Baru dites dengan 3 foto contoh, perlu dites dengan HP dan cahaya kantor.

## Status & langkah berikutnya

1. **Sekarang:** uji face-demo-app di HP (lihat `D:\face-demo-app\AGENTS.md`). Kumpulkan skor untuk kalibrasi threshold, dan pastikan arah toleh kiri/kanan benar.
2. `git init` + push ke GitHub private, lalu deploy ke Coolify (langkah di README).
3. Jalankan `scripts/benchmark.py` di VPS.
4. Integrasi Laravel: service `FaceService` (HTTP client ke `/embed`), migration `employee_faces` (embedding JSON), endpoint enroll & check-in, validasi shift/geofence/device, deteksi fake GPS.

### Pekerjaan server di luar repo ini (catatan pemilik)

- Di VPS: port MySQL 3306 masih terbuka ke publik dan sedang diserang brute force bot. Aplikasi POS sudah dialihkan ke host internal. Setelah pemantauan 24 jam (`/root/mysql-clients.log`), matikan **Make it publicly available** di resource MySQL Coolify.
- Firewall UFW belum aktif, dan dashboard Coolify port 8000 masih bisa diakses lewat HTTP. Rapikan setelah port MySQL ditutup.
