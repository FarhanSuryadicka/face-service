"""
Ukur kecepatan face-service.

Mode lokal (langsung memanggil engine, tanpa HTTP):
    python scripts/benchmark.py --image tests/fixtures/real_face.jpg -n 50

Mode HTTP (menguji service yang sedang berjalan, termasuk antrean saat ramai):
    python scripts/benchmark.py --image foto.jpg -n 200 -c 8 \
        --url http://localhost:8000 --key RAHASIA
"""
import argparse
import statistics
import sys
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def pct(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, int(round(p / 100 * (len(values) - 1))))]


def report(label, latencies, wall):
    print(f"\n{label}")
    print(f"  permintaan : {len(latencies)}")
    print(f"  p50        : {pct(latencies, 50):.1f} ms")
    print(f"  p95        : {pct(latencies, 95):.1f} ms")
    print(f"  maks       : {max(latencies):.1f} ms")
    print(f"  throughput : {len(latencies) / wall:.1f} req/detik")


def run_local(data: bytes, n: int):
    from app.config import Settings
    from app.engine import FaceEngine

    engine = FaceEngine(Settings())
    engine.embed(engine.decode(data))  # pemanasan
    stages = {}
    latencies = []
    start = time.perf_counter()
    for _ in range(n):
        t = time.perf_counter()
        r = engine.embed(engine.decode(data))
        latencies.append((time.perf_counter() - t) * 1000)
        for k, v in r.timing_ms.items():
            stages.setdefault(k, []).append(v)
    report("Engine lokal", latencies, time.perf_counter() - start)
    print("  rata-rata per tahap: " + ", ".join(f"{k}={statistics.mean(v):.1f}ms" for k, v in stages.items()))


def run_http(data: bytes, n: int, concurrency: int, url: str, key: str):
    boundary = uuid.uuid4().hex
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"a.jpg\"\r\n"
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
    headers = {"X-API-Key": key, "Content-Type": f"multipart/form-data; boundary={boundary}"}

    def one(_):
        req = urllib.request.Request(url.rstrip("/") + "/embed", data=body, headers=headers, method="POST")
        t = time.perf_counter()
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp.read()
            ok = resp.status == 200
        return (time.perf_counter() - t) * 1000, ok

    one(0)  # pemanasan
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(one, range(n)))
    wall = time.perf_counter() - start
    failed = sum(1 for _, ok in results if not ok)
    report(f"HTTP {url} (konkurensi {concurrency})", [lat for lat, _ in results], wall)
    print(f"  gagal      : {failed}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("-n", type=int, default=50)
    p.add_argument("-c", "--concurrency", type=int, default=4)
    p.add_argument("--url")
    p.add_argument("--key", default="")
    a = p.parse_args()
    data = Path(a.image).read_bytes()
    if a.url:
        run_http(data, a.n, a.concurrency, a.url, a.key)
    else:
        run_local(data, a.n)


if __name__ == "__main__":
    main()
