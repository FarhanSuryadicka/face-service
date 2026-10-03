FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FACE_MODELS_DIR=/app/models \
    WEB_CONCURRENCY=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# model anti-spoof ikut dari repo; YuNet & SFace diunduh + diverifikasi checksum
COPY models/ models/
COPY scripts/download_models.py scripts/
RUN python scripts/download_models.py /app/models

COPY app/ app/

RUN useradd --system --uid 10001 face && chown -R face /app
USER face

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"

CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers ${WEB_CONCURRENCY} --no-access-log"]
