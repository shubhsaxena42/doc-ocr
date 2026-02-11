# PaddleOCR Service
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FLAGS_use_mkldnn=0 \
    FLAGS_enable_pir_executor=0 \
    FLAGS_use_mkl=0 \
    MKLDNN_DISABLE=1 \
    FLAGS_enable_pir_api=0

# System libraries needed by PaddleOCR / OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 5001

# Gunicorn for production; single worker because PaddleOCR is not thread-safe
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "1", "--timeout", "120", "app:app"]
