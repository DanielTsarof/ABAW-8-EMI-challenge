FROM nvidia/cuda:12.8.0-cudnn9-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

# PyTorch with CUDA 12.8 wheels (matches the host environment)
RUN pip install --no-cache-dir \
    torch==2.9.1 \
    torchaudio==2.9.1 \
    torchvision==0.24.1 \
    --index-url https://download.pytorch.org/whl/cu128

# Remaining ML + API dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir \
    "accelerate==1.12.0" \
    "emotiefflib[torch]==1.1.1" \
    "facenet-pytorch==2.5.3" \
    "faster-whisper==1.2.1" \
    "matplotlib==3.10.8" \
    "numpy==2.2.6" \
    "opencv-python-headless==4.13.0.90" \
    "pillow==12.0.0" \
    "protobuf==6.33.1" \
    "sentencepiece==0.2.1" \
    "soundfile==0.13.1" \
    "tqdm==4.67.1" \
    "transformers==4.57.3" \
    "fastapi==0.115.12" \
    "uvicorn[standard]==0.34.3" \
    "python-multipart==0.0.20" \
    "pydantic-settings==2.9.1"

COPY app/ app/
COPY frontend/ frontend/

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
