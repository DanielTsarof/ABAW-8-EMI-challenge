FROM python:3.10-slim

# Declare GPU capabilities so the NVIDIA Container Toolkit exposes the device.
# PyTorch cu128 wheels bundle CUDA runtime/cuDNN, so no nvidia/cuda base image needed.
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch with CUDA 12.8 wheels (matches the host environment).
# --extra-index-url keeps PyPI as primary so typing-extensions and other
# dependencies resolve from PyPI; torch/torchaudio/torchvision come from
# the cu128 index.
RUN pip install --no-cache-dir \
    torch==2.9.1 \
    torchaudio==2.9.1 \
    torchvision==0.24.1 \
    --extra-index-url https://download.pytorch.org/whl/cu128

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
