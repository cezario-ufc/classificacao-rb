# Base PyTorch + CUDA 12.1 (mesma combinação validada localmente: torch 2.5.1+cu121).
# Requer no host: driver NVIDIA compatível com CUDA 12.1 + nvidia-container-toolkit.
# Rodar com: docker run --gpus all ...
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    PYTHONPATH=/app \
    TORCH_HOME=/app/.torch_cache \
    YOLO_CONFIG_DIR=/app/.ultralytics \
    MPLCONFIGDIR=/app/.mplconfig

# libgl1/libglib: dependências nativas do OpenCV (ultralytics puxa opencv-python).
# awscli: sincronização do dataset a partir do S3 no entrypoint.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 awscli \
    && rm -rf /var/lib/apt/lists/*

# torch/torchvision já vêm na imagem base; instala só o restante (ultralytics, sahi, etc.).
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN sed -i 's/\r$//' /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# O entrypoint sincroniza o DDR do S3 e converte as anotações (idempotente); depois
# executa o comando passado. Rode uma config por vez, ex.:
#   docker run --gpus all ... python scripts/03_nested_cv.py --config A
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python", "scripts/03_nested_cv.py", "--config", "A"]
