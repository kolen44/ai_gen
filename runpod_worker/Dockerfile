# syntax=docker/dockerfile:1.7
#
# Образ воркера для RunPod. База и раскладка кешей — как в WetDreams
# (runpod/wai-nsfw-illustrious-sdxl/Dockerfile): веса лежат в /models, чтобы каталог можно было
# смонтировать сетевым томом и не выкачивать SDXL заново при каждом создании пода.

FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf \
    MODEL_CACHE_DIR=/models/checkpoints \
    TORCH_HOME=/models/torch \
    OUTPUT_DIR=/data/output \
    CHARACTER_DIR=/data/characters \
    LORA_DIR=/data/loras

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    git wget ca-certificates libgl1 libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY runpod_worker/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY worker/ ./worker/
COPY safety/ ./safety/
COPY runpod_worker/ ./runpod_worker/

# /data — сетевой том RunPod: персонажи и результаты обязаны пережить пересоздание пода.
# Эмбеддинг персонажа невосстановим, потерять его нельзя.
VOLUME ["/models", "/data"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=5)"

CMD ["python", "-m", "runpod_worker.worker", "--mode", "both"]
