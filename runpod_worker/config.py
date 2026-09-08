"""Настройки сервиса генерации фото на RunPod.

Разделение намеренное и повторяет WetDreams (`runpod/wai-nsfw-illustrious-sdxl/config.py`):
здесь только конфигурация сервиса — порт, ключ доступа, время жизни задач, каталоги. Всё, что
касается моделей и генерации (PHOTO_MODEL_CONFIGS, IDENTITY_CONFIG, HIRES_CONFIG), живёт в
worker/config.py и переиспользуется как есть — второй копии параметров быть не должно.

Все значения читаются из окружения, потому что под на RunPod настраивается переменными, а не
правкой файлов внутри образа.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- сеть и доступ ---
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
# Пусто = без проверки. На публичном поде обязателен: RunPod отдаёт наружу HTTP-порт, и без
# ключа генерацию сможет запускать кто угодно, а платим за GPU-часы мы.
API_KEY = os.getenv("WORKER_API_KEY", "").strip()

# --- каталоги ---
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(ROOT / "output" / "runpod")))
# Персонаж = эталонный кадр + его ArcFace-эмбеддинг. Хранится на сетевом томе, чтобы пережить
# пересоздание пода: эмбеддинг — это и есть личность, заново её не получить.
CHARACTER_DIR = Path(os.getenv("CHARACTER_DIR", str(ROOT / "characters")))
LORA_DIR = Path(os.getenv("LORA_DIR", str(ROOT / "loras")))

# --- задачи ---
JOB_TTL_SECONDS = int(os.getenv("JOB_TTL_SECONDS", "3600"))
MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE", "64"))
# Один под — одна карта, и генерация всё равно сериализуется локом в ModelManager. Больше одного
# рабочего потока смысла не имеет, только съест VRAM параллельными пайплайнами.
WORKER_CONCURRENCY = int(os.getenv("WORKER_CONCURRENCY", "1"))

# --- качество по умолчанию ---
# Значения из фактического прогона на RTX 4090 (deliverables/BENCHMARK.md): 8.91 с на готовый
# кадр 1536x1536 при метрике 0.727. Менять их стоит только вместе с новым замером.
DEFAULT_HIRES = os.getenv("DEFAULT_HIRES", "1").lower() not in ("0", "false", "no")
DEFAULT_FACE_DETAILER = os.getenv("DEFAULT_FACE_DETAILER", "1").lower() not in ("0", "false", "no")
DEFAULT_MIN_SIMILARITY = float(os.getenv("DEFAULT_MIN_SIMILARITY", "0.5"))

# Порог «тот же человек». Откалиброван на этой же модели: портреты заведомо чужих людей дают до
# 0.387 (scripts/calibrate_similarity.py), поэтому 0.5 — не назначенное наугад число.
IMPOSTOR_CEILING = 0.387

VERSION = "1.0.0"
