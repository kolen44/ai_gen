"""Точка входа воркера.

    python -m runpod_worker.worker --mode api        только HTTP-сервер
    python -m runpod_worker.worker --mode warmup     прогреть веса и выйти
    python -m runpod_worker.worker --mode both       прогреть и поднять сервер (по умолчанию)

Режим warmup существует ради денег, а не ради удобства: первый запрос после старта пода тянет
SDXL, IP-Adapter и InsightFace, и это минуты GPU-времени, которые тарифицируются. Лучше явно
прогреть на старте контейнера, чем оплачивать это внутри первого запроса клиента.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from runpod_worker.config import API_HOST, API_PORT, CHARACTER_DIR, LORA_DIR, OUTPUT_DIR  # noqa: E402


def warmup() -> None:
    import time

    from runpod_worker.service import service
    from worker.config import DEFAULT_PHOTO_MODEL

    start = time.monotonic()
    print(f"прогрев: чекпойнт {DEFAULT_PHOTO_MODEL} + IP-Adapter FaceID…", flush=True)
    gen = service._get_generator()
    gen.model_manager.load_model(DEFAULT_PHOTO_MODEL, with_ip_adapter=True)

    # InsightFace грузится отдельно и тоже не бесплатен — трогаем его здесь же.
    from worker.identity import get_face_analysis
    from worker.config import IDENTITY_CONFIG

    get_face_analysis(IDENTITY_CONFIG["face_analysis_model"])
    print(f"прогрев завершён за {time.monotonic() - start:.0f} с", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="воркер генерации фото персонажа для RunPod")
    ap.add_argument("--mode", choices=["api", "warmup", "both"], default="both")
    ap.add_argument("--host", default=API_HOST)
    ap.add_argument("--port", type=int, default=API_PORT)
    args = ap.parse_args()

    for d in (OUTPUT_DIR, CHARACTER_DIR, LORA_DIR):
        d.mkdir(parents=True, exist_ok=True)

    if args.mode in ("warmup", "both"):
        warmup()
    if args.mode == "warmup":
        return

    from runpod_worker.api import run_api_server

    print(f"API: http://{args.host}:{args.port}  (документация на /docs)", flush=True)
    run_api_server(args.host, args.port)


if __name__ == "__main__":
    main()
