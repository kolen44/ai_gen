"""HTTP API воркера.

Набор маршрутов повторяет WetDreams (`/health`, `/generate`, `/generate/sync`, `/job/{id}`,
`/models`, `/metrics`), чтобы клиентский код и мониторинг у обоих сервисов выглядели одинаково.
Добавлено то, чего там нет и без чего эта задача не решается: реестр персонажей и пакетная
генерация пака одним запросом.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException
from fastapi.responses import PlainTextResponse

from runpod_worker import characters, metrics
from runpod_worker.config import API_KEY, IMPOSTOR_CEILING, OUTPUT_DIR, VERSION
from runpod_worker.job_store import store
from runpod_worker.models import (
    VideoEnhanceRequest,
    VideoEnhanceResult,
    VideoRequest,
    VideoResult,
    BatchPhotoRequest,
    CharacterInfo,
    CreateCharacterRequest,
    GenerationStatus,
    HealthResponse,
    JobResponse,
    PhotoRequest,
    PhotoResult,
)
from runpod_worker.service import ServiceError, service

app = FastAPI(title="Character Photo Worker", version=VERSION)


def _require_api_key(x_api_key: Optional[str] = Header(default=None, alias="X-Api-Key")) -> None:
    # Пустой ключ в конфиге означает «проверки нет» — это нормально для локального запуска,
    # но на поде с публичным портом ключ обязателен: платим за GPU-часы мы.
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="неверный или отсутствующий X-Api-Key")


auth = [Depends(_require_api_key)]


# --- служебное ---

@app.get("/health", response_model=HealthResponse)
async def health():
    import torch

    gpu = vram_total = vram_used = None
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram_total = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        vram_used = round(torch.cuda.memory_allocated(0) / 1e9, 1)

    loaded = []
    if service._generator is not None:
        loaded = list(service._generator.model_manager.loaded_models)

    return HealthResponse(
        status="ok",
        version=VERSION,
        device="cuda" if torch.cuda.is_available() else "cpu",
        gpu=gpu,
        vram_total_gb=vram_total,
        vram_used_gb=vram_used,
        models_loaded=loaded,
        characters=len(characters.list_characters()),
        jobs_active=store.active(),
    )


@app.get("/version")
async def version():
    return {"version": VERSION, "impostor_ceiling": IMPOSTOR_CEILING}


@app.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics():
    return metrics.render()


@app.get("/models", dependencies=auth)
async def list_models():
    from worker.config import DEFAULT_PHOTO_MODEL, PHOTO_MODEL_CONFIGS

    return {"default": DEFAULT_PHOTO_MODEL,
            "models": [{"name": k, "source": v.get("source"), "filename": v.get("filename")}
                       for k, v in PHOTO_MODEL_CONFIGS.items()]}


# --- персонажи ---

@app.get("/characters", response_model=list[CharacterInfo], dependencies=auth)
async def get_characters():
    return [CharacterInfo(**c) for c in characters.list_characters()]


@app.post("/characters", response_model=CharacterInfo, dependencies=auth)
async def create_character(req: CreateCharacterRequest):
    try:
        data = service.create_character(
            req.name, reference_image=req.reference_image, reference_path=req.reference_path,
            prompt=req.prompt, seed=req.seed, overwrite=req.overwrite,
            model_name=req.model_name)
    except (ServiceError, characters.CharacterError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return CharacterInfo(name=data["name"], created_at=data["created_at"],
                         reference_image=data.get("reference_image"), has_embedding=True)


@app.delete("/characters/{name}", dependencies=auth)
async def delete_character(name: str):
    if not characters.delete(name):
        raise HTTPException(status_code=404, detail=f"персонаж {name!r} не найден")
    return {"deleted": name}


# --- генерация ---

@app.post("/generate/sync", response_model=PhotoResult, dependencies=auth)
async def generate_sync(req: PhotoRequest):
    """Один кадр, ответ по готовности. Для отладки и одиночных запросов."""
    _require_character(req.character)
    try:
        with metrics.track():
            result = service.generate_photo(req)
    except (ServiceError, characters.CharacterError) as e:
        metrics.count_failure()
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001 — наружу отдаём текст, внутрь пишем метрику
        metrics.count_failure()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
    metrics.count_result(result.accepted, result.similarity)
    return result


def _run_job(job_id: str, requests: list[PhotoRequest]) -> None:
    store.set_status(job_id, GenerationStatus.PROCESSING)
    try:
        for i, req in enumerate(requests, start=1):
            store.set_status(job_id, GenerationStatus.PROCESSING,
                             progress=f"{i}/{len(requests)}")
            with metrics.track():
                result = service.generate_photo(req)
            metrics.count_result(result.accepted, result.similarity)
            store.add_result(job_id, result)
        store.set_status(job_id, GenerationStatus.COMPLETED, progress="готово")
    except Exception as e:  # noqa: BLE001
        metrics.count_failure()
        store.set_status(job_id, GenerationStatus.FAILED, error=f"{type(e).__name__}: {e}")


def _require_character(name: str) -> None:
    """Проверяем персонажа до постановки задачи.

    Иначе запрос с опечаткой в имени получает 200 и «принято», а падает через полминуты уже
    в фоне — увидеть это можно только опросив статус. Ошибку ввода честнее вернуть сразу.
    """
    if not characters.exists(name):
        known = ", ".join(c["name"] for c in characters.list_characters()) or "ни одного"
        raise HTTPException(status_code=400,
                            detail=f"персонаж {name!r} не найден; есть: {known}")


@app.post("/generate", response_model=JobResponse, dependencies=auth)
async def generate(req: PhotoRequest, background: BackgroundTasks):
    _require_character(req.character)
    job_id = uuid4().hex
    job = store.create(job_id)
    background.add_task(_run_job, job_id, [req])
    return job


@app.post("/generate/batch", response_model=JobResponse, dependencies=auth)
async def generate_batch(req: BatchPhotoRequest, background: BackgroundTasks):
    """Пак кадров одного персонажа — основной сценарий: восемь сцен одним запросом."""
    _require_character(req.character)
    try:
        requests = service.expand_batch(req)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(e))
    job_id = uuid4().hex
    job = store.create(job_id)
    background.add_task(_run_job, job_id, requests)
    return job


@app.get("/job/{job_id}", response_model=JobResponse, dependencies=auth)
async def job_status(job_id: str):
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="задача не найдена или истекла")
    return job


@app.get("/jobs", response_model=list[JobResponse], dependencies=auth)
async def list_jobs(limit: int = 50):
    return store.list_jobs(limit)


@app.get("/video/models", dependencies=auth)
async def list_video_models():
    from worker.config import VIDEO_MODEL_CONFIGS, DEFAULT_VIDEO_MODEL

    return {"models": list(VIDEO_MODEL_CONFIGS), "default": DEFAULT_VIDEO_MODEL}


@app.post("/video/sync", response_model=VideoResult, dependencies=auth)
async def generate_video(req: VideoRequest):
    """Оживление готового кадра. Синхронно: ролик считается минуты, но очередь ради одного
    запроса на под с одной картой — лишний узел (см. runpod_worker/README.md)."""
    try:
        with metrics.track():
            result = service.generate_video(req)
    except ServiceError as e:
        metrics.count_failure()
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001 — наружу отдаём текст, внутрь пишем метрику
        metrics.count_failure()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
    return VideoResult(**result)


@app.get("/file", dependencies=auth)
async def download_file(path: str):
    """Отдаёт готовый файл с пода по HTTP.

    Зачем при живом scp: на реальном прогоне 20-мегабайтный ролик оборвался на 2.6 МБ
    ("Connection closed"), а mp4 хранит метаданные в конце файла — усечённая копия не
    открывается вообще ("moov atom not found"), то есть частичная выгрузка бесполезна. HTTP через
    прокси RunPod докачивает надёжнее и не зависит от живого SSH.

    Путь ограничен OUTPUT_DIR: параметр приходит от клиента, и без проверки это было бы чтение
    любого файла на поде по произвольному пути.
    """
    from pathlib import Path as _Path

    from fastapi.responses import FileResponse

    target = _Path(path).resolve()
    root = _Path(OUTPUT_DIR).resolve()
    if not (target == root or root in target.parents):
        raise HTTPException(status_code=403, detail="путь вне OUTPUT_DIR")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="файл не найден")
    return FileResponse(str(target), filename=target.name)


@app.post("/video/enhance", response_model=VideoEnhanceResult, dependencies=auth)
async def enhance_video(req: VideoEnhanceRequest):
    """Восстановление лица в готовом ролике. Синхронно — как и генерация видео."""
    try:
        with metrics.track():
            result = service.enhance_video(req)
    except (ServiceError, characters.CharacterError) as e:
        metrics.count_failure()
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        metrics.count_failure()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
    return VideoEnhanceResult(**result)


def create_app() -> FastAPI:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return app


def run_api_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(create_app(), host=host, port=port, log_level="info")
