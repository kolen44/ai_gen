"""
Дописывает строки в benchmarking/benchmark_log.csv по факту каждой реальной генерации — раньше
это предлагалось делать вручную через deploy/cost_calculator.py, теперь pipeline/generate_photos.py
и pipeline/image_to_video.py делают это сами, без ручного шага после каждого фото/видео.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deploy.cost_calculator import generation_cost  # noqa: E402

BENCHMARK_LOG_PATH = Path(__file__).resolve().parent.parent / "benchmarking" / "benchmark_log.csv"

FIELDNAMES = [
    "stage", "index", "checkpoint", "ip_adapter_weight", "ip_adapter_scale", "sampler", "steps",
    "resolution", "seed", "elapsed_seconds", "gpu_type", "hourly_rate_usd", "cost_usd",
    "vram_peak_gb", "safety_gate_result", "notes",
]

_cleaned_this_session = False


def _ensure_clean_log() -> None:
    """При первом реальном append в этой сессии: если файл содержит только заготовку-шаблон
    (все строки с пустым elapsed_seconds, как в benchmark_log.csv из коробки) — заменяет его на
    чистый файл с одним заголовком, чтобы реальные цифры не перемешивались визуально с
    плейсхолдерами. Если в файле уже есть реальные прошлые прогоны (elapsed_seconds заполнен) —
    ничего не трогает, только добавляет новые строки."""
    global _cleaned_this_session
    if _cleaned_this_session or not BENCHMARK_LOG_PATH.exists():
        _cleaned_this_session = True
        return

    with open(BENCHMARK_LOG_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    only_placeholders = all(not (row.get("elapsed_seconds") or "").strip() for row in rows)
    if only_placeholders:
        with open(BENCHMARK_LOG_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()

    _cleaned_this_session = True


def append_row(
    *,
    stage: str,
    index: Optional[int],
    checkpoint: str,
    elapsed_seconds: float,
    seed: int,
    resolution: str,
    steps: int,
    gpu_type: str,
    hourly_rate_usd: float,
    safety_gate_result: str,
    sampler: str = "",
    ip_adapter_weight: str = "",
    ip_adapter_scale: str = "",
    vram_peak_gb: str = "",
    notes: str = "",
) -> float:
    """Пишет одну строку и возвращает посчитанную стоимость этой генерации в USD."""
    _ensure_clean_log()

    cost_usd = generation_cost(elapsed_seconds, hourly_rate_usd).cost_usd

    row = {
        "stage": stage,
        "index": index if index is not None else "",
        "checkpoint": checkpoint,
        "ip_adapter_weight": ip_adapter_weight,
        "ip_adapter_scale": ip_adapter_scale,
        "sampler": sampler,
        "steps": steps,
        "resolution": resolution,
        "seed": seed,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "gpu_type": gpu_type,
        "hourly_rate_usd": hourly_rate_usd,
        "cost_usd": round(cost_usd, 6),
        "vram_peak_gb": vram_peak_gb,
        "safety_gate_result": safety_gate_result,
        "notes": notes,
    }

    BENCHMARK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_is_empty = not BENCHMARK_LOG_PATH.exists() or BENCHMARK_LOG_PATH.stat().st_size == 0
    with open(BENCHMARK_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if file_is_empty:
            writer.writeheader()
        writer.writerow(row)

    return cost_usd
