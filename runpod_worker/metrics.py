"""Метрики в формате Prometheus.

Свой минимальный сборщик вместо prometheus_client: зависимость ради четырёх счётчиков не нужна,
а формат вывода простой и стабильный. Набор метрик выбран под то, что реально болит на поде:
сколько кадров отбраковано по лицу и сколько времени занимает кадр — по ним видно и качество,
и сгорающие GPU-часы.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Optional

_lock = threading.Lock()
_state = {
    "generated_total": 0,
    "accepted_total": 0,
    "rejected_total": 0,
    "failed_total": 0,
    "duration_sum_s": 0.0,
    "similarity_sum": 0.0,
    "similarity_count": 0,
    "started_at": time.time(),
}


@contextmanager
def track():
    start = time.monotonic()
    try:
        yield
    finally:
        with _lock:
            _state["duration_sum_s"] += time.monotonic() - start


def count_result(accepted: bool, similarity: Optional[float]) -> None:
    with _lock:
        _state["generated_total"] += 1
        _state["accepted_total" if accepted else "rejected_total"] += 1
        if similarity is not None:
            _state["similarity_sum"] += similarity
            _state["similarity_count"] += 1


def count_failure() -> None:
    with _lock:
        _state["failed_total"] += 1


def snapshot() -> dict:
    with _lock:
        s = dict(_state)
    s["uptime_s"] = time.time() - s.pop("started_at")
    n = s["generated_total"]
    s["duration_avg_s"] = round(s["duration_sum_s"] / n, 2) if n else 0.0
    c = s["similarity_count"]
    s["similarity_avg"] = round(s["similarity_sum"] / c, 4) if c else 0.0
    return s


def render() -> str:
    s = snapshot()
    lines = [
        "# HELP photos_generated_total Сгенерировано кадров",
        "# TYPE photos_generated_total counter",
        f"photos_generated_total {s['generated_total']}",
        "# HELP photos_accepted_total Кадров принято по порогу сходства и безопасности",
        "# TYPE photos_accepted_total counter",
        f"photos_accepted_total {s['accepted_total']}",
        "# HELP photos_rejected_total Кадров отбраковано",
        "# TYPE photos_rejected_total counter",
        f"photos_rejected_total {s['rejected_total']}",
        "# HELP photos_failed_total Ошибок генерации",
        "# TYPE photos_failed_total counter",
        f"photos_failed_total {s['failed_total']}",
        "# HELP photo_duration_avg_seconds Среднее время кадра",
        "# TYPE photo_duration_avg_seconds gauge",
        f"photo_duration_avg_seconds {s['duration_avg_s']}",
        "# HELP photo_similarity_avg Средняя косинусная близость к эталону",
        "# TYPE photo_similarity_avg gauge",
        f"photo_similarity_avg {s['similarity_avg']}",
        "# HELP worker_uptime_seconds Время жизни процесса",
        "# TYPE worker_uptime_seconds gauge",
        f"worker_uptime_seconds {s['uptime_s']:.0f}",
    ]
    return "\n".join(lines) + "\n"
