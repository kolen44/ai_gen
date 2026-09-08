"""Хранилище задач в памяти с TTL.

Аналог job_store.py из WetDreams, но без Redis: там очередь общая на несколько подов, здесь под
один и переживать его перезапуск задачам незачем — результаты уже лежат файлами в OUTPUT_DIR.
Если понадобится несколько подов, сюда встанет тот же интерфейс поверх Redis, а вызывающий код
не изменится.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

from runpod_worker.config import JOB_TTL_SECONDS
from runpod_worker.models import GenerationStatus, JobResponse, PhotoResult


class JobStore:
    def __init__(self, ttl_seconds: int = JOB_TTL_SECONDS):
        self.ttl = ttl_seconds
        self._jobs: Dict[str, JobResponse] = {}
        self._expiry: Dict[str, float] = {}
        self._lock = threading.Lock()

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def create(self, job_id: str) -> JobResponse:
        job = JobResponse(job_id=job_id, status=GenerationStatus.PENDING, created_at=self._now())
        with self._lock:
            self._jobs[job_id] = job
            self._expiry[job_id] = time.monotonic() + self.ttl
        return job

    def get(self, job_id: str) -> Optional[JobResponse]:
        self.purge()
        with self._lock:
            return self._jobs.get(job_id)

    def set_status(self, job_id: str, status: GenerationStatus, *,
                   error: Optional[str] = None, progress: Optional[str] = None) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = status
            if error is not None:
                job.error = error
            if progress is not None:
                job.progress = progress
            if status in (GenerationStatus.COMPLETED, GenerationStatus.FAILED):
                job.finished_at = self._now()

    def add_result(self, job_id: str, result: PhotoResult) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.results.append(result)

    def active(self) -> int:
        with self._lock:
            return sum(1 for j in self._jobs.values()
                       if j.status in (GenerationStatus.PENDING, GenerationStatus.PROCESSING))

    def purge(self) -> int:
        """Убирает истёкшие задачи. Без этого словарь растёт всю жизнь пода."""
        now = time.monotonic()
        with self._lock:
            dead = [k for k, exp in self._expiry.items() if exp < now]
            for k in dead:
                self._jobs.pop(k, None)
                self._expiry.pop(k, None)
        return len(dead)

    def list_jobs(self, limit: int = 50) -> List[JobResponse]:
        self.purge()
        with self._lock:
            return list(self._jobs.values())[-limit:]


store = JobStore()
