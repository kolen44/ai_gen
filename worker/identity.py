"""
FaceID-эмбеддер — единая точка загрузки InsightFace, переиспользуется в двух местах, которым
иначе пришлось бы каждому грузить свою копию модели на GPU:

1. worker/photo_generation.py — эмбеддинг эталонного лица идёт в IP-Adapter FaceID как identity
   conditioning для всех 8 фото.
2. safety/moderation_pipeline.py (IdentitySafetyGate) — тот же эмбеддинг + атрибут age из той же
   модели используются для age-gate и сверки с watchlist.

Раньше (до этого файла) safety/moderation_pipeline.py грузил InsightFace самостоятельно — теперь
оба места должны получать FaceAnalysis через get_face_analysis() ниже, чтобы на поде не тратить
лишние секунды и VRAM на повторную загрузку одной и той же модели.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

_face_analysis_cache: dict[str, "object"] = {}


def get_face_analysis(model_name: str = "buffalo_l", det_size: tuple[int, int] = (640, 640)):
    """Возвращает закешированный InsightFace FaceAnalysis для данного имени модели. Первый вызов
    грузит модель (и качает веса при первом запуске на поде), последующие — переиспользуют."""
    if model_name not in _face_analysis_cache:
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(name=model_name)
        app.prepare(ctx_id=0, det_size=det_size)
        _face_analysis_cache[model_name] = app
    return _face_analysis_cache[model_name]


@dataclass
class DetectedFace:
    embedding: np.ndarray  # ArcFace normed_embedding, идёт и в IP-Adapter, и в similarity-сверку
    age: Optional[float]
    det_score: float
    # Координаты лица (x1, y1, x2, y2) в системе исходного кадра. Нужны везде, где лицо надо не
    # сравнить, а вырезать: face-detailer и подготовка face_video для Wan Animate. Раньше поля не
    # было, и вызывающий код через getattr(face, "bbox", None) молча получал None и уходил в
    # фолбэк — в Animate это давало вместо кропов лица центральный квадрат всего кадра.
    bbox: Optional[tuple[int, int, int, int]] = None


def _pad_for_detection(img_arr: np.ndarray, factor: float = 0.6) -> np.ndarray:
    """Добавляет поля вокруг кадра, уменьшая относительный размер лица.

    Зачем. Детектор RetinaFace ищет лица набором якорей ограниченного размера. Когда лицо занимает
    почти весь кадр (макрошот), оно оказывается крупнее самого большого якоря, и детектор
    возвращает пусто — это не "плохая картинка", а границы применимости детектора. Проверено на
    реальном прогоне: из пяти макрошотов в паке эталонов ни один не был распознан, при том что
    лицо на них видно прекрасно.

    Поля из отражённых краёв (edge-replication), а не чёрные: чёрная рамка сама по себе даёт
    ложные градиенты на границе и может ронять уверенность детектора.
    """
    h, w = img_arr.shape[:2]
    pad_h, pad_w = int(h * factor), int(w * factor)
    return np.pad(img_arr, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode="edge")


def detect_primary_face(image_path: str | Path, model_name: str = "buffalo_l") -> Optional[DetectedFace]:
    """Находит самое уверенно задетектированное лицо. Возвращает None, только если лица
    действительно нет (например, кадр видео, где человек отвернулся или вышел из кадра)."""
    from PIL import Image

    img_arr = np.array(Image.open(image_path).convert("RGB"))
    return detect_primary_face_array(img_arr, model_name)


def detect_primary_face_array(img_arr: np.ndarray, model_name: str = "buffalo_l") -> Optional[DetectedFace]:
    """То же самое для уже загруженного в память кадра — нужно при покадровой проверке видео,
    чтобы не писать каждый кадр во временный файл."""
    app = get_face_analysis(model_name)
    faces = app.get(img_arr)
    pad_h = pad_w = 0

    if not faces:
        # Вторая попытка на кадре с полями — вытягивает макрошоты (см. _pad_for_detection).
        # Эмбеддинг берётся с выровненного по ключевым точкам кропа, поэтому добавленные поля
        # на него не влияют: сравнимость с эмбеддингами обычных кадров сохраняется.
        h, w = img_arr.shape[:2]
        pad_h, pad_w = int(h * 0.6), int(w * 0.6)
        faces = app.get(_pad_for_detection(img_arr))

    if not faces:
        return None

    face = max(faces, key=lambda f: f.det_score)
    age = float(face.age) if getattr(face, "age", None) is not None else None

    # bbox приводим к координатам ИСХОДНОГО кадра: если сработал фолбэк с полями, координаты
    # найдены в системе расширенного изображения и без сдвига указывали бы не туда.
    height, width = img_arr.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in face.bbox)
    bbox = (
        max(0, min(width, x1 - pad_w)),
        max(0, min(height, y1 - pad_h)),
        max(0, min(width, x2 - pad_w)),
        max(0, min(height, y2 - pad_h)),
    )
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        bbox = None

    return DetectedFace(embedding=face.normed_embedding, age=age,
                        det_score=float(face.det_score), bbox=bbox)


def extract_embedding(image_path: str | Path, model_name: str = "buffalo_l") -> np.ndarray:
    """Только эмбеддинг — то, что реально нужно IP-Adapter FaceID при генерации 8 фото.
    Поднимает RuntimeError, если лицо не найдено (для эталонного фото это должно быть исключением,
    не тихим пропуском — без эмбеддинга Stage 2 генерировать нечем)."""
    face = detect_primary_face(image_path, model_name)
    if face is None:
        raise RuntimeError(f"No face detected in reference image: {image_path}")
    return face.embedding
