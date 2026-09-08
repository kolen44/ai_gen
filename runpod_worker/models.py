"""Схемы запросов и ответов HTTP-слоя.

Форма повторяет WetDreams (`runpod/.../models.py`): GenerationStatus, запрос с алиасами полей,
ответ с job_id и статусом. Отличие по существу одно — здесь у генерации есть персонаж, и
идентичность лица является частью контракта: в ответе всегда есть similarity, а не только путь
к картинке.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from worker.config import (
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_HEIGHT,
    DEFAULT_PHOTO_MODEL,
    DEFAULT_STEPS,
    DEFAULT_WIDTH,
)


class GenerationStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class PhotoRequest(BaseModel):
    """Один кадр персонажа."""

    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(default_factory=lambda: uuid4().hex)
    character: str = Field(..., description="Имя персонажа из реестра — источник эмбеддинга лица")
    # Лимит с большим запасом. Прежние 3000 символов упирались в реальные промпты: описание
    # образа из карточки занимает до 1958 символов, плюс фиксация лица (899), мимика, внешность
    # и стиль съёмки — вместе выходит 3100-3200, и запрос отклонялся схемой ещё до генерации.
    # Ограничение по числу токенов при этом снято отдельно (worker/photo_generation.py::
    # _encode_chunked), так что длина промпта упирается только в здравый смысл.
    prompt: str = Field(..., min_length=5, max_length=12000)
    negative_prompt: Optional[str] = Field(None, max_length=4000)
    model_name: str = Field(DEFAULT_PHOTO_MODEL, description="Ключ из worker.config.PHOTO_MODEL_CONFIGS")

    width: int = Field(DEFAULT_WIDTH, ge=512, le=1536)
    height: int = Field(DEFAULT_HEIGHT, ge=512, le=1536)
    num_inference_steps: int = Field(
        DEFAULT_STEPS, ge=10, le=60,
        validation_alias=AliasChoices("num_inference_steps", "steps"),
        serialization_alias="num_inference_steps",
    )
    guidance_scale: float = Field(DEFAULT_GUIDANCE_SCALE, ge=0.0, le=20.0)
    seed: Optional[int] = Field(None, ge=0, le=2**32 - 1)

    ip_adapter_scale: Optional[float] = Field(None, ge=0.0, le=1.5)
    # Режим фиксации лица на КАЖДЫЙ запрос, а не только переменной окружения при старте воркера.
    # Иначе сравнить base/portrait/plusv2 между собой можно только перезапуском сервиса, а это
    # повторная загрузка чекпойнта в память — минуты оплаченного GPU-времени на каждый замер.
    # Пайплайны кешируются по (чекпойнт, режим), так что переключение внутри процесса дешёвое.
    identity_mode: Optional[str] = Field(None, description="base | portrait | plusv2")
    # Сколько ракурсов эталона отдать портретному режиму. Замерено на поде: пять ракурсов дают
    # ХУЖЕ, чем один (0.43 против 0.63) — усреднение профилей и наклонов уводит вектор от
    # фронтальной личности, с которой мы же и сравниваем. Значит число ракурсов — параметр, а не
    # константа, и подбирать его надо замером, а не рассуждением.
    identity_reference_count: Optional[int] = Field(None, ge=1, le=10)
    # Порог, ниже которого лицо переписывается детейлером принудительно. Нужен для двухэтапной
    # схемы: сцена рисуется БЕЗ FaceID (иначе композиция не слушается промпта и всё сваливается в
    # макрошот), и личность вживляется вторым проходом. Размер лица тут не показатель — у
    # ростового кадра он около 0.19 при пороге детейлера 0.12, то есть «лицо крупное, доводить
    # нечего», а похожесть при этом 0.50. Решает именно похожесть.
    detail_below_similarity: Optional[float] = Field(None, ge=0.0, le=1.0)
    hires: Optional[bool] = Field(None, description="Апскейл ×1.5 и img2img-доводка кадра")
    face_detailer: Optional[bool] = Field(None, description="Второй проход по лицу")
    min_similarity: Optional[float] = Field(None, ge=0.0, le=1.0)

    lora: Optional[str] = Field(None, description="Имя файла LoRA в LORA_DIR")
    lora_scale: float = Field(0.8, ge=0.0, le=2.0)


class BatchPhotoRequest(BaseModel):
    """Пак кадров одного персонажа — основной сценарий: восемь сцен одним запросом."""

    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(default_factory=lambda: uuid4().hex)
    character: str
    scenes: List[str] = Field(..., min_length=1, max_length=32,
                              description="Промпты сцен; на каждый — один кадр")
    base_seed: Optional[int] = Field(None, ge=0, le=2**32 - 1,
                                     description="Кадр i получает base_seed + 1000·i")
    defaults: Optional[Dict[str, Any]] = Field(
        None, description="Общие поля PhotoRequest для всех сцен: model_name, steps и прочее")

    @model_validator(mode="after")
    def _strip_scenes(self):
        self.scenes = [s.strip() for s in self.scenes if s and s.strip()]
        if not self.scenes:
            raise ValueError("нужна хотя бы одна непустая сцена")
        return self


class PhotoResult(BaseModel):
    id: str
    request_id: str
    image_path: str
    image_url: Optional[str] = None
    prompt: str
    seed: int
    model_name: str
    width: int
    height: int

    # Идентичность — часть контракта, а не приложение к нему.
    similarity: Optional[float] = Field(None, description="Cosine ArcFace к эталону персонажа")
    # Доля площади кадра, занятая лицом. Без неё похожесть лица — метрика, которую легко
    # «выиграть» неправильным способом: косинус ArcFace тем выше, чем крупнее и фронтальнее лицо,
    # поэтому подбор параметров по одной похожести уводит пайплайн в макрошоты, а сцена при этом
    # перестаёт слушаться промпта. Замерено: ip_adapter_scale 0.85 давал похожесть 0.71 и лицо на
    # пол-кадра там, где в промпте было "walking past a shop window, mid-step".
    face_area_ratio: Optional[float] = Field(None, ge=0.0, le=1.0)
    accepted: bool = Field(True, description="Прошёл ли кадр порог сходства и проверки безопасности")
    reject_reason: Optional[str] = None

    hires_applied: bool = False
    face_detailer_applied: bool = False
    duration_ms: int = 0

    nsfw_score: Optional[float] = None
    estimated_age: Optional[float] = None
    safety_allowed: bool = True
    created_at: str


class JobResponse(BaseModel):
    job_id: str
    status: GenerationStatus
    results: List[PhotoResult] = Field(default_factory=list)
    error: Optional[str] = None
    created_at: str
    finished_at: Optional[str] = None
    progress: str = ""


class VideoRequest(BaseModel):
    """Оживление одного готового кадра.

    Источник — путь к кадру НА ПОДЕ, а не загруженный файл: кадр только что сгенерирован тем же
    воркером и уже лежит в OUTPUT_DIR. Гонять его на клиент и обратно ради видео бессмысленно.
    """

    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(default_factory=lambda: uuid4().hex)
    image_path: str = Field(..., description="Путь к исходному кадру на поде")
    prompt: str = Field("", max_length=4000, description="Описание движения")
    negative_prompt: Optional[str] = Field(None, max_length=4000)
    model_name: Optional[str] = Field(None, description="Ключ из VIDEO_MODEL_CONFIGS")
    num_frames: Optional[int] = Field(None, ge=9, le=241)
    fps: Optional[int] = Field(None, ge=6, le=30)
    num_inference_steps: Optional[int] = Field(None, ge=1, le=60)
    guidance_scale: Optional[float] = Field(None, ge=0.0, le=20.0)
    frame_seed: Optional[int] = Field(None, ge=0, le=2**32 - 1)
    motion_seed: Optional[int] = Field(None, ge=0, le=2**32 - 1)
    # Подрезка исходника вокруг лица перед анимацией: 1.0 как есть, 1.8 примерно поясной план.
    # Чем крупнее человек в кадре, тем больше пикселей достаётся лицу в готовом ролике.
    subject_zoom: float = Field(1.0, ge=1.0, le=3.0)
    # False — полное число шагов без ускоряющей LoRA: дольше впятеро, картинка достовернее.
    fast: bool = True


class VideoResult(BaseModel):
    id: str
    request_id: str
    video_path: str
    model_name: str
    num_frames: int
    fps: int
    frame_seed: int
    motion_seed: int
    duration_ms: int
    # Safety по видео считается по ВЫБОРКЕ кадров, а не по одному: ролик может начаться
    # безобидно и уехать к середине, и проверка только первого кадра это пропустит.
    frames_checked: int = 0
    safety_allowed: bool = True
    # Похожесть лица на среднем кадре ролика — единственный способ увидеть, что видеомодель не
    # подменила человека по ходу движения. Может быть None: на резком повороте головы детектор
    # штатно не находит лицо.
    similarity: Optional[float] = None


class VideoEnhanceRequest(BaseModel):
    """Постобработка готового ролика: восстановление лица покадрово."""

    model_config = ConfigDict(populate_by_name=True)

    video_path: str = Field(..., description="Путь к ролику на поде")
    character: Optional[str] = Field(None, description="Нужен для method=detailer — берём эталон")
    method: str = Field("gfpgan", description="gfpgan | detailer | both")
    # Апскейл всего кадра при GFPGAN. Резкое лицо на мыльном фоне выглядит наклейкой, поэтому
    # увеличивается кадр целиком, а не только лицо.
    upscale: int = Field(1, ge=1, le=2)
    # Доля восстановленного лица при смешивании с исходным кадром. Ниже единицы — меньше
    # мерцания между кадрами ценой резкости.
    blend: float = Field(0.75, ge=0.1, le=1.0)
    model_name: str = Field(DEFAULT_PHOTO_MODEL, description="Чекпойнт для detailer")
    # Обрабатывать каждый N-й кадр. Только для быстрой пробы: пропуски дают «пульсирование»
    # чёткости между обработанными и необработанными кадрами.
    every: int = Field(1, ge=1, le=10)
    seed: int = Field(0, ge=0, le=2**32 - 1)


class VideoEnhanceResult(BaseModel):
    method: str
    video_path: str
    frames_total: int
    frames_processed: int
    elapsed_ms: int
    # Похожесть лица на среднем кадре ПОСЛЕ обработки — по ней видно, вытянул ли метод личность
    # или просто сделал лицо чётким и чужим.
    similarity: Optional[float] = None


class CharacterInfo(BaseModel):
    name: str
    created_at: str
    reference_image: Optional[str] = None
    has_embedding: bool = True
    # Сколько ракурсов эталона сохранено. Портретный режим FaceID усредняет их в одну личность, и
    # на одном ракурсе вырождается в base — по этому числу видно, работает он на полную или нет.
    reference_count: int = 1


class CreateCharacterRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=64, pattern=r"^[\w\-]+$")
    reference_image: Optional[str] = Field(None, description="base64 эталонного кадра")
    reference_path: Optional[str] = Field(None, description="путь к эталону на диске пода")
    prompt: Optional[str] = Field(None, min_length=5, max_length=12000,
                                  description="если эталона нет — сгенерировать его по описанию")
    seed: Optional[int] = Field(None, ge=0, le=2**32 - 1)
    overwrite: bool = False

    @model_validator(mode="after")
    def _need_source(self):
        if not any((self.reference_image, self.reference_path, self.prompt)):
            raise ValueError("нужен reference_image, reference_path или prompt")
        return self


class HealthResponse(BaseModel):
    status: str
    version: str
    device: str
    gpu: Optional[str] = None
    vram_total_gb: Optional[float] = None
    vram_used_gb: Optional[float] = None
    models_loaded: List[str] = Field(default_factory=list)
    characters: int = 0
    jobs_active: int = 0
