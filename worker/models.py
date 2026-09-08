"""
Pydantic-схемы запросов/результатов для фото и видео. Форма намеренно похожа на
GenerationRequest/GenerationResult/GenerateResponse из WetDreams
(WetDreams/runpod/wai-nsfw-illustrious-sdxl/models.py): id/seed/duration_ms/nsfw_score — те же
поля, что у них, чтобы результат этого воркера был совместим по форме, если фичу когда-нибудь
перенесут в их прод-воркер (см. REQUIREMENTS_FROM_USER.md, п.8).

Отличие от их nsfw_score: у них это Optional[float] = None с комментарием в коде "Could implement
NSFW detection here" — то есть объявлено, но никогда не заполняется. Здесь оно реально считается
через safety/moderation_pipeline.py и дополнено age/identity-полями, которых у них нет вообще.
"""

from __future__ import annotations

from typing import Optional, List
from uuid import uuid4

from pydantic import BaseModel, Field


# === Фото ===

class PhotoGenerationRequest(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    model_name: str = Field(..., description="Ключ из worker.config.PHOTO_MODEL_CONFIGS")
    # Лимиты подняты вместе с внешней схемой (runpod_worker/models.py). Схем ДВЕ: HTTP-запрос
    # валидируется одной, а внутренний запрос к генератору — этой. Правка только внешней ничего
    # не дала: запрос проходил в сервис и падал уже там, с тем же «String should have at most
    # 3000 characters», только другим типом ошибки.
    prompt: str = Field(..., min_length=10, max_length=12000)
    negative_prompt: Optional[str] = Field(None, max_length=4000)
    width: int = Field(1024, ge=512, le=1536)
    height: int = Field(1024, ge=512, le=1536)
    num_inference_steps: int = Field(28, ge=10, le=60)
    guidance_scale: float = Field(6.0, ge=0.0, le=20.0)
    seed: Optional[int] = Field(None, ge=0, le=2**32 - 1)
    # face_embedding: ArcFace-эмбеддинг эталонного лица (см. worker/identity.py), None для Stage 0
    # (генерация самого эталонного лица, которому ещё не на что опираться).
    face_embedding: Optional[List[float]] = Field(None)
    # Режим portrait принимает НЕСКОЛЬКО эталонов одного человека (разные ракурсы) вместо одного.
    # Отдельное поле, а не расширение face_embedding до списка: face_embedding читают ещё и
    # safety-гейт со сверкой по watchlist, которым нужен ровно один вектор, и менять его тип
    # значило бы трогать несвязанный код. Если задано — используется вместо face_embedding.
    face_embeddings: Optional[List[List[float]]] = Field(None)
    ip_adapter_scale: Optional[float] = Field(None, ge=0.0, le=1.5)
    # None = взять DEFAULT_IDENTITY_MODE из worker/config.py. Значения: base | portrait | plusv2.
    identity_mode: Optional[str] = Field(None)
    # Только для plusv2: путь к эталонному кадру. PlusV2 помимо ArcFace-эмбеддинга требует CLIP-
    # эмбеддинг ТОГО ЖЕ изображения, а CLIP считается по картинке, не по вектору — поэтому одного
    # face_embedding здесь принципиально недостаточно.
    reference_image_path: Optional[str] = Field(None)


class PhotoGenerationResult(BaseModel):
    id: str
    request_id: str
    image_path: str
    seed: int
    duration_ms: int
    model_name: str
    # Safety — заполняется реально (в отличие от WetDreams-стаба), см. safety/moderation_pipeline.py
    nsfw_score: Optional[float] = None
    estimated_age: Optional[float] = None
    watchlist_similarity: Optional[float] = None
    safety_allowed: bool
    created_at: str


# === Видео ===

class VideoGenerationRequest(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    model_name: str = Field(..., description="Ключ из worker.config.VIDEO_MODEL_CONFIGS")
    input_image_path: str = Field(..., description="Путь к одному из 8 фото — источник для I2V")
    prompt: str = Field("", max_length=4000, description="Доп. описание движения/сцены")
    negative_prompt: Optional[str] = Field(None, max_length=4000)
    num_frames: Optional[int] = None
    fps: Optional[int] = None
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    frame_seed: Optional[int] = Field(None, ge=0, le=2**32 - 1)
    motion_seed: Optional[int] = Field(None, ge=0, le=2**32 - 1)
    # Во сколько раз подрезать исходный кадр вокруг лица перед анимацией. 1.0 — как есть.
    # Смысл в пикселях: видеомодель рендерит в 704x1280 и сжимает через VAE в 16 раз по каждой
    # оси, поэтому на ростовом плане лицу достаётся ~90x120 пикселей ролика и текстуру там рисовать
    # не из чего. Подрезка кадра — единственный способ отдать лицу больше пикселей ДО генерации;
    # постобработка их не создаёт.
    subject_zoom: float = Field(1.0, ge=1.0, le=3.0)
    # Ускоряющая LoRA: 4 шага вместо 40. Быстрее впятеро, но картинка грубее — за 4 шага модель
    # не успевает проработать то, что прорабатывает за 40.
    fast: bool = True


class VideoGenerationResult(BaseModel):
    id: str
    request_id: str
    video_path: str
    frame_seed: int
    motion_seed: int
    duration_ms: int
    model_name: str
    num_frames: int
    frames_checked: int
    safety_allowed: bool
    created_at: str
