"""Safety-слой: четыре проверки, все self-hosted, работают на том же поде.

  1. PromptGate         — до генерации, deny-list по тексту промпта.
  2. ImageSafetyGate    — после, NSFW-классификатор на кадре.
  3. IdentitySafetyGate — после, возраст по лицу и сверка с watchlist реальных людей.

Границы честные: возраст — оценка по лицу, а не документ, а watchlist ограничен списком.
Проверки «не похож ни на кого из живущих» не существует в принципе, поэтому финальное ручное
ревью перед публикацией никто не отменяет.

InsightFace берётся из worker.identity — тот же закешированный экземпляр, что считает эмбеддинг
для FaceID. Одна загрузка на процесс, а не две копии на GPU.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# 1. Prompt gate — работает до того, как потрачена хоть секунда GPU-времени
# ---------------------------------------------------------------------------

# Список должен пополняться по мере находок ложноотрицательных срабатываний, это стартовый набор.
DENYLIST_PATTERNS = [
    r"\bchild(ren)?\b",
    r"\bkid[s]?\b",
    r"\btoddler\b",
    r"\bminor\b",
    r"\bteen(ager)?\b",
    r"\bpreteen\b",
    r"\b(school|high school)\s*(girl|uniform)\b",
    r"\bloli\b",
    r"\bunderage\b",
]
# Числовой возраст ("16 years old", "17yo") НЕ входит в этот список: он проверяется отдельно
# в PromptGate.check() и блокирует только если число ниже min_allowed_age — иначе легитимные
# промпты вида "30 years old" ловились бы просто за наличие цифры перед "years old".

_COMPILED_DENYLIST = [re.compile(p, re.IGNORECASE) for p in DENYLIST_PATTERNS]


@dataclass
class PromptGateResult:
    allowed: bool
    matched_patterns: list[str] = field(default_factory=list)
    numeric_age_found: int | None = None


class PromptGate:
    """Быстрый текстовый фильтр перед запуском генерации."""

    def __init__(self, min_allowed_age: int = 18, extra_denylist: list[str] | None = None):
        self.min_allowed_age = min_allowed_age
        patterns = list(DENYLIST_PATTERNS) + (extra_denylist or [])
        self._compiled = [re.compile(p, re.IGNORECASE) for p in patterns]

    def check(self, prompt: str) -> PromptGateResult:
        matched = [p.pattern for p in self._compiled if p.search(prompt)]

        # Отдельно достаём числовой возраст, если он есть в промпте — это не то же самое,
        # что просто наличие слова "years old" в деньлисте: число ниже порога блокирует
        # конкретно из-за значения, а не из-за факта наличия паттерна.
        age_match = re.search(r"\b(\d{1,2})\s*(?:y/?o|years?\s*old)\b", prompt, re.IGNORECASE)
        numeric_age = int(age_match.group(1)) if age_match else None
        if numeric_age is not None and numeric_age < self.min_allowed_age:
            matched.append(f"numeric_age={numeric_age}")

        return PromptGateResult(
            allowed=len(matched) == 0,
            matched_patterns=matched,
            numeric_age_found=numeric_age,
        )


# ---------------------------------------------------------------------------
# 2. Image NSFW gate
# ---------------------------------------------------------------------------

@dataclass
class ImageSafetyResult:
    allowed: bool
    label: str
    score: float


class ImageSafetyGate:
    """NSFW-классификатор кадра: Falconsai/nsfw_image_detection (ViT, apache-2.0).

    Ленивая загрузка, чтобы модуль импортировался и тестировался без скачивания весов.
    """

    def __init__(
        self,
        model_name: str = "Falconsai/nsfw_image_detection",
        nsfw_threshold: float = 0.5,
        lazy: bool = True,
    ):
        self.model_name = model_name
        self.nsfw_threshold = nsfw_threshold
        self._pipe = None
        if not lazy:
            self._load()

    def _load(self):
        if self._pipe is None:
            from transformers import pipeline as hf_pipeline

            self._pipe = hf_pipeline("image-classification", model=self.model_name)

    def check(self, image: Image.Image) -> ImageSafetyResult:
        self._load()
        results = self._pipe(image)  # [{"label": "nsfw", "score": 0.02}, {"label": "normal", ...}]
        nsfw_score = next((r["score"] for r in results if r["label"].lower() == "nsfw"), 0.0)
        top = max(results, key=lambda r: r["score"])
        return ImageSafetyResult(
            allowed=nsfw_score < self.nsfw_threshold,
            label=top["label"],
            score=nsfw_score,
        )

    def check_video_frames(self, frames: list[Image.Image], sample_every: int = 5) -> list[ImageSafetyResult]:
        """Проверяет каждый sample_every-й кадр видео — не весь клип целиком, чтобы не тратить
        лишнее время на 5-секундный ролик (при 24fps это ~120 кадров, каждый 5-й = 24 проверки)."""
        return [self.check(f) for f in frames[::sample_every]]


# ---------------------------------------------------------------------------
# 3. Identity safety gate — возраст (эвристика) + похожесть на watchlist
# ---------------------------------------------------------------------------

@dataclass
class IdentitySafetyResult:
    allowed: bool
    estimated_age: float | None
    age_flagged: bool
    max_watchlist_similarity: float | None
    similarity_flagged: bool
    matched_watchlist_name: str | None


class IdentitySafetyGate:
    """Возраст по лицу и сверка с watchlist через InsightFace.

    watchlist_dir — папка с фото людей, на которых персонаж быть похож не должен. Эмбеддинги
    считаются один раз и кешируются рядом.

    Порог 0.5 по косинусу — с запасом: одно лицо ArcFace обычно даёт выше 0.6. Лучше лишний раз
    отклонить и посмотреть руками, чем пропустить.
    """

    def __init__(
        self,
        watchlist_dir: str | Path | None = None,
        min_allowed_age: float = 25.0,  # запас от целевых "выглядит на 28-35"
        similarity_threshold: float = 0.5,
        face_model_name: str = "buffalo_l",
        lazy: bool = True,
    ):
        self.watchlist_dir = Path(watchlist_dir) if watchlist_dir else None
        self.min_allowed_age = min_allowed_age
        self.similarity_threshold = similarity_threshold
        self.face_model_name = face_model_name
        self._app = None
        self._watchlist_embeddings: dict[str, np.ndarray] = {}
        if not lazy:
            self._load()

    def _load(self):
        if self._app is not None:
            return
        from worker.identity import get_face_analysis

        self._app = get_face_analysis(self.face_model_name)
        if self.watchlist_dir and self.watchlist_dir.exists():
            self._load_watchlist()

    def _load_watchlist(self):
        cache_path = self.watchlist_dir / "_embeddings_cache.npz"
        if cache_path.exists():
            data = np.load(cache_path, allow_pickle=True)
            self._watchlist_embeddings = {k: data[k] for k in data.files}
            return
        for img_path in self.watchlist_dir.glob("*.*"):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            img = np.array(Image.open(img_path).convert("RGB"))
            faces = self._app.get(img)
            if faces:
                self._watchlist_embeddings[img_path.stem] = faces[0].normed_embedding
        if self._watchlist_embeddings:
            np.savez(cache_path, **self._watchlist_embeddings)

    def check(self, image: Image.Image) -> IdentitySafetyResult:
        self._load()
        from worker.identity import detect_primary_face_array

        img_arr = np.array(image.convert("RGB"))
        # Через worker.identity, а не app.get напрямую: там есть вторая попытка на кадре с полями.
        # Сырой RetinaFace не находит лицо на макропортрете, и гейт молча возвращал allowed=True —
        # то есть возрастная проверка отключалась именно на крупных планах.
        face = detect_primary_face_array(img_arr)
        if face is None:
            # Лица действительно нет — не наш кейс для этого гейта, пропускаем как allowed=True,
            # NSFW/prompt-гейты всё равно отработали отдельно.
            return IdentitySafetyResult(True, None, False, None, False, None)

        estimated_age = float(face.age) if face.age else None
        age_flagged = estimated_age is not None and estimated_age < self.min_allowed_age

        max_sim = None
        matched_name = None
        similarity_flagged = False
        if self._watchlist_embeddings:
            embedding = face.embedding
            sims = {
                name: float(np.dot(embedding, ref)) for name, ref in self._watchlist_embeddings.items()
            }
            matched_name, max_sim = max(sims.items(), key=lambda kv: kv[1])
            similarity_flagged = max_sim >= self.similarity_threshold

        return IdentitySafetyResult(
            allowed=not (age_flagged or similarity_flagged),
            estimated_age=estimated_age,
            age_flagged=age_flagged,
            max_watchlist_similarity=max_sim,
            similarity_flagged=similarity_flagged,
            matched_watchlist_name=matched_name if similarity_flagged else None,
        )


# ---------------------------------------------------------------------------
# Оркестратор — единая точка входа для всех гейтов.
# ---------------------------------------------------------------------------

@dataclass
class FullSafetyReport:
    prompt_result: PromptGateResult
    image_result: ImageSafetyResult | None
    identity_result: IdentitySafetyResult | None

    @property
    def allowed(self) -> bool:
        checks = [self.prompt_result.allowed]
        if self.image_result is not None:
            checks.append(self.image_result.allowed)
        if self.identity_result is not None:
            checks.append(self.identity_result.allowed)
        return all(checks)


class SafetyPipeline:
    """Единая точка входа: run_prompt() перед генерацией, run_image() после каждого кадра/фото."""

    def __init__(
        self,
        watchlist_dir: str | Path | None = None,
        min_allowed_age: float | None = None,
        nsfw_threshold: float | None = None,
        similarity_threshold: float = 0.5,
    ):
        # Как и порог NSFW: None означает "взять из .env". Оба параметра читаются здесь, в
        # единственной точке создания гейтов, чтобы весь пайплайн работал по одной конфигурации.
        if min_allowed_age is None:
            try:
                from pipeline.prompt_config import min_allowed_age as _age_from_env

                min_allowed_age = _age_from_env()
            except Exception:
                min_allowed_age = 18.0
        # nsfw_threshold=None означает "взять из .env (NSFW_THRESHOLD)". Значение берётся здесь,
        # в единственной точке создания гейтов, а не в каждом вызывающем скрипте — иначе часть
        # пайплайна работала бы по .env, а часть по зашитому дефолту, и понять по логу, каким
        # порогом отбирался конкретный кадр, стало бы невозможно.
        if nsfw_threshold is None:
            try:
                from pipeline.prompt_config import nsfw_threshold as _from_env

                nsfw_threshold = _from_env()
            except Exception:
                # safety не должен падать из-за конфига: при недоступном .env возвращаемся
                # к строгому значению, а не к разрешительному.
                nsfw_threshold = 0.5

        self.nsfw_threshold = nsfw_threshold
        self.prompt_gate = PromptGate()
        self.image_gate = ImageSafetyGate(nsfw_threshold=nsfw_threshold)
        self.identity_gate = IdentitySafetyGate(
            watchlist_dir=watchlist_dir,
            min_allowed_age=min_allowed_age,
            similarity_threshold=similarity_threshold,
        )

    def run_prompt(self, prompt: str) -> PromptGateResult:
        return self.prompt_gate.check(prompt)

    def run_image(self, image: Image.Image) -> FullSafetyReport:
        image_result = self.image_gate.check(image)
        identity_result = self.identity_gate.check(image)
        # prompt_result здесь заведомо allowed=True, т.к. до генерации уже прошли run_prompt —
        # включаем "пустой" allowed-результат просто чтобы FullSafetyReport.allowed агрегировал всё.
        return FullSafetyReport(
            prompt_result=PromptGateResult(allowed=True),
            image_result=image_result,
            identity_result=identity_result,
        )


if __name__ == "__main__":
    # Быстрый self-test PromptGate без загрузки тяжёлых моделей.
    gate = PromptGate()
    for test_prompt in [
        "adult woman, 30 years old, studio portrait, editorial lighting",
        "16 years old girl in school uniform",
        "teen model on the beach",
    ]:
        result = gate.check(test_prompt)
        print(f"{test_prompt!r:65} -> allowed={result.allowed} matched={result.matched_patterns}")
