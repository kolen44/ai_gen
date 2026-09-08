"""Клиент Nano Banana Pro (Gemini 3 Pro Image).

Параллельный трек к self-hosted ветке, поэтому модуль изолирован и ничего не знает про worker/.

Что даёт: принимает до 14 референсов за запрос (на этом строится цепочка пака), держит персонажа
между сценами без адаптеров и обучения, даёт «телефонную» фактуру из коробки.
Чего не даёт: воспроизводимости по seed, доступа к весам, работы офлайн, предсказуемой цены.

Нужен GEMINI_API_KEY в окружении или .env.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

# GA-версия, а не -preview: то же самое качество, но без риска, что превью-эндпоинт отключат.
# "Nano Banana Pro" — народное название именно этой модели.
DEFAULT_MODEL = "gemini-3-pro-image"

# Жёсткий предел самого API: больше 14 референсов за запрос он не принимает.
MAX_REFERENCE_IMAGES = 14

# Цена за кадр по официальному прайсу на момент написания. Используется только для отчётности:
# реальный счёт всё равно приходит от Google, здесь — чтобы в логе была оценка стоимости пака.
PRICE_PER_IMAGE_USD = {"1K": 0.134, "2K": 0.134, "4K": 0.24}


class NBPError(RuntimeError):
    """Ошибка вызова API — с текстом причины от Google, а не голым кодом."""


@dataclass
class NBPResult:
    image_path: Path
    prompt: str
    references: list[Path] = field(default_factory=list)
    elapsed_s: float = 0.0
    cost_usd: float = 0.0
    finish_reason: Optional[str] = None


def _load_api_key(env_path: Optional[Path] = None) -> str:
    key = os.getenv("GEMINI_API_KEY")
    if key:
        return key.strip()

    env_path = env_path or Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = re.match(r"^GEMINI_API_KEY=(.+)$", line.strip())
            if m:
                return m.group(1).strip()

    raise NBPError(
        "GEMINI_API_KEY не найден ни в окружении, ни в .env. "
        "Ключ берётся в Google AI Studio."
    )


def _encode_image(path: Path) -> dict:
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    return {
        "inline_data": {
            "mime_type": mime,
            "data": base64.b64encode(path.read_bytes()).decode("ascii"),
        }
    }


class NanoBananaPro:
    """Тонкая обёртка над generateContent. Без SDK — одна зависимость меньше, а нужен ровно
    один эндпоинт."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        image_size: str = "2K",
        timeout: int = 180,
        min_interval_s: float = 45.0,
    ):
        self.api_key = api_key or _load_api_key()
        self.model = model
        self.image_size = image_size
        self.timeout = timeout
        # Пауза между запросами. Лимит здесь поминутный, и серия подряд выбивает 429 уже на
        # втором кадре. Проверено экспериментом с паузами: 6 секунд не хватает, 20 — проходят
        # все варианты запроса подряд. Текст ошибки при этом упоминает free_tier и "limit: 0",
        # что вводит в заблуждение: биллинг включён, упирается именно темп. Если тариф позволит
        # больше RPM, значение можно снизить параметром конструктора.
        self.min_interval_s = min_interval_s
        self._last_request_at = 0.0

    def generate(
        self,
        prompt: str,
        *,
        output_path: Path,
        references: Sequence[Path] = (),
        aspect_ratio: str = "1:1",
        # Восемь попыток, а не три: окно лимита минутное, сервер сам просит ждать ~60 секунд,
        # и при генерации пака подряд в него можно упереться несколько раз кряду. Дешевле
        # подождать, чем уронить сборку пака на середине и потерять уже оплаченные кадры.
        max_retries: int = 8,
    ) -> NBPResult:
        """Один кадр. references — до 14 изображений, которые модель использует как образец
        внешности/стиля; именно так строится цепочка эталонов."""
        references = [Path(r) for r in references]
        if len(references) > MAX_REFERENCE_IMAGES:
            raise NBPError(
                f"референсов {len(references)}, а API принимает максимум {MAX_REFERENCE_IMAGES}. "
                "Отбери лучшие по метрике консистентности."
            )
        for ref in references:
            if not ref.exists():
                raise NBPError(f"референс не найден: {ref}")

        parts: list[dict] = [_encode_image(r) for r in references]
        parts.append({"text": prompt})

        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "responseModalities": ["IMAGE"],
                "imageConfig": {"aspectRatio": aspect_ratio, "imageSize": self.image_size},
            },
        }

        url = f"{API_BASE}/{self.model}:generateContent"
        body = json.dumps(payload).encode("utf-8")

        start = time.monotonic()
        last_error: Optional[str] = None
        for attempt in range(1, max_retries + 1):
            self._respect_interval()
            req = urllib.request.Request(
                url,
                data=body,
                headers={
                    "x-goog-api-key": self.api_key,
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = json.loads(r.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as e:
                raw = e.read().decode("utf-8", "replace")
                try:
                    payload_err = json.loads(raw)["error"]
                    last_error = payload_err.get("message", raw[:300])
                except Exception:
                    payload_err, last_error = {}, raw[:300]

                # 429/5xx — временные, повторяем; остальное бессмысленно, ошибка в самом запросе.
                if e.code in (429, 500, 502, 503, 504) and attempt < max_retries:
                    time.sleep(self._retry_delay(payload_err, attempt, is_rate_limit=e.code == 429))
                    continue
                raise NBPError(f"HTTP {e.code}: {last_error}") from None
            except Exception as e:
                last_error = str(e)
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                    continue
                raise NBPError(f"сетевая ошибка: {last_error}") from None
        else:
            raise NBPError(f"не удалось после {max_retries} попыток: {last_error}")

        elapsed = time.monotonic() - start
        image_bytes, finish_reason = self._extract_image(data)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(image_bytes)

        return NBPResult(
            image_path=output_path,
            prompt=prompt,
            references=references,
            elapsed_s=elapsed,
            cost_usd=PRICE_PER_IMAGE_USD.get(self.image_size, 0.134),
            finish_reason=finish_reason,
        )

    def _respect_interval(self) -> None:
        """Держит минимальный интервал между запросами — лимит здесь поминутный, и серия
        запросов подряд выбивает 429 уже на втором-третьем кадре."""
        if self.min_interval_s <= 0:
            return
        wait = self.min_interval_s - (time.monotonic() - self._last_request_at)
        if wait > 0:
            time.sleep(wait)
        self._last_request_at = time.monotonic()

    @staticmethod
    def _retry_delay(error_payload: dict, attempt: int, *, is_rate_limit: bool) -> float:
        """Сколько ждать перед повтором.

        Google кладёт в details блок RetryInfo с полем retryDelay ("17s") — это точное указание
        сервера, и оно всегда лучше нашей догадки. Если его нет, для 429 берём заведомо длинную
        паузу: окно лимита минутное, и стандартный экспоненциальный откат в 2-4 секунды в него
        просто не попадает.
        """
        for detail in error_payload.get("details", []) or []:
            if detail.get("@type", "").endswith("RetryInfo"):
                raw = str(detail.get("retryDelay", "")).rstrip("s")
                try:
                    return max(1.0, float(raw)) + 1.0
                except ValueError:
                    pass
        if is_rate_limit:
            return min(60.0, 20.0 * attempt)
        return 2.0 ** attempt

    @staticmethod
    def _extract_image(data: dict) -> tuple[bytes, Optional[str]]:
        """Достаёт картинку из ответа. Отдельный метод, потому что содержательная причина отказа
        (сработал фильтр, не та модальность) лежит в finishReason/promptFeedback, и без её
        показа отладка превращается в гадание."""
        candidates = data.get("candidates") or []
        if not candidates:
            feedback = data.get("promptFeedback", {})
            raise NBPError(f"пустой ответ, promptFeedback={feedback}")

        candidate = candidates[0]
        finish_reason = candidate.get("finishReason")
        for part in candidate.get("content", {}).get("parts", []):
            inline = part.get("inlineData") or part.get("inline_data")
            if inline and inline.get("data"):
                return base64.b64decode(inline["data"]), finish_reason

        text = " ".join(
            p.get("text", "") for p in candidate.get("content", {}).get("parts", [])
        ).strip()
        raise NBPError(
            f"в ответе нет изображения (finishReason={finish_reason}). "
            f"Текст ответа: {text[:300] or '<пусто>'}"
        )


def iter_reference_sets(images: Sequence[Path], limit: int = MAX_REFERENCE_IMAGES) -> Iterable[list[Path]]:
    """Нарезает накопленный пак на наборы по <=14 штук — ровно тот предел, что принимает API."""
    images = list(images)
    for i in range(0, len(images), limit):
        yield images[i:i + limit]
