"""
Nano Banana Pro через fal.ai — тот же движок, что и напрямую у Google, но без его квоты.

Зачем этот клиент помимо pipeline/nbp_client.py. Прямой доступ к Gemini упирается в квоту
проекта: в теле отказа метрика называется generate_content_free_tier_requests с limit 0, то есть
платный тариф к проекту не применён. На практике это означает несколько запросов подряд, а потом
блокировку на десятки минут — собрать пак из 12 кадров невозможно. Проверено многократно.

fal раздаёт ту же модель как fal-ai/nano-banana-pro, и по нашему ключу она отвечает стабильно,
как Seedream. Интерфейс здесь намеренно совпадает с seedream_client.Seedream (text_to_image / edit),
чтобы оркестратор мог подставлять любой движок без ветвлений.

Когда нужен прямой nbp_client вместо этого: если у тебя настроен Tier 1 у Google и хочется платить
Google напрямую, а не через посредника. Функционально они эквивалентны.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

from pipeline.seedream_client import Seedream, SeedreamError, _strip_echoed_input

FAL_RUN_BASE = "https://fal.run"

TEXT_TO_IMAGE_MODEL = "fal-ai/nano-banana-pro"
EDIT_MODEL = "fal-ai/nano-banana-pro/edit"


class NBPFalError(RuntimeError):
    pass


@dataclass
class NBPFalResult:
    image_path: Path
    prompt: str
    references: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0


class NanoBananaProFal:
    """Клиент с теми же методами, что у Seedream: text_to_image() и edit()."""

    def __init__(self, *, api_key: Optional[str] = None, timeout: int = 300,
                 image_size: str = "square_hd"):
        # Загрузку файлов, ключ и скачивание результата переиспользуем у Seedream — провайдер
        # один и тот же, дублировать эти сто строк незачем.
        self._fal = Seedream(api_key=api_key, timeout=timeout)
        self.timeout = timeout
        self.image_size = image_size

    @property
    def api_key(self) -> str:
        return self._fal.api_key

    def _post(self, model: str, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{FAL_RUN_BASE}/{model}",
            data=json.dumps(payload).encode(),
            headers={"Authorization": f"Key {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", "replace")
            try:
                detail = _strip_echoed_input(json.loads(raw).get("detail", raw))
            except Exception:
                detail = raw[:400]
            raise NBPFalError(f"HTTP {e.code}: {str(detail)[:400]}") from None

    @staticmethod
    def _extract(data: dict) -> str:
        images = data.get("images") or []
        if not images:
            raise NBPFalError(f"в ответе нет изображений: {list(data)}")
        url = images[0].get("url")
        if not url:
            raise NBPFalError("в ответе нет ссылки на изображение")
        return url

    def text_to_image(self, prompt: str, *, output_path: Path,
                      image_size: Optional[str] = None) -> NBPFalResult:
        payload = {"prompt": prompt, "image_size": image_size or self.image_size}
        start = time.monotonic()
        data = self._post(TEXT_TO_IMAGE_MODEL, payload)
        elapsed = time.monotonic() - start
        path = self._fal._download(self._extract(data), output_path)
        return NBPFalResult(image_path=path, prompt=prompt, elapsed_s=elapsed)

    def edit(self, prompt: str, *, references: Sequence[Path | str], output_path: Path,
             image_size: Optional[str] = None) -> NBPFalResult:
        urls: list[str] = []
        for ref in references:
            if isinstance(ref, str) and ref.startswith(("http://", "https://", "data:")):
                urls.append(ref)
            else:
                urls.append(self._fal.to_reference(Path(ref)))

        if not urls:
            raise NBPFalError("нужен хотя бы один референс — на них держится идентичность")

        payload = {"prompt": prompt, "image_urls": urls,
                   "image_size": image_size or self.image_size}
        start = time.monotonic()
        data = self._post(EDIT_MODEL, payload)
        elapsed = time.monotonic() - start
        path = self._fal._download(self._extract(data), output_path)
        return NBPFalResult(image_path=path, prompt=prompt, references=urls, elapsed_s=elapsed)
