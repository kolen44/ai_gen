"""
Клиент Seedream 4.5 (ByteDance) через fal.ai — шаг перегенерации образов.

Роль в пайплайне. Берём готовый кадр персонажа (из NBP-трека или своей генерации), даём Seedream
как референс и описываем новый образ: одежда, поза, локация, окружение. Модель сохраняет лицо и
фигуру с референса и перерисовывает остальное. Тот же смысл, что у перегенерации на своих весах, но на
чужих весах и заметно быстрее — не нужен свой GPU.

Ключевое отличие от перегенерации на своих весах:
  + не требует пода, работает за секунды, качество из коробки;
  - нет seed-воспроизводимости в том же смысле, платно за кадр, зависит от доступности провайдера
    и его правил по контенту.

Формат ввода. Эндпоинт /edit принимает image_urls — список ссылок на изображения. Локальные файлы
загружаются в хранилище fal методом upload() ниже; data-URI поддерживается не всеми версиями
эндпоинта, поэтому по умолчанию используется явная загрузка.

Требует FAL_KEY в окружении или .env (формат "<id>:<secret>").
"""

from __future__ import annotations

import json
import mimetypes
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

FAL_RUN_BASE = "https://fal.run"
FAL_UPLOAD_INITIATE_URL = "https://rest.alpha.fal.ai/storage/upload/initiate"

# v4.5 — актуальная на момент написания. Если провайдер выкатит новую версию, меняется одна
# строка; проверить доступные пути можно, отправив пустой запрос — живой эндпоинт отвечает 422
# со списком недостающих полей, несуществующий — 404.
DEFAULT_EDIT_MODEL = "fal-ai/bytedance/seedream/v4.5/edit"
DEFAULT_T2I_MODEL = "fal-ai/bytedance/seedream/v4/text-to-image"



def _strip_echoed_input(detail):
    """Убирает из ответа об ошибке эхо самого запроса.

    fal возвращает в 422 весь input, включая картинку в data-URI. Одна такая строка весит
    мегабайты: она попадает в лог прогона, оттуда в /api/status и вешает страницу админки.
    Само сообщение об ошибке при этом в поле msg, и оно короткое.
    """
    if isinstance(detail, list):
        return [_strip_echoed_input(item) for item in detail]
    if isinstance(detail, dict):
        return {k: v for k, v in detail.items() if k != "input"}
    return detail

class SeedreamError(RuntimeError):
    pass


@dataclass
class SeedreamResult:
    image_path: Path
    prompt: str
    references: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0
    seed: Optional[int] = None


def _load_key(env_path: Optional[Path] = None) -> str:
    key = os.getenv("FAL_KEY")
    if key:
        return key.strip()
    env_path = env_path or Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = re.match(r"^FAL_KEY=(.+)$", line.strip())
            if m:
                return m.group(1).strip()
    raise SeedreamError("FAL_KEY не найден ни в окружении, ни в .env")


class Seedream:
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        edit_model: str = DEFAULT_EDIT_MODEL,
        t2i_model: str = DEFAULT_T2I_MODEL,
        timeout: int = 300,
    ):
        self.api_key = api_key or _load_key()
        self.edit_model = edit_model
        self.t2i_model = t2i_model
        self.timeout = timeout

    # --- вспомогательное ---

    @property
    def _auth(self) -> dict:
        return {"Authorization": f"Key {self.api_key}"}

    def to_reference(self, path: Path, max_side: int = 2048, quality: int = 92) -> str:
        """Готовит локальный файл к отправке и возвращает то, что можно положить в image_urls.

        Проверено на живом API: одношаговый POST на rest.alpha.fal.ai/storage/upload обрывает
        соединение (эндпоинт устарел). Рабочих путей два, и используются оба:

          1. data-URI прямо в image_urls — принимается, самый простой путь, без сетевых шагов;
          2. двухшаговая загрузка (initiate -> PUT) — для файлов, которые в data-URI раздувают
             тело запроса до неприличия (base64 добавляет ~33%).

        Кадры при этом ужимаются до max_side и переводятся в JPEG: у Seedream на вход всё равно
        идёт своя нормализация, а гонять по сети 2.5 МБ PNG 1536x1536 на каждый референс незачем.
        """
        import base64
        import io

        from PIL import Image

        path = Path(path)
        if not path.exists():
            raise SeedreamError(f"файл не найден: {path}")

        image = Image.open(path).convert("RGB")
        if max(image.size) > max_side:
            image.thumbnail((max_side, max_side), Image.LANCZOS)
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=quality)
        raw = buf.getvalue()

        # ~1.5 МБ — граница, после которой data-URI уже неудобен: тело запроса пухнет, а часть
        # прокси начинает резать длинные строки.
        if len(raw) <= 1_500_000:
            return "data:image/jpeg;base64," + base64.b64encode(raw).decode()
        return self._upload_two_step(raw, filename=path.stem + ".jpg")

    def _upload_two_step(self, raw: bytes, *, filename: str,
                         content_type: Optional[str] = None) -> str:
        """initiate -> PUT. initiate отдаёт пару (file_url, upload_url): по второму кладём байты,
        первый потом отдаём модели.

        content_type определяется по расширению, если не задан явно: через этот же путь грузятся
        не только картинки, но и драйвер-ролики для Animate, а неверный тип провайдер отвергает.
        """
        if content_type is None:
            content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

        req = urllib.request.Request(
            f"{FAL_UPLOAD_INITIATE_URL}?storage_type=fal-cdn-v3",
            data=json.dumps({"content_type": content_type, "file_name": filename}).encode(),
            headers={**self._auth, "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            raise SeedreamError(
                f"initiate не удался: HTTP {e.code} {e.read(300).decode('utf-8', 'replace')}"
            ) from None

        file_url, upload_url = data.get("file_url"), data.get("upload_url")
        if not file_url or not upload_url:
            raise SeedreamError(f"initiate вернул неожиданный ответ: {list(data)}")

        put = urllib.request.Request(
            upload_url, data=raw, headers={"Content-Type": content_type}, method="PUT")
        try:
            urllib.request.urlopen(put, timeout=self.timeout).read()
        except urllib.error.HTTPError as e:
            raise SeedreamError(
                f"загрузка байтов не удалась: HTTP {e.code} "
                f"{e.read(300).decode('utf-8', 'replace')}"
            ) from None

        return file_url

    def _post(self, model: str, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{FAL_RUN_BASE}/{model}",
            data=json.dumps(payload).encode(),
            headers={**self._auth, "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", "replace")
            try:
                detail = json.loads(raw)
                # 422 от fal — это список конкретных претензий к полям запроса; показываем их,
                # а не голый код, иначе отладка превращается в перебор.
                detail = detail.get("detail", detail)
                detail = _strip_echoed_input(detail)
            except Exception:
                detail = raw[:400]
            raise SeedreamError(f"HTTP {e.code}: {str(detail)[:400]}") from None

    @staticmethod
    def _download(url: str, output_path: Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=180) as r:
            output_path.write_bytes(r.read())
        return output_path

    # --- основное ---

    def edit(
        self,
        prompt: str,
        *,
        references: Sequence[Path | str],
        output_path: Path,
        image_size: str = "square_hd",
        num_images: int = 1,
        seed: Optional[int] = None,
    ) -> SeedreamResult:
        """Перегенерация по референсам: лицо и фигура берутся с них, остальное — из промпта.

        references — локальные файлы (будут загружены) или уже готовые URL.
        """
        urls: list[str] = []
        for ref in references:
            if isinstance(ref, str) and ref.startswith(("http://", "https://", "data:")):
                urls.append(ref)
            else:
                urls.append(self.to_reference(Path(ref)))

        if not urls:
            raise SeedreamError("нужен хотя бы один референс — на них и держится идентичность")

        payload: dict = {
            "prompt": prompt,
            "image_urls": urls,
            "image_size": image_size,
            "num_images": num_images,
        }
        if seed is not None:
            payload["seed"] = seed

        start = time.monotonic()
        data = self._post(self.edit_model, payload)
        elapsed = time.monotonic() - start

        images = data.get("images") or []
        if not images:
            raise SeedreamError(f"в ответе нет изображений: {list(data)}")

        path = self._download(images[0]["url"], output_path)
        return SeedreamResult(
            image_path=path, prompt=prompt, references=urls,
            elapsed_s=elapsed, seed=data.get("seed"),
        )

    def text_to_image(
        self,
        prompt: str,
        *,
        output_path: Path,
        image_size: str = "square_hd",
        seed: Optional[int] = None,
    ) -> SeedreamResult:
        payload: dict = {"prompt": prompt, "image_size": image_size}
        if seed is not None:
            payload["seed"] = seed

        start = time.monotonic()
        data = self._post(self.t2i_model, payload)
        elapsed = time.monotonic() - start

        images = data.get("images") or []
        if not images:
            raise SeedreamError(f"в ответе нет изображений: {list(data)}")

        path = self._download(images[0]["url"], output_path)
        return SeedreamResult(image_path=path, prompt=prompt, elapsed_s=elapsed,
                              seed=data.get("seed"))
