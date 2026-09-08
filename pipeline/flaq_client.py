"""Seedream через flaq.ai — асинхронный API с опросом задачи.

Интерфейс повторяет seedream_client (text_to_image, edit, elapsed_s), чтобы Engine в
run_pipeline мог подставить его вместо Seedream без других правок.

Схема восстановлена пробами по живому эндпоинту, документации на неё нет:

    POST /api/v1/image/task
        model_name, prompt, seed
        width, height  — ЧАСТИ СООТНОШЕНИЯ СТОРОН, не пиксели: 1/1, 9/16, 16/9
        resolution     — "1k" | "2k"
        image_url      — один референс, только http(s)
      -> {"data": {"task_id"}}
    GET /api/v1/image/{task_id}
      -> {"data": {"task_status", "task_result": {"credit", "images": [{"url"}]}}}

Три грабли:
1. Cloudflare отбивает запросы без браузерного User-Agent, и выглядит это как отказ авторизации.
2. image_url принимает только настоящий адрес: data-URI не влезает в колонку их базы. Поэтому
   локальный файл сначала кладётся в хранилище fal.
3. Референс ровно один, остальные игнорируются. Это ухудшает удержание лица.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

TASK_URL = "https://api.flaq.ai/api/v1/image/task"
POLL_URL = "https://api.flaq.ai/api/v1/image/{task_id}"
DEFAULT_MODEL = "seedream-v5.0-pro"

# Cloudflare перед API режет клиентов без браузерного User-Agent.
BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")

# Имена размеров у fal -> (ширина, высота, разрешение) здесь. Держим единый словарь имён,
# чтобы --image-size в оркестраторе значил одно и то же для обоих провайдеров.
SIZE_MAP: dict[str, tuple[int, int, str]] = {
    "square_hd": (1, 1, "2k"),
    "square": (1, 1, "1k"),
    "portrait_4_3": (3, 4, "2k"),
    "portrait_16_9": (9, 16, "2k"),
    "landscape_4_3": (4, 3, "2k"),
    "landscape_16_9": (16, 9, "2k"),
    "auto_2K": (1, 1, "2k"),
    "auto_4K": (1, 1, "2k"),
}


class FlaqError(RuntimeError):
    pass


@dataclass
class FlaqResult:
    image_path: Path
    prompt: str
    references: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0
    seed: Optional[int] = None
    credit: Optional[float] = None


def _load_key(env_path: Optional[Path] = None) -> str:
    key = os.getenv("FLAQ_API_KEY")
    if key:
        return key.strip()
    env_path = env_path or Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = re.match(r"^FLAQ_API_KEY=(.+)$", line.strip())
            if m:
                return m.group(1).strip()
    raise FlaqError("FLAQ_API_KEY не найден ни в окружении, ни в .env")


class Flaq:
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        image_size: str = "square_hd",
        timeout: int = 300,
        poll_interval: float = 4.0,
    ):
        self.api_key = api_key or _load_key()
        self.model = model
        self.image_size = image_size
        self.timeout = timeout
        self.poll_interval = poll_interval

    # --- вспомогательное ---

    @property
    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": BROWSER_UA,
                "Accept": "application/json"}

    @staticmethod
    def _aspect(image_size: str) -> tuple[int, int, str]:
        if image_size not in SIZE_MAP:
            raise FlaqError(
                f"неизвестный размер {image_size!r}; допустимы: {', '.join(SIZE_MAP)}")
        return SIZE_MAP[image_size]

    def _upload(self, path: Path) -> str:
        """Кладёт локальный кадр в хранилище fal и возвращает ссылку.

        Своего аплоада у flaq нет, а data-URI их API не принимает. Загрузка в fal уже
        написана и протестирована, поэтому переиспользуется как есть, а не пишется заново.
        """
        from PIL import Image

        from pipeline.seedream_client import Seedream, SeedreamError

        import io

        im = Image.open(path).convert("RGB")
        im.thumbnail((2048, 2048), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=92)
        try:
            return Seedream()._upload_two_step(
                buf.getvalue(), filename=f"{Path(path).stem}.jpg", content_type="image/jpeg")
        except SeedreamError as e:
            raise FlaqError(f"не удалось выложить референс {Path(path).name}: {e}") from None

    def _post(self, body: dict) -> str:
        req = urllib.request.Request(TASK_URL, data=json.dumps(body).encode(),
                                     headers=self._headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            raise FlaqError(
                f"HTTP {e.code}: {e.read(400).decode('utf-8', 'replace')}") from None

        # Свои ошибки flaq отдаёт с HTTP 200 и ненулевым code — без этой проверки сбой
        # выглядел бы как успех, а падало бы позже и не в том месте.
        if data.get("code") not in (0, None):
            raise FlaqError(f"code={data.get('code')}: "
                            f"{str(data.get('message') or data.get('msg'))[:300]}")
        task = (data.get("data") or {}).get("task_id")
        if not task:
            raise FlaqError(f"в ответе нет task_id: {str(data)[:300]}")
        return task

    def _wait(self, task_id: str) -> dict:
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            req = urllib.request.Request(POLL_URL.format(task_id=task_id),
                                         headers=self._headers)
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = json.loads(r.read().decode())
            except urllib.error.HTTPError as e:
                raise FlaqError(f"опрос задачи: HTTP {e.code}") from None

            payload = data.get("data") or {}
            status = payload.get("task_status")
            if status == "succeed":
                return payload
            if status == "failed":
                raise FlaqError(f"задача провалена: {payload.get('task_status_msg')}")
            time.sleep(self.poll_interval)
        raise FlaqError(f"задача {task_id} не завершилась за {self.timeout} с")

    @staticmethod
    def _download(url: str, output_path: Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": BROWSER_UA})
        with urllib.request.urlopen(req, timeout=180) as r:
            output_path.write_bytes(r.read())
        return output_path

    def _run(self, body: dict, output_path: Path, prompt: str,
             references: list[str], seed: Optional[int]) -> FlaqResult:
        started = time.monotonic()
        payload = self._wait(self._post(body))
        result = payload.get("task_result") or {}
        images = result.get("images") or []
        if not images or not images[0].get("url"):
            raise FlaqError(f"в ответе нет изображений: {str(result)[:200]}")
        path = self._download(images[0]["url"], output_path)
        return FlaqResult(image_path=path, prompt=prompt, references=references,
                          elapsed_s=time.monotonic() - started, seed=seed,
                          credit=result.get("credit"))

    # --- публичное ---

    def text_to_image(self, prompt: str, output_path: Path, *,
                      image_size: Optional[str] = None,
                      seed: Optional[int] = None) -> FlaqResult:
        w, h, res = self._aspect(image_size or self.image_size)
        body = {"model_name": self.model, "prompt": prompt,
                "width": w, "height": h, "resolution": res}
        if seed is not None:
            body["seed"] = int(seed)
        return self._run(body, output_path, prompt, [], seed)

    def edit(self, prompt: str, references: list[Path], output_path: Path, *,
             image_size: Optional[str] = None,
             seed: Optional[int] = None) -> FlaqResult:
        if not references:
            raise FlaqError("нужен хотя бы один референс — на них и держится идентичность")

        # Референс здесь ровно один, в отличие от Seedream 4.5 на fal. Берём первый: в
        # пайплайне референсы отсортированы по метрике, то есть первый — лучший.
        url = self._upload(Path(references[0]))
        w, h, res = self._aspect(image_size or self.image_size)
        body = {"model_name": self.model, "prompt": prompt, "image_url": url,
                "width": w, "height": h, "resolution": res}
        if seed is not None:
            body["seed"] = int(seed)
        return self._run(body, output_path, prompt, [url], seed)


if __name__ == "__main__":
    import argparse
    import sys

    # Запуск файлом, а не как модуль пакета: корень проекта иначе не виден и ленивый
    # импорт pipeline.seedream_client внутри _upload падает на ModuleNotFoundError.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    ap = argparse.ArgumentParser(description="разовая генерация через flaq.ai")
    ap.add_argument("prompt")
    ap.add_argument("--out", type=Path, default=Path("flaq_out.png"))
    ap.add_argument("--ref", type=Path, default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--image-size", default="square_hd", choices=list(SIZE_MAP))
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()

    client = Flaq(model=a.model, image_size=a.image_size)
    r = (client.edit(a.prompt, [a.ref], a.out, seed=a.seed) if a.ref
         else client.text_to_image(a.prompt, a.out, seed=a.seed))
    print(f"{r.elapsed_s:.0f}s  ${r.credit}  -> {r.image_path}")
