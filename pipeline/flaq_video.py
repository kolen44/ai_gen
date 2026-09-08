"""Видео из готового кадра через flaq.ai — image-to-video.

Отдельно от fal_video, потому что провайдеры отличаются не ценой: Kling через fal отклоняет
обнажённый кадр на входе (HTTP 422 content_policy_violation), flaq те же кадры принимает.
Замерено на выходе: nsfw 0.9993 по выборке кадров ролика, лицо к эталону 0.714.

Схема восстановлена по живому эндпоинту, с картиночной не совпадает:

    POST /api/v1/video/task
        model_name    <семейство>-image-to-video, суффикс обязателен: без него 404
        image_url     стартовый кадр, только http(s)
        prompt        описание движения, до 5000 символов
        duration, resolution ("480p"|"720p"|"1080p"), aspect_ratio, seed
      -> {"data": {"task_id"}}
    GET /api/v1/video/{task_id}
      -> {"data": {"task_status", "task_result": {"credit", "videos": [{"url"}]}}}

Имена моделей проверены запросом: существующая жалуется на недостающие поля, несуществующая
отвечает кодом 1203. Список — в VIDEO_MODELS ниже.

Цена за СЕКУНДУ ролика, не за задачу: у Wan 3.0 это $0.045/$0.09/$0.18 за 480p/720p/1080p.

    python pipeline/flaq_video.py --image <кадр> --output-dir <папка> \
        --model wan-3.0-image-to-video --resolution 720p --duration 5
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Запускается и как модуль (`python -m`), и напрямую файлом — во втором случае корень проекта
# в sys.path не попадает сам.
if __package__ in (None, ""):
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.flaq_client import BROWSER_UA, Flaq, FlaqError, _load_key

TASK_URL = "https://api.flaq.ai/api/v1/video/task"
POLL_URL = "https://api.flaq.ai/api/v1/video/{task_id}"

DEFAULT_MODEL = "wan-3.0-image-to-video"

# Цена за секунду готового ролика. Держим здесь, а не в документе: стоимость прогона считается
# кодом и попадает в отчёт, а число, которое живёт только в тексте, устаревает молча.
PRICE_PER_SECOND = {"480p": 0.045, "720p": 0.09, "1080p": 0.18}

VIDEO_MODELS = (
    "wan-3.0-image-to-video",
    "wan-2.6-image-to-video",
    "seedance-v2.5-image-to-video",
    "seedance-v2.0-mini-image-to-video",
    "flux-3.0-image-to-video",
    "minimax-h3-image-to-video",
    "veo3.1-image-to-video",
)


@dataclass
class FlaqVideoResult:
    path: Path
    url: str
    task_id: str
    model: str
    elapsed_s: float
    credit: Optional[float]
    resolution: str
    duration: int

    @property
    def estimated_cost_usd(self) -> Optional[float]:
        rate = PRICE_PER_SECOND.get(self.resolution)
        return round(rate * self.duration, 4) if rate else None


class FlaqVideo:
    def __init__(self, *, api_key: Optional[str] = None, model: str = DEFAULT_MODEL,
                 timeout: int = 1800, poll_interval: float = 6.0):
        self.api_key = api_key or _load_key()
        self.model = model
        self.timeout = timeout
        self.poll_interval = poll_interval

    @property
    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": BROWSER_UA,
                "Accept": "application/json"}

    def _post(self, body: dict) -> str:
        req = urllib.request.Request(TASK_URL, data=json.dumps(body).encode(),
                                     headers=self._headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                payload = json.loads(response.read().decode())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            # Отдельным сообщением, потому что причина частая и лечится не кодом: у flaq
            # поминутная оплата, и дорогая модель может не влезть в остаток.
            if "1101" in detail or "Insufficient balance" in detail:
                raise FlaqError(f"на счёте flaq не хватает средств под {body['model_name']}: "
                                f"{detail}") from None
            raise FlaqError(f"HTTP {e.code} при создании задачи: {detail}") from None

        data = payload.get("data") or {}
        task_id = data.get("task_id")
        if not task_id:
            raise FlaqError(f"ответ без task_id: {payload}")
        return task_id

    def _wait(self, task_id: str) -> dict:
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            req = urllib.request.Request(POLL_URL.format(task_id=task_id),
                                         headers=self._headers)
            try:
                with urllib.request.urlopen(req, timeout=60) as response:
                    payload = json.loads(response.read().decode())
            except urllib.error.HTTPError as e:
                raise FlaqError(f"HTTP {e.code} при опросе задачи: "
                                f"{e.read().decode(errors='replace')[:200]}") from None

            data = payload.get("data") or {}
            status = (data.get("task_status") or "").lower()
            if status in ("succeed", "success", "succeeded"):
                return data
            if status in ("failed", "fail", "error"):
                raise FlaqError(f"задача провалилась: {data.get('task_status_msg') or data}")
            time.sleep(self.poll_interval)
        raise FlaqError(f"задача {task_id} не завершилась за {self.timeout} с")

    @staticmethod
    def _download(url: str, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": BROWSER_UA})
        with urllib.request.urlopen(req, timeout=900) as response:
            output_path.write_bytes(response.read())
        return output_path

    def image_to_video(
        self,
        image_path: Path,
        output_path: Path,
        *,
        prompt: str,
        duration: int = 5,
        resolution: str = "720p",
        aspect_ratio: str = "9:16",
        seed: Optional[int] = None,
    ) -> FlaqVideoResult:
        """Оживляет ГОТОВЫЙ кадр: картинка не перегенерируется и денег не стоит.

        Кадр сначала выкладывается ссылкой — data-URI их API не принимает, как и у картиночного
        эндпоинта. Своего хранилища у flaq нет, поэтому переиспользуется уже написанная загрузка
        в хранилище fal: там лежит только файл, генерация идёт у flaq, и контентная политика fal
        к нему не применяется.
        """
        started = time.monotonic()
        image_url = Flaq(api_key=self.api_key)._upload(Path(image_path))

        body = {
            "model_name": self.model,
            "image_url": image_url,
            "prompt": prompt,
            "duration": duration,
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
        }
        if seed is not None:
            body["seed"] = seed

        task_id = self._post(body)
        data = self._wait(task_id)

        result = data.get("task_result") or {}
        videos = result.get("videos") or []
        if not videos or not videos[0].get("url"):
            raise FlaqError(f"задача завершилась без файла: {data}")

        path = self._download(videos[0]["url"], Path(output_path))
        return FlaqVideoResult(
            path=path, url=videos[0]["url"], task_id=task_id, model=self.model,
            elapsed_s=round(time.monotonic() - started, 1),
            credit=result.get("credit"), resolution=resolution, duration=duration,
        )


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", type=Path, required=True, help="готовый кадр — старт ролика")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=list(VIDEO_MODELS))
    parser.add_argument("--prompt", default=None,
                        help="движение; по умолчанию берётся VIDEO_PROMPT из админки")
    parser.add_argument("--duration", type=int, default=5)
    parser.add_argument("--resolution", default="720p", choices=list(PRICE_PER_SECOND))
    parser.add_argument("--aspect-ratio", default="9:16")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    prompt = args.prompt
    if prompt is None:
        from pipeline import prompt_config

        prompt = prompt_config._get(prompt_config._read_env_file(), "VIDEO_PROMPT") or ""
    if not prompt:
        raise SystemExit("промпт движения пуст: заполните «Движение в видео» в админке")

    out = Path(args.output_dir) / f"video_{args.model.split('-image')[0]}.mp4"
    print(f"модель {args.model} · {args.resolution} · {args.duration} с")
    print(f"кадр: {args.image}")
    print(f"движение: {prompt[:150]}…")

    client = FlaqVideo(model=args.model)
    result = client.image_to_video(
        Path(args.image), out, prompt=prompt, duration=args.duration,
        resolution=args.resolution, aspect_ratio=args.aspect_ratio, seed=args.seed)

    print(f"\nготово за {result.elapsed_s:.0f} с -> {result.path}")
    if result.credit is not None:
        print(f"списано кредитов: {result.credit}")
    if result.estimated_cost_usd is not None:
        print(f"по прайсу: ${result.estimated_cost_usd}")


if __name__ == "__main__":
    _cli()
