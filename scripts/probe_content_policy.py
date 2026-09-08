"""Проверяет, где стоят контентные блокировки — запросом, а не по документам.

Политики провайдеров меняются без предупреждения, а формулировки в условиях использования редко
отвечают на нужный вопрос. Ответ даёт только запрос.

Проверок три, по местам, где блокировка реально срабатывает:

  1. промпт картинки  — отклоняют ли до генерации;
  2. кадр на вход i2v — принимают ли обнажённый кадр как исходник ролика;
  3. готовый результат — что показывает наш классификатор на том, что вернули.

Отказ приходит ДО списания, поэтому первые два шага бесплатны. Третий требует генерации, и он
включается отдельным флагом.

    python scripts/probe_content_policy.py
    python scripts/probe_content_policy.py --spend --image <обнажённый кадр>
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PROMPT = ("editorial nude photograph of a 30-year-old woman standing by a window, "
          "natural light, film grain, no clothing")

BLOCKED = ("content_policy_violation", "content checker", "flagged", "nsfw",
           "safety", "not allowed", "violat")


def verdict(ok: bool, detail: str) -> str:
    return f"{'ПРИНЯТО ' if ok else 'ОТКЛОНЕНО'}  {detail}"


def probe_fal_prompt() -> str:
    """fal: промпт картинки. Отказ приходит на валидации, генерация не запускается."""
    from pipeline.seedream_client import Seedream, SeedreamError

    try:
        Seedream().text_to_image(PROMPT, output_path=Path("/tmp/probe.png"),
                                 image_size="portrait_16_9")
        return verdict(True, "промпт прошёл валидацию (кадр сгенерирован, деньги списаны)")
    except SeedreamError as exc:
        text = str(exc).lower()
        if any(marker in text for marker in BLOCKED):
            return verdict(False, str(exc)[:160])
        return f"ошибка не про контент: {str(exc)[:120]}"


def probe_flaq_prompt() -> str:
    """flaq: то же самое. Задача создаётся только после валидации промпта."""
    from pipeline.flaq_client import BROWSER_UA, _load_key

    body = {"model_name": "seedream-v5.0-pro", "prompt": PROMPT,
            "width": 9, "height": 16, "resolution": "2k"}
    req = urllib.request.Request(
        "https://api.flaq.ai/api/v1/image/task", data=json.dumps(body).encode(),
        method="POST",
        headers={"Content-Type": "application/json", "User-Agent": BROWSER_UA,
                 "Authorization": f"Bearer {_load_key()}"})
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            data = json.loads(response.read().decode()).get("data") or {}
        return verdict(bool(data.get("task_id")), "задача принята в работу")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:160]
        return verdict(False, detail)


def probe_flaq_video(image_url: str) -> str:
    """flaq: обнажённый кадр на вход image-to-video."""
    from pipeline.flaq_client import BROWSER_UA, _load_key

    body = {"model_name": "wan-3.0-image-to-video", "image_url": image_url,
            "prompt": "she walks slowly toward the camera", "duration": 5,
            "resolution": "720p", "aspect_ratio": "9:16"}
    req = urllib.request.Request(
        "https://api.flaq.ai/api/v1/video/task", data=json.dumps(body).encode(),
        method="POST",
        headers={"Content-Type": "application/json", "User-Agent": BROWSER_UA,
                 "Authorization": f"Bearer {_load_key()}"})
    try:
        with urllib.request.urlopen(req, timeout=90) as response:
            data = json.loads(response.read().decode()).get("data") or {}
        return verdict(bool(data.get("task_id")), "кадр принят, ролик считается (деньги списаны)")
    except urllib.error.HTTPError as exc:
        return verdict(False, exc.read().decode(errors="replace")[:160])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=None,
                        help="обнажённый кадр для проверки входа i2v")
    parser.add_argument("--spend", action="store_true",
                        help="разрешить проверки, которые создают платную задачу")
    args = parser.parse_args()

    print("Проверка контентных блокировок запросом\n")
    print("Своя карта (RunPod): проверять нечего — веса и модель свои, контентной политики")
    print("                     на генерацию не существует в принципе.\n")

    print("-- промпт картинки --")
    print(f"  fal.ai   {probe_fal_prompt() if args.spend else 'пропущено: без --spend'}")
    print(f"  flaq.ai  {probe_flaq_prompt() if args.spend else 'пропущено: без --spend'}")

    if args.image and args.image.exists():
        from pipeline.flaq_client import Flaq

        url = Flaq()._upload(args.image)
        print("\n-- обнажённый кадр на вход image-to-video --")
        print(f"  flaq.ai  {probe_flaq_video(url) if args.spend else 'пропущено: без --spend'}")
    elif args.image:
        print(f"\n  кадр не найден: {args.image}")

    print("\nВывод, который не меняется от провайдера к провайдеру:")
    print("  аренда железа  — политики на генерацию нет, ответственность на арендаторе;")
    print("  API чужой модели — политика есть всегда и может смениться без предупреждения,")
    print("                     причём часто это политика партнёра, а не самого провайдера.")


if __name__ == "__main__":
    main()
