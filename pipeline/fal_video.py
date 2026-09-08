"""Видео из кадра через fal.ai: Kling, Wan, Seedance, Hailuo.

Внимание: контент-чекер fal отклоняет обнажённый кадр на входе (HTTP 422
content_policy_violation). Для взрослого контента используется pipeline/flaq_video.py.

Сверх голого вызова API модуль меряет identity покадрово против исходника и эталона, прогоняет
safety по выборке кадров и пишет строку в benchmark-лог.

Разные модели ждут разные имена поля для картинки (image_url против start_image_url) — учтено
в MODELS, проверено опросом эндпоинтов.

    python pipeline/fal_video.py --image <кадр> --reference <эталон> \
        --output-dir ./output/video_fal --model kling3 --prompt "she walks toward the camera"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.seedream_client import (  # noqa: E402
    Seedream, SeedreamError, _strip_echoed_input)
from worker.benchmark import append_row  # noqa: E402

FAL_RUN_BASE = "https://fal.run"
# Очередь для долгих задач: Animate считается минутами, синхронный запрос
# отваливается по таймауту сокета уже после начала генерации.
FAL_QUEUE_BASE = "https://queue.fal.run"

# image_field различается между семействами моделей — проверено опросом живых эндпоинтов:
# у Kling v3 поле называется start_image_url, у остальных image_url.
MODELS = {
    "kling3": {
        "path": "fal-ai/kling-video/v3/pro/image-to-video",
        "image_field": "start_image_url",
        "supports_duration": True,
    },
    "kling25": {
        "path": "fal-ai/kling-video/v2.5-turbo/pro/image-to-video",
        "image_field": "image_url",
        "supports_duration": True,
    },
    "kling21": {
        "path": "fal-ai/kling-video/v2.1/pro/image-to-video",
        "image_field": "image_url",
        "supports_duration": True,
    },
    "wan22": {
        "path": "fal-ai/wan/v2.2-a14b/image-to-video",
        "image_field": "image_url",
        "supports_duration": False,
    },
    "seedance": {
        "path": "fal-ai/bytedance/seedance/v1/pro/image-to-video",
        "image_field": "image_url",
        "supports_duration": True,
    },
    "hailuo": {
        "path": "fal-ai/minimax/hailuo-02/pro/image-to-video",
        "image_field": "image_url",
        "supports_duration": False,
    },
    # Wan 2.2 Animate — перенос движения с драйвер-ролика на нашего персонажа.
    # Отличается от остальных тем, что требует ВТОРОЙ вход: video_url с движением.
    # Препроцессинг (скелет позы, кропы лица) провайдер делает сам — в отличие от self-hosted
    # версии, где это ложится на нас и упирается в память пода.
    #
    # move    — персонаж с кадра начинает двигаться как человек в ролике, фон берётся с кадра;
    # replace — персонаж подставляется в сам ролик вместо человека там, со светом и цветом сцены.
    "wan-animate-move": {
        "path": "fal-ai/wan/v2.2-14b/animate/move",
        "image_field": "image_url",
        "supports_duration": False,
        "needs_driver": True,
    },
    "wan-animate-replace": {
        "path": "fal-ai/wan/v2.2-14b/animate/replace",
        "image_field": "image_url",
        "supports_duration": False,
        "needs_driver": True,
    },
}


class FalVideoError(RuntimeError):
    pass


def _submit_and_wait(model_path: str, payload: dict, *, api_key: str,
                     timeout: int = 3600, poll_every: int = 15) -> dict:
    """Ставит задачу в очередь fal и ждёт результата, опрашивая статус.

    Почему не синхронный fal.run. Wan Animate считается долго: режим move занял 584 секунды,
    replace не уложился и в 900 — синхронный запрос отвалился по таймауту сокета уже после того,
    как модель начала работать, то есть время и деньги потрачены, а результат потерян.
    Очередь для таких задач и предназначена: соединение не держится, статус опрашивается, и
    падение сети не отменяет саму генерацию.
    """
    headers = {"Authorization": f"Key {api_key}", "Content-Type": "application/json"}

    def request(url: str, data: Optional[bytes] = None) -> dict:
        req = urllib.request.Request(url, data=data, headers=headers,
                                     method="POST" if data else "GET")
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", "replace")
            try:
                detail = _strip_echoed_input(json.loads(raw).get("detail", raw))
            except Exception:
                detail = raw[:400]
            raise FalVideoError(f"HTTP {e.code}: {str(detail)[:400]}") from None

    submitted = request(f"{FAL_QUEUE_BASE}/{model_path}", json.dumps(payload).encode())
    status_url = submitted.get("status_url")
    response_url = submitted.get("response_url")
    if not status_url or not response_url:
        raise FalVideoError(f"очередь не вернула ссылки статуса: {list(submitted)}")

    print(f"задача поставлена в очередь: {submitted.get('request_id', '')[:12]}…")

    deadline = time.monotonic() + timeout
    last_state = ""
    while time.monotonic() < deadline:
        time.sleep(poll_every)
        state = request(status_url)
        status = state.get("status", "")
        if status != last_state:
            print(f"  статус: {status.lower()}")
            last_state = status
        if status == "COMPLETED":
            return request(response_url)
        if status in ("FAILED", "CANCELLED"):
            raise FalVideoError(f"задача завершилась статусом {status}: "
                                f"{str(state.get('error') or state)[:300]}")

    raise FalVideoError(f"не дождались результата за {timeout} секунд")


def _upload_video(client: Seedream, path: Path) -> str:
    """Кладёт ролик в хранилище fal и возвращает ссылку.

    Отдельно от картинок: у Seedream-клиента загрузка сначала пробует data-URI, а ролик на
    десятки мегабайт в тело запроса не запихнуть — только двухшаговая загрузка.
    """
    raw = path.read_bytes()
    size_mb = len(raw) / 1e6
    print(f"загружаю драйвер {path.name} ({size_mb:.1f} MB)")
    return client._upload_two_step(raw, filename=path.name)


def generate(
    *,
    model_key: str,
    image_path: Path,
    prompt: str,
    output_path: Path,
    duration: int = 5,
    driver_path: Optional[Path] = None,
    timeout: int = 3600,
) -> tuple[Path, float, dict]:
    cfg = MODELS[model_key]
    client = Seedream()  # переиспользуем его загрузку файлов и ключ — API один и тот же

    image_ref = client.to_reference(image_path)

    payload: dict = {"prompt": prompt, cfg["image_field"]: image_ref}
    if cfg["supports_duration"]:
        payload["duration"] = str(duration)

    # Animate-режимы принимают второй вход — ролик, откуда берётся движение.
    if cfg.get("needs_driver"):
        if driver_path is None:
            raise FalVideoError(
                f"модель {model_key} переносит движение с ролика, поэтому нужен --driver"
            )
        if not Path(driver_path).exists():
            raise FalVideoError(f"драйвер-ролик не найден: {driver_path}")
        # Ролик грузим через хранилище fal: в data-URI он не помещается (десятки мегабайт),
        # а картинки клиент сам решает, как передать.
        payload["video_url"] = _upload_video(client, Path(driver_path))

    start = time.monotonic()
    data = _submit_and_wait(cfg["path"], payload, api_key=client.api_key, timeout=timeout)
    elapsed = time.monotonic() - start

    video = data.get("video") or {}
    url = video.get("url")
    if not url:
        raise FalVideoError(f"в ответе нет видео: {list(data)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=600) as r:
        output_path.write_bytes(r.read())

    return output_path, elapsed, data


def measure(video_path: Path, *, source: Path, reference: Optional[Path], stride: int = 10) -> dict:
    """Покадровая метрика и safety. Меряем против двух баз: исходного кадра (собственный дрейф
    видеомодели) и эталона персонажа (итоговая идентичность). Эти числа отвечают на разные
    вопросы, и смешивать их нельзя."""
    import imageio.v3 as iio
    from PIL import Image

    from safety.moderation_pipeline import SafetyPipeline
    from worker.identity import detect_primary_face, detect_primary_face_array

    src_face = detect_primary_face(source)
    ref_face = detect_primary_face(reference) if reference else None
    safety = SafetyPipeline()

    def cosine(a, b):
        a = np.asarray(a, np.float32)
        b = np.asarray(b, np.float32)
        return float(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b)))

    vs_src, vs_ref = [], []
    blocked = 0
    checked = 0
    for i, frame in enumerate(iio.imiter(video_path, plugin="pyav")):
        if i % stride:
            continue
        checked += 1
        face = detect_primary_face_array(np.asarray(frame))
        if face is not None:
            if src_face:
                vs_src.append(cosine(src_face.embedding, face.embedding))
            if ref_face:
                vs_ref.append(cosine(ref_face.embedding, face.embedding))
        if not safety.run_image(Image.fromarray(frame)).allowed:
            blocked += 1

    def stats(values):
        if not values:
            return None
        return {"min": min(values), "avg": sum(values) / len(values), "max": max(values),
                "n": len(values)}

    return {"vs_source": stats(vs_src), "vs_reference": stats(vs_ref),
            "blocked": blocked, "checked": checked}


def run(
    *,
    image: Path,
    reference: Optional[Path],
    output_dir: Path,
    model_key: str,
    prompt: str,
    duration: int,
    driver: Optional[Path] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"video_{model_key}.mp4"

    print(f"движок: {MODELS[model_key]['path']}")
    print(f"исходный кадр: {image.name}")
    if MODELS[model_key].get("needs_driver"):
        print(f"драйвер движения: {driver.name if driver else 'НЕ ЗАДАН'}")
    print(f"промпт: {prompt[:110]}...")

    path, elapsed, _ = generate(model_key=model_key, image_path=image, prompt=prompt,
                                output_path=out_path, duration=duration, driver_path=driver)
    size_mb = path.stat().st_size / 1e6
    print(f"\nготово за {elapsed:.1f}s -> {path.name} ({size_mb:.1f} MB)")

    m = measure(path, source=image, reference=reference)
    if m["vs_source"]:
        s = m["vs_source"]
        print(f"identity к исходному кадру: min={s['min']:.3f} avg={s['avg']:.3f} "
              f"max={s['max']:.3f} (n={s['n']})")
    if m["vs_reference"]:
        s = m["vs_reference"]
        print(f"identity к эталону:         min={s['min']:.3f} avg={s['avg']:.3f} "
              f"max={s['max']:.3f} (n={s['n']})")
    verdict = "пройдено" if m["blocked"] == 0 else "есть отклонённые кадры"
    print(f"safety: {verdict} ({m['blocked']} заблокировано из {m['checked']} проверенных)")

    notes = f"model={model_key} duration={duration}"
    if m["vs_source"]:
        notes += f" cos_src_avg={m['vs_source']['avg']:.3f}"
    if m["vs_reference"]:
        notes += f" cos_ref_avg={m['vs_reference']['avg']:.3f}"

    append_row(
        stage="video_fal", index=1, checkpoint=MODELS[model_key]["path"],
        elapsed_seconds=elapsed, seed=0, resolution="провайдер", steps=0,
        gpu_type="fal.ai (внешний провайдер)", hourly_rate_usd=0.0,
        safety_gate_result="passed" if m["blocked"] == 0 else "rejected",
        notes=notes,
    )


def _cli():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", type=Path, required=True, help="кадр персонажа — старт видео")
    parser.add_argument("--reference", type=Path, default=None, help="эталон для метрики")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", dest="model_key", type=str, default="kling3", choices=list(MODELS))
    parser.add_argument("--duration", type=int, default=5)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--driver", type=Path, default=None,
                        help="ролик-источник движения; обязателен для wan-animate-*")
    args = parser.parse_args()

    run(image=args.image, reference=args.reference, output_dir=args.output_dir,
        model_key=args.model_key, prompt=args.prompt, duration=args.duration,
        driver=args.driver)


if __name__ == "__main__":
    _cli()
