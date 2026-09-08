"""Замер связки «чекпойнт + режим фиксации лица + scale» на живом поде.

Зачем отдельный скрипт, а не кнопка в админке: подобрать рабочие параметры нужно один раз, но
перебором в несколько десятков кадров, и смотреть на это надо таблицей, а не по одному кадру в
браузере.

Метрика — cosine ArcFace к эталону персонажа, ту же считает сам воркер. Шкала откалибрована:
портреты заведомо разных людей на этой модели дают до 0.387 (scripts/calibrate_similarity.py),
поэтому 0.5 — измеренная граница «тот же человек», а не назначенная.

ВАЖНО про интерпретацию. Низкая похожесть не всегда означает плохую фиксацию лица: ArcFace резко
проседает на профиле, сильном контровом свете и мелком лице в кадре. Поэтому промпты сцен здесь
задаются явно и держат лицо крупно и фронтально — иначе замеряется не адаптер, а удачность позы.
Разброс по сценам меряется отдельно, уже подобранными параметрами.

    python scripts/pod_benchmark.py --pod <id> --character test
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Сцены для замера идентичности: лицо крупно, фронтально, ровный свет. Это НЕ продакшн-сцены —
# их задача исключить влияние ракурса, чтобы в числе осталась только работа адаптера.
IDENTITY_SCENES = [
    "portrait photo, looking directly at the camera, soft even daylight, plain background, "
    "head and shoulders, sharp focus on the face",
    "casual photo standing indoors, facing the camera, natural window light, upper body in frame, "
    "relaxed expression",
    "photo sitting at a cafe table, facing the camera, warm indoor light, head and shoulders visible",
]


def generate(base: str, key: str, body: dict, timeout: int = 900) -> dict:
    response = requests.post(f"{base}/generate/sync", json=body,
                             headers={"X-Api-Key": key}, timeout=timeout)
    if response.status_code != 200:
        return {"error": response.text[:200]}
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pod", required=True)
    parser.add_argument("--character", default="test")
    parser.add_argument("--model", default=None, help="ключ чекпойнта; по умолчанию — из пода")
    parser.add_argument("--modes", default="base,portrait,plusv2")
    parser.add_argument("--scales", default="0.7,0.85,0.95")
    parser.add_argument("--scenes", type=int, default=2, help="сколько сцен на комбинацию")
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--refs", default=None,
                        help="перебор числа ракурсов эталона для portrait, например 1,2,3,5")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    from admin import runpod_pods as rp
    from pipeline import prompt_config

    record = rp.get_record(args.pod)
    model = args.model or (record.photo_model if record else "lustify")
    env = prompt_config.read_env_file()
    key = prompt_config._get(env, "WORKER_API_KEY") or args.pod
    # Через SSH-туннель, а не через HTTP-прокси RunPod: у прокси свой таймаут, короче
    # создания персонажа и генерации видео, и при его срабатывании приходит HTML-страница
    # Cloudflare вместо ответа воркера — выглядит как сломанный JSON, хотя на поде всё
    # досчиталось. Найдено на реальном прогоне.
    base = rp.worker_base(args.pod, long_running=True)

    health = rp.worker_health(args.pod)
    if not health:
        print(f"воркер на {args.pod} не отвечает")
        raise SystemExit(1)
    print(f"под {args.pod}: {health.get('gpu')}, чекпойнт {model}\n")

    ref_counts = [int(x) for x in args.refs.split(",")] if args.refs else [None]

    rows = []
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
      for refs in ref_counts:
        for scale in [float(x) for x in args.scales.split(",") if x.strip()]:
            sims, times, errors = [], [], []
            for i, scene in enumerate(IDENTITY_SCENES[: args.scenes]):
                body = {
                    "character": args.character,
                    "prompt": scene,
                    "model_name": model,
                    "identity_mode": mode,
                    "ip_adapter_scale": scale,
                    "num_inference_steps": args.steps,
                    # Один и тот же seed на всю комбинацию: сравниваем параметры, а не удачу
                    # конкретного шума. Смещение по сцене — чтобы кадры не были копией друг друга.
                    "seed": args.seed + i * 1000,
                    "width": 1024, "height": 1024,
                    # Без hires и детейлера: они улучшают лицо и замазали бы разницу между
                    # режимами, которую мы как раз и меряем.
                    "hires": False, "face_detailer": False,
                }
                if refs is not None:
                    body["identity_reference_count"] = refs
                started = time.monotonic()
                result = generate(base, key, body)
                if "error" in result:
                    errors.append(result["error"])
                    print(f"  {mode:9} scale={scale:<5} сцена {i + 1}: ОШИБКА {result['error'][:90]}")
                    continue
                sim = result.get("similarity")
                if sim is not None:
                    sims.append(sim)
                times.append(time.monotonic() - started)
                print(f"  {mode:9} scale={scale:<5} сцена {i + 1}: "
                      f"лицо {sim:.3f} за {times[-1]:.1f} с" if sim is not None else
                      f"  {mode:9} scale={scale:<5} сцена {i + 1}: лицо не найдено")

            rows.append({
                "mode": mode, "scale": scale, "refs": refs,
                "similarity_avg": round(statistics.mean(sims), 3) if sims else None,
                "similarity_min": round(min(sims), 3) if sims else None,
                "seconds_avg": round(statistics.mean(times), 1) if times else None,
                "errors": errors,
            })

    print("\n" + "=" * 62)
    print(f"{'режим':10} {'scale':>6} {'ракурсов':>9} {'лицо ср.':>9} {'лицо мин.':>10} {'сек':>6}")
    print("-" * 62)
    for row in sorted(rows, key=lambda r: -(r["similarity_avg"] or 0)):
        avg = f"{row['similarity_avg']:.3f}" if row["similarity_avg"] is not None else "—"
        low = f"{row['similarity_min']:.3f}" if row["similarity_min"] is not None else "—"
        sec = f"{row['seconds_avg']:.0f}" if row["seconds_avg"] is not None else "—"
        refs = str(row.get("refs") or "все")
        print(f"{row['mode']:10} {row['scale']:>6} {refs:>9} {avg:>9} {low:>10} {sec:>6}")

    out = Path(args.out) if args.out else ROOT / "benchmarking" / f"pod_{args.pod}_{model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"pod": args.pod, "model": model, "rows": rows},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nсохранено: {out}")


if __name__ == "__main__":
    main()
