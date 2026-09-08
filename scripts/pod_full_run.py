"""Полный прогон на своём поде: персонаж → пак кадров → видео → отчёт.

Тот же путь, что админка проходит кнопками, но одной командой. Нужен, чтобы проверять пайплайн
целиком: половина проблем вылезает на стыках этапов.

Умолчания — замеренные, см. deliverables/RUNPOD_TUNING.md.

    python scripts/pod_full_run.py --pod <id> --frames 8 --videos 1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from worker.seed_manager import SeedPlan  # noqa: E402

# Промпт целиком собирается из полей админки, ничего своего скрипт не подмешивает. Кадрирование
# и поза уже написаны в самих образах («framed from the knees up», «mid-step»), и приписка от
# скрипта их перебивала: сцена «идёт мимо витрины» превращалась в макрошот лица.
#
#   образ / сцена    — RESTYLE_* / SCENE_*, задаёт картинку целиком
#   мимика           — EXPRESSION_*, раздаётся кадрам по кругу
#   внешность        — CHARACTER_*, то, что FaceID не держит: волосы, глаза, приметы
#   стиль съёмки     — STYLE_SUFFIX
#   фиксация лица    — IDENTITY_LOCK
#   движение в видео — VIDEO_PROMPT

# Сцены, где человек сидит или снимает селфи, походкой не оживить. Скрипт их не переписывает —
# текст движения задаёт VIDEO_PROMPT, — но предупреждает в логе, если одно противоречит другому.
STATIC_SCENE_MARKERS = (
    "sitting", "seated", "sits", "selfie", "mirror", "lying", "lies", "leaning",
)
MOTION_MARKERS = ("walk", "step", "stride", "running", "runs")


def is_static_scene(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in STATIC_SCENE_MARKERS)


def build_prompt(subject: str, character: dict, env: dict, index: int,
                 *, with_lock: bool = True) -> str:
    """Промпт кадра: образ или сцена + мимика + внешность + стиль + фиксация лица.

    Первым идёт то, что определяет картинку. Пустые поля админки просто выпадают.

    with_lock=False убирает текстовую фиксацию лица — она нужна только там, где эталон подаётся
    картинкой. На своей карте первый проход идёт без эталона, и фраза «preserve the exact same
    person from the reference image» ссылается на несуществующее. Плюс она состоит из запретов
    («no face morphing», «no identity drift»), а CLIP отрицания не знает — в положительном
    промпте это те же понятия, внесённые в кадр. Место таких формулировок — в негативном.
    Заодно это 899 символов из 2729.
    """
    from pipeline import prompt_config

    parts = [
        subject,
        prompt_config.expression_for(index, env),
        describe_character(character),
        prompt_config.style_suffix(env),
        prompt_config.identity_lock(env) if with_lock else "",
    ]
    return ", ".join(part.strip().rstrip(",. ") for part in parts if part and part.strip())


def describe_character(character: dict) -> str:
    """Внешность, дописывается к каждой сцене.

    FaceID держит геометрию лица, но не цвет волос и не сложение. Без описания кадр выходит с
    другими волосами, и косинус этого не замечает: 0.655 и с описанием, и без.
    """
    return ("{age}-year-old {ethnicity} woman, {hair}, {eyes}, "
            "{distinguishing_features}, {build}").format(**character)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pod", required=True)
    parser.add_argument("--character", default="hero")
    # «Образы» дают пак фотографий, «Сцены» — ролики. Разделение не косметическое: это два
    # разных списка в карточке персонажа с разным назначением, и раньше пак ошибочно собирался
    # из «Сцен», а «Образы» на своей карте не использовались вообще.
    parser.add_argument("--frames", type=int, default=8,
                        help="сколько ОБРАЗОВ (RESTYLE_*) взять в фотопак")
    parser.add_argument("--videos", type=int, default=1,
                        help="сколько роликов снять; на каждый — свой кадр-исходник")
    # Откуда брать кадр под ролик. Разница не в качестве, а в содержании: видеомодель анимирует
    # ровно то, что видит, и одетый исходник даёт одетый ролик, сколько ни описывай движение.
    #
    # Зум подрезает кадр вокруг лица. 1.8 — поясной план: лицу достаётся 214x296 пикселей ролика
    # против 84x115 на ростовом. Постобработка столько не добавит, пикселей просто нет.
    parser.add_argument("--zoom", type=float, default=1.25,
                        help="подрезка кадра под видео вокруг лица: 1.0 как есть, 1.8 поясной")
    # Полное качество видео: 40 шагов вместо 4. Замерено на одном исходнике — 61.5 минуты
    # против 13, но картинка достовернее. По умолчанию берём качество: скорость выбирается явно.
    parser.add_argument("--fast-video", action="store_true",
                        help="ускоряющая LoRA: 4 шага вместо 40, впятеро быстрее, грубее")
    parser.add_argument("--video-source", default="look", choices=["look", "scene"],
                        help="кадр под ролик из ОБРАЗОВ (RESTYLE_*) или из СЦЕН (SCENE_*)")
    # lustify, а не realvis-xl: проект про взрослый контент, и чекпойнт без ограничений здесь
    # рабочий по умолчанию. realvis-xl остаётся выбором для обычной съёмки.
    parser.add_argument("--photo-model", default="lustify")
    parser.add_argument("--identity-mode", default="portrait")
    # scale=0 — сцена рисуется БЕЗ FaceID. Это не «выключить идентичность», а двухэтапная схема:
    # личность вживляет детейлер вторым проходом. Замерено на поде: с FaceID на сцене лицо
    # занимало 16-20% кадра и промпт по композиции игнорировался (сцена «идёт мимо витрины»
    # давала макрошот), без него — 1.9-4.3%, то есть настоящий ростовой кадр.
    parser.add_argument("--scale", type=float, default=0.0)
    # Порог, ниже которого детейлер включается принудительно. Собственный порог детейлера смотрит
    # на размер лица, а у ростового кадра лицо формально «крупное» (0.19 при пороге 0.12) —
    # решать должна похожесть.
    parser.add_argument("--detail-below", type=float, default=0.62)
    # Вертикаль, а не квадрат: квадрат сам тянет композицию к портрету.
    # 1024x1536 — верх, который SDXL держит без потери связности; после hires выходит 1536x2304.
    # Выше модель начинает дублировать части тела, шагами это не лечится.
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1536)
    # 40 шагов вместо 28. На DPM++ 2M Karras прирост заметен до ~40 и дальше выходит на полку;
    # 28 — компромисс ради скорости, а он тут не нужен: кадр всё равно считается секунды.
    parser.add_argument("--steps", type=int, default=40)
    # Сколько вариантов считать на образ, оставляя лучший по похожести лица. Разброс между
    # попытками на одном промпте (0.52-0.68) больше, чем прибавка от лишних десяти шагов.
    # Отклонённые ложатся в rejected/ — видно, что именно отбор посчитал хуже.
    parser.add_argument("--candidates", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--video-model", default=None)
    parser.add_argument("--video-frames", type=int, default=None)
    # Постобработка ролика: detailer тянет лицо к эталону (0.709 против 0.530), gfpgan втрое
    # быстрее, но про личность не знает и даёт +0.03. both — оба по очереди, дороже, но иначе
    # не получить сразу и похожесть, и резкость.
    parser.add_argument("--enhance", default="both",
                        choices=["both", "detailer", "gfpgan", "none"])
    parser.add_argument("--upscale", type=int, default=2, choices=[1, 2],
                        help="увеличение кадра целиком при GFPGAN")
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--base-seed", type=int, default=20260907)
    parser.add_argument("--hourly-rate", type=float, default=None,
                        help="ставка $/час; по умолчанию берётся из живых данных RunPod")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    from admin import runpod_pods as rp
    from pipeline import prompt_config

    env = prompt_config.read_env_file()
    key = prompt_config._get(env, "WORKER_API_KEY") or args.pod
    # Через SSH-туннель, а не через HTTP-прокси RunPod: у прокси свой таймаут, короче
    # создания персонажа и генерации видео, и при его срабатывании приходит HTML-страница
    # Cloudflare вместо ответа воркера — выглядит как сломанный JSON, хотя на поде всё
    # досчиталось. Найдено на реальном прогоне.
    base = rp.worker_base(args.pod, long_running=True)
    headers = {"X-Api-Key": key}

    health = rp.worker_health(args.pod)
    if not health:
        print(f"воркер на {args.pod} не отвечает")
        raise SystemExit(1)
    print(f"под {args.pod}: {health.get('gpu')}\n")

    character = prompt_config.load_character(env)
    # Эталон: то же описание внешности плюс фиксация лица. Своего шаблона у скрипта нет —
    # эталонный кадр описывается теми же полями админки, что и все остальные.
    # Фиксация лица сюда не идёт: эталон — это САМ первый кадр, ссылаться ему не на что.
    reference_prompt = ", ".join(x for x in (
        "candid close-up photo, looking straight at the camera, relaxed natural expression",
        describe_character(character),
        prompt_config.style_suffix(env),
    ) if x)
    # Движение в ролике — поле «Движение в видео» админки.
    motion_prompt = prompt_config._get(env, "VIDEO_PROMPT") or ""
    description = describe_character(character)
    suffix = prompt_config.style_suffix(env)

    # Фотопак — из «Образов» (RESTYLE_*): это и есть картинки персонажа в разных образах.
    all_looks = prompt_config.load_restyle_prompts(env)
    looks = all_looks[: args.frames]
    # Источник кадров под ролики — по выбору: образы или сцены.
    if args.video_source == "look":
        video_scenes = all_looks[: max(0, args.videos)]
    else:
        video_scenes = prompt_config.load_scenes(env)[0][: max(0, args.videos)]

    if not looks:
        print("в карточке нет ОБРАЗОВ (RESTYLE_*) — фотопак собирать не из чего")
        raise SystemExit(1)

    # Каталог результатов берётся из OUTPUT_DIR в .env — там же, где его читает админка
    # (admin/app.py::runs_root). Раньше здесь было жёстко ROOT/"runs", и прогоны с пода
    # оседали внутри проекта, а не там, куда настроен вывод пайплайна.
    root = prompt_config.output_root(env)
    out_dir = Path(args.out) if args.out else root / f"podrun_{datetime.now():%m%d_%H%M}"
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)
    (out_dir / "rejected").mkdir(parents=True, exist_ok=True)
    # Ставка за час аренды — из живых данных RunPod, а не из константы: у одной и той же карты
    # цена отличается между регионами и типами облака, и посчитанная по константе стоимость
    # генерации была бы просто неправдой.
    seeds = SeedPlan(character_name=args.character, base_seed=args.base_seed)

    hourly_rate = args.hourly_rate
    if hourly_rate is None:
        hourly_rate = 0.0
        try:
            for pod in rp.live_pods():
                if pod.get("id") == args.pod:
                    hourly_rate = float(pod.get("costPerHr") or 0)
                    break
        except Exception as exc:  # noqa: BLE001
            print(f"  ставка за час не получена ({type(exc).__name__}), стоимость будет нулевой")
    print(f"ставка: ${hourly_rate:.3f}/час")

    report: dict = {"pod": args.pod, "video_source": args.video_source,
                    "photo_model": args.photo_model,
                    "identity_mode": args.identity_mode, "scale": args.scale,
                    "steps": args.steps, "width": args.width, "height": args.height,
                    "candidates": args.candidates, "base_seed": args.base_seed,
                    "fast_video": bool(args.fast_video), "enhance": args.enhance,
                    "hourly_rate_usd": hourly_rate, "started_at": datetime.now().isoformat(),
                    "frames": []}

    # --- персонаж ---
    existing = requests.get(f"{base}/characters", headers=headers, timeout=60)
    known = [c.get("name") for c in existing.json()] if existing.status_code == 200 else []
    if args.character not in known:
        print(f"создаю персонажа {args.character} (эталон + ракурсы)…")
        started = time.monotonic()
        created = requests.post(
            f"{base}/characters", headers=headers, timeout=3600,
            json={"name": args.character, "prompt": reference_prompt,
                  "seed": seeds.reference_seed(), "overwrite": True,
                  # Тот же чекпойнт, что у пака: иначе похожесть считается к чужому лицу.
                  "model_name": args.photo_model})
        if created.status_code not in (200, 201):
            print(f"эталон не создан: {created.text[:300]}")
            raise SystemExit(1)
        print(f"  готов за {time.monotonic() - started:.0f} с")
    else:
        print(f"персонаж {args.character} уже есть")

    # Эталон — часть результата, а не служебный файл: по нему судят, на кого вообще похожи
    # кадры, и без него метрика похожести повисает в воздухе.
    reference_remote = f"/workspace/characters/{args.character}.png"
    if rp.fetch_file(args.pod, reference_remote, out_dir / "reference.png"):
        report["reference"] = str(out_dir / "reference.png")

    # --- пак кадров: образы ---
    print(f"\nфотопак — образы ({len(looks)}):")
    for index, scene in enumerate(looks, start=1):
        prompt = build_prompt(scene, character, env, index,
                              with_lock=args.scale > 0)
        body = {
            "character": args.character, "prompt": prompt,
            "model_name": args.photo_model, "identity_mode": args.identity_mode,
            "ip_adapter_scale": args.scale, "identity_reference_count": 1,
            "num_inference_steps": args.steps,
            # seed проставляется на каждую попытку отдельно, ниже в цикле отбора.
            "width": args.width, "height": args.height,
            "hires": True, "face_detailer": True,
            "detail_below_similarity": args.detail_below,
        }
        started = time.monotonic()
        attempts = []
        for attempt in range(args.candidates):
            body["seed"] = seeds.photo_seed(index, attempt)
            response = requests.post(f"{base}/generate/sync", json=body,
                                     headers=headers, timeout=1800)
            if response.status_code != 200:
                print(f"  {index}: ОШИБКА {response.text[:150]}")
                continue
            attempts.append(response.json())

        if not attempts:
            report["frames"].append({"index": index, "error": "все попытки не удались"})
            continue

        # Лучший по похожести лица. Прочие метрики сюда не мешаем: доля кадра у вариантов одного
        # промпта примерно одинакова, а safety уже отработал внутри каждого.
        attempts.sort(key=lambda r: r.get("similarity") or 0, reverse=True)
        result = attempts[0]
        if len(attempts) > 1:
            others = ", ".join(f"{r.get('similarity') or 0:.3f}" for r in attempts[1:])
            print(f"  {index}: выбран {result.get('similarity') or 0:.3f} из [{others}]")
            for spare in attempts[1:]:
                rp.fetch_file(args.pod, spare["image_path"],
                              out_dir / "rejected" / f"frame_{index:02d}_{spare['id'][:6]}.png")
        mark = "принят" if result.get("accepted") else f"брак ({result.get('reject_reason')})"
        print(f"  {index}: лицо {result['similarity']:.3f} · {mark} · "
              f"{time.monotonic() - started:.0f} с")
        # Брак кладём отдельной папкой, а не выбрасываем: по отклонённым кадрам видно, ЧТО
        # именно не сошлось (ракурс, свет, чужое лицо), и без них причину не разобрать.
        subdir = "frames" if result.get("accepted") else "rejected"
        local = out_dir / subdir / f"frame_{index:02d}.png"
        rp.fetch_file(args.pod, result["image_path"], local)
        # gpu_ms — чистая работа модели, wall_s — кадр целиком с попытками и передачей файлов.
        # Стоимость считаем по первой: за скачивание аренда денег не берёт. По сумме попыток,
        # а не одной выбранной — оплачены все.
        gpu_ms = sum(int(a.get("duration_ms") or 0) for a in attempts)
        report["frames"].append({
            "index": index, "prompt": prompt, "similarity": result.get("similarity"),
            # Доля кадра, занятая лицом — вторая ось качества. Без неё похожесть можно «выиграть»
            # макрошотом, что и произошло на первом заходе.
            "face_area_ratio": result.get("face_area_ratio"),
            "detailer_applied": result.get("face_detailer_applied"),
            "accepted": result.get("accepted"), "seed": result.get("seed"),
            "attempts": len(attempts),
            "gpu_seconds": round(gpu_ms / 1000.0, 2),
            "wall_seconds": round(time.monotonic() - started, 2),
            "cost_usd": round(gpu_ms / 1000.0 / 3600.0 * hourly_rate, 6),
            "nsfw_score": result.get("nsfw_score"),
            "estimated_age": result.get("estimated_age"),
            "safety_allowed": result.get("safety_allowed"),
            "remote": result["image_path"], "local": str(local),
        })

    accepted = [f for f in report["frames"] if f.get("accepted")]
    sims = [f["similarity"] for f in report["frames"] if f.get("similarity") is not None]
    ratios = [f["face_area_ratio"] for f in report["frames"]
              if f.get("face_area_ratio") is not None]
    if sims:
        print(f"\nпринято {len(accepted)} из {len(report['frames'])}, "
              f"лицо в среднем {sum(sims) / len(sims):.3f} (мин {min(sims):.3f})"
              + (f", лицо занимает {sum(ratios) / len(ratios) * 100:.1f}% кадра" if ratios else ""))

    # --- видео: по ролику на сцену ---
    # Кадр-исходник рисуется отдельно от фотопака: видеомодель анимирует то, что видит, и с
    # портрета выходит поворот головы, а не действие. Нужен кадр с уже начатым движением,
    # снятый поясным планом.
    report["videos"] = []
    if not args.skip_video and video_scenes:
        if motion_prompt and any(m in motion_prompt.lower() for m in MOTION_MARKERS):
            static = [s for s in video_scenes if is_static_scene(s)]
            if static:
                print(f"  ВНИМАНИЕ: в «Движении в видео» описана ходьба, а {len(static)} из "
                      f"{len(video_scenes)} сцен статичные (сидит/селфи) — модель на таком "
                      f"противоречии ломается")

        for number, scene in enumerate(video_scenes, start=1):
            print(f"\nсцена {number}/{len(video_scenes)}: {scene[:70]}…")

            prompt = build_prompt(scene, character, env, number,
                                  with_lock=args.scale > 0)
            body = {
                "character": args.character, "prompt": prompt,
                "model_name": args.photo_model, "identity_mode": args.identity_mode,
                "ip_adapter_scale": args.scale, "identity_reference_count": 1,
                "num_inference_steps": args.steps,
                "seed": seeds.video_source_seed(number),
                "width": args.width, "height": args.height,
                "hires": True, "face_detailer": True,
                "detail_below_similarity": args.detail_below,
            }
            # Печатаем реально отправленный промпт: без этого непонятно, что победило в
            # результате — текст образа, хвост карточки или стилевая приписка.
            print(f"  промпт кадра ({len(prompt)} символов): {prompt[:220]}…")
            response = requests.post(f"{base}/generate/sync", json=body,
                                     headers=headers, timeout=1800)
            if response.status_code != 200:
                print(f"  кадр не вышел: {response.text[:150]}")
                continue

            source = response.json()
            ratio = source.get("face_area_ratio")
            print(f"  кадр: лицо {source['similarity']:.3f}"
                  + (f" · занимает {ratio * 100:.1f}% кадра" if ratio else ""))
            rp.fetch_file(args.pod, source["image_path"],
                          out_dir / f"video_{number:02d}_source.png")

            print(f"  промпт движения: {motion_prompt[:180]}…" if motion_prompt
                  else "  промпт движения пуст — поле «Движение в видео» в админке не заполнено")
            started = time.monotonic()
            video_body = {"image_path": source["image_path"], "prompt": motion_prompt,
                          "subject_zoom": args.zoom, "fast": bool(args.fast_video)}
            if args.video_model:
                video_body["model_name"] = args.video_model
            if args.video_frames:
                video_body["num_frames"] = args.video_frames
            # Таймаут вдвое больше обычного: A14B вдвое тяжелее 5B, и первый ролик на нём вместе
            # со скачиванием 126 GB весов идёт заметно дольше.
            response = requests.post(f"{base}/video/sync", headers=headers, timeout=9000,
                                     json=video_body)
            if response.status_code != 200:
                print(f"  ролик не вышел: {response.text[:200]}")
                continue

            result = response.json()
            sim = result.get("similarity")
            print(f"  ролик: {result['num_frames']} кадров за "
                  f"{(time.monotonic() - started) / 60:.1f} мин"
                  + (f" · лицо в движении {sim:.3f}" if sim is not None else ""))
            local = out_dir / f"video_{number:02d}.mp4"
            rp.fetch_file(args.pod, result["video_path"], local)
            # Стоимость ролика складывается из ТРЁХ статей, и по отдельности они выглядят
            # безобидно: сам кадр-исходник, генерация ролика и покадровая доводка лица. Доводка
            # на «both» сопоставима по времени с самой генерацией, поэтому считать её отдельно —
            # не педантизм: без этого стоимость ролика занижена вдвое.
            source_ms = int(source.get("duration_ms") or 0)
            video_ms = int(result.get("duration_ms") or 0)
            entry = {"scene": scene, "index": number, **result, "local": str(local),
                     "source_gpu_seconds": round(source_ms / 1000.0, 2),
                     "video_gpu_seconds": round(video_ms / 1000.0, 2),
                     "video_wall_seconds": round(time.monotonic() - started, 2),
                     "fast": bool(args.fast_video),
                     "cost_usd": round((source_ms + video_ms) / 1000.0 / 3600.0 * hourly_rate, 6)}

            if args.enhance != "none":
                print(f"  доводка лица ({args.enhance}):")
                started = time.monotonic()
                enhanced = requests.post(
                    f"{base}/video/enhance", headers=headers, timeout=7200,
                    json={"video_path": result["video_path"], "character": args.character,
                          "method": args.enhance, "model_name": args.photo_model,
                          "upscale": args.upscale})
                if enhanced.status_code == 200:
                    ed = enhanced.json()
                    esim = ed.get("similarity")
                    print(f"    {ed['frames_processed']}/{ed['frames_total']} кадров за "
                          f"{(time.monotonic() - started) / 60:.1f} мин"
                          + (f" · лицо {esim:.3f}" if esim is not None else ""))
                    local_enh = out_dir / f"video_{number:02d}_enhanced.mp4"
                    rp.fetch_file(args.pod, ed["video_path"], local_enh)
                    enh_ms = int(ed.get("elapsed_ms") or 0)
                    entry["enhanced"] = {**ed, "local": str(local_enh),
                                         "gpu_seconds": round(enh_ms / 1000.0, 2),
                                         "cost_usd": round(enh_ms / 1000.0 / 3600.0
                                                           * hourly_rate, 6)}
                    entry["cost_usd"] = round(entry["cost_usd"]
                                              + entry["enhanced"]["cost_usd"], 6)
                else:
                    print(f"    ОШИБКА {enhanced.text[:150]}")

            report["videos"].append(entry)

    _write_video_strip(out_dir)
    # Сводка: то, что попадёт в таблицу отчёта. Считается здесь, а не при чтении отчёта, чтобы
    # цифры в документе и в run.json не могли разойтись.
    photo_costs = [f["cost_usd"] for f in report["frames"] if f.get("cost_usd") is not None]
    photo_gpu = [f["gpu_seconds"] for f in report["frames"] if f.get("gpu_seconds")]
    video_costs = [v["cost_usd"] for v in report["videos"] if v.get("cost_usd") is not None]
    video_gpu = [v["video_gpu_seconds"] for v in report["videos"] if v.get("video_gpu_seconds")]
    report["totals"] = {
        "hourly_rate_usd": hourly_rate,
        "photo": {
            "count": len(photo_costs),
            "gpu_seconds_avg": round(sum(photo_gpu) / len(photo_gpu), 2) if photo_gpu else None,
            "cost_usd_avg": round(sum(photo_costs) / len(photo_costs), 6) if photo_costs else None,
            "cost_usd_total": round(sum(photo_costs), 6),
        },
        "video": {
            "count": len(video_costs),
            "gpu_seconds_avg": round(sum(video_gpu) / len(video_gpu), 2) if video_gpu else None,
            "cost_usd_avg": round(sum(video_costs) / len(video_costs), 6) if video_costs else None,
            "cost_usd_total": round(sum(video_costs), 6),
        },
        "run_cost_usd": round(sum(photo_costs) + sum(video_costs), 6),
    }

    _write_contact_sheet(out_dir, report)
    _write_report(out_dir, report, character)
    (out_dir / "run.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print(f"\nрезультаты: {out_dir}")


def _write_video_strip(out_dir: Path) -> None:
    """Раскадровка ролика: пять моментов подряд в натуральном разрешении.

    По ней видно движение и композицию целиком. Кропы лица — отдельно, они про резкость.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return

    # Раскадровка делается для КАЖДОГО ролика прогона. Роликов теперь столько, сколько сцен, и
    # одного файла с фиксированным именем больше не хватает.
    videos = sorted(out_dir.glob("video_*.mp4"))
    for path in videos:
        capture = cv2.VideoCapture(str(path))
        frames = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
        capture.release()
        if not frames:
            continue
        picks = [0, len(frames) // 4, len(frames) // 2, 3 * len(frames) // 4, len(frames) - 1]
        strip = np.concatenate([frames[i] for i in picks], axis=1)
        cv2.imencode(".png", strip)[1].tofile(str(out_dir / f"{path.stem}_strip.png"))


def _write_contact_sheet(out_dir: Path, report: dict) -> None:
    """Все принятые кадры одной картинкой.

    Смотреть пак по одному файлу неудобно, а расхождения личности между кадрами видно именно в
    сравнении рядом — на отдельных кадрах глаз их не ловит.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return

    images = []
    for frame in report.get("frames", []):
        path = frame.get("local")
        if not path or not Path(path).exists() or not frame.get("accepted"):
            continue
        image = cv2.imread(path)
        if image is not None:
            images.append(cv2.resize(image, (512, 512)))
    if not images:
        return
    cv2.imwrite(str(out_dir / "contact_sheet.png"), np.concatenate(images, axis=1))


def _write_report(out_dir: Path, report: dict, character: dict) -> None:
    """Отчёт по прогону: чем генерировали, что получилось, что забраковано.

    Пишется всегда, в том числе когда всё прошло гладко: через неделю по одним файлам уже не
    восстановить, на каких параметрах они сделаны.
    """
    sims = [f["similarity"] for f in report["frames"] if f.get("similarity") is not None]
    accepted = [f for f in report["frames"] if f.get("accepted")]
    lines = [
        f"# Прогон {out_dir.name}",
        "",
        f"- Персонаж: {character.get('name')}, {character.get('age')} лет, "
        f"{character.get('ethnicity')}",
        f"- Внешность: {character.get('hair')}, {character.get('eyes')}, "
        f"{character.get('distinguishing_features')}, {character.get('build')}",
        f"- Чекпойнт: `{report['photo_model']}`",
        f"- Фиксация лица: "
        + ("двухэтапная (сцена без FaceID, лицо детейлером)" if report["scale"] == 0
           else f"`{report['identity_mode']}`, scale {report['scale']}"),
        "",
        "Доля кадра, занятая лицом, — вторая ось качества наравне с похожестью. Без неё подбор",
        "параметров уходит в макрошоты: косинус ArcFace тем выше, чем крупнее лицо, и сцена",
        "перестаёт слушаться промпта.",
        f"- Под: `{report['pod']}`",
        "",
        "## Фотопак (образы)",
        "",
        "Собирается из списка «Образы» (`RESTYLE_*`) — по кадру на образ. Список «Сцены»",
        "(`SCENE_*`) на своей карте отвечает за видео, а не за фотографии.",
        "",
        f"Принято **{len(accepted)} из {len(report['frames'])}**."
        + (f" Похожесть лица: в среднем **{sum(sims) / len(sims):.3f}**, "
           f"минимум {min(sims):.3f}, максимум {max(sims):.3f}." if sims else ""),
        "",
        "Порог приёмки 0.5 не назначен, а измерен: портреты заведомо разных людей на этой же",
        "модели дают до 0.387 (`scripts/calibrate_similarity.py`).",
        "",
        "| # | лицо | доля кадра | детейлер | GPU, с | $ | статус |",
        "|---|---|---|---|---|---|---|",
    ]
    for frame in report["frames"]:
        similarity = frame.get("similarity")
        ratio = frame.get("face_area_ratio")
        cost = frame.get("cost_usd")
        gpu = frame.get("gpu_seconds")
        status = "принят" if frame.get("accepted") else f"брак — {frame.get('reject_reason', '?')}"
        cells = [
            str(frame["index"]),
            f"{similarity:.3f}" if similarity is not None else "—",
            f"{ratio * 100:.1f}%" if ratio is not None else "—",
            "да" if frame.get("detailer_applied") else "нет",
            f"{gpu:.1f}" if gpu else "—",
            f"{cost:.4f}" if cost is not None else "—",
            status,
        ]
        lines.append("| " + " | ".join(cells) + " |")

    videos = report.get("videos") or []
    if videos:
        lines += [
            "",
            "## Видео",
            "",
            f"Роликов: **{len(videos)}**, по одному на сцену. Движение: "
            f"{report.get('motion', '—')}.",
            "",
            "Кадр-исходник под каждый ролик рисуется отдельно от фотопака: видеомодель анимирует",
            "ровно то, что видит в кадре, и с портрета получается поворот головы, а не действие.",
            "",
            "| # | сцена | кадров | лицо в движении | после доводки | GPU, мин | $ |",
            "|---|---|---|---|---|---|---|",
        ]
        for video in videos:
            similarity = video.get("similarity")
            enhanced_block = video.get("enhanced") or {}
            enhanced = enhanced_block.get("similarity")
            gpu_s = (video.get("video_gpu_seconds") or 0) + (video.get("source_gpu_seconds") or 0)
            gpu_s += enhanced_block.get("gpu_seconds") or 0
            cost = video.get("cost_usd")
            cells = [
                str(video["index"]),
                video["scene"][:60].replace("|", "/") + "…",
                str(video.get("num_frames", "—")),
                f"{similarity:.3f}" if similarity is not None else "—",
                f"{enhanced:.3f}" if enhanced is not None else "—",
                f"{gpu_s / 60:.1f}" if gpu_s else "—",
                f"{cost:.4f}" if cost is not None else "—",
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines += [
            "",
            "Похожесть считается по СРЕДНЕМУ кадру ролика, а не по первому: первый кадр почти",
            "равен исходному фото и о том, удержалось ли лицо в движении, не говорит ничего.",
        ]

    totals = report.get("totals")
    if totals:
        photo, video = totals["photo"], totals["video"]
        lines += [
            "",
            "## Время и стоимость",
            "",
            f"Карта арендована по **${totals['hourly_rate_usd']:.3f}/час**. Время — чистое время",
            "работы GPU, которое возвращает воркер; передача файлов на машину оператора в стоимость",
            "не входит, потому что арендованная карта за неё денег не берёт.",
            "",
            "| | штук | GPU на штуку | $ за штуку | $ всего |",
            "|---|---|---|---|---|",
        ]
        if photo["count"]:
            lines.append(
                f"| фото | {photo['count']} | {photo['gpu_seconds_avg']:.1f} с "
                f"| {photo['cost_usd_avg']:.4f} | {photo['cost_usd_total']:.4f} |")
        if video["count"]:
            lines.append(
                f"| видео | {video['count']} | {video['gpu_seconds_avg'] / 60:.1f} мин "
                f"| {video['cost_usd_avg']:.4f} | {video['cost_usd_total']:.4f} |")
        lines += [
            "",
            f"Прогон целиком: **${totals['run_cost_usd']:.4f}**.",
            "",
            "В стоимость фото входят ВСЕ попытки отбора, а не только выбранная: оплачены обе. "
            f"Попыток на кадр — {report.get('candidates', '?')}.",
            "В стоимость видео входят три статьи: кадр-исходник, сам ролик и покадровая доводка "
            "лица. Доводка сопоставима по времени с генерацией, и без неё стоимость занижена вдвое.",
        ]

    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
