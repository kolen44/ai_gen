"""Админка пайплайна: правка .env через браузер и запуск прогона.

Полный паритет с .env: всё, что есть в файле, видно и правится здесь, и наоборот. Отдельного
формата настроек нет специально — иначе появилось бы два источника правды.

Разделы собираются по префиксу переменной:
    CHARACTER_*                        карточка персонажа
    SCENE_*, RESTYLE_*, EXPRESSION_*   нумерованные списки
    *_KEY, *_TOKEN                     ключи, скрыты до нажатия
    остальное                          настройки как есть

    python admin/app.py  →  http://127.0.0.1:5000

Слушает только localhost.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
from collections import deque
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import requests  # noqa: E402
from flask import Flask, abort, jsonify, render_template, request, send_file  # noqa: E402

from pipeline import prompt_config  # noqa: E402

app = Flask(__name__)

# POSE убран из админки: на своей карте он не читался никогда, а у flaq есть
# встроенный DEFAULT_POSES (pipeline/run_pipeline.py:54), на который шаг 2 и
# опирается при пустом списке. Держать в интерфейсе список, который правится, но
# нужен ровно одному из двух путей, — лишний повод для путаницы.
LIST_PREFIXES = ["SCENE", "RESTYLE", "EXPRESSION"]

# Максимальная длина строки лога, отдаваемой в браузер.
MAX_LOG_LINE = 400

# Лог прогона на диске: процесс отвязан от админки и переживает её перезапуск, поэтому вывод
# должен жить в файле, а не в трубе, которая закрывается вместе с родителем.
RUN_LOG = ROOT / "runs" / "current_run.log"
RUN_PID = ROOT / "runs" / "current_run.pid"

# Заголовок предупреждения Python: «путь/файл.py:123: UserWarning: текст».
WARNING_HEADER_RE = re.compile(r":\d+:\s+\w*Warning:")

# В подписи каждого списка указано, какой пайплайн его читает. Это не украшение: на своей карте
# используются только «Сцены», и без пометки правки в остальных трёх выглядят как настройка, а на
# деле ни на что не влияют. Разбор с ссылками на код — deliverables/PROMPT_LISTS.md.
LIST_TITLES = {
    "SCENE": ("Сцены",
              "Ситуации: где происходит и что на ней надето. "
              "flaq — шаг 3, вместе с образами. Своя карта — ИСТОЧНИК ВИДЕО: на каждую сцену "
              "рисуется свой кадр-исходник и снимается свой ролик."),
    "RESTYLE": ("Образы",
                "Один образ — один кадр. "
                "flaq — шаг 3, переодевание готовых кадров. Своя карта — ФОТОПАК: именно из "
                "этого списка собираются восемь фотографий персонажа."),
    "EXPRESSION": ("Мимика",
                   "Только flaq: раздаётся кадрам по кругу, чтобы набор не вышел с одним "
                   "выражением. На своей карте не используется."),
}

# Подписи и подсказки для одиночных полей. Всё, чего здесь нет, выводится под своим именем —
# так новая переменная в .env появляется в интерфейсе сама, без правки кода.
LABELS = {
    "OUTPUT_DIR": ("Папка результатов",
                   "куда админка складывает кадры, ролики и отчёты каждого прогона"),
    "CHARACTER_NAME": ("Имя", "внутреннее, для имён файлов"),
    "CHARACTER_AGE": ("Возраст", "число; сверяется с возрастным гейтом"),
    "CHARACTER_ETHNICITY": ("Этничность", ""),
    "CHARACTER_HAIR": ("Волосы", "цвет, длина, тип"),
    "CHARACTER_EYES": ("Глаза", ""),
    "CHARACTER_FEATURES": ("Особые приметы", "родинки, веснушки, тату"),
    "CHARACTER_BUILD": ("Телосложение", ""),
    "STYLE_SUFFIX": ("Стиль съёмки", "дописывается к каждой сцене — главный рычаг «вайба»"),
    "IDENTITY_LOCK": ("Фиксация лица", "держит лицо между кадрами; оставь разрешение менять мимику"),
    "VIDEO_PROMPT": ("Движение в видео", "мягкое работает лучше резких поворотов головы"),
    "NSFW_THRESHOLD": ("Порог NSFW", "0.5 строго · 0.85 для fashion · 1.0 не блокирует"),
    "MIN_ALLOWED_AGE": ("Минимальный возраст", "оценка по лицу; ниже 18 нельзя"),
}

LONG_FIELDS = {"STYLE_SUFFIX", "IDENTITY_LOCK", "VIDEO_PROMPT"}


def _is_secret(key: str) -> bool:
    return key.endswith(("_KEY", "_TOKEN"))


def _is_list_item(key: str) -> bool:
    return any(re.match(rf"^{p}_\d+$", key) for p in LIST_PREFIXES)


# --- прогон -----------------------------------------------------------------

class RunTracker:
    """Один активный прогон и его лог. Больше одного не нужно: они конкурируют за один .env
    и за лимиты провайдера."""

    def __init__(self):
        self.process: subprocess.Popen | None = None
        self.lines: deque[str] = deque(maxlen=400)
        self._lock = threading.Lock()
        self._log = None
        self._skip_next = False

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self, script: str, args: list[str], *, title: str, folder: str = "pipeline") -> None:
        """Запускает скрипт пайплайна. Один на всю админку: прогоны конкурируют за один .env и за
        лимиты провайдера.

        folder — подкаталог со скриптом; без него прогон на своём поде из админки было не запустить.

        Процесс отвязан от админки, вывод идёт в файл, а не в трубу: прогон на поде длится больше
        часа, и перезапуск админки убивал его вместе с собой — так был потерян готовый ролик.
        """
        with self._lock:
            if self.running:
                raise RuntimeError("прогон уже идёт")
            self.lines.clear()
            cmd = [sys.executable, "-u", str(ROOT / folder / script), *args]
            self.lines.append(f"→ {title}")

            RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
            self._log = open(RUN_LOG, "w", encoding="utf-8", errors="replace")
            self._log.write(f"→ {title}\n")
            self._log.flush()

            # На Windows отвязка — это CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS; на POSIX
            # start_new_session. Без этого дочерний процесс входит в группу админки и получает
            # сигнал завершения вместе с ней.
            kwargs = {}
            if sys.platform == "win32":
                kwargs["creationflags"] = (subprocess.CREATE_NEW_PROCESS_GROUP
                                           | getattr(subprocess, "DETACHED_PROCESS", 0x00000008))
            else:
                kwargs["start_new_session"] = True

            self.process = subprocess.Popen(
                cmd, cwd=str(ROOT), stdout=self._log, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", **kwargs,
            )
            RUN_PID.write_text(str(self.process.pid), encoding="utf-8")
            threading.Thread(target=self._tail, daemon=True).start()

    def _tail(self) -> None:
        """Читает файл лога по мере появления строк — так же, как раньше читалась труба."""
        import time as _time

        with open(RUN_LOG, "r", encoding="utf-8", errors="replace") as handle:
            while True:
                line = handle.readline()
                if not line:
                    if not self.running:
                        break
                    _time.sleep(0.4)
                    continue
                self._consume(line)

    def _consume(self, line: str) -> None:
        noise = ("Applied providers", "find model", "set det-size", "it/s]", "s/it]",
                 "Device set", "use_fast", "sequentially on GPU")
        if True:
            line = line.rstrip()

            # Предупреждение Python занимает ДВЕ строки: заголовок «файл:строка: Категория:
            # текст» и эхо самой строки исходника под ним. Раньше отсеивался только заголовок,
            # и в логе оставался огрызок вида «tform.estimate(lmk, dst)» — он выглядит как
            # ошибка, хотя это хвост безобидного FutureWarning из insightface.
            if self._skip_next:
                self._skip_next = False
                return
            if WARNING_HEADER_RE.search(line):
                self._skip_next = True
                return

            if not line or any(s in line for s in noise):
                return
            # Предел на строку. Ответы провайдеров об ошибке иногда содержат эхо запроса вместе
            # с картинкой в data-URI — такая строка весит мегабайты и, попав в /api/status,
            # вешает страницу: она опрашивается раз в две секунды.
            if len(line) > MAX_LOG_LINE:
                line = line[:MAX_LOG_LINE] + f" …[обрезано, всего {len(line)} символов]"
            self.lines.append(line)

    def stop(self) -> None:
        with self._lock:
            if self.running and self.process:
                self.process.terminate()
                self.lines.append("— остановлено —")


tracker = RunTracker()


# --- сборка конфигурации ----------------------------------------------------

def _collect_config() -> dict:
    env = prompt_config.read_env_file()

    # Значения по умолчанию, которые живут в коде, показываем как заполненные — иначе в админке
    # поле выглядит пустым, хотя пайплайн что-то использует.
    fallbacks = {
        "STYLE_SUFFIX": prompt_config.DEFAULT_STYLE_SUFFIX,
        "IDENTITY_LOCK": prompt_config.DEFAULT_IDENTITY_LOCK,
        "NSFW_THRESHOLD": "0.85",
        "MIN_ALLOWED_AGE": "18",
    }
    for key, value in fallbacks.items():
        env.setdefault(key, value)

    lists: dict[str, list[str]] = {}
    for prefix in LIST_PREFIXES:
        items, i = [], 1
        while True:
            value = env.get(f"{prefix}_{i:02d}") or env.get(f"{prefix}_{i}")
            if not value:
                break
            items.append(value)
            i += 1
        lists[prefix] = items

    character, secrets, settings = {}, {}, {}
    for key, value in sorted(env.items()):
        if _is_list_item(key):
            continue
        if _is_secret(key):
            secrets[key] = value
        elif key.startswith("CHARACTER_"):
            character[key] = value
        else:
            settings[key] = value

    def described(items: dict) -> list[dict]:
        out = []
        for key, value in items.items():
            title, hint = LABELS.get(key, (key, ""))
            out.append({"key": key, "value": value, "title": title, "hint": hint,
                        "long": key in LONG_FIELDS})
        return out

    return {
        "character": described(character),
        "settings": described(settings),
        "secrets": described(secrets),
        "lists": lists,
        "list_titles": LIST_TITLES,
    }


@app.route("/")
def index():
    return render_template("index.html", config=_collect_config(), runs=_list_runs())


@app.route("/api/save", methods=["POST"])
def save():
    payload = request.get_json(force=True)
    merged: dict[str, str] = {}

    for key, value in payload.get("fields", {}).items():
        if value is not None and str(value).strip():
            merged[key] = " ".join(str(value).split())

    for prefix, items in payload.get("lists", {}).items():
        for i, item in enumerate([s for s in items if s.strip()], start=1):
            merged[f"{prefix}_{i:02d}"] = " ".join(item.split())

    # Проверки до записи: испорченное число здесь означает падение прогона через полчаса,
    # когда пайплайн дойдёт до соответствующего гейта.
    try:
        if float(merged.get("MIN_ALLOWED_AGE", "18")) < 18:
            return jsonify(ok=False, error="Минимальный возраст не может быть ниже 18"), 400
        if not 0 <= float(merged.get("NSFW_THRESHOLD", "0.85")) <= 1:
            return jsonify(ok=False, error="Порог NSFW должен быть от 0 до 1"), 400
        int(merged.get("CHARACTER_AGE", "30"))
    except ValueError as e:
        return jsonify(ok=False, error=f"Неверное число: {e}"), 400

    prompt_config.write_env_file(merged)
    return jsonify(ok=True, at=datetime.now().strftime("%H:%M:%S"))


@app.route("/api/check", methods=["POST"])
def check_prompt():
    """Проверка формулировки фильтрами до запуска — чтобы не выяснять на середине прогона."""
    from safety.moderation_pipeline import PromptGate

    text = request.get_json(force=True).get("prompt", "")
    result = PromptGate(min_allowed_age=int(prompt_config.min_allowed_age())).check(text)
    return jsonify(allowed=result.allowed, matched=result.matched_patterns)


@app.route("/api/run", methods=["POST"])
def run_pipeline():
    payload = request.get_json(force=True)
    name = re.sub(r"[^\w\-]+", "_", payload.get("name") or datetime.now().strftime("%m%d_%H%M"))

    extra: list[str] = []
    if payload.get("skip_video"):
        extra.append("--skip-video")
    if payload.get("force"):
        extra.append("--force")
    if payload.get("pack_target"):
        extra += ["--pack-target", str(int(payload["pack_target"]))]

    # Движок шагов 1-2. По умолчанию Seedream — на нём собран основной путь (пак 03_dataset_v3),
    # он же даёт лучшую метрику лица. NBP остаётся альтернативой и выбирается в интерфейсе.
    from pipeline.run_pipeline import ENGINE_CHOICES

    engine = str(payload.get("engine") or "seedream")
    # Имя движка уходит в командную строку дочернего процесса — сверяем со списком,
    # а не полагаемся на то, что браузер прислал одно из значений выпадающего списка.
    for key in ("engine", "restyle_engine"):
        val = payload.get(key)
        if val and val not in ENGINE_CHOICES:
            return jsonify(ok=False, error=f"неизвестный движок: {val}"), 400
    extra += ["--pack-engine", engine, "--pose-engine", engine]
    # Шаг 3 отдельным выбором: у Seedream 5.0 через flaq другой набор ограничений (один
    # референс вместо нескольких), и держать его отдельно от шагов 1-2 полезно.
    extra += ["--restyle-engine", str(payload.get("restyle_engine") or engine)]
    if payload.get("image_size"):
        extra += ["--image-size", str(payload["image_size"])]
    # Умолчание — Wan 3.0 у flaq: Kling через fal отклоняет обнажённый кадр на входе
    # (HTTP 422 content_policy_violation), а взрослый контент здесь целевой.
    model = str(payload.get("video_model") or "wan-3.0-image-to-video")
    extra += ["--video-model", model]
    if payload.get("video_resolution"):
        extra += ["--video-resolution", str(payload["video_resolution"])]

    # Animate-режимы переносят движение с ролика — без него шаг видео просто пропустится,
    # поэтому проверяем путь здесь, а не через полчаса на середине прогона.
    if model.startswith("wan-animate"):
        driver = Path(str(payload.get("driver", "")).strip('" '))
        if not str(driver) or not driver.exists():
            return jsonify(ok=False,
                           error=f"для {model} нужен драйвер-ролик; не найден: {driver}"), 400
        extra += ["--driver", str(driver)]

    output_dir = runs_root() / name
    try:
        tracker.start("run_pipeline.py", ["--output-dir", str(output_dir), *extra],
                      title=f"полный прогон · {name}")
    except RuntimeError as e:
        return jsonify(ok=False, error=str(e)), 409
    return jsonify(ok=True)


@app.route("/api/run-animate", methods=["POST"])
def run_animate():
    """Покадровая анимация: движение берётся с драйвер-ролика, персонаж — с референсов."""
    payload = request.get_json(force=True)

    driver = Path(payload.get("driver", "").strip('" '))
    if not driver.exists():
        return jsonify(ok=False, error=f"драйвер-ролик не найден: {driver}"), 400

    refs = [Path(p.strip('" ')) for p in payload.get("references", []) if p.strip()]
    missing = [str(p) for p in refs if not p.exists()]
    if not refs:
        return jsonify(ok=False, error="нужен хотя бы один референс персонажа"), 400
    if missing:
        return jsonify(ok=False, error=f"референсы не найдены: {', '.join(missing)}"), 400

    name = re.sub(r"[^\w\-]+", "_", payload.get("name") or datetime.now().strftime("%m%d_%H%M"))
    output_dir = runs_root() / f"animate_{name}"

    args = [
        "--references", *[str(p) for p in refs],
        "--driver", str(driver),
        "--output-dir", str(output_dir),
        "--frames", str(int(payload.get("frames") or 24)),
        "--fps", str(int(payload.get("fps") or 8)),
    ]
    if payload.get("scene"):
        args += ["--scene", payload["scene"]]
    if payload.get("limit"):
        args += ["--limit", str(int(payload["limit"]))]
    if payload.get("no_chain"):
        args.append("--no-chain")

    try:
        tracker.start("seedream_animate.py", args, title=f"анимация · {name}")
    except RuntimeError as e:
        return jsonify(ok=False, error=str(e)), 409
    return jsonify(ok=True)


@app.route("/api/images")
def list_images():
    """Готовые кадры персонажа, из которых удобно выбрать референсы, не набирая путь руками."""
    found: list[dict] = []
    for base in (ROOT / "runs", Path.home() / "Downloads" / "Synthetic_Character_WIP"):
        if not base.exists():
            continue
        for p in sorted(base.rglob("*.png"))[:400]:
            if any(s in p.name.lower() for s in ("sheet", "strip", "pose", "check")):
                continue
            found.append({"path": str(p), "label": f"{p.parent.name}/{p.name}"})
    return jsonify(images=found[:200])


def _first_scene_prompt(env: dict) -> str:
    """Промпт пробного кадра: внешность + первая сцена + стиль съёмки.

    Внешность обязательна: FaceID держит геометрию лица, но не цвет волос. Без описания кадр
    выходил рыжим при тёмно-каштановых в карточке, и косинус этого не замечал (0.655 в обоих).

    Стиль берётся из style_suffix(), а не из второго значения load_scenes() — там источник сцен,
    строка вида ".env (SCENE_01..SCENE_08)", и она однажды уехала прямо в промпт.
    """
    from pipeline import prompt_config

    character = prompt_config.load_character(env)
    description = ("{age}-year-old {ethnicity} woman, {hair}, {eyes}, "
                   "{distinguishing_features}, {build}").format(**character)

    scenes, _source = prompt_config.load_scenes(env)
    scene = scenes[0] if scenes else (
        "standing by a window in a plain apartment, natural daylight, casual clothes")
    return f"{description}, {prompt_config.compose(scene, prompt_config.style_suffix(env))}"


# --- поды RunPod ------------------------------------------------------------
# За каждым вызовом здесь стоят деньги: карта тарифицируется с создания до удаления, работает она
# или простаивает. Поэтому список подов отдаёт наработку и стоимость, а не только статус.

@app.route("/api/pods")
def api_pods():
    from admin import runpod_pods as rp

    try:
        live = {p["id"]: p for p in rp.live_pods()}
    except Exception as e:  # noqa: BLE001 — сеть/ключ; страница должна открыться и без RunPod
        return jsonify(ok=False, error=str(e), pods=[])

    known = rp._load_registry()
    out = []
    # Идём по живым подам аккаунта, а не по нашему реестру: под, созданный мимо админки или
    # оставшийся от прошлой сессии, тоже тратит деньги и обязан быть виден.
    for pod_id, pod in live.items():
        record = known.get(pod_id)
        runtime = pod.get("runtime") or {}
        uptime = runtime.get("uptimeInSeconds") or 0
        cost = pod.get("costPerHr") or 0
        out.append({
            "id": pod_id,
            "name": pod.get("name"),
            "status": pod.get("desiredStatus"),
            "gpu": (pod.get("machine") or {}).get("gpuDisplayName"),
            "cost_per_hr": cost,
            "uptime_min": round(uptime / 60),
            "spent": round(cost * uptime / 3600, 2),
            "known": record is not None,
            "photo_model": record.photo_model if record else None,
            "identity_mode": record.identity_mode if record else None,
            # Состояние развёртывания хранится в нашем реестре и живёт своей жизнью: RunPod может
            # погасить карту (кончился баланс, вытеснение), а в реестре останется "ready". Поэтому
            # для неработающей карты состояние подменяется на "stopped" — иначе интерфейс выбирает
            # мёртвый под и запуск падает с «воркер не отвечает».
            "deploy_state": (record.deploy_state if record else "unmanaged")
                            if pod.get("desiredStatus") == "RUNNING" else "stopped",
            "deploy_log": record.deploy_log if record else "",
            "worker_url": rp.worker_url(pod_id),
        })
    return jsonify(ok=True, pods=out)


@app.route("/api/pods/catalog")
def api_pods_catalog():
    """Карты с ценами и наличием + список чекпойнтов, которые воркер умеет грузить."""
    from admin import runpod_pods as rp
    from worker.config import PHOTO_MODEL_CONFIGS, IDENTITY_MODES

    try:
        gpus = rp.gpu_catalog()
    except Exception as e:  # noqa: BLE001
        return jsonify(ok=False, error=str(e), gpus=[], models=[], modes=[])

    models = []
    for key, cfg in PHOTO_MODEL_CONFIGS.items():
        # Civitai-чекпойнты требуют токен, которого может не быть — помечаем, чтобы выбор
        # не приводил к падению уже на поде.
        models.append({
            "key": key,
            "repo": cfg.get("hf_repo_id") or f"civitai:{cfg.get('model_id')}",
            "faceid": cfg.get("supports_faceid", True),
            "needs_civitai": cfg.get("source") == "civitai",
        })
    return jsonify(ok=True, gpus=gpus, models=models, modes=list(IDENTITY_MODES))


@app.route("/api/pods/create", methods=["POST"])
def api_pods_create():
    from admin import runpod_pods as rp

    payload = request.get_json(force=True)
    name = re.sub(r"[^\w\-]+", "_", payload.get("name") or datetime.now().strftime("pod_%m%d_%H%M"))
    try:
        record = rp.create_pod(
            name=name,
            profile=payload.get("profile") or "a6000",
            photo_model=payload.get("photo_model") or "lustify",
            identity_mode=payload.get("identity_mode") or "portrait",
        )
    except Exception as e:  # noqa: BLE001
        return jsonify(ok=False, error=str(e)), 400

    # Развёртывание идёт в фоне: заливка кода, pip и первый /health занимают 10-20 минут, и
    # держать на них HTTP-запрос браузера нельзя.
    rp.deploy_async(record.pod_id)
    return jsonify(ok=True, pod_id=record.pod_id)


@app.route("/api/pods/<pod_id>/redeploy", methods=["POST"])
def api_pods_redeploy(pod_id: str):
    """Перезалить код и перезапустить воркер на уже арендованном поде.

    Нужно чаще, чем кажется: правка в worker/ не требует новой карты, а пересоздание пода стоит
    и денег, и десяти минут на скачивание образа.
    """
    from admin import runpod_pods as rp

    if not rp.get_record(pod_id):
        return jsonify(ok=False, error="под не найден в реестре админки"), 404
    rp.deploy_async(pod_id)
    return jsonify(ok=True)


@app.route("/api/pods/<pod_id>/health")
def api_pods_health(pod_id: str):
    from admin import runpod_pods as rp

    health = rp.worker_health(pod_id)
    return jsonify(ok=health is not None, health=health)


@app.route("/api/pods/<pod_id>", methods=["DELETE"])
def api_pods_delete(pod_id: str):
    from admin import runpod_pods as rp

    try:
        rp.delete_pod(pod_id)
    except Exception as e:  # noqa: BLE001
        return jsonify(ok=False, error=str(e)), 400
    return jsonify(ok=True)


@app.route("/api/pods/<pod_id>/generate", methods=["POST"])
def api_pods_generate(pod_id: str):
    """Пробный кадр на поде — проверка, что связка «карта + чекпойнт + identity» реально работает.

    Синхронный маршрут воркера (/generate/sync), а не очередь: это диагностика на один кадр, и
    ответ нужен сразу, вместе с метрикой похожести лица.
    """
    from admin import runpod_pods as rp

    record = rp.get_record(pod_id)
    if not record:
        return jsonify(ok=False, error="под не найден в реестре"), 404

    payload = request.get_json(force=True)
    from pipeline import prompt_config

    env = prompt_config.read_env_file()
    key = prompt_config._get(env, "WORKER_API_KEY") or pod_id

    character = payload.get("character") or "test"
    headers = {"X-Api-Key": key}
    # Длинная операция: создание персонажа рисует пять кадров, и HTTP-прокси RunPod
    # успевает отвалиться по своему таймауту, отдав HTML вместо ответа воркера.
    base = rp.worker_base(pod_id, long_running=True)

    # Персонаж = эталонный кадр + его ArcFace-эмбеддинг. Без него воркер вернёт 404: генерация с
    # фиксацией лица не имеет смысла, если фиксировать не на что. Заводим его тем же вызовом, что
    # использует полный пайплайн, — по описанию из карточки персонажа в .env, чтобы пробный кадр
    # проверял ту же связку, что пойдёт в работу, а не абстрактный промпт.
    try:
        existing = requests.get(f"{base}/characters", headers=headers, timeout=30)
        known = [c.get("name") for c in existing.json()] if existing.status_code == 200 else []
    except requests.RequestException as e:
        return jsonify(ok=False, error=f"воркер не ответил: {e}"), 502

    if character not in known:
        described = prompt_config.load_character(env)
        reference_prompt = (
            "candid close-up photo of a {age}-year-old {ethnicity} woman, {hair}, {eyes}, "
            "{distinguishing_features}, {build}, looking straight at the camera, "
            "relaxed natural expression, soft window light, natural skin texture"
        ).format(**described)
        try:
            created = requests.post(
                f"{base}/characters", headers=headers,
                json={"name": character, "prompt": reference_prompt},
                # Здесь тянется сам чекпойнт (несколько ГБ) и грузится в память — самый долгий
                # шаг за всю проверку.
                timeout=2400,
            )
        except requests.RequestException as e:
            return jsonify(ok=False, error=f"эталон не создан: {e}"), 502
        if created.status_code not in (200, 201):
            return jsonify(ok=False, error=f"эталон не создан: {created.text[:400]}"), 502

    body = {
        "character": character,
        # Первая сцена из карточки, а не выдуманный промпт: пробный кадр должен проверять ту
        # же связку, что пойдёт в работу. compose() дописывает стиль съёмки — тот же рычаг
        # «вайба», что и в полном прогоне.
        "prompt": payload.get("prompt") or _first_scene_prompt(env),
        "model_name": record.photo_model,
        "width": int(payload.get("width") or 1024),
        "height": int(payload.get("height") or 1024),
    }
    try:
        response = requests.post(
            f"{base}/generate/sync", json=body, headers=headers, timeout=1800,
        )
    except requests.RequestException as e:
        return jsonify(ok=False, error=f"воркер не ответил: {e}"), 502
    if response.status_code != 200:
        return jsonify(ok=False, error=response.text[:400]), response.status_code

    result = response.json()
    # Кадр приходит путём на диске пода, а не файлом — забираем по scp и кладём рядом с
    # прогонами, чтобы он открывался в браузере тем же /api/file, что и всё остальное.
    local_dir = runs_root() / f"pod_{pod_id}"
    saved = None
    remote = result.get("image_path")
    if remote:
        candidate = local_dir / f"{result.get('id', 'frame')}.png"
        if rp.fetch_file(pod_id, remote, candidate):
            saved = candidate

    # Эталон персонажа — рядом с кадром: без него метрику похожести не на что смотреть глазами,
    # а именно она решает, годится связка или нет.
    reference = None
    ref_remote = f"/workspace/characters/{character}.png"
    ref_local = local_dir / f"reference_{character}.png"
    if not ref_local.exists():
        rp.fetch_file(pod_id, ref_remote, ref_local)
    if ref_local.exists():
        reference = str(ref_local)

    return jsonify(ok=True, result=result,
                   path=str(saved) if saved else None, reference=reference)


@app.route("/api/pods/<pod_id>/video", methods=["POST"])
def api_pods_video(pod_id: str):
    """Оживление кадра, уже лежащего на поде.

    Путь к кадру передаётся как есть: он вернулся из /generate/sync того же пода, гонять картинку
    туда-обратно ради видео незачем. Готовый ролик забираем по scp — как и кадры.
    """
    from admin import runpod_pods as rp

    record = rp.get_record(pod_id)
    if not record:
        return jsonify(ok=False, error="под не найден в реестре"), 404

    payload = request.get_json(force=True)
    remote_image = payload.get("image_path")
    if not remote_image:
        return jsonify(ok=False, error="нужен image_path — путь к кадру на поде"), 400

    from pipeline import prompt_config

    env = prompt_config.read_env_file()
    key = prompt_config._get(env, "WORKER_API_KEY") or pod_id

    body = {
        "image_path": remote_image,
        # Движение описывается отдельно от сцены: мягкое работает заметно лучше резких поворотов
        # головы, на которых видеомодели рвут лицо (VIDEO_PROMPT в .env — тот же рычаг).
        "prompt": payload.get("prompt") or prompt_config._get(env, "VIDEO_PROMPT") or "",
        "model_name": payload.get("model_name"),
        "num_frames": payload.get("num_frames"),
        "fps": payload.get("fps"),
        "frame_seed": payload.get("frame_seed"),
    }
    body = {k: v for k, v in body.items() if v is not None}

    try:
        response = requests.post(
            f"{rp.worker_base(pod_id, long_running=True)}/video/sync", json=body,
            headers={"X-Api-Key": key},
            # Первый ролик тянет веса видео-модели (гигабайты) и только потом считает.
            timeout=3600,
        )
    except requests.RequestException as e:
        return jsonify(ok=False, error=f"воркер не ответил: {e}"), 502
    if response.status_code != 200:
        return jsonify(ok=False, error=response.text[:400]), response.status_code

    result = response.json()
    local_dir = runs_root() / f"pod_{pod_id}"
    saved = None
    if result.get("video_path"):
        candidate = local_dir / f"{result.get('id', 'video')}.mp4"
        if rp.fetch_file(pod_id, result["video_path"], candidate):
            saved = candidate
    return jsonify(ok=True, result=result, path=str(saved) if saved else None)


@app.route("/api/pods/<pod_id>/video-models")
def api_pods_video_models(pod_id: str):
    from admin import runpod_pods as rp
    from pipeline import prompt_config

    env = prompt_config.read_env_file()
    key = prompt_config._get(env, "WORKER_API_KEY") or pod_id
    try:
        response = requests.get(f"{rp.worker_url(pod_id)}/video/models",
                                headers={"X-Api-Key": key}, timeout=30)
        return jsonify(ok=response.status_code == 200, **response.json())
    except (requests.RequestException, ValueError) as e:
        return jsonify(ok=False, error=str(e), models=[]), 502


@app.route("/api/pods/<pod_id>/full-run", methods=["POST"])
def api_pods_full_run(pod_id: str):
    """Весь пайплайн на своём поде одной кнопкой: персонаж → пак кадров → видео.

    Запускается тем же трекером, что и облачные прогоны, поэтому лог и кнопка «стоп» работают
    одинаково для обоих путей — в интерфейсе это один и тот же блок.
    """
    from admin import runpod_pods as rp

    record = rp.get_record(pod_id)
    if not record:
        return jsonify(ok=False, error="под не найден в реестре"), 404
    if rp.worker_health(pod_id) is None:
        return jsonify(ok=False, error="воркер на поде не отвечает — дождитесь готовности"), 409

    payload = request.get_json(force=True) or {}
    args = [
        "--pod", pod_id,
        "--character", re.sub(r"[^\w\-]+", "_", payload.get("character") or "hero"),
        "--frames", str(int(payload.get("frames") or 8)),
        # Чекпойнт можно переопределить на запуск: он выбирается при аренде карты, но менять его
        # между прогонами дешевле, чем поднимать новый под — веса уже в кеше на диске.
        "--photo-model", payload.get("photo_model") or record.photo_model,
    ]
    if payload.get("fast_video"):
        args.append("--fast-video")
    if payload.get("candidates"):
        args += ["--candidates", str(int(payload["candidates"]))]
    if payload.get("video_model"):
        args += ["--video-model", str(payload["video_model"])]
    if payload.get("video_frames"):
        args += ["--video-frames", str(int(payload["video_frames"]))]
    if payload.get("zoom"):
        args += ["--zoom", str(float(payload["zoom"]))]
    if payload.get("video_source"):
        args += ["--video-source", str(payload["video_source"])]
    if payload.get("videos") is not None:
        args += ["--videos", str(int(payload["videos"]))]
    if payload.get("enhance"):
        args += ["--enhance", str(payload["enhance"])]
    if payload.get("skip_video"):
        args.append("--skip-video")

    try:
        tracker.start("pod_full_run.py", args, folder="scripts",
                      title=f"прогон на поде {pod_id}")
    except RuntimeError as e:
        return jsonify(ok=False, error=str(e)), 409
    return jsonify(ok=True)


@app.route("/api/status")
def status():
    return jsonify(running=tracker.running, lines=list(tracker.lines))


@app.route("/api/stop", methods=["POST"])
def stop():
    tracker.stop()
    return jsonify(ok=True)


def runs_root() -> Path:
    """Папка, куда складываются прогоны.

    Задаётся в .env через OUTPUT_DIR — тогда её видно и можно поменять прямо в админке.
    Пусто или не задано — runs/ внутри проекта, то есть поведение по умолчанию не меняется.
    """
    # Общий помощник, а не своя копия логики: раньше путь вычисляли в трёх местах по-разному,
    # и результаты расползались по каталогам.
    return prompt_config.output_root()


def _allowed_roots() -> list[Path]:
    """Откуда админке разрешено отдавать файлы в браузер.

    Список закрытый специально: путь приходит от клиента, и без проверки страница
    превратилась бы в чтение любого файла на диске по произвольному пути.
    """
    roots = [runs_root(), ROOT / "runs", Path.home() / "Downloads" / "Synthetic_Character_WIP"]
    out = []
    for r in roots:
        try:
            out.append(r.resolve())
        except OSError:
            continue
    return out


@app.route("/api/file")
def serve_file():
    """Отдаёт кадр или ролик из прогона — чтобы их было видно на странице, а не только путём."""
    raw = (request.args.get("path") or "").strip('" ')
    if not raw:
        abort(400)
    try:
        target = Path(raw).resolve()
    except OSError:
        abort(400)
    if not any(target == r or target.is_relative_to(r) for r in _allowed_roots()):
        abort(403)
    if not target.is_file():
        abort(404)
    return send_file(target)


def _run_summary(d: Path) -> dict:
    """Сводка по прогону для списка: без чтения картинок, только run.json и подсчёт файлов."""
    info = {
        "name": d.name,
        "path": str(d),
        "images": len(list(d.rglob("*.png"))),
        "videos": len(list(d.rglob("*.mp4"))),
        "when": datetime.fromtimestamp(d.stat().st_mtime).strftime("%d.%m %H:%M"),
        "engines": {}, "accepted": 0, "total": 0, "avg": None, "video_model": None,
    }
    run_json = d / "run.json"
    if not run_json.exists():
        return info
    try:
        data = json.loads(run_json.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return info
    frames = data.get("frames", [])
    sims = [f["similarity"] for f in frames
            if f.get("similarity") is not None and f.get("accepted")]
    info["engines"] = data.get("engines", {})
    info["total"] = len(frames)
    info["accepted"] = sum(1 for f in frames if f.get("accepted"))
    info["avg"] = round(sum(sims) / len(sims), 3) if sims else None
    info["video_model"] = (data.get("video") or {}).get("model")
    return info


def _list_runs() -> list[dict]:
    runs_dir = runs_root()
    if not runs_dir.exists():
        return []
    return [_run_summary(d)
            for d in sorted((p for p in runs_dir.iterdir() if p.is_dir()),
                            key=lambda p: -p.stat().st_mtime)[:30]]


@app.route("/api/runs")
def api_runs():
    return jsonify(runs=_list_runs(), root=str(runs_root()))


@app.route("/api/runs/<name>")
def api_run_detail(name: str):
    """Всё, что известно о прогоне: промпты, метрики, файлы по шагам, отчёт."""
    safe = re.sub(r"[^\w\-]+", "_", name)
    d = runs_root() / safe
    if not d.is_dir():
        abort(404)

    data: dict = {}
    run_json = d / "run.json"
    if run_json.exists():
        try:
            data = json.loads(run_json.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}

    stages: dict[str, list] = {}
    for sub in sorted(p for p in d.iterdir() if p.is_dir()):
        files = sorted(list(sub.glob("*.png")) + list(sub.glob("*.mp4")))
        if files:
            stages[sub.name] = [{"name": f.name, "path": str(f),
                                 "video": f.suffix.lower() == ".mp4"} for f in files]

    sheets = [{"name": f.name, "path": str(f)} for f in sorted(d.glob("*_sheet.png"))]
    report = ""
    if (d / "REPORT.md").exists():
        report = (d / "REPORT.md").read_text(encoding="utf-8", errors="replace")

    return jsonify(name=safe, path=str(d), data=data,
                   stages=stages, sheets=sheets, report=report)


if __name__ == "__main__":
    print("админка: http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
