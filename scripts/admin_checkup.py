"""Проверка админки: ведут ли кнопки туда, куда написано.

Зачем отдельная проверка. Админка склеена из HTML, JavaScript и Flask-маршрутов, и связи между
ними нигде не типизированы: кнопка ссылается на функцию по имени, функция читает поле по строковому
id, маршрут разбирает JSON по строковым ключам. Опечатка в любом из трёх мест не ломает запуск —
она молча приводит к тому, что кнопка ничего не делает или уходит не по тому адресу. Ровно так уже
случалось: поля чужого пайплайна оставались на экране, и выглядело это как «работают другие
сервисы», хотя маршрутизация была правильной.

Скрипт проверяет статически (по исходникам) и, если админка запущена, вживую по HTTP.

    python scripts/admin_checkup.py            # только статика
    python scripts/admin_checkup.py --live     # плюс живые запросы к запущенной админке
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

APP = ROOT / "admin" / "app.py"
HTML = ROOT / "admin" / "templates" / "index.html"

OK, FAIL, WARN = "OK  ", "СБОЙ", "ВНИМ"
results: list[tuple[str, str, str]] = []


def check(name: str, condition: bool, detail: str = "", *, warn_only: bool = False) -> bool:
    status = OK if condition else (WARN if warn_only else FAIL)
    results.append((status, name, detail))
    return condition


# --- 1. маршруты Flask ---

def flask_routes(source: str) -> dict:
    """Маршруты и имена функций из исходника, без импорта приложения.

    Разбор через ast, а не импортом: импорт admin.app тянет за собой Flask, чтение .env и
    инициализацию, и проверка начала бы падать по причинам, к маршрутам отношения не имеющим.
    """
    tree = ast.parse(source)
    routes = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            func = decorator.func
            if not (isinstance(func, ast.Attribute) and func.attr == "route"):
                continue
            if decorator.args and isinstance(decorator.args[0], ast.Constant):
                methods = ["GET"]
                for kw in decorator.keywords:
                    if kw.arg == "methods" and isinstance(kw.value, ast.List):
                        methods = [e.value for e in kw.value.elts
                                   if isinstance(e, ast.Constant)]
                routes[decorator.args[0].value] = (node.name, methods)
    return routes


def check_routes(app_source: str, html: str) -> None:
    routes = flask_routes(app_source)
    check("маршруты админки найдены", len(routes) > 5, f"{len(routes)} шт.")

    # Каждый fetch(...) из интерфейса должен попадать в существующий маршрут. Динамические куски
    # вида '/api/pods/' + id приводим к шаблону с <pod_id>.
    called = set()
    for raw in re.findall(r"fetch\(\s*('[^']+'|\"[^\"]+\")", html):
        called.add(raw.strip("'\""))
    for raw in re.findall(r"fetch\(\s*'([^']+)'\s*\+\s*\w+\s*\+\s*'([^']*)'", html):
        called.add(raw[0] + "<pod_id>" + raw[1])
    for raw in re.findall(r"fetch\(\s*'([^']+)'\s*\+\s*\w+[,)]", html):
        called.add(raw + "<pod_id>")

    known = set(routes)
    for url in sorted(called):
        clean = url.split("?")[0].rstrip("/") or "/"
        matched = any(
            clean == r or re.fullmatch(re.sub(r"<[^>]+>", "[^/]+", r), clean)
            for r in known
        )
        check(f"вызов {url}", matched, "" if matched else "маршрута нет")


# --- 2. связи «кнопка → функция → поле» ---

def check_ui_wiring(html: str) -> None:
    ids = set(re.findall(r"id=[\"']([\w\-]+)[\"']", html))
    handlers = set(re.findall(r"function\s+(\w+)\s*\(", html))
    handlers |= set(re.findall(r"async\s+function\s+(\w+)\s*\(", html))

    # onclick/onchange ссылаются на существующие функции
    for call in sorted(set(re.findall(r'on(?:click|change|toggle)="(\w+)\(', html))):
        check(f"обработчик {call}()", call in handlers or call in {"if"},
              "" if call in handlers else "функции нет")

    # getElementById читает существующие id
    missing = sorted({m for m in re.findall(r"getElementById\('([\w\-]+)'\)", html)
                      if m not in ids})
    check("все getElementById находят элемент", not missing, ", ".join(missing))


# --- 3. разведение пайплайнов ---

def check_pipeline_split(html: str, app_source: str) -> None:
    check("селектор пайплайна есть", "id=\"pipeline-choice\"" in html)
    check("варианта два: flaq и ранпод",
          'value="flaq"' in html and 'value="runpod"' in html)

    # Ключевая проверка: при выборе «ранпод» запуск уходит на под ДО облачной ветки.
    match = re.search(r"async function startRun\(\)\s*\{(.*?)\n\}", html, re.S)
    body = match.group(1) if match else ""
    routes_to_pod = bool(re.search(r"pipeline-choice.*?runpod.*?return startRunOnPod", body, re.S))
    check("«ранпод» уходит на под до облачной ветки", routes_to_pod,
          "" if routes_to_pod else "startRun не проверяет выбор первым делом")

    cloud_call = body.index("'/api/run'") if "'/api/run'" in body else -1
    pod_return = body.index("startRunOnPod") if "startRunOnPod" in body else -1
    check("облачный вызов идёт ПОСЛЕ проверки выбора",
          pod_return >= 0 and (cloud_call < 0 or pod_return < cloud_call))

    # Поля чужого пайплайна должны прятаться, и hidden обязан побеждать собственный display.
    check("класс flaq-only используется", "flaq-only" in html)
    check("класс runpod-only используется", "runpod-only" in html)
    check("switchPipeline прячет обе группы",
          "flaq-only').forEach" in html and "runpod-only').forEach" in html)
    check("hidden перебивает display", "[hidden]{display:none !important}" in html,
          "иначе .pick{display:flex} оставляет поля видимыми")
    check("состояние применяется при загрузке", re.search(r"switchPipeline\(\);\s*\npoll\(\);", html) is not None)

    # Порядок правил в CSS: [hidden] должен идти после правил с display
    if "[hidden]{display:none !important}" in html:
        check("правило [hidden] объявлено после .pick",
              html.index("[hidden]{display:none !important}") > html.index(".pick{display:flex"))


# --- 4. параметры запуска на поде ---

def check_pod_run_contract(html: str, app_source: str) -> None:
    match = re.search(r"async function startRunOnPod\(\)\s*\{(.*?)\n\}", html, re.S)
    body = match.group(1) if match else ""
    sent = set(re.findall(r"(\w+):\s*(?:Number\()?document\.getElementById", body))
    sent |= set(re.findall(r"(\w+):\s*document\.getElementById", body))

    route = re.search(r'def api_pods_full_run\(pod_id: str\):(.*?)\n@app\.route',
                      app_source, re.S)
    accepted = set(re.findall(r'payload\.get\("(\w+)"\)', route.group(1) if route else ""))

    for field in sorted(sent):
        check(f"поле {field} принимается маршрутом", field in accepted,
              "" if field in accepted else "маршрут его игнорирует")

    # И обратная сторона: маршрут не должен ждать того, чего интерфейс не шлёт.
    unused = sorted(accepted - sent - {"character"})
    check("маршрут не ждёт лишних полей", not unused, ", ".join(unused), warn_only=True)


# --- 5. соответствие списков пайплайнам ---

def check_lists() -> None:
    from pipeline import prompt_config

    env = prompt_config.read_env_file()
    looks = prompt_config.load_restyle_prompts(env)
    scenes, _ = prompt_config.load_scenes(env)

    check("«Образы» заполнены (фотопак ранпода)", len(looks) > 0, f"{len(looks)} шт.")
    check("«Сцены» заполнены (видео ранпода)", len(scenes) > 0, f"{len(scenes)} шт.")

    # Ни один этап не имеет права подставлять свой текст вместо пустого списка админки:
    # это платные генерации по промпту, которого в карточке нет и который не поправить.
    runner = (ROOT / "pipeline" / "run_pipeline.py").read_text(encoding="utf-8")
    check("этап поз не подставляет зашитый текст",
          'load_numbered("POSE") or DEFAULT_POSES' not in runner,
          "нет списка поз в админке — этап должен пропускаться, а не идти по своему тексту")

    source = (ROOT / "scripts" / "pod_full_run.py").read_text(encoding="utf-8")
    check("фотопак ранпода читает RESTYLE", "load_restyle_prompts" in source)
    check("видео ранпода читает SCENE", "load_scenes" in source)

    app_source = APP.read_text(encoding="utf-8")
    check("POSE убран из админки", '"POSE"' not in app_source)
    check("POSE_* нет в .env", not prompt_config.load_numbered("POSE", env),
          "иначе они молча влияют на flaq")

    # DEFAULT_POSES — заготовка для карточки, а НЕ запасное значение: при пустом списке этап
    # поз пропускается целиком. Проверка держит эту роль: список должен существовать как
    # образец, но не подставляться (это отдельно проверено ниже).
    from pipeline.run_pipeline import DEFAULT_POSES
    check("образец списка поз на месте", len(DEFAULT_POSES) > 0,
          f"{len(DEFAULT_POSES)} шт., подставляются только вручную через админку")

    # Промпты длиннее 77 токенов режутся текстовыми энкодерами SDXL молча. Проверяем, что
    # служебный хвост из карточки отсекается и что для остатка есть compel.
    import importlib.util

    spec = importlib.util.spec_from_file_location("pfr", ROOT / "scripts" / "pod_full_run.py")
    pfr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pfr)

    # Промпт должен собираться ТОЛЬКО из полей админки: свой текст скрипт не подмешивает, чужой
    # не выбрасывает. Проверяем, что образ доходит целиком и что в промпте нет зашитых кусков.
    prompt = pfr.build_prompt(looks[0], prompt_config.load_character(env), env, 1)
    check("образ уходит целиком, без обрезки", looks[0].strip().rstrip(",. ") in prompt)
    check("стиль съёмки в промпте", prompt_config.style_suffix(env) in prompt)
    check("мимика в промпте", prompt_config.expression_for(1, env) in prompt)

    # Фиксация лица подаётся ТОЛЬКО когда в пайплайн реально уходит эталонная картинка.
    # В двухэтапном режиме (scale=0) эталона в первом проходе нет, и фраза «preserve the exact
    # same person from the reference image» ссылается на пустоту, а её запреты («no face
    # morphing», «no identity drift») CLIP читает как обычные понятия — то есть вносит в кадр
    # ровно то, что они запрещают.
    lock = prompt_config.identity_lock(env)
    with_reference = pfr.build_prompt(looks[0], prompt_config.load_character(env), env, 1,
                                      with_lock=True)
    without = pfr.build_prompt(looks[0], prompt_config.load_character(env), env, 1,
                               with_lock=False)
    check("фиксация лица есть, когда эталон подаётся", lock in with_reference)
    check("фиксации лица нет, когда эталона нет", lock not in without)
    check("фиксация лица привязана к наличию эталона",
          "with_lock=args.scale > 0" in (ROOT / "scripts" / "pod_full_run.py").read_text(
              encoding="utf-8"))

    source = (ROOT / "scripts" / "pod_full_run.py").read_text(encoding="utf-8")
    for hardcoded in ("FRAMING_PREFIX", "MOTION_PRESETS", "REFERENCE_TEMPLATE"):
        check(f"зашитого {hardcoded} нет", hardcoded not in source,
              "промпт должен собираться из полей админки")
    check("движение берётся из VIDEO_PROMPT", 'VIDEO_PROMPT' in source)

    # Умолчания интерфейса должны совпадать с умолчаниями скрипта — иначе кнопка «Запустить»
    # даёт не то, что прогонялось и замерялось вручную.
    html = HTML.read_text(encoding="utf-8")
    expected = {
        "run-video-model": "wan2.2-i2v-a14b",
        "run-video-source": "look",
        "run-zoom": "1.25",
        "run-enhance": "both",
        # Отбор лучшего из двух попыток и ПОЛНОЕ число шагов видео. Обе ручки появились позже
        # остальных и в сверку не входили — а именно они отличают проверенный прогон от быстрого
        # черновика, и молча съехавшее умолчание здесь заметить не по чему.
        "run-candidates": "2",
        "run-fast-video": "0",
    }
    for element, value in expected.items():
        block = re.search(rf'id="{element}">(.*?)</select>', html, re.S)
        selected = re.search(r'value="([^"]+)" selected', block.group(1)) if block else None
        got = selected.group(1) if selected else None
        check(f"умолчание {element} = {value}", got == value, f"в интерфейсе {got}")

    for argument, value in (("--zoom", "1.25"), ("--enhance", '"both"'),
                            ("--video-source", '"look"'), ("--candidates", "2"),
                            ("--width", "1024"), ("--height", "1536"), ("--steps", "40")):
        found = re.search(rf'add_argument\("{re.escape(argument)}"[^)]*?default=([^,)]+)',
                          source, re.S)
        got = found.group(1).strip() if found else None
        check(f"умолчание скрипта {argument} = {value}", got == value, f"в скрипте {got}")

    check("видео по умолчанию на полных шагах",
          'add_argument("--fast-video", action="store_true"' in source,
          "флаг без default: без него прогон идёт 40 шагов, а не 4")

    # Без .env на поде воркер не видит настроек админки и берёт запасные — а они строгие:
    # порог NSFW откатывается к 0.5, и кадры бракуются фильтром, хотя в админке он выключен.
    deploy = (ROOT / "admin" / "runpod_pods.py").read_text(encoding="utf-8")
    check("карточка заливается на под", '".env"' in deploy,
          "иначе воркер работает на строгих значениях по умолчанию")
    check("pipeline заливается на под", '"pipeline"' in deploy,
          "из него SafetyPipeline читает пороги; без него они молча строгие")

    # Схема воркера ограничивает длину промпта, и это ограничение легко перерасти: описания
    # образов в карточке длинные, а к ним добавляются мимика, внешность, стиль и фиксация лица.
    # Запрос отклоняется схемой ещё до генерации, то есть кадр просто не появляется.
    # Схем ДВЕ: HTTP-запрос валидируется runpod_worker/models.py, внутренний запрос к
    # генератору — worker/models.py. Правка одной ничего не даёт: запрос падает на второй.
    limits = []
    for path in (ROOT / "runpod_worker" / "models.py", ROOT / "worker" / "models.py"):
        found = re.search(r'prompt: str = Field\(\.\.\., min_length=\d+, max_length=(\d+)\)',
                          path.read_text(encoding="utf-8"))
        limits.append(int(found.group(1)) if found else 0)
    limit = min(limits)
    character = prompt_config.load_character(env)
    longest = max(len(pfr.build_prompt(x, character, env, i))
                  for i, x in enumerate(looks + scenes, 1))
    check("промпты укладываются в лимит воркера", longest < limit,
          f"самый длинный {longest}, лимит {limit}")

    worker_source = (ROOT / "worker" / "photo_generation.py").read_text(encoding="utf-8")
    check("лимит в 77 токенов снят", "_encode_chunked" in worker_source)
    # Разбиение должно применяться ВЕЗДЕ, где промпт уходит в модель. Проход hires переписывает
    # весь кадр заново, и обрезанный там промпт означал бы доводку по другому описанию.
    hires_source = (ROOT / "worker" / "hires.py").read_text(encoding="utf-8")
    check("hires тоже без лимита", "_long_prompt_embeds" in hires_source)
    # Ищем именно импорт, а не слово: упоминание compel осталось в комментарии, где объяснено,
    # почему он не используется — и это полезный текст, а не повод считать проверку упавшей.
    check("кодирование без сторонних зависимостей",
          "import compel" not in worker_source and "from compel" not in worker_source,
          "compel несовместим с transformers 5.x")

    # Противоречие «движение про ходьбу + статичная сцена» скрипт не исправляет молча, а
    # предупреждает: подменять текст из админки он не должен.
    static = [s for s in scenes if pfr.is_static_scene(s)]
    check("статичные сцены распознаются для предупреждения", len(static) > 0,
          f"{len(static)} из {len(scenes)}")


# --- 6. скрипты, которые запускает админка ---

def check_scripts(app_source: str) -> None:
    for script, folder in re.findall(r'tracker\.start\(\s*"([\w.]+)"[^)]*?folder="(\w+)"',
                                     app_source, re.S):
        path = ROOT / folder / script
        check(f"скрипт {folder}/{script} существует", path.exists())

    for script in re.findall(r'tracker\.start\(\s*"([\w.]+)"(?![^)]*folder=)',
                             app_source, re.S):
        path = ROOT / "pipeline" / script
        check(f"скрипт pipeline/{script} существует", path.exists())

    # Аргументы, которые админка передаёт скрипту прогона на поде, должны им приниматься.
    run_source = (ROOT / "scripts" / "pod_full_run.py").read_text(encoding="utf-8")
    known_args = set(re.findall(r'add_argument\("(--[\w\-]+)"', run_source))
    route = re.search(r'def api_pods_full_run\(pod_id: str\):(.*?)\n@app\.route',
                      app_source, re.S)
    passed = set(re.findall(r'"(--[\w\-]+)"', route.group(1) if route else ""))
    for arg in sorted(passed):
        check(f"аргумент {arg} есть у pod_full_run", arg in known_args)


# --- 7. живые запросы ---

def check_live(port: int = 5000) -> None:
    import requests

    base = f"http://127.0.0.1:{port}"
    try:
        page = requests.get(base + "/", timeout=10)
    except requests.RequestException as e:
        check("админка отвечает", False, str(e)[:80])
        return
    check("главная страница открывается", page.status_code == 200, f"HTTP {page.status_code}")

    # Таймаут щедрый: /api/pods/catalog ходит в GraphQL RunPod за наличием и ценами карт, и на
    # холодную это занимает больше половины минуты. Короткий таймаут давал бы ложный сбой.
    for path in ("/api/runs", "/api/pods", "/api/pods/catalog", "/api/status", "/api/images"):
        try:
            response = requests.get(base + path, timeout=90)
            check(f"GET {path}", response.status_code == 200, f"HTTP {response.status_code}")
        except requests.RequestException as e:
            check(f"GET {path}", False, str(e)[:60])

    # Запуск на поде без готового пода обязан отказать понятной ошибкой, а не молча стартовать.
    try:
        response = requests.post(base + "/api/pods/nonexistent/full-run",
                                 json={"frames": 1}, timeout=30)
        check("запуск на несуществующем поде отклонён", response.status_code >= 400,
              f"HTTP {response.status_code}")
    except requests.RequestException as e:
        check("запуск на несуществующем поде отклонён", False, str(e)[:60])


def check_layers() -> None:
    """Направление зависимостей между слоями.

    Правило одно: нижний слой не знает о верхнем. worker/ считает и ничего не знает ни про HTTP,
    ни про админку — поэтому его можно запустить локально и в тесте. runpod_worker/ — это HTTP
    поверх worker/, он не лезет ни в облачный путь, ни в админку. pipeline/ и scripts/ стоят
    выше и могут звать всё, что ниже.

    Проверка не косметическая: один такой импорт в обратную сторону превращает вычислительный
    модуль в кусок, который не поднять без Flask и ключей RunPod, и обнаруживается это в тот
    момент, когда что-то уже сломалось.

    Единственное исключение — safety/ читает пороги из pipeline/prompt_config: конфигурация
    живёт там, и брать её из второго места означало бы, что часть пайплайна работает по админке,
    а часть по зашитым значениям.
    """
    forbidden = {
        "worker": ("admin", "pipeline", "runpod_worker", "scripts"),
        "runpod_worker": ("admin", "pipeline", "scripts"),
        "pipeline": ("admin", "scripts"),
        "safety": ("admin", "runpod_worker", "scripts"),
    }
    for layer, banned in forbidden.items():
        offenders = []
        for path in (ROOT / layer).glob("*.py"):
            text = path.read_text(encoding="utf-8", errors="replace")
            for other in banned:
                if re.search(rf"^\s*(?:from|import)\s+{other}[\s.]", text, re.M):
                    offenders.append(f"{path.name} -> {other}")
        check(f"слой {layer}/ не тянет верхние слои", not offenders,
              "; ".join(offenders) if offenders else f"запрещены: {', '.join(banned)}")


def check_cloud_video(html: str, app_source: str) -> None:
    """Видеомодель облачного пути — одна и та же в трёх местах.

    Умолчание живёт в интерфейсе, в запасном значении маршрута и в аргументе пайплайна. Разъезд
    здесь не заметен на глаз и стоит дорого: Kling через fal отклоняет обнажённый кадр на входе
    (HTTP 422 content_policy_violation), то есть при съехавшем умолчании прогон молча
    возвращается к провайдеру, который целевой контент не пропускает.
    """
    runner = (ROOT / "pipeline" / "run_pipeline.py").read_text(encoding="utf-8")
    expected = "wan-3.0-image-to-video"

    selected = re.search(r'id="video-model".*?<option value="([^"]+)"[^>]*selected', html, re.S)
    check("умолчание видеомодели в интерфейсе",
          bool(selected) and selected.group(1) == expected,
          f"в интерфейсе {selected.group(1) if selected else None}")

    in_route = re.search(r'payload\.get\("video_model"\) or "([^"]+)"', app_source)
    check("запасное значение маршрута совпадает",
          bool(in_route) and in_route.group(1) == expected,
          f"в маршруте {in_route.group(1) if in_route else None}")

    in_cli = re.search(r'"--video-model", type=str, default="([^"]+)"', runner)
    check("умолчание пайплайна совпадает",
          bool(in_cli) and in_cli.group(1) == expected,
          f"в пайплайне {in_cli.group(1) if in_cli else None}")

    check("выбор качества ролика есть в интерфейсе", 'id="video-resolution"' in html)
    check("качество ролика уходит в запрос", "video_resolution:" in html)
    check("маршрут передаёт качество ролика", '"--video-resolution"' in app_source)

    # Имена моделей flaq должны совпадать со списком в клиенте: опечатка в интерфейсе даёт 404
    # уже во время прогона, через полчаса после запуска.
    from pipeline.flaq_video import VIDEO_MODELS

    options = set(re.findall(r'<option value="([^"]+)"', html))
    unknown = [m for m in options if m.endswith("-image-to-video") and m not in VIDEO_MODELS]
    check("имена моделей flaq в интерфейсе известны клиенту", not unknown,
          f"нет в клиенте: {unknown}" if unknown else f"проверено {len(VIDEO_MODELS)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="плюс запросы к запущенной админке")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    app_source = APP.read_text(encoding="utf-8")
    html = HTML.read_text(encoding="utf-8")

    print("=== маршруты ===")
    check_routes(app_source, html)
    print("=== связи интерфейса ===")
    check_ui_wiring(html)
    print("=== разведение пайплайнов ===")
    check_pipeline_split(html, app_source)
    print("=== контракт запуска на поде ===")
    check_pod_run_contract(html, app_source)
    print("=== списки промптов ===")
    check_lists()
    print("=== скрипты ===")
    check_scripts(app_source)
    print("=== слои ===")
    check_layers()
    print("=== видео на облачном пути ===")
    check_cloud_video(html, app_source)
    if args.live:
        print("=== живые запросы ===")
        check_live(args.port)

    print()
    for status, name, detail in results:
        print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))

    failed = [r for r in results if r[0] == FAIL]
    warned = [r for r in results if r[0] == WARN]
    print(f"\nвсего {len(results)}, сбоев {len(failed)}, предупреждений {len(warned)}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
