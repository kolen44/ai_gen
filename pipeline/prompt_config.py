"""Промпты и карточка персонажа — из .env, а не из кода.

Единственный источник промптов для всего проекта. Всё содержательное правится в админке.

    CHARACTER_*            внешность
    SCENE_01..NN           сцены под видео, сплошная нумерация с 1
    RESTYLE_01..NN         образы для фотопака
    EXPRESSION_01..NN      мимика, раздаётся кадрам по кругу
    STYLE_SUFFIX           как снято: камера, свет, фактура кожи
    IDENTITY_LOCK          фиксация лица
    VIDEO_PROMPT           движение в ролике
    NEGATIVE_PROMPT        перекрывает дефолт из worker/config.py

Многострочные значения не поддерживаются: одна строка — одно значение, иначе .env превращается
в формат со своими правилами экранирования.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

DEFAULT_CHARACTER = {
    "name": "nova",
    "age": "30",
    "ethnicity": "Eastern European",
    "hair": "dark brown wavy shoulder-length hair",
    "eyes": "hazel eyes",
    "distinguishing_features": "small beauty mark above her left lip, fair skin",
    "build": "slim athletic build",
}

# Хвост по умолчанию — то, что переводит картинку из каталожной в лайфстайл. Держим его коротким:
# длинные хвосты съедают лимит в 77 токенов у CLIP и вытесняют саму сцену.
DEFAULT_STYLE_SUFFIX = (
    "shot on iPhone, natural skin texture with visible pores, no beauty filter, "
    "candid photo, slight grain"
)

# Блок фиксации личности. Длинный намеренно: у моделей, работающих по референсам (Seedream, NBP),
# короткое "keep the same face" работает заметно слабее перечисления конкретных признаков.
# Вторая половина блока так же важна, как первая: без явного разрешения менять мимику модель
# копирует с референса и выражение, и весь пак выходит с одним положением губ.
DEFAULT_IDENTITY_LOCK = (
    "preserve the exact same person from the reference image, facial identity must remain "
    "consistent across every generation, keep the same facial structure, facial proportions, "
    "eye shape and spacing, nose shape, lips, jawline, cheekbones, chin, forehead, hairline, "
    "skin tone and distinctive facial features, the face must remain clearly recognizable as the "
    "same person, allow natural variation in facial expression, mood and emotion from image to "
    "image, subtle changes such as smiling, neutral expression, relaxed expression, slight "
    "laughter, thoughtful expression or confident expression are allowed, allow natural "
    "micro-expressions and slight changes in gaze direction, but never change the underlying "
    "face, facial geometry, proportions, age, ethnicity, identity or distinctive features, "
    "no face morphing, no identity drift, no generic replacement face, no beautification that "
    "changes facial structure"
)

# Варианты мимики, раздаются кадрам по кругу. Задача — разнообразие внутри пака при неизменном
# лице; поэтому здесь только выражения и взгляд, ничего про черты.
DEFAULT_EXPRESSIONS = [
    "relaxed neutral expression, lips slightly parted",
    "soft closed-lip smile, warm eyes",
    "laughing openly, teeth visible, eyes crinkled",
    "thoughtful look, gazing away from the camera",
    "confident half-smile, chin slightly raised",
    "calm expression, mouth closed, direct eye contact",
    "surprised raised eyebrows, mouth slightly open",
    "tired soft expression, eyes half closed",
]

DEFAULT_SCENES = [
    "candid photo, wearing an oversized beige knit sweater, sitting by a large window in a "
    "sunlit apartment, morning light across her face, holding a mug, looking out the window",
    "candid street photo, wearing a denim jacket over a white tee, walking past a shop window, "
    "late afternoon sun flare, mid-step, hair caught by the wind, looking away from camera",
    "mirror selfie in a bathroom with warm bulb lighting, wearing a grey tank top and jeans, "
    "phone partially visible in hand, relaxed neutral expression, waist-up",
    "candid photo sitting on a park bench in autumn, wearing a cream trench coat, overcast soft "
    "daylight, blurred golden leaves behind her, laughing at something off-camera",
    "candid photo in a cosy cafe, wearing a fitted black turtleneck, warm tungsten light mixed "
    "with window daylight, chin resting on her hand, faint smile at the camera",
    "candid beach photo at sunset, wearing a one-piece swimsuit, strong golden backlight, "
    "wet hair pushed back, squinting slightly against the sun, waist-up",
    "candid gym mirror selfie, wearing a fitted sports top and leggings, harsh overhead "
    "fluorescent light, slightly flushed after training, strands of hair on her forehead",
    "candid photo on a rooftop at dusk, wearing a fitted black slip dress, city lights bokeh "
    "behind her, warm string lights above, glancing back over her shoulder",
]


def _read_env_file(path: Path = ENV_PATH) -> dict[str, str]:
    """Разбирает .env. Специально не использует python-dotenv: одна зависимость на десять строк
    кода не нужна, а поведение здесь предсказуемое — комментарии и пустые строки пропускаются,
    кавычки по краям снимаются."""
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$", line)
        if not m:
            continue
        key, value = m.group(1), m.group(2).strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def write_env_file(values: dict[str, str], path: Path = ENV_PATH) -> None:
    """Перезаписывает .env, сохраняя порядок разделов и пояснения.

    Порядок фиксированный — ключи, настройки, длинные промпты: иначе NSFW_THRESHOLD уезжает
    в конец файла за строки промптов по 600 символов и глазами не находится.
    """
    secrets = ["RUNPOD_API_KEY", "HF_TOKEN", "FAL_KEY", "GEMINI_API_KEY", "CIVITAI_TOKEN",
               "OPENROUTER_API_KEY"]
    settings = ["NSFW_THRESHOLD", "MIN_ALLOWED_AGE", "STYLE_SUFFIX", "VIDEO_PROMPT",
                "IDENTITY_LOCK"]
    character = ["CHARACTER_NAME", "CHARACTER_AGE", "CHARACTER_ETHNICITY", "CHARACTER_HAIR",
                 "CHARACTER_EYES", "CHARACTER_FEATURES", "CHARACTER_BUILD"]

    def section(title: str) -> list[str]:
        return ["", "# " + "=" * 60, f"# {title}", "# " + "=" * 60]

    out: list[str] = ["# Файл сгенерирован админкой. Можно править руками — формат обычный .env."]

    out += section("КЛЮЧИ ДОСТУПА")
    for k in secrets:
        if values.get(k):
            out.append(f"{k}={values[k]}")

    out += section("ПЕРСОНАЖ")
    for k in character:
        if values.get(k):
            out.append(f"{k}={values[k]}")

    out += section("НАСТРОЙКИ")
    out.append("# NSFW_THRESHOLD — порог фильтра готовых кадров: выше значение, мягче отбор.")
    out.append("# MIN_ALLOWED_AGE — порог возрастного гейта (оценка по лицу, не документ).")
    for k in settings:
        if values.get(k):
            out.append(f"{k}={values[k]}")

    numbered_groups = [
        ("SCENE", "СЦЕНЫ"),
        ("RESTYLE", "ОБРАЗЫ ДЛЯ ПЕРЕГЕНЕРАЦИИ"),
        ("EXPRESSION", "ВАРИАНТЫ МИМИКИ"),
    ]
    for prefix, title in numbered_groups:
        keys = sorted(k for k in values
                      if re.match(rf"^{prefix}_\d+$", k) and values[k].strip())
        if not keys:
            continue
        out += section(title)
        # Перенумеровываем подряд: дыра в нумерации ломает чтение (сборщик останавливается
        # на первом пропуске), а после удаления строк в админке дыры появляются неизбежно.
        for i, key in enumerate(sorted(keys, key=lambda k: int(k.split("_")[-1])), start=1):
            out.append(f"{prefix}_{i:02d}={values[key]}")

    known = set(secrets) | set(settings) | set(character)
    rest = [k for k in values
            if k not in known
            and not any(re.match(rf"^{p}_\d+$", k) for p, _ in numbered_groups)
            and values[k].strip()]
    if rest:
        out += section("ПРОЧЕЕ")
        for k in sorted(rest):
            out.append(f"{k}={values[k]}")

    path.write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")


def read_env_file(path: Path = ENV_PATH) -> dict[str, str]:
    """Публичная обёртка над разбором .env — для админки."""
    return _read_env_file(path)


def _get(env: dict[str, str], key: str, default: Optional[str] = None) -> Optional[str]:
    """Переменная окружения имеет приоритет над .env — так удобно переопределять на поде
    одной командой, не редактируя файл."""
    return os.getenv(key) or env.get(key) or default


def output_root(env: Optional[dict[str, str]] = None) -> Path:
    """Куда складывать любые результаты. Единственный источник правды по этому пути.

    Раньше каждый скрипт заводил свой, и результаты расползались по трём каталогам.
    Не задано — runs/ внутри проекта.
    """
    env = env if env is not None else _read_env_file()
    raw = _get(env, "OUTPUT_DIR")
    return Path(raw).expanduser() if raw else ENV_PATH.parent / "runs"


def load_character(env: Optional[dict[str, str]] = None) -> dict:
    env = env if env is not None else _read_env_file()
    character = {
        "name": _get(env, "CHARACTER_NAME", DEFAULT_CHARACTER["name"]),
        "age": _get(env, "CHARACTER_AGE", DEFAULT_CHARACTER["age"]),
        "ethnicity": _get(env, "CHARACTER_ETHNICITY", DEFAULT_CHARACTER["ethnicity"]),
        "hair": _get(env, "CHARACTER_HAIR", DEFAULT_CHARACTER["hair"]),
        "eyes": _get(env, "CHARACTER_EYES", DEFAULT_CHARACTER["eyes"]),
        "distinguishing_features": _get(env, "CHARACTER_FEATURES",
                                        DEFAULT_CHARACTER["distinguishing_features"]),
        "build": _get(env, "CHARACTER_BUILD", DEFAULT_CHARACTER["build"]),
    }

    # Возраст участвует в возрастном гейте safety, поэтому обязан быть числом — падаем сразу с
    # понятным текстом, а не через двадцать минут генерации на форматировании промпта.
    try:
        character["age"] = int(str(character["age"]).strip())
    except ValueError:
        raise ValueError(
            f"CHARACTER_AGE должен быть числом, получено: {character['age']!r}. "
            "Это значение сверяется с порогом возрастного гейта в safety/moderation_pipeline.py."
        )
    return character


def _numbered(env: dict[str, str], prefix: str) -> list[str]:
    """Собирает PREFIX_01, PREFIX_02, ... до первого пропуска. Пропуск — это почти всегда опечатка
    в нумерации, и молча проглотить его хуже, чем остановиться: иначе часть сцен просто исчезнет
    из пака, и заметишь ты это уже по числу файлов на выходе."""
    items: list[str] = []
    index = 1
    while True:
        value = _get(env, f"{prefix}_{index:02d}") or _get(env, f"{prefix}_{index}")
        if not value:
            break
        items.append(value.strip())
        index += 1

    stray = [k for k in list(env) + list(os.environ)
             if re.match(rf"^{prefix}_\d+$", k) and int(k.split("_")[-1]) > len(items)]
    if stray:
        raise ValueError(
            f"в нумерации {prefix}_* пропуск: собрано {len(items)} подряд, "
            f"но дальше найдены {sorted(set(stray))}. Нумерация должна идти без дыр, с 1."
        )
    return items


def load_scenes(env: Optional[dict[str, str]] = None) -> tuple[list[str], str]:
    """Возвращает (сцены, источник). Источник нужен, чтобы в логе прогона было видно, откуда
    взялись промпты — из .env или дефолтные; иначе при разборе результатов не понять, что
    именно генерировалось."""
    env = env if env is not None else _read_env_file()
    scenes = _numbered(env, "SCENE")
    if scenes:
        return scenes, f".env (SCENE_01..SCENE_{len(scenes):02d})"
    return list(DEFAULT_SCENES), "встроенные значения по умолчанию"


def load_numbered(prefix: str, env: Optional[dict[str, str]] = None) -> list[str]:
    """Любой нумерованный список из .env: POSE_01, SCENE_01, RESTYLE_01 и так далее.
    Публичная обёртка — чтобы новый вид списка не требовал новой функции здесь."""
    env = env if env is not None else _read_env_file()
    return _numbered(env, prefix)


def load_restyle_prompts(env: Optional[dict[str, str]] = None) -> list[str]:
    """Образы для перегенерации готового кадра — шаг 3."""
    env = env if env is not None else _read_env_file()
    return _numbered(env, "RESTYLE")


def style_suffix(env: Optional[dict[str, str]] = None) -> str:
    env = env if env is not None else _read_env_file()
    return _get(env, "STYLE_SUFFIX", DEFAULT_STYLE_SUFFIX)


def restyle_style_suffix(env: Optional[dict[str, str]] = None) -> str:
    """Стилевой хвост шага 3, если он отличается от общего.

    Пак снимается под репортажный реализм, образы могут — под кино. Один общий хвост тогда воюет
    сам с собой: «candid, slight grain» и «cinematic» гасят друг друга.
    Не задано — берётся общий STYLE_SUFFIX.
    """
    env = env if env is not None else _read_env_file()
    return _get(env, "RESTYLE_STYLE_SUFFIX") or style_suffix(env)


def nsfw_threshold(env: Optional[dict[str, str]] = None) -> float:
    """Порог NSFW-гейта из .env.

    Настройка фильтра, а не генерации: содержимое кадра задаёт промпт, порог решает только,
    браковать ли готовый кадр. Поднять его — значит пропускать больше, а не рисовать другое.

    Классификатор бинарный и на fashion-съёмке ошибается: плечи, спина, купальник уверенно уходят
    в «откровенное». Отсюда дефолт 0.85 вместо 0.5. Значение 1.0 пропускает всё, но score всё
    равно считается и пишется в лог по каждому кадру.
    """
    env = env if env is not None else _read_env_file()
    raw = _get(env, "NSFW_THRESHOLD", "0.85")
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"NSFW_THRESHOLD должен быть числом от 0 до 1, получено: {raw!r}")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"NSFW_THRESHOLD вне диапазона 0..1: {value}")
    return value


def min_allowed_age(env: Optional[dict[str, str]] = None) -> float:
    """Порог возрастного гейта из .env.

    Оценка по лицу, а не документ, и оценщик шумный: на одном лице давал от 26 до 63 в
    зависимости от света и ракурса. Значение выше 18 — запас против этого шума.
    """
    env = env if env is not None else _read_env_file()
    raw = _get(env, "MIN_ALLOWED_AGE", "18")
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"MIN_ALLOWED_AGE должен быть числом, получено: {raw!r}")
    if value < 18:
        raise ValueError(
            f"MIN_ALLOWED_AGE={value}: ниже 18 не поддерживается. "
            "Возрастной гейт существует именно для этой границы."
        )
    return value


def identity_lock(env: Optional[dict[str, str]] = None) -> str:
    """Блок фиксации лица для промптов, где персонаж должен остаться тем же.

    Отдельно от STYLE_SUFFIX: тот про то, как снято, этот — про то, кто в кадре.

    Блок явно разрешает менять мимику: на длинном списке запретов «не меняй лицо» модель
    копирует и выражение, и весь пак выходит с одним положением губ.
    """
    env = env if env is not None else _read_env_file()
    return _get(env, "IDENTITY_LOCK", DEFAULT_IDENTITY_LOCK)


def expression_for(index: int, env: Optional[dict[str, str]] = None) -> str:
    """Выражение лица для кадра с номером index.

    Нужно, чтобы пак не выходил с одинаковой мимикой. Список циклический: даже когда сцен больше,
    чем вариантов, соседние кадры гарантированно отличаются.
    """
    env = env if env is not None else _read_env_file()
    custom = _numbered(env, "EXPRESSION")
    variants = custom or DEFAULT_EXPRESSIONS
    return variants[(index - 1) % len(variants)]


def negative_prompt_override(env: Optional[dict[str, str]] = None) -> Optional[str]:
    env = env if env is not None else _read_env_file()
    return _get(env, "NEGATIVE_PROMPT")


def compose(scene: str, suffix: str) -> str:
    """Сцена + стилевой хвост. Хвост не дублируется, если он уже вписан в саму сцену вручную."""
    scene = scene.strip().rstrip(",")
    if not suffix or suffix.lower() in scene.lower():
        return scene
    return f"{scene}, {suffix}"


def describe(env: Optional[dict[str, str]] = None) -> str:
    """Человекочитаемая сводка активной конфигурации — печатается в начале прогона, чтобы в логе
    было зафиксировано, чем именно генерировали."""
    env = env if env is not None else _read_env_file()
    character = load_character(env)
    scenes, source = load_scenes(env)
    restyle = load_restyle_prompts(env)

    lines = [
        f"персонаж: {character['name']}, {character['age']} лет, {character['ethnicity']}",
        f"  внешность: {character['hair']}, {character['eyes']}, "
        f"{character['distinguishing_features']}, {character['build']}",
        f"сцен: {len(scenes)} (источник: {source})",
        f"стилевой хвост: {style_suffix(env)}",
    ]
    if restyle:
        lines.append(f"образов для restyle: {len(restyle)}")
    if negative_prompt_override(env):
        lines.append("негативный промпт: переопределён в .env")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
    print()
    scenes, _ = load_scenes()
    suffix = style_suffix()
    for i, scene in enumerate(scenes, start=1):
        print(f"[{i:02d}] {compose(scene, suffix)}\n")
