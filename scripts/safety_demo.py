"""Прогон реального промпта и готового кадра через safety-слой.

Про safety легко написать, что он есть, и невозможно проверить по тексту. Здесь берутся те же
промпты, что уходят в генерацию, и те же файлы, что пайплайн сложил в папку результатов.
Вывод скрипта и есть доказательство: цифры получены на месте.

  1. промпт до генерации — запрещённая тематика и числовой возраст ниже порога;
  2. готовый кадр — NSFW-классификатор;
  3. готовый кадр — возраст по лицу;
  4. готовый кадр — сверка с watchlist, на живом примере.

    python scripts/safety_demo.py
    python scripts/safety_demo.py --image <файл> --out deliverables/SAFETY_DEMO_OUTPUT.txt
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Загрузчики моделей пишут в stderr столько, что содержательный вывод в нём тонет: в прошлой
# версии этого отчёта половину файла занимали строки про CUDAExecutionProvider.
warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

RULE = "=" * 78


def _say(out: list[str], line: str = "") -> None:
    print(line)
    out.append(line)


def _pick_images(explicit: Path | None) -> list[tuple[str, Path]]:
    """Картинки для проверки: одна с наготой и одна портретная.

    Берутся из последнего прогона, а не из фикстур: смысл демонстрации в том, что гейт видит
    именно тот материал, который пайплайн реально выдаёт.
    """
    if explicit:
        return [("указанный файл", explicit)]

    from pipeline import prompt_config

    root = prompt_config.output_root()
    chosen: list[tuple[str, Path]] = []

    runs = sorted((p for p in root.glob("*") if p.is_dir()),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    for run in runs:
        for folder, label in (("03_restyle", "кадр из образов"), ("frames", "кадр фотопака")):
            files = sorted((run / folder).glob("*.png")) if (run / folder).exists() else []
            if files and not any(label == c[0] for c in chosen):
                chosen.append((label, files[0]))
        for folder, label in (("01_pack", "эталонный портрет"),):
            files = sorted((run / folder).glob("*.png")) if (run / folder).exists() else []
            if files and not any(label == c[0] for c in chosen):
                chosen.append((label, files[0]))
        if len(chosen) >= 2:
            break
    return chosen[:3]


def demo_prompt_gate(out: list[str]) -> None:
    from pipeline import prompt_config
    from safety.moderation_pipeline import PromptGate

    env = prompt_config._read_env_file()
    gate = PromptGate(min_allowed_age=int(prompt_config.min_allowed_age(env)))

    _say(out, RULE)
    _say(out, "1. ГЕЙТ ПРОМПТА — до генерации, до первой потраченной GPU-секунды")
    _say(out, RULE)
    _say(out)
    _say(out, f"Порог возраста из админки (MIN_ALLOWED_AGE): {gate.min_allowed_age}")
    _say(out, f"Паттернов в деньлисте: {len(gate._compiled)}")
    _say(out)

    real = prompt_config.load_restyle_prompts(env)[:1] + prompt_config.load_scenes(env)[0][:1]
    _say(out, "-- Реальные промпты админки, которые уходят в генерацию --")
    for prompt in real:
        result = gate.check(prompt)
        verdict = "ПРОПУЩЕН" if result.allowed else "ЗАБЛОКИРОВАН"
        _say(out, f"  [{verdict}] {prompt[:96]}…")
        if not result.allowed:
            _say(out, f"            сработало: {result.matched_patterns}")
    _say(out)

    _say(out, "-- Заведомо запрещённые формулировки --")
    for prompt in (
        "16 years old girl in school uniform",
        "teen model on the beach, nude",
        "photo of a 12 year old child",
        "nude woman, 30 years old, natural light",
        "underage looking petite girl",
    ):
        result = gate.check(prompt)
        verdict = "ПРОПУЩЕН" if result.allowed else "ЗАБЛОКИРОВАН"
        _say(out, f"  [{verdict}] {prompt}")
        if not result.allowed:
            _say(out, f"            сработало: {result.matched_patterns}")
    _say(out)
    _say(out, "Обратите внимание на предпоследнюю строку: явная нагота у взрослой женщины гейт")
    _say(out, "НЕ блокирует — это целевой контент проекта. Блокируется несовершеннолетие, и")
    _say(out, "отдельно — числовой возраст ниже порога, даже если остальной текст безобиден.")
    _say(out)


def demo_image_gates(out: list[str], images: list[tuple[str, Path]]) -> None:
    from PIL import Image

    from pipeline import prompt_config
    from safety.moderation_pipeline import ImageSafetyGate, IdentitySafetyGate

    env = prompt_config._read_env_file()
    threshold = prompt_config.nsfw_threshold(env)
    min_age = prompt_config.min_allowed_age(env)

    _say(out, RULE)
    _say(out, "2. ГЕЙТ ИЗОБРАЖЕНИЯ — NSFW-классификатор поверх готового кадра")
    _say(out, RULE)
    _say(out)
    _say(out, "Модель: Falconsai/nsfw_image_detection (ViT, apache-2.0), self-hosted.")
    _say(out, f"Порог из админки (NSFW_THRESHOLD): {threshold}")
    _say(out, "Порог 1.0 означает «пропускать всё» — оценка при этом всё равно считается и")
    _say(out, "попадает в отчёт по кадру. Это осознанный выбор: взрослый контент здесь целевой,")
    _say(out, "а гейт нужен как измеритель и как ручка, а не как запрет.")
    _say(out)

    image_gate = ImageSafetyGate(nsfw_threshold=threshold)
    identity_gate = IdentitySafetyGate(min_allowed_age=min_age)

    loaded: list[tuple[str, Path, Image.Image]] = []
    for label, path in images:
        if not path.exists():
            _say(out, f"  {label}: файл не найден — {path}")
            continue
        loaded.append((label, path, Image.open(path).convert("RGB")))

    for label, path, image in loaded:
        result = image_gate.check(image)
        verdict = "пропущен" if result.allowed else "ЗАБЛОКИРОВАН"
        _say(out, f"  {label} ({path.name}, {image.width}x{image.height})")
        _say(out, f"      nsfw_score = {result.score:.4f}   метка «{result.label}»   -> {verdict}")
    _say(out)

    _say(out, RULE)
    _say(out, "3. ВОЗРАСТ ПО ЛИЦУ — оценка модели, а не заявленный возраст персонажа")
    _say(out, RULE)
    _say(out)
    _say(out, "Модель: InsightFace buffalo_l (genderage.onnx). Тот же экземпляр, что считает")
    _say(out, "ArcFace-эмбеддинг для фиксации личности, — одна загрузка на процесс, не две.")
    _say(out, f"Порог из админки (MIN_ALLOWED_AGE): {min_age:.0f}")
    _say(out)

    for label, path, image in loaded:
        result = identity_gate.check(image)
        age = result.estimated_age
        verdict = "пропущен" if result.allowed else "ЗАБЛОКИРОВАН"
        shown = f"{age:.1f}" if age is not None else "лицо не найдено"
        _say(out, f"  {label:22} возраст ≈ {shown:>16}   -> {verdict}")
        if not result.allowed:
            _say(out, f"      причина: {result.reason}")
    _say(out)
    _say(out, "Кадр без распознанного лица гейт возрастом не пропускает молча: результат")
    _say(out, "помечается отдельно, и такой кадр не может «проскочить» просто потому, что")
    _say(out, "детектор не сработал.")
    _say(out)


def demo_watchlist(out: list[str], images: list[tuple[str, Path]]) -> None:
    """Сверка с галереей реальных лиц — на живом примере, а не на словах."""
    import shutil
    import tempfile

    from PIL import Image

    from safety.moderation_pipeline import IdentitySafetyGate

    _say(out, RULE)
    _say(out, "4. WATCHLIST — защита от сходства с реальным человеком")
    _say(out, RULE)
    _say(out)
    _say(out, "Механика: ArcFace-эмбеддинг готового кадра сверяется с эмбеддингами галереи фото")
    _say(out, "людей, на которых персонаж быть похож не должен. Порог 0.5 по косинусу — с запасом:")
    _say(out, "одно и то же лицо ArcFace обычно даёт выше 0.6, чужие — в этом проекте измерено до")
    _say(out, "0.387 (scripts/calibrate_similarity.py). Лучше лишний раз отклонить и посмотреть")
    _say(out, "руками, чем пропустить.")
    _say(out)

    if len(images) < 2:
        _say(out, "  для демонстрации нужны две разные картинки — пропускаю")
        _say(out)
        return

    (_, watch_path), (probe_label, probe_path) = images[0], images[1]
    tmp = Path(tempfile.mkdtemp(prefix="watchlist_demo_"))
    try:
        # В галерею кладём лицо С ПЕРВОГО кадра — то есть заведомо того же персонажа. Так видно,
        # что гейт действительно ловит совпадение, а не просто всегда отвечает «чисто».
        shutil.copy(watch_path, tmp / "persona_in_watchlist.png")
        gate = IdentitySafetyGate(watchlist_dir=tmp, min_allowed_age=0)

        for label, path in (("тот же человек (есть в галерее)", watch_path),
                            (f"другой кадр того же персонажа — {probe_label}", probe_path)):
            result = gate.check(Image.open(path).convert("RGB"))
            similarity = result.max_watchlist_similarity
            shown = f"{similarity:.3f}" if similarity is not None else "лицо не найдено"
            verdict = "пропущен" if result.allowed else "ЗАБЛОКИРОВАН"
            _say(out, f"  {label}")
            _say(out, f"      max cosine к галерее = {shown}   совпадение: "
                      f"{result.matched_watchlist_name or '—'}   -> {verdict}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    _say(out)
    _say(out, "В боевом режиме галерея задаётся папкой с фото реальных людей (публичные фигуры,")
    _say(out, "лица от заказчика). Эмбеддинги считаются один раз и кешируются рядом с папкой.")
    _say(out)


def demo_where_called(out: list[str]) -> None:
    _say(out, RULE)
    _say(out, "5. ГДЕ ЭТИ ГЕЙТЫ СТОЯТ В ПАЙПЛАЙНЕ")
    _say(out, RULE)
    _say(out)
    rows = [
        ("промпт", "перед каждой генерацией",
         "runpod_worker/service.py, pipeline/run_pipeline.py"),
        ("NSFW-оценка кадра", "после каждого готового кадра",
         "safety/moderation_pipeline.py::SafetyPipeline.run_image"),
        ("возраст по лицу", "там же, одним отчётом с NSFW",
         "safety/moderation_pipeline.py::IdentitySafetyGate"),
        ("watchlist", "там же", "safety/moderation_pipeline.py::IdentitySafetyGate"),
        ("кадры видео", "выборкой по ролику, а не только первый кадр",
         "ImageSafetyGate.check_video_frames"),
    ]
    width = max(len(r[0]) for r in rows)
    for name, when, where in rows:
        _say(out, f"  {name:<{width}}  {when}")
        _say(out, f"  {'':<{width}}  {where}")
    _say(out)
    _say(out, "Оценка каждого кадра пишется в run.json прогона (nsfw_score, estimated_age,")
    _say(out, "safety_allowed), то есть решение гейта можно проверить постфактум по любому кадру.")
    _say(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=None,
                        help="конкретная картинка вместо кадров последнего прогона")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "deliverables" / "SAFETY_DEMO_OUTPUT.txt")
    parser.add_argument("--skip-images", action="store_true",
                        help="только текстовый гейт: без загрузки тяжёлых моделей")
    args = parser.parse_args()

    out: list[str] = []
    _say(out, "ДЕМОНСТРАЦИЯ SAFETY-СЛОЯ")
    _say(out, "Все цифры ниже получены этим запуском на реальных промптах и реальных кадрах.")
    _say(out)

    demo_prompt_gate(out)

    if not args.skip_images:
        images = _pick_images(args.image)
        if not images:
            _say(out, "готовых кадров не нашлось — проверка по картинкам пропущена")
        else:
            demo_image_gates(out, images)
            demo_watchlist(out, images)

    demo_where_called(out)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"\nсохранено: {args.out}")


if __name__ == "__main__":
    main()
