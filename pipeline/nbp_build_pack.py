"""Сборка пака эталонных образов через Nano Banana Pro, цепочкой.

Первый кадр из текстового описания, дальше каждый следующий генерируется с уже накопленными
кадрами как референсами. У NBP нет FaceID и вообще адаптеров — референсы на входе единственный
механизм удержания лица, и цепочка есть способ им пользоваться.

Каждый кадр меряется по косинусу к первому, и в референсы следующих шагов идут только принятые:
иначе одна неудачная генерация протаскивает дрейф дальше и пак уезжает в другого человека.

Макрошоты стоят первыми: чем крупнее лицо, тем больше информации о чертах попадает в пак.

Нужен GEMINI_API_KEY с биллингом, на free tier квота на image-модели нулевая. ~$0.134 за кадр.

    python pipeline/nbp_build_pack.py --output-dir ./dataset/nova_nbp --target 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import prompt_config  # noqa: E402
from pipeline.nbp_client import MAX_REFERENCE_IMAGES, NanoBananaPro, NBPError  # noqa: E402
from safety.moderation_pipeline import SafetyPipeline  # noqa: E402
from worker.identity import detect_primary_face  # noqa: E402

# Порог отбора. Откалиброван на портретах заведомо чужих людей: они дают до 0.387
# (scripts/calibrate_similarity.py), поэтому 0.55 — уверенный запас, но не настолько строгий,
# чтобы выбрасывать нормальные кадры с другим ракурсом.
DEFAULT_MIN_SIMILARITY = 0.55

# Первый кадр — фронтальный крупный план: с него снимается эталонный эмбеддинг, и если он выйдет
# неудачным, кривым окажется весь пак. Поэтому он генерируется без референсов и отдельно.
SEED_PROMPT = (
    "A candid close-up photo of a {age}-year-old {ethnicity} woman with {hair}, {eyes}, "
    "{distinguishing_features}. {build}. She is looking straight into the camera with a relaxed "
    "natural expression. Soft window light from the side, plain interior background. "
    "Shot on an iPhone. Realistic skin with visible pores and fine texture, no beauty filter, "
    "no retouching, slight sensor grain, sharp focus on the eyes. Natural candid photo, "
    "not a studio portrait."
)

# Порядок важен: сначала макро и крупные планы (максимум информации о чертах), потом поясные,
# потом общие. К моменту общих планов в референсах уже накоплено много лица.
CHAIN_PROMPTS = [
    "Extreme close-up macro photo of the same woman's face from the reference images, keeping her "
    "exact features. Three-quarter angle, looking slightly away. Soft daylight. Visible skin pores "
    "and fine texture, shot on iPhone.",
    "Close-up photo of the same woman from the reference images, keeping her exact face. Profile "
    "view from the side, warm indoor lamp light, natural skin texture, shot on iPhone.",
    "Close-up photo of the same woman from the reference images, keeping her exact face. Head "
    "tilted slightly down, looking up at the camera, soft shadow across one cheek, shot on iPhone.",
    "Close-up photo of the same woman from the reference images, keeping her exact face. Laughing "
    "with her eyes half closed, bright outdoor daylight, natural skin texture, shot on iPhone.",
    "Close-up photo of the same woman from the reference images, keeping her exact face. Hair "
    "pulled back, plain wall behind, flat even daylight, no makeup, shot on iPhone.",
    "Close-up photo of the same woman from the reference images, keeping her exact face. Wet hair "
    "after a shower, bathroom light, no makeup, natural skin texture, shot on iPhone.",
    "Close-up photo of the same woman from the reference images, keeping her exact face. Side lit "
    "by a window at golden hour, warm rim light on her cheek, shot on iPhone.",
    "Close-up photo of the same woman from the reference images, keeping her exact face. Looking "
    "back over her shoulder at the camera, indoor daylight, shot on iPhone.",
    "Waist-up candid photo of the same woman from the reference images, keeping her exact face. "
    "Wearing a plain white t-shirt, standing against a plain wall, flat daylight, shot on iPhone.",
    "Waist-up candid photo of the same woman from the reference images, keeping her exact face. "
    "Wearing an oversized grey hoodie, sitting on a sofa, warm indoor light, shot on iPhone.",
    "Waist-up mirror selfie of the same woman from the reference images, keeping her exact face. "
    "Wearing a black tank top, bathroom mirror, warm bulb light, phone in hand, shot on iPhone.",
    "Waist-up candid photo of the same woman from the reference images, keeping her exact face. "
    "Wearing a beige knit sweater by a window, morning light, holding a mug, shot on iPhone.",
    "Waist-up photo of the same woman from the reference images, keeping her exact face. Arms "
    "crossed, slight smile, plain background, soft daylight, shot on iPhone.",
    "Medium shot candid photo of the same woman from the reference images, keeping her exact face. "
    "Wearing a denim jacket, walking outdoors on a cloudy day, looking away, shot on iPhone.",
    "Medium shot candid photo of the same woman from the reference images, keeping her exact face. "
    "Wearing a summer dress in a park, dappled sunlight through leaves, shot on iPhone.",
    "Medium shot candid photo of the same woman from the reference images, keeping her exact face. "
    "Sitting at a cafe table, chin resting on her hand, window light, shot on iPhone.",
    "Full body candid photo of the same woman from the reference images, keeping her exact face. "
    "Wearing jeans and a white t-shirt in a bright empty room, daylight from a large window, "
    "shot on iPhone.",
    "Full body candid photo of the same woman from the reference images, keeping her exact face. "
    "Wearing a coat on a city street at dusk, mixed street lighting, shot on iPhone.",
    "Close-up photo of the same woman from the reference images, keeping her exact face. Eyes "
    "closed, face turned toward the sun, warm outdoor light, shot on iPhone.",
    "Waist-up candid photo of the same woman from the reference images, keeping her exact face. "
    "Wearing a striped shirt, leaning on a balcony railing, evening light, shot on iPhone.",
]


def _cosine(a, b) -> float:
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    return float(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b)))


def _pick_references(accepted: list[Path], limit: int) -> list[Path]:
    """Какие кадры отдать модели как референсы на следующем шаге.

    Первый кадр (эталон) остаётся всегда — он якорь всей цепочки. Остальные берутся из хвоста,
    то есть самые свежие. Так пак постепенно обогащается новыми ракурсами, но не теряет привязку
    к исходному лицу.
    """
    if len(accepted) <= limit:
        return list(accepted)
    return [accepted[0]] + accepted[-(limit - 1):]


def run(
    *,
    output_dir: Path,
    target: int,
    min_similarity: float,
    image_size: str,
    max_references: int,
    model: Optional[str],
) -> None:
    character = prompt_config.load_character()
    safety = SafetyPipeline()
    client = NanoBananaPro(image_size=image_size, **({"model": model} if model else {}))

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir.parent / f"{output_dir.name}_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"движок: {client.model}, размер: {image_size}")
    print(f"цель: {target} кадров, порог отбора: {min_similarity}\n")

    # --- Кадр 0: эталон, без референсов ---
    seed_prompt = SEED_PROMPT.format(**character)
    check = safety.run_prompt(seed_prompt)
    if not check.allowed:
        raise SystemExit(f"эталонный промпт заблокирован: {check.matched_patterns}")

    # Эталон переснимаем до трёх раз: на нём держится вся цепочка, и уронить сборку из-за одной
    # неудачи (лимит запросов или кадр без распознанного лица) — самый обидный способ потерять
    # уже оплаченные генерации.
    seed_path = output_dir / "01.png"
    seed_face = None
    result = None
    for attempt in range(1, 4):
        try:
            result = client.generate(seed_prompt, output_path=seed_path, aspect_ratio="1:1")
        except NBPError as e:
            print(f"[эталон] попытка {attempt}/3 не удалась: {e}")
            continue
        seed_face = detect_primary_face(seed_path)
        if seed_face is not None:
            break
        print(f"[эталон] попытка {attempt}/3: лицо на кадре не распознано, пересниму")

    if seed_face is None or result is None:
        raise SystemExit(
            "не удалось получить эталонный кадр с распознанным лицом за три попытки. "
            "Если в логе 429 — упёрлись в лимит запросов, подожди несколько минут и запусти снова."
        )
    print(f"[эталон] {result.elapsed_s:.1f}s ~${result.cost_usd} -> {seed_path.name}")

    accepted: list[Path] = [seed_path]
    total_cost = result.cost_usd
    dropped = 0

    # --- Цепочка: каждый следующий кадр опирается на уже принятые ---
    for i, prompt in enumerate(CHAIN_PROMPTS, start=2):
        if len(accepted) >= target:
            break

        check = safety.run_prompt(prompt)
        if not check.allowed:
            print(f"[BLOCKED] {i}: {check.matched_patterns}")
            continue

        refs = _pick_references(accepted, max_references)
        tmp_path = raw_dir / f"candidate_{i:02d}.png"
        try:
            result = client.generate(prompt, output_path=tmp_path, references=refs,
                                     aspect_ratio="1:1")
        except NBPError as e:
            print(f"[ERROR] {i}: {e}")
            continue

        total_cost += result.cost_usd
        face = detect_primary_face(tmp_path)
        sim = _cosine(seed_face.embedding, face.embedding) if face else None

        from PIL import Image
        report = safety.run_image(Image.open(tmp_path).convert("RGB"))

        ok = sim is not None and sim >= min_similarity and report.allowed
        if ok:
            dst = output_dir / f"{len(accepted) + 1:02d}.png"
            tmp_path.replace(dst)
            accepted.append(dst)
            print(f"[KEEP {len(accepted):02d}] {result.elapsed_s:.1f}s cos={sim:.3f} "
                  f"refs={len(refs)} -> {dst.name}")
        else:
            dropped += 1
            if not report.allowed:
                reason = "safety"
            elif sim is None:
                reason = "лицо не найдено"
            else:
                reason = f"cos={sim:.3f} < {min_similarity}"
            print(f"[DROP] {i}: {reason}")

    print(f"\nпак: {len(accepted)} кадров принято, {dropped} отброшено")
    print(f"стоимость: ~${total_cost:.2f} за {len(accepted) + dropped} генераций")
    print(f"каталог: {output_dir}")
    if len(accepted) < 10:
        print("ВНИМАНИЕ: меньше 10 кадров — для устойчивого пака мало, "
              "снизь порог или добавь промптов в CHAIN_PROMPTS.")


def _cli():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target", type=int, default=20, help="сколько принятых кадров нужно")
    parser.add_argument("--min-similarity", type=float, default=DEFAULT_MIN_SIMILARITY)
    parser.add_argument("--image-size", type=str, default="2K", choices=["1K", "2K", "4K"])
    parser.add_argument("--max-references", type=int, default=MAX_REFERENCE_IMAGES,
                        help=f"референсов на запрос, максимум API — {MAX_REFERENCE_IMAGES}")
    parser.add_argument("--model", type=str, default=None,
                        help="переопределить модель (по умолчанию gemini-3-pro-image)")
    args = parser.parse_args()

    run(output_dir=args.output_dir, target=args.target, min_similarity=args.min_similarity,
        image_size=args.image_size, max_references=args.max_references, model=args.model)


if __name__ == "__main__":
    _cli()
