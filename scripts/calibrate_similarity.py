"""Калибровка шкалы cosine similarity на конкретной модели.

Генерирует портреты заведомо других людей (без identity-conditioning) и меряет их похожесть на
эталон. Полученный диапазон и есть база сравнения: всё, что заметно выше, — тот же человек.

Прогон на RTX 4090, buffalo_l/w600k_r50:

    эталон сам к себе:   1.000
    чужие люди:          min 0.072  avg 0.214  max 0.387
    наши кадры:          min 0.506  avg 0.576  max 0.662
    зазор:               +0.120

Портрет, на котором детектор не нашёл лица, из статистики выпадает — это штатно.

    python scripts/calibrate_similarity.py --reference <эталон> --genuine-dir <кадры> \
        --output-dir output/impostors
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from safety.moderation_pipeline import SafetyPipeline  # noqa: E402
from worker.identity import detect_primary_face  # noqa: E402
from worker.models import PhotoGenerationRequest  # noqa: E402
from worker.photo_generation import ModelManager, PhotoGenerator  # noqa: E402

# Заведомо другие люди: другой пол, возраст, этничность, цвет волос/глаз, форма лица. Никакого
# identity-conditioning — чистый txt2img тем же чекпойнтом, что и эталон, чтобы разница в метрике
# объяснялась личностью, а не сменой модели или стиля рендера.
IMPOSTOR_PROMPTS = [
    "professional studio portrait photo of a 45-year-old West African man with a shaved head, "
    "dark brown eyes, broad face, neutral expression, soft studio lighting, photorealistic",
    "professional studio portrait photo of a 22-year-old East Asian woman, straight black hair, "
    "dark eyes, round face, neutral expression, soft studio lighting, photorealistic",
    "professional studio portrait photo of a 60-year-old Scandinavian man with white beard, "
    "blue eyes, weathered face, neutral expression, soft studio lighting, photorealistic",
    "professional studio portrait photo of a 35-year-old Latin American woman, curly red hair, "
    "green eyes, freckles, neutral expression, soft studio lighting, photorealistic",
    "professional studio portrait photo of a 28-year-old South Asian man, black wavy hair, "
    "dark eyes, thin face, neutral expression, soft studio lighting, photorealistic",
]


def cosine(a, b) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b)))


def embedding_of(path, model_name: str = "buffalo_l"):
    face = detect_primary_face(path, model_name)
    return None if face is None else face.embedding


def _summary(name: str, scores: list[float]) -> None:
    if not scores:
        print(f"{name}: нет измерений")
        return
    print(f"{name}: min={min(scores):.3f} avg={sum(scores) / len(scores):.3f} "
          f"max={max(scores):.3f} n={len(scores)}")


def run(*, reference: Path, genuine_dir: Path | None, output_dir: Path, model_name: str) -> None:
    ref = embedding_of(reference)
    if ref is None:
        raise SystemExit(f"нет лица на эталоне {reference}")
    print(f"self (эталон vs он же): {cosine(ref, ref):.3f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    generator = PhotoGenerator(ModelManager(), SafetyPipeline(), output_dir=output_dir)

    impostor_scores: list[float] = []
    for i, prompt in enumerate(IMPOSTOR_PROMPTS, start=1):
        result = generator.generate(PhotoGenerationRequest(
            model_name=model_name, prompt=prompt, seed=770000 + i * 137, width=1024, height=1024,
        ))
        emb = embedding_of(result.image_path)
        if emb is None:
            print(f"impostor {i}: лицо не найдено, пропуск")
            continue
        score = cosine(ref, emb)
        impostor_scores.append(score)
        print(f"impostor {i} (другой человек): {score:.3f}")

    genuine_scores: list[float] = []
    if genuine_dir is not None:
        for path in sorted(genuine_dir.glob("*.png")):
            if path.resolve() == reference.resolve():
                continue
            emb = embedding_of(path)
            if emb is not None:
                genuine_scores.append(cosine(ref, emb))

    print()
    _summary("IMPOSTOR (разные люди) ", impostor_scores)
    _summary("GENUINE  (наши кадры)  ", genuine_scores)
    if impostor_scores and genuine_scores:
        print(f"\nЗазор между худшим своим и лучшим чужим: "
              f"{min(genuine_scores) - max(impostor_scores):+.3f}")


def _cli():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--genuine-dir", type=Path, default=None,
                        help="папка с финальными кадрами персонажа для сравнения диапазонов")
    parser.add_argument("--output-dir", type=Path, default=Path("./output/impostors"))
    parser.add_argument("--photo-model", type=str, default="sdxl-base")
    args = parser.parse_args()
    run(reference=args.reference, genuine_dir=args.genuine_dir,
        output_dir=args.output_dir, model_name=args.photo_model)


if __name__ == "__main__":
    _cli()
