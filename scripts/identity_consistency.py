"""Числовое подтверждение, что на всех кадрах и в видео — один человек.

Cosine similarity ArcFace между эталонным лицом и каждым кадром. Эмбеддинг берётся из той же
модели (buffalo_l/w600k_r50), что даёт вектор в FaceID, — меряем ровно то пространство, в котором
личность и задавалась.

Пороги откалиброваны на этой же модели (scripts/calibrate_similarity.py), а не назначены:

    чужие люди:  min 0.072  avg 0.214  max 0.387
    наши кадры:  min 0.506  avg 0.576  max 0.662

Распределения не пересекаются, зазор +0.120, граница решения проходит по его середине:

    >= 0.50   уверенно тот же человек
    0.40-0.50 пограничная зона
    <  0.40   диапазон чужих лиц

    python scripts/identity_consistency.py --reference <эталон> --photos-dir <папка> \
        --video <ролик> --report deliverables/IDENTITY_CONSISTENCY.md
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from worker.identity import detect_primary_face, detect_primary_face_array  # noqa: E402

SAME_PERSON_THRESHOLD = 0.50
DRIFT_THRESHOLD = 0.40
# Максимум, показанный заведомо чужими лицами при калибровке — печатается в отчёте как база
# сравнения, чтобы читатель видел не абстрактный порог, а измеренный диапазон "не тот человек".
IMPOSTOR_MAX_MEASURED = 0.387

PHOTO_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
# Файлы, которые лежат в той же папке, но кадрами персонажа не являются — контрольные листы,
# полосы кадров и прочие сборки, где лиц несколько и метрика по "главному" лицу бессмысленна.
NON_PHOTO_MARKERS = ("contact_sheet", "frame_strip", "kps_check", "frame_check", "full_frame")


@dataclass
class SimilarityRow:
    source: str          # "photo" | "video_frame"
    label: str           # имя файла или номер кадра
    similarity: Optional[float]
    verdict: str


def _classify(similarity: Optional[float]) -> str:
    if similarity is None:
        return "лицо не найдено"
    if similarity >= SAME_PERSON_THRESHOLD:
        return "тот же человек"
    if similarity >= DRIFT_THRESHOLD:
        return "тот же человек, пограничная зона"
    return "в диапазоне чужих лиц"


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Эмбеддинги из InsightFace уже нормированы (normed_embedding), но нормируем повторно —
    дёшево и защищает от того, что кто-то передаст сюда ненормированный вектор."""
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


def _iter_photos(photos_dir: Path) -> Iterator[Path]:
    for path in sorted(photos_dir.iterdir()):
        if path.suffix.lower() not in PHOTO_EXTENSIONS:
            continue
        if any(marker in path.stem.lower() for marker in NON_PHOTO_MARKERS):
            continue
        yield path


def _iter_video_frames(video_path: Path, stride: int) -> Iterator[tuple[int, np.ndarray]]:
    """Кадры видео как numpy-массивы RGB. imageio идёт в зависимостях diffusers (export_to_video
    использует его же), так что отдельной установки на поде не требует; cv2 — фолбэк, если
    imageio собран без ffmpeg-плагина."""
    try:
        import imageio.v3 as iio

        for index, frame in enumerate(iio.imiter(video_path)):
            if index % stride == 0:
                yield index, np.asarray(frame)
        return
    except Exception:
        pass

    import cv2

    capture = cv2.VideoCapture(str(video_path))
    try:
        index = 0
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            if index % stride == 0:
                yield index, cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            index += 1
    finally:
        capture.release()


def _embedding_from_array(image_array: np.ndarray, model_name: str) -> Optional[np.ndarray]:
    """Тот же путь, что detect_primary_face(), но для уже загруженного в память кадра видео —
    без записи во временный файл на каждый из проверяемых кадров. Через общую функцию, чтобы
    крупные планы обрабатывались тем же фолбэком с полями, что и везде (см. worker/identity.py)."""
    face = detect_primary_face_array(image_array, model_name)
    return None if face is None else face.embedding


def run(
    *,
    reference: Path,
    photos_dir: Optional[Path],
    video: Optional[Path],
    frame_stride: int,
    report_path: Optional[Path],
    csv_path: Optional[Path],
    model_name: str,
) -> list[SimilarityRow]:
    ref_face = detect_primary_face(reference, model_name)
    if ref_face is None:
        raise RuntimeError(
            f"На эталонном изображении не найдено лицо: {reference}. "
            "Без эталона сверять не с чем — проверь путь к reference_face.png."
        )
    ref_embedding = ref_face.embedding

    rows: list[SimilarityRow] = []

    if photos_dir is not None:
        for photo_path in _iter_photos(photos_dir):
            if photo_path.resolve() == reference.resolve():
                continue
            face = detect_primary_face(photo_path, model_name)
            similarity = _cosine(ref_embedding, face.embedding) if face else None
            rows.append(SimilarityRow("photo", photo_path.name, similarity, _classify(similarity)))

    if video is not None:
        for frame_index, frame in _iter_video_frames(video, frame_stride):
            embedding = _embedding_from_array(frame, model_name)
            similarity = _cosine(ref_embedding, embedding) if embedding is not None else None
            rows.append(
                SimilarityRow("video_frame", f"frame {frame_index}", similarity, _classify(similarity))
            )

    _print_summary(rows)
    if csv_path:
        _write_csv(rows, csv_path)
    if report_path:
        _write_report(rows, report_path, reference=reference, video=video, frame_stride=frame_stride)
    return rows


def _measured(rows: list[SimilarityRow], source: str) -> list[float]:
    return [r.similarity for r in rows if r.source == source and r.similarity is not None]


def _print_summary(rows: list[SimilarityRow]) -> None:
    for row in rows:
        value = f"{row.similarity:.3f}" if row.similarity is not None else "  —  "
        print(f"[{row.source:11}] {row.label:34} cos={value}  {row.verdict}")

    for source, title in (("photo", "Фото"), ("video_frame", "Кадры видео")):
        values = _measured(rows, source)
        if not values:
            continue
        print(
            f"\n{title}: min={min(values):.3f} avg={sum(values) / len(values):.3f} "
            f"max={max(values):.3f} (n={len(values)})"
        )
        below = [r.label for r in rows if r.source == source and r.similarity is not None
                 and r.similarity < DRIFT_THRESHOLD]
        if below:
            print(f"  НИЖЕ ПОРОГА {DRIFT_THRESHOLD}: {', '.join(below)}")


def _write_csv(rows: list[SimilarityRow], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "label", "cosine_similarity", "verdict"])
        for row in rows:
            writer.writerow([
                row.source,
                row.label,
                "" if row.similarity is None else round(row.similarity, 4),
                row.verdict,
            ])
    print(f"\nCSV: {csv_path}")


def _write_report(
    rows: list[SimilarityRow],
    report_path: Path,
    *,
    reference: Path,
    video: Optional[Path],
    frame_stride: int,
) -> None:
    lines = [
        "# Консистентность лица — измерение",
        "",
        "Cosine similarity ArcFace-эмбеддингов (InsightFace `buffalo_l`, recognition-модель",
        "w600k_r50) между эталонным лицом и каждым сгенерированным кадром. Это то же",
        "эмбеддинг-пространство, в котором identity задавалась через IP-Adapter FaceID, и та же",
        "модель, что используется в safety-гейте для сверки с watchlist.",
        "",
        f"Эталон: `{reference.name}`",
        "",
        "**Шкала откалибрована на этой же модели, а не назначена.** Сгенерировано 5 портретов",
        "заведомо других людей (другой пол/возраст/этничность, без identity-conditioning) и",
        "измерена их похожесть на тот же эталон — это даёт реальный диапазон «не тот человек»:",
        "",
        f"| Группа | min | avg | max |",
        "|---|---|---|---|",
        "| чужие лица (калибровка) | 0.072 | 0.214 | **0.387** |",
        "",
        "Отсюда пороги:",
        "",
        "| Порог | Трактовка |",
        "|---|---|",
        f"| ≥ {SAME_PERSON_THRESHOLD:.2f} | уверенно тот же человек |",
        f"| {DRIFT_THRESHOLD:.2f} – {SAME_PERSON_THRESHOLD:.2f} | пограничная зона |",
        f"| < {DRIFT_THRESHOLD:.2f} | попадает в измеренный диапазон чужих лиц (max {IMPOSTOR_MAX_MEASURED}) |",
        "",
    ]

    photo_rows = [r for r in rows if r.source == "photo"]
    if photo_rows:
        lines += ["## Фото", "", "| Кадр | cosine similarity | Вывод |", "|---|---|---|"]
        for row in photo_rows:
            value = f"{row.similarity:.3f}" if row.similarity is not None else "—"
            lines.append(f"| {row.label} | {value} | {row.verdict} |")
        values = _measured(rows, "photo")
        if values:
            lines += [
                "",
                f"**min {min(values):.3f} / среднее {sum(values) / len(values):.3f} / "
                f"max {max(values):.3f}** по {len(values)} кадрам.",
                "",
            ]

    frame_rows = [r for r in rows if r.source == "video_frame"]
    if frame_rows:
        lines += [
            "## Видео",
            "",
            f"Источник: `{video.name if video else ''}`, проверялся каждый {frame_stride}-й кадр.",
            "",
            "| Кадр | cosine similarity | Вывод |",
            "|---|---|---|",
        ]
        for row in frame_rows:
            value = f"{row.similarity:.3f}" if row.similarity is not None else "—"
            lines.append(f"| {row.label} | {value} | {row.verdict} |")
        values = _measured(rows, "video_frame")
        if values:
            lines += [
                "",
                f"**min {min(values):.3f} / среднее {sum(values) / len(values):.3f} / "
                f"max {max(values):.3f}** по {len(values)} проверенным кадрам.",
                "",
                "Отсутствие лица на отдельных кадрах (`—`) — не дрейф идентичности, а кадры, где",
                "лицо ушло из поля зрения детектора (поворот головы, размытие движением); такие",
                "кадры исключаются из статистики, а не считаются провалом.",
                "",
            ]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Отчёт: {report_path}")


def _cli():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", type=Path, required=True, help="Эталонное лицо (Stage 0)")
    parser.add_argument("--photos-dir", type=Path, default=None, help="Папка с 8 фото")
    parser.add_argument("--video", type=Path, default=None, help="Видеоклип для покадровой сверки")
    parser.add_argument("--frame-stride", type=int, default=5, help="Проверять каждый N-й кадр видео")
    parser.add_argument("--report", type=Path, default=None, help="Куда записать markdown-отчёт")
    parser.add_argument("--csv", type=Path, default=None, help="Куда записать CSV")
    parser.add_argument("--model", type=str, default="buffalo_l")
    args = parser.parse_args()

    if args.photos_dir is None and args.video is None:
        parser.error("нужен хотя бы один из --photos-dir / --video")

    run(
        reference=args.reference,
        photos_dir=args.photos_dir,
        video=args.video,
        frame_stride=args.frame_stride,
        report_path=args.report,
        csv_path=args.csv,
        model_name=args.model,
    )


if __name__ == "__main__":
    _cli()
