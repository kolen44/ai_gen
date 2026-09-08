"""Покадровая анимация: движение с драйвер-ролика, персонаж наш.

Каждый кадр генерируется отдельно: на вход идут референсы персонажа плюс кадр драйвера, в промпте
написано «та же женщина, поза как на последнем изображении». Дальше кадры собираются в видео.

Против Wan Animate: тот же смысл, но без 82 GB весов и без пода. Хуже временная связность —
видеомодель генерирует последовательность целиком, а здесь каждый кадр независим, и фон с
волосами могут дрожать.

Что с этим делаем:
  1. предыдущий кадр подаётся референсом в следующий;
  2. описание сцены неизменно, меняется только поза;
  3. низкий fps плюс интерполяция при сборке — дрожание размазывается, кадров нужно втрое меньше.

Стоимость линейна по кадрам, поэтому сначала --limit 6.

    python pipeline/seedream_animate.py --references <кадры> --driver <ролик> \
        --output-dir ./output/sd_animate --frames 24 --fps 8 --limit 6
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import prompt_config  # noqa: E402
from pipeline.seedream_client import Seedream, SeedreamError  # noqa: E402

# Промпт кадра. Порядок частей важен: сначала кто, потом поза, потом неизменная сцена.
# Слово "last image" указывает на кадр драйвера — он всегда идёт последним в списке референсов.
FRAME_PROMPT = (
    "The same woman from the first reference images, preserving her exact face, hair and body. "
    "She is in exactly the same body pose, position and camera angle as in the LAST reference "
    "image, but it is her, not the person from that image. {scene} "
    "Consistent clothing and background across the sequence. Photorealistic, shot on iPhone, "
    "natural skin texture."
)

DEFAULT_SCENE = "Outdoors on a bright dry salt flat under a clear blue sky, midday sun."


def _cosine(a, b) -> float:
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    return float(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b)))


def sample_driver_frames(driver: Path, count: int, out_dir: Path) -> list[Path]:
    """Равномерно выбирает кадры из драйвера и складывает на диск.

    Читаем потоково: ролик 1080p на 400+ кадров целиком в память не помещается (проверено —
    MemoryError в декодере).
    """
    import imageio.v3 as iio
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for _ in iio.imiter(driver, plugin="pyav"):
        total += 1
    if total == 0:
        raise SystemExit(f"не удалось прочитать кадры из {driver}")

    wanted = {int(round(i * (total - 1) / max(1, count - 1))) for i in range(count)}
    saved: list[Path] = []
    for index, frame in enumerate(iio.imiter(driver, plugin="pyav")):
        if index in wanted:
            im = Image.fromarray(frame)
            im.thumbnail((1280, 1280), Image.LANCZOS)
            path = out_dir / f"driver_{len(saved):03d}.jpg"
            im.save(path, quality=92)
            saved.append(path)
        del frame

    print(f"[animate] драйвер: {total} кадров всего, взято {len(saved)}")
    return saved


def run(
    *,
    references: list[Path],
    driver: Path,
    output_dir: Path,
    frames: int,
    fps: int,
    limit: Optional[int],
    scene: str,
    image_size: str,
    chain_previous: bool,
) -> None:
    from PIL import Image

    from safety.moderation_pipeline import SafetyPipeline
    from worker.identity import detect_primary_face

    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    driver_frames = sample_driver_frames(driver, frames, output_dir / "driver_frames")
    if limit:
        driver_frames = driver_frames[:limit]
        print(f"[animate] пробный режим: только {len(driver_frames)} кадров")

    ref_face = detect_primary_face(references[0])
    if ref_face is None:
        raise SystemExit(f"на референсе не найдено лицо: {references[0]}")

    client = Seedream()
    safety = SafetyPipeline()
    prompt = FRAME_PROMPT.format(scene=scene)

    check = safety.run_prompt(prompt)
    if not check.allowed:
        raise SystemExit(f"промпт заблокирован фильтром: {check.matched_patterns}")

    generated: list[Path] = []
    sims: list[float] = []
    previous: Optional[Path] = None
    started = time.monotonic()

    for i, driver_frame in enumerate(driver_frames, start=1):
        # Порядок референсов: персонаж, затем предыдущий готовый кадр (для связности),
        # затем кадр драйвера — он последний, и промпт ссылается именно на последний.
        refs: list[Path] = list(references)
        if chain_previous and previous is not None:
            refs.append(previous)
        refs.append(driver_frame)

        out = frames_dir / f"frame_{i:03d}.png"
        try:
            res = client.edit(prompt, references=refs, output_path=out, image_size=image_size)
        except SeedreamError as e:
            print(f"      [{i:03d}] ошибка: {e}")
            continue

        face = detect_primary_face(out)
        sim = _cosine(ref_face.embedding, face.embedding) if face else None
        if sim is not None:
            sims.append(sim)
        generated.append(out)
        previous = out

        print(f"      [{i:03d}/{len(driver_frames)}] {res.elapsed_s:.0f}s "
              f"cos={'—' if sim is None else f'{sim:.3f}'}")

    if not generated:
        raise SystemExit("не сгенерировано ни одного кадра")

    # Полоса кадров — быстрее понять, есть ли дрожание, чем открывать видео.
    strip_src = generated[:6]
    h = 260
    ims = [Image.open(p).convert("RGB") for p in strip_src]
    ims = [im.resize((int(im.width * h / im.height), h), Image.LANCZOS) for im in ims]
    strip = Image.new("RGB", (sum(i.width for i in ims), h), "white")
    x = 0
    for im in ims:
        strip.paste(im, (x, 0))
        x += im.width
    strip.save(output_dir / "frames_strip.png")

    video_path = output_dir / "animate_seedream.mp4"
    _assemble(generated, video_path, fps=fps)

    elapsed = (time.monotonic() - started) / 60
    print(f"\n[animate] кадров: {len(generated)}, время: {elapsed:.1f} мин")
    if sims:
        print(f"[animate] identity: min={min(sims):.3f} avg={sum(sims)/len(sims):.3f} "
              f"max={max(sims):.3f}")
    print(f"[animate] видео: {video_path}")
    print(f"[animate] полоса кадров: {output_dir / 'frames_strip.png'}")


def _assemble(frames: list[Path], out_path: Path, *, fps: int) -> None:
    """Собирает кадры в mp4. quality=9 обязателен: дефолтные настройки кодека дают ~400 kbps
    и съедают всю детализацию, ради которой всё и делалось."""
    import imageio.v3 as iio
    from PIL import Image

    # Приводим к одному размеру: Seedream может вернуть кадры разного разрешения, а кодек
    # на разнородных кадрах падает.
    first = Image.open(frames[0]).convert("RGB")
    size = first.size
    arrays = []
    for p in frames:
        im = Image.open(p).convert("RGB")
        if im.size != size:
            im = im.resize(size, Image.LANCZOS)
        arrays.append(np.asarray(im))

    iio.imwrite(out_path, np.stack(arrays), fps=fps, quality=9, codec="libx264")


def _cli():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--references", type=Path, nargs="+", required=True,
                        help="кадры персонажа, 2-3 достаточно")
    parser.add_argument("--driver", type=Path, required=True, help="ролик-источник движения")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=24, help="сколько кадров взять из драйвера")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None,
                        help="сгенерировать только первые N кадров — для пробы перед полным прогоном")
    parser.add_argument("--scene", type=str, default=DEFAULT_SCENE,
                        help="неизменное описание сцены; одинаково во всех кадрах")
    parser.add_argument("--image-size", type=str, default="square_hd")
    parser.add_argument("--no-chain", action="store_true",
                        help="не подавать предыдущий кадр референсом (быстрее, но дрожание сильнее)")
    args = parser.parse_args()

    run(references=list(args.references), driver=args.driver, output_dir=args.output_dir,
        frames=args.frames, fps=args.fps, limit=args.limit, scene=args.scene,
        image_size=args.image_size, chain_previous=not args.no_chain)


if __name__ == "__main__":
    _cli()
