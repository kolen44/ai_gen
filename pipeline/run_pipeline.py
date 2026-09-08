"""Облачный прогон персонажа одной командой.

  1. ЛИЦО И ПАК    — первый кадр из описания внешности, дальше цепочкой: каждый следующий кадр
                     с уже накопленными как референсами.
  2. ПОЗЫ          — тот же персонаж в позах под анимацию. Идёт только если список поз задан
                     в админке.
  3. ПЕРЕГЕНЕРАЦИЯ — образы и сцены: одежда, локация, окружение. Лицо берётся с референсов.
  4. ВИДЕО         — из лучшего кадра образов. Провайдер по имени модели: flaq или fal.
  5. ОТЧЁТ         — контрольные листы, метрики, benchmark-лог.

Движок каждого шага переключается флагом — квоты у провайдеров кончаются не вовремя.
Прогон возобновляемый: готовые результаты этап пропускает, падение на видео не означает
переделку пака за деньги.

Всё содержательное — в .env админки: внешность, позы, образы, сцены, пороги.

    python pipeline/run_pipeline.py --output-dir ./runs/nova_01
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import prompt_config  # noqa: E402

# Образец списка поз для карточки, а НЕ значение по умолчанию: при пустом списке этап поз
# пропускается целиком. Подобраны под анимацию — ясный ракурс, видимое тело.
DEFAULT_POSES = [
    "standing straight facing the camera, arms relaxed at her sides, full body in frame",
    "standing three-quarter turned, one hand on her hip, looking at the camera, full body",
    "walking toward the camera mid-step, natural arm swing, full body",
    "standing side profile, head turned toward the camera over her shoulder, full body",
    "sitting on a simple chair, hands on her knees, facing the camera, full body",
    "leaning against a plain wall, arms crossed, relaxed posture, waist-up",
]


# Движки flaq.ai: имя в пайплайне -> имя модели в их API. Держим отдельными движками, а не
# одним с параметром: в админке это выпадающий список, и человек выбирает конкретную модель,
# а не «flaq плюс где-то ещё настройка версии».
FLAQ_MODELS = {
    "flaq-5pro": "seedream-v5.0-pro",
    "flaq-5": "seedream-v5.0",
    "flaq-45": "seedream-v4.5",
    "flaq": "seedream-v5.0-pro",   # прежнее имя, чтобы старые команды не сломались
}

ENGINE_CHOICES = ["nbp", "seedream", *FLAQ_MODELS]


@dataclass
class FrameRecord:
    """Одна сгенерированная картинка со всем, что о ней известно."""
    name: str
    stage: str
    engine: str
    prompt: str
    similarity: Optional[float]
    nsfw_score: Optional[float]
    estimated_age: Optional[float]
    elapsed_s: float
    resolution: str
    accepted: bool
    reject_reason: Optional[str] = None
    # Списано провайдером за эту генерацию, в долларах. Есть только у flaq — он возвращает
    # credit в ответе. У остальных None: цену за кадр взять неоткуда, кроме счёта в кабинете.
    credit: Optional[float] = None


@dataclass
class RunState:
    started_at: str
    character: dict
    engines: dict = field(default_factory=dict)
    frames: list[dict] = field(default_factory=list)
    best_references: list[str] = field(default_factory=list)
    video: Optional[dict] = None

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")


def _cosine(a, b) -> float:
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    return float(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b)))


def _contact_sheet(paths: list[Path], out_path: Path, height: int = 320, per_row: int = 4) -> None:
    from PIL import Image

    if not paths:
        return
    ims = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        ims.append(im.resize((int(im.width * height / im.height), height), Image.LANCZOS))
    rows = [ims[i:i + per_row] for i in range(0, len(ims), per_row)]
    width = max(sum(i.width for i in r) for r in rows)
    sheet = Image.new("RGB", (width, height * len(rows)), "white")
    for ri, row in enumerate(rows):
        x = 0
        for im in row:
            sheet.paste(im, (x, ri * height))
            x += im.width
    sheet.save(out_path)


# ---------------------------------------------------------------------------
# Движки
# ---------------------------------------------------------------------------

class Engine:
    """Единый интерфейс поверх NBP и Seedream: оба умеют «сгенерируй из текста» и
    «перерисуй по референсам», но с разными сигнатурами и разными ограничениями."""

    def __init__(self, name: str, image_size: str):
        self.name = name
        self.image_size = image_size
        # Стоимость последней генерации: провайдер возвращает её в ответе, и это единственный
        # способ узнать цену кадра, не сверяясь со счётом вручную.
        self.last_credit: Optional[float] = None

        if name == "nbp":
            # Nano Banana Pro через fal — предпочтительный путь. Та же модель, что у Google,
            # но без его квоты: прямой доступ упирается в generate_content_free_tier_requests
            # с limit 0 и блокируется после нескольких запросов, а пак из 12 кадров так не собрать.
            from pipeline.nbp_fal_client import NanoBananaProFal

            self._client = NanoBananaProFal(image_size=image_size)
        elif name == "seedream":
            from pipeline.seedream_client import Seedream

            self._client = Seedream()
        elif name in FLAQ_MODELS:
            # Seedream через flaq.ai. Отличие от seedream на fal: референс тут ровно один
            # (их API принимает единственный image_url), поэтому цепочка идентичности
            # опирается на лучший кадр, а не на несколько сразу.
            from pipeline.flaq_client import Flaq

            self._client = Flaq(model=FLAQ_MODELS[name], image_size=image_size)
        else:
            raise ValueError(f"неизвестный движок: {name}")

    def text_to_image(self, prompt: str, out: Path) -> float:
        result = self._client.text_to_image(prompt, output_path=out,
                                            image_size=self.image_size)
        self.last_credit = getattr(result, "credit", None)
        return result.elapsed_s

    def edit(self, prompt: str, references: list[Path], out: Path) -> float:
        result = self._client.edit(prompt, references=references, output_path=out,
                                   image_size=self.image_size)
        self.last_credit = getattr(result, "credit", None)
        return result.elapsed_s

    @property
    def errors(self):
        if self.name == "nbp":
            from pipeline.nbp_fal_client import NBPFalError

            return NBPFalError
        if self.name in FLAQ_MODELS:
            from pipeline.flaq_client import FlaqError

            return FlaqError
        from pipeline.seedream_client import SeedreamError

        return SeedreamError


# ---------------------------------------------------------------------------
# Шаг 1: лицо и пак эталонов
# ---------------------------------------------------------------------------

def stage_pack(
    *, run_dir: Path, engine: Engine, target: int, min_similarity: float,
    max_references: int, force: bool,
) -> tuple[list[Path], object, list["FrameRecord"]]:
    from PIL import Image

    from pipeline.nbp_build_pack import CHAIN_PROMPTS, SEED_PROMPT, _pick_references
    from safety.moderation_pipeline import SafetyPipeline
    from worker.identity import detect_primary_face

    pack_dir = run_dir / "01_pack"
    rejected = run_dir / "01_pack_rejected"
    pack_dir.mkdir(parents=True, exist_ok=True)
    rejected.mkdir(parents=True, exist_ok=True)

    existing = sorted(pack_dir.glob("*.png"))
    if existing and not force:
        print(f"[1/5] пак уже собран ({len(existing)} кадров), пропускаю")
        return existing, detect_primary_face(existing[0]), []

    character = prompt_config.load_character()
    lock = prompt_config.identity_lock()
    safety = SafetyPipeline()

    print(f"[1/5] лицо и пак · движок {engine.name} · цель {target} кадров")

    seed_path = pack_dir / "01.png"
    seed_face = None
    for attempt in range(1, 4):
        try:
            engine.text_to_image(SEED_PROMPT.format(**character), seed_path)
        except engine.errors as e:
            print(f"      эталон, попытка {attempt}/3: {e}")
            continue
        seed_face = detect_primary_face(seed_path)
        if seed_face:
            break
        print(f"      эталон, попытка {attempt}/3: лицо не распознано")

    if seed_face is None:
        raise SystemExit("не удалось получить эталонный кадр с распознанным лицом")

    # Эталон меряется к самому себе (cos = 1.0 по определению) — строка в отчёте нужна не ради
    # метрики, а ради проверки изображения: без неё первый кадр пака был бы единственным,
    # который прошёл только фильтр промпта, но не классификатор картинки.
    seed_image = Image.open(seed_path).convert("RGB")
    seed_report = safety.run_image(seed_image)
    records: list[FrameRecord] = [FrameRecord(
        "01", "pack", engine.name, "эталон (SEED_PROMPT)", 1.0,
        seed_report.image_result.score if seed_report.image_result else None,
        seed_report.identity_result.estimated_age if seed_report.identity_result else None,
        0.0, f"{seed_image.width}x{seed_image.height}", seed_report.allowed,
        None if seed_report.allowed else "safety",
    )]
    if not seed_report.allowed:
        raise SystemExit("эталонный кадр не прошёл проверку безопасности")
    print(f"      эталон готов")

    accepted = [seed_path]
    for i, scene in enumerate(CHAIN_PROMPTS, start=2):
        if len(accepted) >= target:
            break
        full = f"{scene} {prompt_config.expression_for(i)}. {lock}"
        name = f"pack_{i:02d}"
        check = safety.run_prompt(full)
        if not check.allowed:
            records.append(FrameRecord(name, "pack", engine.name, scene, None, None, None,
                                       0.0, "", False,
                                       f"промпт-фильтр: {check.matched_patterns}"))
            print(f"      [{i}] заблокирован фильтром")
            continue

        tmp = rejected / f"candidate_{i:02d}.png"
        try:
            elapsed = engine.edit(full, _pick_references(accepted, max_references), tmp)
        except engine.errors as e:
            records.append(FrameRecord(name, "pack", engine.name, scene, None, None, None,
                                       0.0, "", False, str(e)[:120]))
            print(f"      [{i}] {e}")
            continue

        image = Image.open(tmp).convert("RGB")
        face = detect_primary_face(tmp)
        sim = _cosine(seed_face.embedding, face.embedding) if face else None
        report = safety.run_image(image)
        ok = report.allowed and sim is not None and sim >= min_similarity

        if ok:
            dst = pack_dir / f"{len(accepted) + 1:02d}.png"
            tmp.replace(dst)
            accepted.append(dst)
            name = dst.stem
            print(f"      [{len(accepted):02d}] cos={sim:.3f}")
        else:
            reason = ("safety" if not report.allowed
                      else "нет лица" if sim is None
                      else f"cos {sim:.3f} < {min_similarity}")
            print(f"      [пропуск] {reason}")

        records.append(FrameRecord(
            name, "pack", engine.name, scene, sim,
            report.image_result.score if report.image_result else None,
            report.identity_result.estimated_age if report.identity_result else None,
            elapsed, f"{image.width}x{image.height}", ok,
            None if ok else reason, engine.last_credit,
        ))

    _contact_sheet(accepted, run_dir / "01_pack_sheet.png")
    print(f"[1/5] пак готов: {len(accepted)} кадров")
    return accepted, seed_face, records


# ---------------------------------------------------------------------------
# Шаг 2: позы под видео
# ---------------------------------------------------------------------------

def sample_driver_poses(driver: Path, count: int, out_dir: Path) -> list[Path]:
    """Равномерно вытаскивает кадры из драйвер-ролика — это и есть источник поз.

    Читаем потоково и сразу уменьшаем: ролик 1080p на 400+ кадров целиком в память не помещается
    (проверено — MemoryError в декодере).
    """
    import imageio.v3 as iio
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("*.jpg"))
    if len(existing) >= count:
        return existing[:count]

    total = sum(1 for _ in iio.imiter(driver, plugin="pyav"))
    if total == 0:
        raise SystemExit(f"не удалось прочитать кадры из {driver}")

    wanted = {int(round(i * (total - 1) / max(1, count - 1))) for i in range(count)}
    saved: list[Path] = []
    for index, frame in enumerate(iio.imiter(driver, plugin="pyav")):
        if index in wanted:
            im = Image.fromarray(frame)
            im.thumbnail((1280, 1280), Image.LANCZOS)
            path = out_dir / f"driver_{len(saved):02d}.jpg"
            im.save(path, quality=92)
            saved.append(path)
        del frame

    print(f"      драйвер: {total} кадров всего, взято {len(saved)} как источник поз")
    return saved


def stage_poses(
    *, run_dir: Path, engine: Engine, references: list[Path], seed_face,
    force: bool, driver: Optional[Path] = None, pose_count: int = 8,
    pose_source: str = "text",
) -> list[FrameRecord]:
    from PIL import Image

    from safety.moderation_pipeline import SafetyPipeline
    from worker.identity import detect_primary_face

    poses_dir = run_dir / "02_poses"
    poses_dir.mkdir(parents=True, exist_ok=True)

    if list(poses_dir.glob("*.png")) and not force:
        print("[2/5] позы уже сгенерированы, пропускаю")
        return []

    lock = prompt_config.identity_lock()
    suffix = prompt_config.style_suffix()
    safety = SafetyPipeline()
    records: list[FrameRecord] = []

    # С драйвер-ролика позы берутся покадрово, а не придумываются из текста: описанная словами
    # поза не совпадёт с движением, которое потом анимируется.
    # ControlNet к закрытому API не подключить, поэтому кадр драйвера идёт последним в списке
    # референсов, а промпт ссылается на последнее изображение.
    driver_frames: list[Path] = []
    if pose_source == "driver" and driver and Path(driver).exists():
        driver_frames = sample_driver_poses(Path(driver), pose_count,
                                            run_dir / "02_poses_driver")
        tasks = [(f"поза с кадра {p.stem}", p) for p in driver_frames]
    else:
        # В текстовом режиме количество кадров задаёт длина списка POSE_*, а не pose_count:
        # список — это явно перечисленные позы, и молча отрезать часть означало бы, что человек
        # добавил позу в админке, а она не сгенерировалась. pose_count управляет только выборкой
        # кадров из драйвер-ролика выше, где перечислять нечего.
        poses = prompt_config.load_numbered("POSE")
        if not poses:
            # Нет списка в админке — нет этапа. Раньше подставлялся DEFAULT_POSES: шесть платных
            # генераций по тексту, которого в карточке нет и который не поправить.
            # Шаг 3 без поз работает по лучшим кадрам пака.
            print("[2/5] списка поз в админке нет — этап пропускаю")
            return []
        tasks = [(p, None) for p in poses]

    source = "кадры драйвера" if driver_frames else "текстовые описания"
    print(f"[2/5] позы · движок {engine.name} · {len(tasks)} штук · источник: {source}")

    for i, (pose, pose_frame) in enumerate(tasks, start=1):
        if pose_frame is not None:
            # Роли референсов заданы явно: иначе модель смешает лица или возьмёт внешность с
            # кадра позы, где посторонний человек.
            # Прямая формулировка «это она, а не тот человек» отклоняется контент-чекером fal как
            # замена лица, поэтому последний кадр описан как схема композиции: берём позу,
            # внешность явно перечислена как игнорируемая.
            n = len(references)
            full = (
                f"The subject is the woman shown in the first {n} reference images. "
                "Render her, keeping her exact face, hairstyle, hair colour, skin tone and build. "
                "The final image is a body-pose diagram only: copy its body posture, limb "
                "positions, body orientation, camera angle and framing. "
                "Ignore everything else about the final image — do not take its face, facial "
                "features, hair, age, body type or clothing. "
                f"{suffix} {lock}"
            )
            refs = list(references) + [pose_frame]
        else:
            full = f"The same woman from the reference images. {pose}. {suffix} {lock}"
            refs = list(references)

        check = safety.run_prompt(full)
        if not check.allowed:
            records.append(FrameRecord(f"pose_{i:02d}", "pose", engine.name, pose,
                                       None, None, None, 0.0, "", False,
                                       f"промпт-фильтр: {check.matched_patterns}"))
            print(f"      [pose_{i:02d}] заблокирован фильтром")
            continue

        out = poses_dir / f"pose_{i:02d}.png"
        try:
            elapsed = engine.edit(full, refs, out)
        except engine.errors as e:
            records.append(FrameRecord(f"pose_{i:02d}", "pose", engine.name, pose,
                                       None, None, None, 0.0, "", False, str(e)[:120]))
            print(f"      [pose_{i:02d}] {e}")
            continue

        image = Image.open(out).convert("RGB")
        face = detect_primary_face(out)
        sim = _cosine(seed_face.embedding, face.embedding) if face else None
        report = safety.run_image(image)
        ok = report.allowed and sim is not None

        records.append(FrameRecord(
            f"pose_{i:02d}", "pose", engine.name, pose, sim,
            report.image_result.score if report.image_result else None,
            report.identity_result.estimated_age if report.identity_result else None,
            elapsed, f"{image.width}x{image.height}", ok,
            None if ok else ("safety" if not report.allowed else "нет лица"),
            engine.last_credit,
        ))
        print(f"      [pose_{i:02d}] {'OK' if ok else 'отклонён'} "
              f"cos={'—' if sim is None else f'{sim:.3f}'} {elapsed:.0f}s")

    good = [poses_dir / f"{r.name}.png" for r in records if r.accepted]
    _contact_sheet(good, run_dir / "02_poses_sheet.png")
    print(f"[2/5] принято {len(good)} из {len(tasks)}")
    return records


# ---------------------------------------------------------------------------
# Шаг 3: перегенерация образов через Seedream
# ---------------------------------------------------------------------------

def stage_restyle(
    *, run_dir: Path, engine: Engine, pose_records: list[FrameRecord], references: list[Path],
    seed_face, image_size: str, force: bool,
) -> list[FrameRecord]:
    from PIL import Image

    from safety.moderation_pipeline import SafetyPipeline
    from worker.identity import detect_primary_face

    out_dir = run_dir / "03_restyle"
    out_dir.mkdir(parents=True, exist_ok=True)

    if list(out_dir.glob("*.png")) and not force:
        print("[3/5] образы уже сгенерированы, пропускаю")
        return []

    looks = prompt_config.load_restyle_prompts()
    scenes, _ = prompt_config.load_scenes()
    tasks = [("look", p) for p in looks] + [("scene", p) for p in scenes]
    if not tasks:
        print("[3/5] в .env нет ни RESTYLE_*, ни SCENE_* — пропускаю")
        return []

    # Референсами берём лучшие кадры поз, если они есть: у них уже нужный ракурс, и Seedream
    # меняет только одежду и окружение, а не композицию.
    poses_dir = run_dir / "02_poses"
    pose_refs = [poses_dir / f"{r.name}.png"
                 for r in sorted((r for r in pose_records if r.accepted),
                                 key=lambda r: -(r.similarity or 0))[:3]]
    refs = (pose_refs or []) + references[:2]

    lock = prompt_config.identity_lock()
    suffix = prompt_config.restyle_style_suffix()
    safety = SafetyPipeline()
    records: list[FrameRecord] = []

    print(f"[3/5] перегенерация · движок {engine.name} · {len(tasks)} кадров, "
          f"референсов {len(refs)}")

    for i, (kind, prompt) in enumerate(tasks, start=1):
        name = f"{kind}_{i:02d}"
        full = f"{prompt_config.compose(prompt, suffix)} {prompt_config.expression_for(i)}. {lock}"

        out = out_dir / f"{name}.png"
        try:
            elapsed = engine.edit(full, refs, out)
        except engine.errors as e:
            records.append(FrameRecord(name, "restyle", engine.name, prompt, None, None, None,
                                       0.0, "", False, str(e)[:120]))
            print(f"      [{name}] {e}")
            continue

        image = Image.open(out).convert("RGB")
        face = detect_primary_face(out)
        sim = _cosine(seed_face.embedding, face.embedding) if face else None
        report = safety.run_image(image)
        ok = report.allowed and sim is not None

        records.append(FrameRecord(
            name, "restyle", engine.name, prompt, sim,
            report.image_result.score if report.image_result else None,
            report.identity_result.estimated_age if report.identity_result else None,
            elapsed, f"{image.width}x{image.height}", ok,
            None if ok else ("safety" if not report.allowed else "нет лица"),
            engine.last_credit,
        ))
        print(f"      [{name}] {'OK' if ok else 'отклонён'} "
              f"cos={'—' if sim is None else f'{sim:.3f}'} {elapsed:.0f}s")

    good = [out_dir / f"{r.name}.png" for r in records if r.accepted]
    _contact_sheet(good, run_dir / "03_restyle_sheet.png")
    print(f"[3/5] принято {len(good)} из {len(tasks)}")
    return records


# ---------------------------------------------------------------------------
# Шаг 4: видео
# ---------------------------------------------------------------------------

def stage_video(
    *, run_dir: Path, records: list[FrameRecord], pack: list[Path], seed_face,
    model_key: str, duration: int, driver: Optional[Path], force: bool,
    video_resolution: str = "720p", video_aspect: str = "9:16",
) -> Optional[dict]:
    from pipeline.fal_video import MODELS, FalVideoError, generate, measure
    from pipeline.flaq_video import VIDEO_MODELS as FLAQ_VIDEO_MODELS

    # Провайдер — по имени модели: у flaq имена всегда с суффиксом -image-to-video.
    # Разница не в цене. Kling через fal отклоняет обнажённый кадр на входе (HTTP 422
    # content_policy_violation), flaq те же кадры принимает: замерено на выходе nsfw 0.9993,
    # лицо к эталону 0.714.
    on_flaq = model_key in FLAQ_VIDEO_MODELS

    video_dir = run_dir / "04_video"
    video_dir.mkdir(parents=True, exist_ok=True)
    out = video_dir / f"video_{model_key}.mp4"

    if out.exists() and not force:
        print("[4/5] видео уже есть, пропускаю")
        return None

    if not on_flaq and MODELS[model_key].get("needs_driver"):
        if driver is None or not Path(driver).exists():
            print(f"[4/5] {model_key} берёт движение с ролика, драйвер не задан — пропускаю")
            return None

    # Источник — кадр с максимальной метрикой среди всех сгенерированных: дрейф видеомодели
    # складывается с дрейфом исходника, поэтому каждая сотая на входе переносится в ролик.
    candidates: list[tuple[float, Path]] = []
    _candidate_stages: list[str] = []
    for r in records:
        if not r.accepted or r.similarity is None:
            continue
        _candidate_stages.append(r.stage)
        # Стадий три, а не две: pack, pose и restyle — и у каждой своя папка. Раньше здесь
        # стояло «pose или иначе restyle», из-за чего кадры пака искались в 03_restyle и шаг
        # видео падал с «файл не найден», как только лучшим кандидатом оказывался кадр пака.
        folder = {"pack": "01_pack", "pose": "02_poses"}.get(r.stage, "03_restyle")
        candidates.append((r.similarity, run_dir / folder / f"{r.name}.png"))
    if not candidates:
        from worker.identity import detect_primary_face

        for p in pack:
            face = detect_primary_face(p)
            if face:
                candidates.append((_cosine(seed_face.embedding, face.embedding), p))
    if not candidates:
        print("[4/5] нет подходящего кадра, пропускаю")
        return None

    # Сначала стадия, потом косинус. Косинус тем выше, чем крупнее лицо, поэтому портрет и поза
    # всегда обгоняют полноростовой образ: позы 0.79-0.89, пак 0.74-0.83, образы 0.65-0.76.
    # Ролик снимался с портрета, и содержание образа в видео не попадало никогда.
    # Внутри стадии образов косинус по-прежнему решает, какой кадр лучше.
    if len(candidates) == len(_candidate_stages):
        preferred = [c for c, stage in zip(candidates, _candidate_stages) if stage == "restyle"]
    else:
        preferred = []
    pool = preferred or candidates
    pool.sort(key=lambda kv: -kv[0])
    best_sim, source = pool[0]
    if preferred:
        print(f"      кадр берётся из образов ({len(preferred)} шт), а не по одному косинусу")
    print(f"[4/5] видео · {model_key} · из {source.name} (cos={best_sim:.3f})")
    if driver:
        print(f"      драйвер движения: {Path(driver).name}")

    motion = prompt_config._get(prompt_config._read_env_file(), "VIDEO_PROMPT") or (
        "She smiles softly and tilts her head slightly, hair moving gently in the breeze, "
        "natural blinking, subtle lifelike motion, handheld phone camera feel, static framing"
    )

    credit = None
    try:
        if on_flaq:
            from pipeline.flaq_client import FlaqError
            from pipeline.flaq_video import FlaqVideo

            try:
                result = FlaqVideo(model=model_key).image_to_video(
                    source, out, prompt=motion, duration=duration,
                    resolution=video_resolution, aspect_ratio=video_aspect)
            except FlaqError as e:
                print(f"[4/5] видео не сгенерировалось: {e}")
                return None
            path, elapsed, credit = result.path, result.elapsed_s, result.credit
        else:
            path, elapsed, _ = generate(model_key=model_key, image_path=source, prompt=motion,
                                        output_path=out, duration=duration, driver_path=driver)
    except FalVideoError as e:
        print(f"[4/5] видео не сгенерировалось: {e}")
        return None

    m = measure(path, source=source, reference=pack[0] if pack else None)
    print(f"      готово за {elapsed:.0f}s")
    for label, key in (("к исходнику", "vs_source"), ("к эталону", "vs_reference")):
        if m.get(key):
            s = m[key]
            print(f"      identity {label}: min={s['min']:.3f} avg={s['avg']:.3f}")
    print(f"      safety: {m['blocked']} заблокировано из {m['checked']}")

    if credit is not None:
        print(f"      списано кредитов: {credit}")

    return {"file": str(path), "source": str(source), "elapsed_s": elapsed,
            "model": model_key, "provider": "flaq" if on_flaq else "fal",
            "credit": credit, "resolution": video_resolution if on_flaq else None,
            "driver": str(driver) if driver else None, **m}


# ---------------------------------------------------------------------------
# Шаг 5: отчёт
# ---------------------------------------------------------------------------

def _saved_records(run_dir: Path, stage: str) -> list[FrameRecord]:
    """Записи пропущенного этапа из run.json прошлого запуска.

    Без них перезапуск давал шагу видео пустой список кандидатов, и тот сваливался на кадры
    пака, а отчёт выходил без единой цифры.
    """
    path = run_dir / "run.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []

    restored = []
    for row in data.get("frames", []):
        if row.get("stage") != stage:
            continue
        try:
            restored.append(FrameRecord(**row))
        except TypeError:
            # Формат записи изменился между запусками — пропускаем строку, а не падаем.
            continue
    return restored


def stage_report(*, run_dir: Path, state: RunState) -> None:
    records = [FrameRecord(**r) for r in state.frames]
    # Эталон исключён из агрегата: он меряется сам к себе и даёт 1.0 по определению,
    # то есть завышал бы среднее, ничего не сообщая о консистентности.
    sims = [r.similarity for r in records
            if r.similarity is not None and not (r.stage == "pack" and r.name == "01")]

    lines = [
        f"# Прогон {run_dir.name}",
        "",
        f"Запущен: {state.started_at}",
        "",
        "## Движки по шагам",
        "",
        f"- лицо и пак: **{state.engines.get('pack', '—')}**",
        f"- позы: **{state.engines.get('poses', '—')}**",
        f"- перегенерация образов: **{state.engines.get('restyle', '—')}**",
        f"- видео: **{state.engines.get('video', '—')}**",
        "",
        "## Персонаж",
        "",
        f"{state.character['age']} лет, {state.character['ethnicity']}, "
        f"{state.character['hair']}, {state.character['eyes']}",
        "",
        "## Консистентность",
        "",
        "Cosine similarity ArcFace к эталонному кадру. Для калибровки: портреты заведомо чужих "
        "людей на этой же модели дают до 0.387.",
        "",
    ]
    if sims:
        lines += [f"- кадров с найденным лицом: {len(sims)}",
                  f"- min {min(sims):.3f} / среднее {sum(sims)/len(sims):.3f} / max {max(sims):.3f}",
                  ""]

    lines += ["| Кадр | Шаг | Движок | cos | Время | Результат |", "|---|---|---|---|---|---|"]
    for r in records:
        lines.append(
            f"| {r.name} | {r.stage} | {r.engine} "
            f"| {'—' if r.similarity is None else f'{r.similarity:.3f}'} "
            f"| {r.elapsed_s:.0f}s "
            f"| {'принят' if r.accepted else (r.reject_reason or 'отклонён')} |"
        )

    if state.video:
        v = state.video
        lines += ["", "## Видео", "",
                  f"- движок: `{v['model']}`",
                  f"- источник: `{Path(v['source']).name}`"]
        if v.get("driver"):
            lines.append(f"- драйвер движения: `{Path(v['driver']).name}`")
        lines.append(f"- время: {v['elapsed_s']:.0f} сек")
        for label, key in (("к исходному кадру", "vs_source"), ("к эталону", "vs_reference")):
            if v.get(key):
                s = v[key]
                lines.append(f"- identity {label}: min {s['min']:.3f} / среднее {s['avg']:.3f}")
        lines.append(f"- safety: заблокировано {v['blocked']} из {v['checked']} кадров")

    lines += ["", "## Настройки", "",
              f"- порог NSFW: {prompt_config.nsfw_threshold()}",
              f"- минимальный возраст: {prompt_config.min_allowed_age()}",
              "", "Задаётся в `.env`, редактируется в админке."]

    (run_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[5/5] отчёт: {run_dir / 'REPORT.md'}")


# ---------------------------------------------------------------------------

def run(
    *,
    output_dir: Path,
    pack_engine: str,
    pose_engine: str,
    restyle_engine: str,
    pack_target: int,
    pack_min_similarity: float,
    reference_count: int,
    image_size: str,
    video_model: str,
    video_duration: int,
    video_resolution: str,
    video_aspect: str,
    driver: Optional[Path],
    pose_count: int,
    pose_source: str,
    skip_video: bool,
    force: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    print(prompt_config.describe())
    print()

    pack_eng = Engine(pack_engine, image_size)
    pack, seed_face, pack_records = stage_pack(
        run_dir=output_dir, engine=pack_eng, target=pack_target,
        min_similarity=pack_min_similarity, max_references=6, force=force,
    )

    from worker.identity import detect_primary_face

    scored = [(_cosine(seed_face.embedding, f.embedding), p)
              for p in pack if (f := detect_primary_face(p))]
    scored.sort(key=lambda kv: -kv[0])
    references = [p for _, p in scored[:reference_count]]
    print(f"      референсы: {', '.join(f'{p.name}({s:.2f})' for s, p in scored[:reference_count])}")

    pose_eng = pack_eng if pose_engine == pack_engine else Engine(pose_engine, image_size)
    pose_records = stage_poses(run_dir=output_dir, engine=pose_eng, references=references,
                               seed_face=seed_face, force=force, driver=driver,
                               pose_count=pose_count, pose_source=pose_source)

    restyle_eng = (pose_eng if restyle_engine == pose_engine
                   else pack_eng if restyle_engine == pack_engine
                   else Engine(restyle_engine, image_size))
    restyle_records = stage_restyle(run_dir=output_dir, engine=restyle_eng,
                                    pose_records=pose_records,
                                    references=references, seed_face=seed_face,
                                    image_size=image_size, force=force)

    # Пропущенный этап возвращает пустой список — подставляем то, что он насчитал в прошлый раз.
    # Иначе перезапуск ради одного шага стирает результаты всех предыдущих.
    pack_records = pack_records or _saved_records(output_dir, "pack")
    pose_records = pose_records or _saved_records(output_dir, "pose")
    restyle_records = restyle_records or _saved_records(output_dir, "restyle")

    all_records = pack_records + pose_records + restyle_records
    state = RunState(
        started_at=datetime.now(timezone.utc).isoformat(),
        character=prompt_config.load_character(),
        engines={"pack": pack_engine, "poses": pose_engine, "restyle": restyle_engine,
                 "video": video_model if not skip_video else "—"},
        frames=[asdict(r) for r in all_records],
        best_references=[str(p) for p in references],
    )

    if not skip_video:
        state.video = stage_video(run_dir=output_dir, records=all_records, pack=pack,
                                  seed_face=seed_face, model_key=video_model,
                                  duration=video_duration, driver=driver, force=force,
                                  video_resolution=video_resolution, video_aspect=video_aspect)

    state.save(output_dir / "run.json")
    stage_report(run_dir=output_dir, state=state)
    print(f"\nготово за {(time.monotonic() - started) / 60:.1f} мин -> {output_dir}")


def _cli():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pack-engine", type=str, default="seedream", choices=ENGINE_CHOICES,
                        help="движок шага 1")
    parser.add_argument("--pose-engine", type=str, default="seedream",
                        choices=ENGINE_CHOICES)
    parser.add_argument("--restyle-engine", type=str, default="seedream",
                        choices=ENGINE_CHOICES,
                        help="движок шага 3; flaq — Seedream 5.0 Pro, один референс вместо нескольких")
    parser.add_argument("--pack-target", type=int, default=8)
    parser.add_argument("--pack-min-similarity", type=float, default=0.55)
    parser.add_argument("--reference-count", type=int, default=4)
    parser.add_argument("--image-size", type=str, default="square_hd")
    # Умолчание — Wan 3.0 у flaq, а не Kling у fal. Причина замерена: Kling отклоняет
    # обнажённый кадр на входе (HTTP 422 content_policy_violation), flaq те же кадры принимает,
    # считает вдвое быстрее (156 с против 207) и дешевле.
    parser.add_argument("--video-model", type=str, default="wan-3.0-image-to-video")
    parser.add_argument("--video-duration", type=int, default=5)
    parser.add_argument("--video-resolution", type=str, default="720p",
                        choices=["480p", "720p", "1080p"],
                        help="только для моделей flaq; цена считается за секунду ролика")
    parser.add_argument("--video-aspect", type=str, default="9:16")
    parser.add_argument("--driver", type=Path, default=None,
                        help="ролик-источник движения; нужен режимам wan-animate-*")
    parser.add_argument("--pose-count", type=int, default=8,
                        help="сколько поз генерировать")
    parser.add_argument("--pose-source", type=str, default="text",
                        choices=["text", "driver"],
                        help="text = позы из POSE_* в .env (лицо держится, замерено 0.809); "
                             "driver = позы с кадров ролика (картиночные модели позу не "
                             "копируют, а лицо просаживают до 0.41 — см. README)")
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run(output_dir=args.output_dir, pack_engine=args.pack_engine, pose_engine=args.pose_engine,
        pack_target=args.pack_target, pack_min_similarity=args.pack_min_similarity,
        reference_count=args.reference_count, image_size=args.image_size,
        video_model=args.video_model, video_duration=args.video_duration,
        video_resolution=args.video_resolution, video_aspect=args.video_aspect,
        restyle_engine=args.restyle_engine,
        driver=args.driver, pose_count=args.pose_count, pose_source=args.pose_source,
        skip_video=args.skip_video, force=args.force)


if __name__ == "__main__":
    _cli()
