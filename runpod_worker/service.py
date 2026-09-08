"""Сборка кадра целиком: генерация → hires → детейлер лица → метрика → safety.

У результата обязательно есть similarity: задача — один и тот же человек на всех кадрах, и кадр
без замера ничего не подтверждает. Порог 0.5 не назначен на глаз — портреты заведомо чужих людей
на этой же модели дают до 0.387.

Веса грузятся лениво и кешируются: первый запрос после старта пода платит за загрузку.
"""

from __future__ import annotations

import base64
import io
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from runpod_worker import characters
from runpod_worker.config import (
    DEFAULT_FACE_DETAILER,
    DEFAULT_HIRES,
    DEFAULT_MIN_SIMILARITY,
    LORA_DIR,
    OUTPUT_DIR,
)
from runpod_worker.models import BatchPhotoRequest, PhotoRequest, PhotoResult
from worker.config import DEFAULT_PHOTO_MODEL, IDENTITY_CONFIG


# Ракурсы многокадрового эталона. Фронтальный кадр создаётся отдельно и идёт первым, здесь —
# только дополнения к нему.
REFERENCE_ANGLES = [
    "three-quarter view, head turned slightly to the left",
    "three-quarter view, head turned slightly to the right",
    "head slightly tilted down, looking up at the camera",
    "head slightly tilted up, chin raised",
]


class ServiceError(RuntimeError):
    pass


def _cosine(a, b) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b)))


class PhotoService:
    """Один экземпляр на под. Держит ModelManager и построенные поверх него проходы."""

    def __init__(self):
        self._generator = None
        self._safety = None
        self._hires: Dict[str, Any] = {}
        self._detailer: Dict[str, Any] = {}
        self._active_lora: Optional[str] = None
        # Видео-генератор ленивый и отдельный от фото: у видео-моделей другие классы пайплайнов и
        # другой профиль памяти, а под может всю жизнь генерировать одни фото — тянуть ради этого
        # веса Wan не за что.
        self._video_generator = None

    # --- ленивые части ---

    def _get_generator(self):
        if self._generator is None:
            from safety.moderation_pipeline import SafetyPipeline
            from worker.photo_generation import PhotoGenerator, get_model_manager

            self._safety = SafetyPipeline()
            self._generator = PhotoGenerator(get_model_manager(), self._safety, OUTPUT_DIR)
        return self._generator

    def _get_hires(self, model_name: str, scale: Optional[float]):
        key = f"{model_name}::{scale}"
        if key not in self._hires:
            from worker.hires import HiresFix

            self._hires[key] = HiresFix(self._get_generator().model_manager,
                                        model_name=model_name, ip_adapter_scale=scale)
        return self._hires[key]

    def _get_detailer(self, model_name: str, scale: Optional[float]):
        key = f"{model_name}::{scale}"
        if key not in self._detailer:
            from worker.face_detailer import FaceDetailer

            self._detailer[key] = FaceDetailer(self._get_generator().model_manager,
                                               model_name=model_name, ip_adapter_scale=scale)
        return self._detailer[key]

    # --- LoRA ---

    def _apply_lora(self, model_name: str, lora: Optional[str], scale: float) -> None:
        """Подключает или снимает LoRA.

        LoRA должна быть обучена под тот же базовый чекпойнт: адаптер от FLUX на SDXL не встанет,
        ключи тензоров разные.
        """
        gen = self._get_generator()
        pipeline = gen.model_manager.load_model(model_name, with_ip_adapter=True)["pipeline"]

        if lora is None:
            if self._active_lora is not None and hasattr(pipeline, "unload_lora_weights"):
                pipeline.unload_lora_weights()
                self._active_lora = None
            return

        path = (LORA_DIR / lora) if not Path(lora).is_absolute() else Path(lora)
        if not path.exists():
            raise ServiceError(f"LoRA не найдена: {path}")

        # Снимаем предыдущую: diffusers складывает адаптеры друг на друга, и без выгрузки
        # вторая LoRA применилась бы поверх первой.
        if self._active_lora and self._active_lora != str(path):
            if hasattr(pipeline, "unload_lora_weights"):
                pipeline.unload_lora_weights()
            self._active_lora = None

        if self._active_lora != str(path):
            try:
                pipeline.load_lora_weights(str(path.parent), weight_name=path.name)
            except Exception as e:
                raise ServiceError(
                    f"LoRA {path.name} не подошла к чекпойнту {model_name}: {str(e)[:200]}"
                ) from None
            self._active_lora = str(path)

        if hasattr(pipeline, "set_adapters"):
            try:
                pipeline.set_adapters(["default_0"], adapter_weights=[scale])
            except Exception:
                # У части версий diffusers имя адаптера другое — сила задаётся при вызове.
                pass

    # --- эталон персонажа ---

    def create_character(self, name: str, *, reference_image: Optional[str] = None,
                         reference_path: Optional[str] = None, prompt: Optional[str] = None,
                         seed: Optional[int] = None, overwrite: bool = False) -> dict:
        from worker.identity import detect_primary_face

        gen = self._get_generator()
        ref: Optional[Path] = None

        if reference_image:
            raw = base64.b64decode(reference_image.split(",", 1)[-1])
            ref = OUTPUT_DIR / f"_ref_{name}.png"
            ref.parent.mkdir(parents=True, exist_ok=True)
            from PIL import Image

            Image.open(io.BytesIO(raw)).convert("RGB").save(ref)
        elif reference_path:
            ref = Path(reference_path)
            if not ref.exists():
                raise ServiceError(f"эталон не найден: {ref}")
        extra_embeddings: list = []
        if reference_image or reference_path:
            pass
        else:
            # Эталона нет — рисуем его текстом, без IP-Adapter: опираться ещё не на что.
            from worker.models import PhotoGenerationRequest

            req = PhotoGenerationRequest(model_name=DEFAULT_PHOTO_MODEL, prompt=prompt, seed=seed)
            result = gen.generate(req)
            ref = Path(result.image_path)

            # Дополнительные ракурсы — вход портретного режима FaceID: без них portrait
            # вырождается в base с лишним слоем.
            # Приписки задают геометрию съёмки, а не эмоцию: проектор усредняет и мимику, и разные
            # выражения дали бы смазанное лицо на всём паке.
            for index, angle in enumerate(REFERENCE_ANGLES, start=1):
                angled = gen.generate(PhotoGenerationRequest(
                    model_name=DEFAULT_PHOTO_MODEL,
                    prompt=f"{prompt}, {angle}",
                    # Смещение 100 на ракурс: не пересекается с шагом 1000 между кадрами пака,
                    # поэтому прогон остаётся воспроизводимым.
                    seed=(seed + index * 100) if seed is not None else None,
                ))
                angled_face = detect_primary_face(Path(angled.image_path))
                if angled_face is None:
                    # Один потерянный ракурс из четырёх не повод валить создание персонажа:
                    # детектор штатно не находит лицо на резком профиле.
                    continue
                extra_embeddings.append(list(map(float, angled_face.embedding)))

        face = detect_primary_face(ref)
        if face is None:
            raise ServiceError("на эталонном кадре не найдено лицо — персонаж не создан")

        primary = list(map(float, face.embedding))
        return characters.save(name, primary, ref, overwrite=overwrite,
                               embeddings=[primary, *extra_embeddings] if extra_embeddings else None)

    # --- постобработка видео ---

    def enhance_video(self, req) -> dict:
        """Покадровое восстановление лица в ролике.

        gfpgan   — быстро и чётко, но про конкретного человека не знает и усредняет черты.
        detailer — тянет лицо к эталону, но это диффузия на каждый кадр, на порядок дольше.
        """
        from pathlib import Path as _Path

        from worker.video_enhance import enhance_video

        source = _Path(req.video_path)
        if not source.exists():
            raise ServiceError(f"ролик не найден на поде: {source}")

        detailer = None
        embedding = None
        prompt = ""

        if req.method in ("detailer", "both"):
            if not req.character:
                raise ServiceError("для method=detailer нужен character — из него берётся эталон")
            character = characters.load(req.character)
            embedding = character["embedding"]
            detailer = self._get_detailer(req.model_name, None)

            # Видео-модель занимает десятки гигабайт и после генерации остаётся в кеше. Диффузия
            # по кадрам поверх неё не поместится — освобождаем карту заранее, а не по факту OOM.
            from worker.video_generation import get_video_model_manager

            get_video_model_manager().unload_all()

        output = source.with_name(f"{source.stem}_{req.method}.mp4")
        result = enhance_video(
            source, output, method=req.method, face_embedding=embedding,
            detailer=detailer, prompt=prompt, seed=req.seed, every=req.every,
            upscale=req.upscale, blend=req.blend,
        )

        # Похожесть после обработки — единственный способ отличить «лицо стало чётким» от «лицо
        # стало чётким и чужим». Считаем по среднему кадру, как и при генерации ролика.
        result["similarity"] = None
        if req.character:
            try:
                import imageio.v3 as iio
                import numpy as np

                from worker.identity import detect_primary_face_array

                reference = characters.load(req.character)["embedding"]
                frames = iio.imread(result["output_path"], plugin="pyav")
                face = detect_primary_face_array(np.asarray(frames[len(frames) // 2]))
                if face is not None:
                    result["similarity"] = _cosine(reference, face.embedding)
            except Exception:  # noqa: BLE001 — метрика не должна ронять готовый ролик
                pass

        result["video_path"] = result.pop("output_path")
        return result

    # --- генерация видео ---

    def generate_video(self, req) -> dict:
        """Оживляет готовый кадр. Возвращает словарь под VideoResult.

        Видео-модели живут в ОТДЕЛЬНОМ менеджере (worker/video_generation.py), а не в том же, что
        фото: это другие классы пайплайнов и другой профиль памяти. Держать их в одном кеше
        значило бы вытеснять SDXL ради Wan и обратно на каждом кадре пака.
        """
        import time
        from pathlib import Path as _Path

        from worker.models import VideoGenerationRequest
        from worker.video_generation import VideoGenerator, get_video_model_manager
        from worker.config import DEFAULT_VIDEO_MODEL

        source = _Path(req.image_path)
        if not source.exists():
            raise ServiceError(f"исходный кадр не найден на поде: {source}")

        # Видео грузится только при первом обращении: под может всю жизнь считать одни фото.
        # Фото и видео на карте одновременно не живут: два SDXL с адаптерами занимают ~44 GB
        # из 48, и видео поверх них падает. Выгружаем до загрузки, а не после ошибки.
        from worker.photo_generation import get_model_manager, _release_vram

        # HiresFix и FaceDetailer держат свои пайплайны вне кеша ModelManager: unload_all()
        # чистил только его, а на карте оставалось ~38 GB. Сбрасываем и их.
        self._hires.clear()
        self._detailer.clear()
        freed = get_model_manager().unload_all()
        _release_vram()
        print(f"[video] освобождена карта, выгружено: {freed} + hires/detailer")

        if self._video_generator is None:
            # _get_generator() заодно создаёт SafetyPipeline — он общий у фото и видео, и второй
            # экземпляр означал бы вторую загрузку NSFW-классификатора и InsightFace в память.
            self._get_generator()
            self._video_generator = VideoGenerator(
                get_video_model_manager(), self._safety, output_dir=OUTPUT_DIR)

        started = time.monotonic()
        result = self._video_generator.generate(VideoGenerationRequest(
            id=req.id,
            model_name=req.model_name or DEFAULT_VIDEO_MODEL,
            input_image_path=str(source),
            prompt=req.prompt or "",
            negative_prompt=req.negative_prompt,
            num_frames=req.num_frames,
            fps=req.fps,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            frame_seed=req.frame_seed,
            motion_seed=req.motion_seed,
            subject_zoom=req.subject_zoom,
            fast=req.fast,
        ))

        # Похожесть считаем по СРЕДНЕМУ кадру ролика, а не по первому: первый кадр — это почти
        # исходное фото, и его метрика ничего не говорит о том, удержалось ли лицо в движении.
        similarity = self._video_similarity(_Path(result.video_path), source)

        return {
            "id": result.id,
            "request_id": result.request_id,
            "video_path": result.video_path,
            "model_name": result.model_name,
            "num_frames": result.num_frames,
            "fps": req.fps or 24,
            "frame_seed": result.frame_seed,
            "motion_seed": result.motion_seed,
            "duration_ms": int((time.monotonic() - started) * 1000),
            "frames_checked": result.frames_checked,
            "safety_allowed": result.safety_allowed,
            "similarity": similarity,
        }

    @staticmethod
    def _video_similarity(video_path, source_image):
        """Cosine ArcFace между лицом на исходном кадре и лицом в середине ролика.

        None — не ошибка: на резком повороте головы детектор штатно не находит лицо, и это
        отличается от «лицо нашлось, но чужое».
        """
        try:
            import imageio.v3 as iio
            import numpy as np

            from worker.identity import detect_primary_face_array, detect_primary_face

            reference = detect_primary_face(source_image)
            if reference is None:
                return None

            frames = iio.imread(video_path, plugin="pyav")
            middle = frames[len(frames) // 2]
            face = detect_primary_face_array(np.asarray(middle))
            if face is None:
                return None
            a, b = reference.embedding, face.embedding
            return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) or 1.0))
        except Exception:  # noqa: BLE001 — метрика не должна ронять готовый ролик
            return None

    # --- генерация кадра ---

    def generate_photo(self, req: PhotoRequest) -> PhotoResult:
        from PIL import Image

        from worker.identity import detect_primary_face
        from worker.models import PhotoGenerationRequest

        started = time.monotonic()
        character = characters.load(req.character)
        embedding = character["embedding"]

        gen = self._get_generator()
        # Обратная сторона того же: после ролика Wan остаётся в кеше и занимает десятки гигабайт,
        # и следующий кадр пака упёрся бы в нехватку памяти с другой стороны.
        if self._video_generator is not None:
            from worker.video_generation import get_video_model_manager

            get_video_model_manager().unload_all()

        from worker.config import IDENTITY_MODES, DEFAULT_IDENTITY_MODE

        mode_name = req.identity_mode or DEFAULT_IDENTITY_MODE
        if mode_name not in IDENTITY_MODES:
            raise ServiceError(f"неизвестный режим фиксации лица: {mode_name}")
        # Умолчание берём из режима, а не из общего конфига: у portrait и plusv2 свои диапазоны.
        # Проверка на None, а не "or": 0.0 — ложное значение, и через "or" оно подменялось
        # дефолтом, то есть отключить FaceID запросом было невозможно. А отключать нужно — он
        # тянет композицию к лицу, и ростовой кадр не снять, сколько ни пиши "full body".
        scale = (req.ip_adapter_scale if req.ip_adapter_scale is not None
                 else IDENTITY_MODES[mode_name]["default_scale"])
        self._apply_lora(req.model_name, req.lora, req.lora_scale)

        # Портретный режим принимает несколько ракурсов одного человека. Берём их из карточки
        # персонажа, если они там есть: именно ради этого режим и существует, на одном эталоне он
        # вырождается в base с лишним слоем.
        embeddings = character.get("embeddings") or None
        if embeddings and req.identity_reference_count:
            embeddings = embeddings[: req.identity_reference_count]

        # scale=0 означает «сцену рисуем без фиксации лица»: эмбеддинг не передаём вовсе, иначе
        # пайплайн всё равно грузится с адаптером и теряет текстовый negative_prompt (на SDXL их
        # нельзя передавать вместе, см. worker/photo_generation.py).
        identity_off = scale <= 0.0

        base = gen.generate(PhotoGenerationRequest(
            id=req.id,
            model_name=req.model_name,
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            width=req.width,
            height=req.height,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            seed=req.seed,
            face_embedding=None if identity_off else embedding,
            face_embeddings=None if identity_off else embeddings,
            identity_mode=mode_name,
            reference_image_path=character.get("reference_image"),
            ip_adapter_scale=scale,
        ))

        path = Path(base.image_path)
        hires_applied = False
        detailer_applied = False

        want_hires = DEFAULT_HIRES if req.hires is None else req.hires
        if want_hires:
            out = self._get_hires(req.model_name, scale).refine(
                path, embedding, prompt=req.prompt, seed=base.seed, output_path=path)
            path = Path(out.get("output_path", path))
            hires_applied = True

        want_detailer = DEFAULT_FACE_DETAILER if req.face_detailer is None else req.face_detailer
        if want_detailer:
            # Похожесть ДО второго прохода — по ней и решаем, нужен ли он. Собственный порог
            # детейлера смотрит только на размер лица, а у ростового кадра лицо формально
            # «крупное» (0.19 при пороге 0.12), хотя личность уехала. Поэтому меряем сами и при
            # просадке включаем force.
            before = detect_primary_face(path)
            similarity_before = _cosine(embedding, before.embedding) if before is not None else None
            floor = (req.detail_below_similarity if req.detail_below_similarity is not None
                     else DEFAULT_MIN_SIMILARITY)
            force = similarity_before is not None and similarity_before < floor

            # Детейлеру передаём НЕ нулевой scale: в двухэтапной схеме именно он вживляет
            # личность в уже правильно скомпонованный кадр, и с нулём он бы ничего не делал.
            detailer_scale = scale if scale > 0 else IDENTITY_MODES[mode_name]["default_scale"]
            out = self._get_detailer(req.model_name, detailer_scale).refine(
                path, embedding, prompt=req.prompt, seed=base.seed, output_path=path, force=force)
            detailer_applied = bool(out.get("applied"))
            path = Path(out.get("output_path", path))
            print(f"[detailer] похожесть до {similarity_before}, порог {floor}, "
                  f"force={force} -> {out.get('reason') or 'применён'}"
                  + (f", кадр увеличен x{out['upscale']}, лицо {out['face_px_after']}px"
                     if out.get('upscale', 1) > 1.01 else ""))

        # Метрика считается по финальному кадру — то есть после обоих проходов, иначе она
        # описывала бы промежуточный результат, которого в выдаче нет.
        face = detect_primary_face(path)
        similarity = _cosine(embedding, face.embedding) if face is not None else None

        # Доля кадра, занятая лицом — вторая половина картины качества. Считается по тому же
        # детектированному лицу, лишней работы не добавляет.
        face_area_ratio = None
        if face is not None and getattr(face, "bbox", None):
            x1, y1, x2, y2 = face.bbox
            with Image.open(path) as probe:
                frame_area = probe.width * probe.height
            if frame_area:
                face_area_ratio = max(0.0, min(1.0, abs((x2 - x1) * (y2 - y1)) / frame_area))

        threshold = DEFAULT_MIN_SIMILARITY if req.min_similarity is None else req.min_similarity
        if not base.safety_allowed:
            accepted, reason = False, "safety"
        elif similarity is None:
            accepted, reason = False, "лицо не найдено"
        elif similarity < threshold:
            accepted, reason = False, f"cos {similarity:.3f} < {threshold}"
        else:
            accepted, reason = True, None

        with Image.open(path) as im:
            width, height = im.size

        return PhotoResult(
            id=f"{req.id}_result",
            request_id=req.id,
            image_path=str(path),
            prompt=req.prompt,
            seed=base.seed,
            model_name=req.model_name,
            width=width,
            height=height,
            similarity=similarity,
            face_area_ratio=face_area_ratio,
            accepted=accepted,
            reject_reason=reason,
            hires_applied=hires_applied,
            face_detailer_applied=detailer_applied,
            duration_ms=int((time.monotonic() - started) * 1000),
            nsfw_score=base.nsfw_score,
            estimated_age=base.estimated_age,
            safety_allowed=base.safety_allowed,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def expand_batch(self, req: BatchPhotoRequest) -> list[PhotoRequest]:
        """Разворачивает пакетный запрос в отдельные кадры.

        Сид: base_seed + 1000·i. Шаг в тысячу, а не единица, — чтобы соседние кадры не делили
        близкий шум и не выходили похожими композициями; при этом каждый кадр воспроизводим,
        а весь пак повторяется одним числом.
        """
        defaults = dict(req.defaults or {})
        defaults.pop("character", None)
        out = []
        for i, scene in enumerate(req.scenes):
            fields = {"character": req.character, "prompt": scene, **defaults}
            if req.base_seed is not None:
                fields["seed"] = req.base_seed + 1000 * i
            out.append(PhotoRequest(**fields))
        return out


service = PhotoService()
