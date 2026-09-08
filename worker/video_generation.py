"""Генерация видео из кадра. Структура зеркалит photo_generation.py:
load_model -> generate -> тайминг -> safety.

pipeline_class резолвится по имени из diffusers лениво, а не импортом в config.py — иначе модуль
падал бы на окружении со старой версией diffusers, где видео вообще не нужно.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import os
from threading import Lock
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from worker.config import VIDEO_MODEL_CONFIGS, DEVICE, OUTPUT_DIR  # noqa: E402
from worker.models import VideoGenerationRequest, VideoGenerationResult  # noqa: E402
from safety.moderation_pipeline import SafetyPipeline  # noqa: E402


def _frame_to_pil(frame: Any):
    """Найдено на реальном прогоне: LTX-Video отдаёт кадры уже как PIL.Image (output_type="pil"
    по умолчанию), а WanImageToVideoPipeline — как numpy-массив (output_type="np" по умолчанию,
    float32 в [0,1]) — SafetyPipeline.run_image() ждёт PIL.Image (.convert("RGB")) и падал на
    втором пайплайне с AttributeError. Нормализует оба варианта к PIL.Image, ничего не делая с
    уже-PIL кадром."""
    from PIL import Image
    import numpy as np

    if isinstance(frame, Image.Image):
        return frame
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8) if arr.max() <= 1.0 else arr.clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _resolve_pipeline_class(class_name: str):
    import diffusers

    pipeline_class = getattr(diffusers, class_name, None)
    if pipeline_class is None:
        raise RuntimeError(
            f"diffusers.{class_name} not found in installed diffusers version. "
            "Video pipelines change name/availability between diffusers releases - "
            "check `pip show diffusers` on the pod against the model's HF README."
        )
    return pipeline_class


def _zoom_to_subject(image, zoom: float):
    """Подрезает кадр вокруг человека, чтобы лицу досталось больше пикселей.

    Центр — по найденному лицу: на ростовом плане человек редко стоит посередине, и подрезка
    от центра кадра отрезала бы ему голову. Окно смещается так, чтобы лицо оказалось в верхней
    трети — тогда в кадр попадает корпус, а не потолок. Лицо не найдено — режем от центра.
    """
    from worker.identity import detect_primary_face_array

    width, height = image.size
    new_width = int(width / zoom)
    new_height = int(height / zoom)

    center_x, center_y = width // 2, height // 2
    try:
        face = detect_primary_face_array(np.array(image))
    except Exception:  # noqa: BLE001 — детекция не должна ронять генерацию ролика
        face = None
    if face is not None and getattr(face, "bbox", None):
        x1, y1, x2, y2 = face.bbox
        center_x = (x1 + x2) // 2
        # Лицо в верхней трети окна: смещаем центр окна ВНИЗ от лица.
        center_y = (y1 + y2) // 2 + new_height // 6

        # Но не настолько, чтобы срезать макушку. Над лицом оставляем половину его высоты —
        # без этого при сильном зуме голова уходила за верхний край кадра, что и было видно на
        # первых кадрах ролика. Ограничение сильнее правила «лицо в верхней трети»: срезанная
        # голова портит кадр заметнее, чем неидеальная композиция.
        headroom = (y2 - y1) // 2
        max_center_y = y1 - headroom + new_height // 2
        center_y = min(center_y, max(new_height // 2, max_center_y))

    left = max(0, min(width - new_width, center_x - new_width // 2))
    top = max(0, min(height - new_height, center_y - new_height // 2))
    return image.crop((left, top, left + new_width, top + new_height))


def _available_vram_gb():
    """Память видеокарты в гигабайтах, None если CUDA недоступна.

    Берём total, а не свободную: решение принимается один раз при загрузке, а свободная зависит
    от случайного момента замера.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:  # noqa: BLE001 — не смогли определить, значит идём консервативным путём
        return None


class VideoModelManager:
    """Загрузка/кеш video-пайплайнов. Один под, одна модель за раз — как и в
    worker/photo_generation.ModelManager, здесь нет multi-GPU device map."""

    def __init__(self):
        self.loaded_models: Dict[str, Dict[str, Any]] = {}
        self._locks: Dict[str, Lock] = {}
        self._generation_lock = Lock()

    def load_model(self, model_name: str, fast: bool = True) -> Dict[str, Any]:
        """fast=True подключает ускоряющую LoRA: 4 шага вместо 40.

        Флаг входит в ключ кеша — это разные пайплайны, и отключить LoRA после загрузки нельзя,
        веса уже влиты в модули.

        Замерено на одном исходнике: 61.5 минуты без неё против 13.2 с ней. Разница не в
        настройках, а в числе шагов — за 40 модель прорабатывает то, что за 4 остаётся эскизом.
        """
        cache_key = f"{model_name}::fast={bool(fast)}"
        if cache_key in self.loaded_models:
            return self.loaded_models[cache_key]

        lock = self._locks.setdefault(cache_key, Lock())
        with lock:
            if cache_key in self.loaded_models:
                return self.loaded_models[cache_key]

            if model_name not in VIDEO_MODEL_CONFIGS:
                raise ValueError(f"Unknown video model: {model_name}. Available: {list(VIDEO_MODEL_CONFIGS)}")

            config = VIDEO_MODEL_CONFIGS[model_name]
            pipeline_class = _resolve_pipeline_class(config["pipeline_class_name"])
            pipeline_kwargs: Dict[str, Any] = {"torch_dtype": config["torch_dtype"]}
            if config.get("needs_wan_vae_override"):
                # Wan2.2-TI2V-5B: официальный пример модели-карты грузит VAE отдельно в float32
                # (AutoencoderKLWan) — обычный VAE из общего from_pretrained даёт хуже стабильность
                # по их же документации.
                import torch as _torch
                from diffusers import AutoencoderKLWan

                vae = AutoencoderKLWan.from_pretrained(config["hf_repo_id"], subfolder="vae", torch_dtype=_torch.float32)
                pipeline_kwargs["vae"] = vae
            pipeline = pipeline_class.from_pretrained(config["hf_repo_id"], **pipeline_kwargs)
            # Либо .to(DEVICE), либо enable_model_cpu_offload() — ровно одно из двух.
            # Решаем по фактической памяти карты, а не по флагу конфига: флаг задаёт только порог,
            # ниже которого оффлоад обязателен. На карте пожирнее оффлоад вреден — на A6000 48GB
            # он держал GPU на 0% при 117% CPU и не выдал ролик за двадцать минут.
            needs_offload = config.get("requires_cpu_offload", False)
            if needs_offload:
                required_gb = config.get("min_vram_gb_for_direct", 40)
                available_gb = _available_vram_gb()
                if available_gb and available_gb >= required_gb:
                    needs_offload = False
                    print(f"[video] {model_name}: {available_gb:.0f} GB VRAM >= {required_gb} GB — "
                          f"грузим напрямую на GPU, без CPU-оффлоада")

            if needs_offload:
                pipeline.enable_model_cpu_offload()
            else:
                pipeline.to(DEVICE)
            # Ускоряющая LoRA грузится ДО оффлоада и до всех оптимизаций: иначе веса адаптера
            # окажутся на другом устройстве, чем модуль, в который их вливают.
            speed = config.get("speed_lora") if fast else None
            if speed:
                try:
                    pipeline.load_lora_weights(
                        speed["repo"], weight_name=speed["high"], adapter_name="high")
                    pipeline.load_lora_weights(
                        speed["repo"], weight_name=speed["low"], adapter_name="low",
                        load_into_transformer_2=True)
                    pipeline.set_adapters(["high", "low"], adapter_weights=[1.0, 1.0])
                    print(f"[video] {model_name}: ускоряющая LoRA подключена, "
                          f"{speed['steps']} шагов вместо {config['default_num_inference_steps']}")
                    config = {**config,
                              "default_num_inference_steps": speed["steps"],
                              "default_guidance_scale": speed["guidance_scale"]}
                except Exception as exc:  # noqa: BLE001
                    # Без LoRA модель работает, просто медленно — это не повод ронять прогон.
                    print(f"[video] ускоряющая LoRA не подключилась ({type(exc).__name__}: "
                          f"{exc}), идём на полном числе шагов")

            if hasattr(pipeline, "enable_vae_tiling"):
                try:
                    pipeline.enable_vae_tiling()
                except Exception:
                    pass
            if hasattr(pipeline, "set_progress_bar_config"):
                pipeline.set_progress_bar_config(disable=True)

            model_info = {"pipeline": pipeline, "config": config,
                          "loaded_at": datetime.now(timezone.utc).isoformat()}
            self.loaded_models[cache_key] = model_info
            return model_info

    def unload_all(self) -> str:
        """Симметрично фото-менеджеру: освобождает карту под фото-пайплайны.

        Wan2.2-TI2V-5B занимает десятки гигабайт и после ролика остаётся в кеше — следующий кадр
        пака упёрся бы в ту же нехватку памяти, только с другой стороны.
        """
        from worker.photo_generation import _release_vram

        freed = list(self.loaded_models)
        for key in freed:
            info = self.loaded_models.pop(key)
            pipeline = info.pop("pipeline", None)
            if pipeline is not None:
                try:
                    pipeline.to("cpu")
                except Exception:  # noqa: BLE001
                    pass
                del pipeline
            del info
        _release_vram()
        return ", ".join(freed) if freed else "нечего выгружать"

    def get_generation_lock(self) -> Lock:
        return self._generation_lock


class VideoGenerator:
    """Аналог PhotoGenerator/ImageGenerator: generate() = собрать kwargs -> засечь время ->
    вызвать pipeline -> сохранить -> прогнать safety по выборке кадров (Stage 5, PLAN.md)."""

    def __init__(self, model_manager: VideoModelManager, safety: SafetyPipeline, output_dir: Optional[Path] = None):
        self.model_manager = model_manager
        self.safety = safety
        self.output_dir = Path(output_dir) if output_dir else OUTPUT_DIR

    def generate(self, request: VideoGenerationRequest, *, safety_frame_sample_every: int = 5) -> VideoGenerationResult:
        import torch
        from diffusers.utils import export_to_video
        from PIL import Image

        model_info = self.model_manager.load_model(request.model_name, fast=request.fast)
        pipeline = model_info["pipeline"]
        config = model_info["config"]

        # Ориентация — по исходному фото, а не из конфига вслепую: ландшафтный дефолт 1280x704
        # на вертикальном кадре срезал макушку и подбородок.
        # Меняем стороны местами, а не берём произвольный размер: модели обучены на своей сетке.
        width = config["default_width"]
        height = config["default_height"]
        try:
            from PIL import Image as _Image

            with _Image.open(request.input_image_path) as probe:
                source_is_portrait = probe.height >= probe.width
        except Exception:  # noqa: BLE001 — не смогли прочитать, остаёмся на конфиге
            source_is_portrait = False

        if source_is_portrait and width > height:
            width, height = height, width
        num_frames = request.num_frames or config["default_num_frames"]
        fps = request.fps or config["default_fps"]
        num_inference_steps = request.num_inference_steps or config["default_num_inference_steps"]
        guidance_scale = request.guidance_scale or config["default_guidance_scale"]

        frame_seed = request.frame_seed if request.frame_seed is not None else 0
        motion_seed = request.motion_seed if request.motion_seed is not None else frame_seed + 500_000
        generator = torch.Generator(device=DEVICE).manual_seed(motion_seed)

        # Сначала кадрируем под нужное соотношение, потом масштабируем: прямой resize плющит
        # лицо и пропорции, и модель анимирует уже искажённого человека.
        # LANCZOS вместо BICUBIC — на понижении заметно чётче волосы и ткань.
        source_image = Image.open(request.input_image_path).convert("RGB")

        # Подрезка вокруг лица ДО кадрирования под соотношение сторон. Порядок важен: сначала
        # решаем, какую часть кадра берём, и только потом приводим к формату видео — иначе зум
        # считался бы от уже обрезанного кадра и уезжал бы вниз.
        if request.subject_zoom > 1.01:
            source_image = _zoom_to_subject(source_image, request.subject_zoom)

        target_ratio = width / height
        src_ratio = source_image.width / source_image.height
        if src_ratio > target_ratio:
            crop_w = int(round(source_image.height * target_ratio))
            left = (source_image.width - crop_w) // 2
            source_image = source_image.crop((left, 0, left + crop_w, source_image.height))
        elif src_ratio < target_ratio:
            crop_h = int(round(source_image.width / target_ratio))
            # Смещаем окно кадрирования к верхней трети, а не к геометрическому центру: на
            # портретных кадрах в центре оказывается торс, а голова уезжает за границу.
            top = max(0, (source_image.height - crop_h) // 3)
            source_image = source_image.crop((0, top, source_image.width, top + crop_h))
        source_image = source_image.resize((width, height), Image.LANCZOS)

        gen_kwargs: Dict[str, Any] = {
            "image": source_image,
            "prompt": request.prompt,
            "num_frames": num_frames,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
            "generator": generator,
        }
        if request.negative_prompt:
            gen_kwargs["negative_prompt"] = request.negative_prompt

        start = time.monotonic()
        with self.model_manager.get_generation_lock():
            with torch.no_grad():
                output = pipeline(**gen_kwargs)
        duration_ms = int((time.monotonic() - start) * 1000)

        frames = output.frames[0]
        video_path = self.output_dir / f"{request.id}.mp4"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Найдено на реальном прогоне: export_to_video() по умолчанию quality=5 (шкала 0-10) дал
        # ~400 kbps на 768x512@24fps — заметно недостаточно, видео выглядело сжатым/блочным при
        # реальном воспроизведении (хотя отдельные превью-кадры на глаз казались нормальными).
        # quality=9 — почти максимум по этой шкале, даёт битрейт в разы выше при том же кодеке.
        export_to_video(frames, str(video_path), fps=fps, quality=9)

        # Видеопайплайн занимает почти весь VRAM, и InsightFace потом не может выделить свой
        # cublas-контекст ("CUBLAS failure"). Освобождаем память перед safety-проверкой кадров.
        # У моделей с оффлоадом .to("cpu") напрямую нельзя — конфликтует с хуками accelerate,
        # поэтому только чистим кеш аллокатора.
        if not config.get("requires_cpu_offload"):
            pipeline.to("cpu")
        torch.cuda.empty_cache()

        safety_allowed, checked = self._run_frame_safety(frames, safety_frame_sample_every)

        return VideoGenerationResult(
            id=f"{request.id}_result",
            request_id=request.id,
            video_path=str(video_path),
            frame_seed=frame_seed,
            motion_seed=motion_seed,
            duration_ms=duration_ms,
            model_name=request.model_name,
            num_frames=len(frames),
            frames_checked=checked,
            safety_allowed=safety_allowed,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def _run_frame_safety(self, frames: List[Any], sample_every: int) -> tuple[bool, int]:
        """Проверяет каждый sample_every-й кадр тем же SafetyPipeline, что и фото (Stage 5,
        PLAN.md) — не весь клип целиком: при 24fps и 5 секундах это ~120 кадров, каждый 5-й даёт
        24 проверки, чего достаточно, чтобы поймать явный дрейф лица или контента по ходу ролика."""
        sampled = frames[::sample_every]
        allowed = True
        for frame in sampled:
            report = self.safety.run_image(_frame_to_pil(frame))
            if not report.allowed:
                allowed = False
        return allowed, len(sampled)


_video_model_manager: Optional[VideoModelManager] = None


def get_video_model_manager() -> VideoModelManager:
    global _video_model_manager
    if _video_model_manager is None:
        _video_model_manager = VideoModelManager()
    return _video_model_manager
