"""Генерация фото: ModelManager кеширует SDXL-пайплайны, PhotoGenerator считает кадр.

Порядок в generate(): загрузить модель -> seed -> собрать kwargs -> засечь время -> вызвать
пайплайн -> прогнать safety -> сохранить локально.

Личность подключается через IP-Adapter FaceID, safety вызывается реально и заполняет
nsfw_score/estimated_age/watchlist_similarity в результате.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import os
from threading import Lock
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from worker.config import (  # noqa: E402
    PHOTO_MODEL_CONFIGS,
    IDENTITY_CONFIG,
    IDENTITY_MODES,
    DEFAULT_IDENTITY_MODE,
    DEVICE,
    MODEL_CACHE_DIR,
    OUTPUT_DIR,
    CIVITAI_TOKEN,
)
from worker.models import PhotoGenerationRequest, PhotoGenerationResult  # noqa: E402
from worker.identity import get_face_analysis  # noqa: E402
from safety.moderation_pipeline import SafetyPipeline  # noqa: E402


def _next_seed(seed: Optional[int]) -> int:
    """Тот же паттерн, что _next_seed() в WetDreams generation.py: если seed не задан явно,
    сгенерировать случайный. В нашем пайплайне seed обычно приходит из scripts/seed_manager.py,
    так что этот путь — fallback, а не основной режим."""
    if seed is not None:
        return seed
    import torch

    return int(torch.randint(0, 2**31 - 1, (1,)).item())


def _encode_chunked(pipeline, text: str, chunks_needed: Optional[int] = None):
    """Кодирует текст любой длины кусками по 75 токенов.

    SDXL принимает 77 токенов и всё сверх этого отбрасывает молча. Описания образов занимают
    95-105 токенов, то есть у каждого терялся хвост со светом и деталями сцены.

    Свой код, а не compel: тот падает на transformers 5.x. Плюс выравнивание длин нужно держать
    самим — промпт и негатив обязаны дать одинаковое число кусков, иначе не собрать батч под CFG.

    Возвращает (эмбеддинги, pooled, число кусков).
    """
    import torch

    tokenizer = pipeline.tokenizer
    max_chunk = tokenizer.model_max_length - 2  # два служебных токена на кусок
    bos, eos = tokenizer.bos_token_id, tokenizer.eos_token_id

    ids = tokenizer(text, truncation=False, add_special_tokens=False).input_ids
    chunks = [ids[i:i + max_chunk] for i in range(0, len(ids), max_chunk)] or [[]]
    if chunks_needed:
        # Добиваем пустыми кусками до нужного числа: у промпта и негатива длина обязана совпадать.
        while len(chunks) < chunks_needed:
            chunks.append([])
        chunks = chunks[:chunks_needed]

    embeds, pooled = [], None
    for index, chunk in enumerate(chunks):
        padded = [bos] + chunk + [eos] * (max_chunk - len(chunk) + 1)
        for encoder_index, (tok, encoder) in enumerate(
                ((pipeline.tokenizer, pipeline.text_encoder),
                 (pipeline.tokenizer_2, pipeline.text_encoder_2))):
            tensor = torch.tensor([padded], device=encoder.device)
            output = encoder(tensor, output_hidden_states=True)
            # Предпоследнее скрытое состояние — так устроен сам SDXL-пайплайн, последнее даёт
            # заметно худший результат.
            hidden = output.hidden_states[-2]
            if encoder_index == 0:
                first = hidden
            else:
                second = hidden
                if index == 0:
                    # pooled берётся только с ПЕРВОГО куска и только со второго энкодера: это
                    # один вектор на весь промпт, а не на кусок.
                    pooled = output[0]
        embeds.append(torch.cat([first, second], dim=-1))

    return torch.cat(embeds, dim=1), pooled, len(chunks)


def _long_prompt_embeds(pipeline, prompt: str, negative_prompt: Optional[str]):
    """Эмбеддинги промпта и негатива одинаковой длины, без усечения по 77 токенам.

    None при любой ошибке — вызывающий код тогда идёт обычным путём.
    """
    try:
        conditioning, pooled, chunks = _encode_chunked(pipeline, prompt)
        if not negative_prompt:
            return conditioning, pooled, None, None
        negative, negative_pooled, negative_chunks = _encode_chunked(
            pipeline, negative_prompt, chunks_needed=chunks)
        if negative_chunks > chunks:
            # Негатив оказался длиннее — перекодируем промпт под его длину.
            conditioning, pooled, _ = _encode_chunked(
                pipeline, prompt, chunks_needed=negative_chunks)
        return conditioning, pooled, negative, negative_pooled
    except Exception as exc:  # noqa: BLE001
        print(f"[prompt] длинный промпт закодировать не удалось ({type(exc).__name__}: {exc}), "
              f"идём обычным путём — хвост будет усечён")
        return None


def _release_vram() -> None:
    """Отдаёт освобождённую память аллокатору.

    Порядок важен: empty_cache() работает только после того, как последняя ссылка на пайплайн
    исчезла и отработал сборщик. Раньше — не освобождает ничего, и следующая загрузка снова OOM.
    """
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 — освобождение памяти не должно ронять запрос
        pass


class ModelManager:
    """Загрузка и кеш SDXL-пайплайнов + IP-Adapter FaceID. Однопоточный/один-под вариант их
    ModelManager: без multi-GPU device map (см. worker/config.py — DEVICE один на весь под) и без
    ControlNet/img2img веток, которые здесь не нужны."""

    def __init__(self):
        self.loaded_models: Dict[str, Dict[str, Any]] = {}
        self._locks: Dict[str, Lock] = {}
        self._generation_lock = Lock()
        # Сколько пайплайнов держим в VRAM. Без предела кеш забивал 48 GB на переборе четырёх
        # чекпойнтов и давал OOM, а следующий падал уже с невнятным "Input type (c10::Half)".
        # Два, а не один: пайплайн без адаптера и он же с адаптером — штатная пара одного прогона.
        self.max_loaded = int(os.getenv("MAX_LOADED_PIPELINES", "2"))

    def load_model(
        self, model_name: str, *, with_ip_adapter: bool, identity_mode: Optional[str] = None
    ) -> Dict[str, Any]:
        """Пайплайн с подключённым IP-Adapter или без него.

        Кешируются отдельно: load_ip_adapter необратимо меняет объект. Режим FaceID входит в
        ключ кеша — у режимов разные веса и разный проектор."""
        mode_name = identity_mode or DEFAULT_IDENTITY_MODE
        if with_ip_adapter and mode_name not in IDENTITY_MODES:
            raise ValueError(f"Unknown identity mode: {mode_name}. Available: {list(IDENTITY_MODES)}")
        cache_key = f"{model_name}::ip_adapter={with_ip_adapter}::mode={mode_name if with_ip_adapter else '-'}"
        lock = self._locks.setdefault(cache_key, Lock())
        with lock:
            if cache_key in self.loaded_models:
                return self.loaded_models[cache_key]

            if model_name not in PHOTO_MODEL_CONFIGS:
                raise ValueError(f"Unknown photo model: {model_name}. Available: {list(PHOTO_MODEL_CONFIGS)}")

            self._evict_if_needed()
            config = PHOTO_MODEL_CONFIGS[model_name]
            # Явная ошибка вместо тихого игнорирования: FaceID обучен на SDXL UNet и к DiT
            # (Chroma, Flux) не подключается. Хуже всего вариант, когда load_ip_adapter отработает,
            # а личность до модели не долетит — это видно только по случайным лицам в выдаче.
            if with_ip_adapter and not config.get("supports_faceid", True):
                raise ValueError(
                    f"Чекпойнт '{model_name}' не поддерживает IP-Adapter FaceID (архитектура не "
                    f"SDXL UNet). Личность на нём достигается обучением персонажной LoRA, а не "
                    f"адаптером на инференсе. Для FaceID выберите SDXL-чекпойнт, например lustify."
                )
            mode = IDENTITY_MODES[mode_name] if with_ip_adapter else None
            pipeline = self._build_pipeline(model_name, config, mode)

            if with_ip_adapter:
                # Веса лежат в корне репозитория, папки image_encoder там нет. Оба параметра
                # обязательны, иначе 404 и попытка скачать несуществующий энкодер.
                pipeline.load_ip_adapter(
                    IDENTITY_CONFIG["ip_adapter_repo"],
                    subfolder=None,
                    weight_name=mode["weight_name"],
                    image_encoder_folder=None,
                )
                pipeline.set_ip_adapter_scale(mode["default_scale"])

                # У base и plusv2 в том же репозитории лежит парная LoRA; portrait своей не имеет.
                # Без LoRA веса грузятся и работают, но авторы обучали пару — сходство ниже.
                lora_name = mode.get("lora_weight_name")
                if lora_name:
                    pipeline.load_lora_weights(
                        IDENTITY_CONFIG["ip_adapter_repo"], weight_name=lora_name
                    )
                    # fuse, а не оставить активной: незафьюженная LoRA складывается с кастомным
                    # attention-процессором IP-Adapter на каждом шаге и даёт заметный оверхед,
                    # а менять её вес между кадрами мы всё равно не собираемся.
                    pipeline.fuse_lora()

            pipeline.to(DEVICE)
            # attention_slicing и xformers после load_ip_adapter ломают attention-процессор
            # адаптера (AttributeError: 'tuple' object has no attribute 'shape'). По VRAM они тут
            # и не нужны, поэтому к пайплайну с адаптером просто не применяем.
            if not with_ip_adapter:
                self._apply_optimizations(pipeline)

            model_info = {"pipeline": pipeline, "config": config, "loaded_at": datetime.now(timezone.utc).isoformat()}
            self.loaded_models[cache_key] = model_info
            return model_info

    def unload_all(self) -> str:
        """Выгружает все фото-пайплайны и возвращает VRAM.

        Нужно перед загрузкой видеомодели: карта одна. Два SDXL с адаптерами занимают ~44 GB
        из 48, и видео поверх них падает с OOM, хотя само помещается свободно.
        """
        freed = list(self.loaded_models)
        for key in freed:
            info = self.loaded_models.pop(key)
            pipeline = info.pop("pipeline", None)
            if pipeline is not None:
                try:
                    pipeline.to("cpu")
                except Exception:  # noqa: BLE001 — выгрузка не должна ронять запрос
                    pass
                del pipeline
            del info
        _release_vram()
        return ", ".join(freed) if freed else "нечего выгружать"

    def _evict_if_needed(self) -> None:
        """Выгружает самый давний пайплайн под новый.

        Порядок: убрать из словаря, снять ссылку, отдать память. empty_cache() раньше этого не
        освобождает ничего, и OOM повторяется на следующем чекпойнте.
        """
        while len(self.loaded_models) >= self.max_loaded:
            # Python 3.7+ хранит порядок вставки, поэтому первый ключ — самый давно загруженный.
            oldest = next(iter(self.loaded_models))
            info = self.loaded_models.pop(oldest)
            pipeline = info.pop("pipeline", None)
            if pipeline is not None:
                try:
                    # На CPU перед удалением: иначе освобождение блоков VRAM откладывается до
                    # сборки мусора, а она может не случиться до следующей аллокации.
                    pipeline.to("cpu")
                except Exception:  # noqa: BLE001 — выгрузка не должна ронять генерацию
                    pass
                del pipeline
            del info
            _release_vram()

    def _build_pipeline(self, model_name: str, config: Dict[str, Any], mode: Optional[Dict[str, Any]] = None):
        import torch

        pipeline_kwargs: Dict[str, Any] = {"torch_dtype": config["torch_dtype"]}
        if "variant" in config:
            pipeline_kwargs["variant"] = config["variant"]

        # PlusV2 работает на паре эмбеддингов, поэтому CLIP-энкодер нужен в пайплайне ДО
        # load_ip_adapter: досоздать потом нельзя, prepare_ip_adapter_image_embeds ищет
        # pipeline.image_encoder. В самом репозитории FaceID его нет — тянем отдельным.
        if mode and mode.get("needs_clip_encoder"):
            from transformers import CLIPVisionModelWithProjection

            pipeline_kwargs["image_encoder"] = CLIPVisionModelWithProjection.from_pretrained(
                mode["clip_encoder_repo"], torch_dtype=config["torch_dtype"]
            )

        if config["source"] == "huggingface":
            pipeline = config["pipeline_class"].from_pretrained(config["hf_repo_id"], **pipeline_kwargs)
        elif config["source"] == "civitai":
            model_path = self._ensure_civitai_model_downloaded(model_name, config)
            pipeline = config["pipeline_class"].from_single_file(
                str(model_path), use_safetensors=True, **pipeline_kwargs
            )
        else:
            raise ValueError(f"Unknown model source: {config['source']}")

        if "scheduler_class" in config:
            scheduler_kwargs = config.get("scheduler_kwargs", {})
            pipeline.scheduler = config["scheduler_class"].from_config(pipeline.scheduler.config, **scheduler_kwargs)

        if hasattr(pipeline, "set_progress_bar_config"):
            pipeline.set_progress_bar_config(disable=True)
        return pipeline

    def _ensure_civitai_model_downloaded(self, model_name: str, config: Dict[str, Any]) -> Path:
        """Аналог _ensure_model_downloaded() из WetDreams generation.py — тот же CIVITAI_TOKEN,
        та же схема каталогов (MODEL_CACHE_DIR/<model_name>/<filename>)."""
        model_dir = MODEL_CACHE_DIR / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        expected_path = model_dir / config["filename"]
        if expected_path.exists():
            return expected_path

        if not CIVITAI_TOKEN:
            raise RuntimeError(
                f"CIVITAI_TOKEN не задан, а модель '{model_name}' скачивается с Civitai. "
                "См. REQUIREMENTS_FROM_USER.md."
            )

        import requests

        url = f"https://civitai.com/api/download/models/{config['model_id']}?token={CIVITAI_TOKEN}"
        response = requests.get(url, stream=True, timeout=600)
        response.raise_for_status()
        with open(expected_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        return expected_path

    def _apply_optimizations(self, pipeline) -> None:
        if hasattr(pipeline, "enable_vae_tiling"):
            try:
                pipeline.enable_vae_tiling()
            except Exception:
                pass
        if hasattr(pipeline, "enable_attention_slicing"):
            try:
                pipeline.enable_attention_slicing()
            except Exception:
                pass
        if hasattr(pipeline, "enable_xformers_memory_efficient_attention"):
            try:
                pipeline.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

    def get_generation_lock(self) -> Lock:
        return self._generation_lock


def _build_face_embeds(request, mode: Dict[str, Any]):
    """Тензор ArcFace-эмбеддингов в форме, которую ждёт проектор выбранного режима.

    Половина тензора — нулевая (uncond) для CFG: при передаче через ip_adapter_image_embeds
    пайплайн её не достраивает, и без неё лицо не долетает до UNet.

    Разница режимов в средней оси:
      base / plusv2 — один референс, (2,1,1,512);
      portrait      — N референсов одного человека, (2,N,512).
    """
    import numpy as np
    import torch

    if mode.get("multi_reference"):
        # Ветвимся по режиму, а не по числу референсов: портретный проектор ждёт (b, n, 512)
        # всегда, в том числе при n=1. Иначе один эталон молча уходил бы в base-форму.
        vectors = request.face_embeddings or [request.face_embedding]
        limit = mode.get("default_reference_count")
        if limit and len(vectors) > limit:
            # Лишние ракурсы размывают личность: проектор усредняет их вместе с профилями.
            # Замерено: пять ракурсов дают 0.43 против 0.63 на одном. Берём первые N, первым
            # идёт фронтальный.
            vectors = vectors[:limit]
        cond = torch.cat(
            [
                torch.from_numpy(np.array(v, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
                for v in vectors
            ],
            dim=1,
        )
    else:
        source = request.face_embeddings[0] if request.face_embeddings else request.face_embedding
        embedding = torch.from_numpy(np.array(source, dtype=np.float32))
        cond = torch.stack([embedding.unsqueeze(0)], dim=0).unsqueeze(0)

    uncond = torch.zeros_like(cond)
    return torch.cat([uncond, cond]).to(dtype=torch.float16, device=DEVICE)


def _attach_clip_embeds(pipeline, request, mode: Dict[str, Any]) -> None:
    """PlusV2: CLIP-эмбеддинг эталона кладётся прямо в проекционный слой.

    Через публичный API его не передать — официальный пример diffusers тоже присваивает атрибутом.
    Отсюда требование reference_image_path: CLIP считается по картинке, вектора лица мало.
    """
    import torch
    from diffusers.utils import load_image

    if not request.reference_image_path:
        raise ValueError(
            "identity_mode=plusv2 требует reference_image_path (CLIP-эмбеддинг считается по "
            "изображению эталона, не по face_embedding). Для режима без эталонного кадра "
            "используйте identity_mode=base или portrait."
        )

    reference = load_image(str(request.reference_image_path))
    clip_embeds = pipeline.prepare_ip_adapter_image_embeds(
        [reference], None, torch.device(DEVICE), 1, True
    )[0]
    projection = pipeline.unet.encoder_hid_proj.image_projection_layers[0]
    projection.clip_embeds = clip_embeds.to(dtype=torch.float16)
    projection.shortcut = bool(mode.get("shortcut", False))


class PhotoGenerator:
    """Аналог ImageGenerator из WetDreams generation.py. generate() покрывает и Stage 0 (эталонное
    лицо, face_embedding=None) и Stage 2 (8 фото, face_embedding задан) — разница только в том,
    подключён ли IP-Adapter у загруженного пайплайна."""

    def __init__(self, model_manager: ModelManager, safety: SafetyPipeline, output_dir: Optional[Path] = None):
        self.model_manager = model_manager
        self.safety = safety
        self.output_dir = Path(output_dir) if output_dir else OUTPUT_DIR

    def generate(self, request: PhotoGenerationRequest) -> PhotoGenerationResult:
        import torch

        prompt_check = self.safety.run_prompt(request.prompt)
        if not prompt_check.allowed:
            raise ValueError(f"Prompt blocked by PromptGate: {prompt_check.matched_patterns}")

        with_ip_adapter = request.face_embedding is not None or bool(request.face_embeddings)
        mode_name = request.identity_mode or DEFAULT_IDENTITY_MODE
        model_info = self.model_manager.load_model(
            request.model_name, with_ip_adapter=with_ip_adapter, identity_mode=mode_name
        )
        pipeline = model_info["pipeline"]
        config = model_info["config"]

        seed = _next_seed(request.seed)
        generator = torch.Generator(device=DEVICE).manual_seed(seed)

        # Pony-чекпойнты обучены с качественными тегами в начале промпта (score_9, ...). Префикс
        # объявлен в конфиге модели, а не в шаблонах промптов, потому что это свойство конкретного
        # чекпойнта: на не-Pony моделях он только засоряет промпт.
        prompt = config.get("prompt_prefix", "") + request.prompt

        # Длинный промпт кодируется отдельно, иначе всё после 77-го токена отбрасывается молча.
        embeds = None
        text_negative = request.negative_prompt or config.get("negative_prompt")
        if len(prompt) > 280:
            embeds = _long_prompt_embeds(pipeline, prompt, text_negative if not with_ip_adapter else None)

        gen_kwargs: Dict[str, Any] = {
            "prompt": prompt,
            "width": request.width,
            "height": request.height,
            "num_inference_steps": request.num_inference_steps,
            "guidance_scale": request.guidance_scale,
            "generator": generator,
        }
        # Текстовый negative_prompt — только без IP-Adapter. Вместе с ip_adapter_image_embeds он
        # ломает attention_processor.py (известные баги diffusers #6832/#6914/#8863/#9448).
        # Наш тензор эмбеддингов уже несёт uncond-половину, так что CFG не страдает.
        if not with_ip_adapter:
            gen_kwargs["negative_prompt"] = text_negative

        if embeds is not None:
            conditioning, pooled, negative, negative_pooled = embeds
            # prompt и prompt_embeds взаимоисключающи: пайплайн падает, если передать оба.
            gen_kwargs.pop("prompt", None)
            gen_kwargs.pop("negative_prompt", None)
            gen_kwargs["prompt_embeds"] = conditioning
            gen_kwargs["pooled_prompt_embeds"] = pooled
            if negative is not None:
                gen_kwargs["negative_prompt_embeds"] = negative
                gen_kwargs["negative_pooled_prompt_embeds"] = negative_pooled

        if with_ip_adapter:
            mode = IDENTITY_MODES[mode_name]
            gen_kwargs["ip_adapter_image_embeds"] = [
                _build_face_embeds(request, mode)
            ]
            # PlusV2 помимо face-эмбеддинга требует CLIP-эмбеддинг эталонного кадра, проставленный
            # напрямую в проекционный слой UNet — параметра пайплайна для этого нет. Ровно как в
            # официальном примере diffusers для Plus/PlusV2.
            if mode.get("needs_clip_encoder"):
                _attach_clip_embeds(pipeline, request, mode)
            if request.ip_adapter_scale is not None:
                pipeline.set_ip_adapter_scale(request.ip_adapter_scale)

        start = time.monotonic()
        with self.model_manager.get_generation_lock():
            with torch.no_grad():
                output = pipeline(**gen_kwargs)
        duration_ms = int((time.monotonic() - start) * 1000)

        image = output.images[0]
        if image.mode != "RGB":
            image = image.convert("RGB")

        image_path = self.output_dir / f"{request.id}.png"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        image.save(image_path)

        # --- Safety: заполняем реально, а не оставляем None как в WetDreams ImageGenerator ---
        report = self.safety.run_image(image)
        nsfw_score = report.image_result.score if report.image_result else None
        estimated_age = report.identity_result.estimated_age if report.identity_result else None
        watchlist_sim = report.identity_result.max_watchlist_similarity if report.identity_result else None

        return PhotoGenerationResult(
            id=f"{request.id}_result",
            request_id=request.id,
            image_path=str(image_path),
            seed=seed,
            duration_ms=duration_ms,
            model_name=request.model_name,
            nsfw_score=nsfw_score,
            estimated_age=estimated_age,
            watchlist_similarity=watchlist_sim,
            safety_allowed=report.allowed,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def extract_face_embedding(self, image_path: str | Path) -> list[float]:
        """Эмбеддинг эталонного лица (Stage 0 -> вход Stage 2). Обёртка над worker.identity —
        отдельный метод здесь просто чтобы вызывающему коду (pipeline/generate_photos.py) не
        нужно было импортировать worker.identity напрямую."""
        from worker.identity import extract_embedding

        embedding = extract_embedding(image_path, IDENTITY_CONFIG["face_analysis_model"])
        return embedding.tolist()


_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager
