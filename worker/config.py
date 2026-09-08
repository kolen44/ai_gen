"""Чекпойнты фото и видео, режимы фиксации лица, настройки из ENV.

Только объявления: ничего не скачивает и не грузит при импорте. Загрузка —
в photo_generation.py и video_generation.py.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from diffusers import (
    StableDiffusionXLPipeline,
    ChromaPipeline,
    EulerAncestralDiscreteScheduler,
    DPMSolverSDEScheduler,
    DPMSolverMultistepScheduler,
)

# === ENV / общие настройки ===
CIVITAI_TOKEN = os.getenv("CIVITAI_TOKEN")  # нужен только для чекпойнтов с source="civitai" ниже
HF_TOKEN = os.getenv("HF_TOKEN")  # нужен для gated: h94/IP-Adapter-FaceID, InsightFace antelopev2
MODEL_CACHE_DIR = Path(os.getenv("MODEL_CACHE_DIR", "./models"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./output"))

DEFAULT_WIDTH = int(os.getenv("DEFAULT_WIDTH", "1024"))
DEFAULT_HEIGHT = int(os.getenv("DEFAULT_HEIGHT", "1024"))
DEFAULT_STEPS = int(os.getenv("DEFAULT_STEPS", "28"))
DEFAULT_GUIDANCE_SCALE = float(os.getenv("DEFAULT_GUIDANCE_SCALE", "6.0"))

# Не падаем, если CUDA недоступна (например, при локальном прогоне логики без GPU) — но реальная
# генерация без cuda просто не будет вызвана, это чисто чтобы модуль импортировался везде.
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Негативный промпт для реалистичного стиля, перенесён из проверенного на проде профиля WetDreams
# (tg-bot/tg_wetdreams/services/image_models.py, ключ "realistic") — общий anatomy/quality négative,
# ничего специфичного для конкретного персонажа или контента.
REALISTIC_NEGATIVE_PROMPT = (
    "(worst quality, low quality:1.3), logo, watermark, signature, text, bad anatomy, bad hands, "
    "text, error, missing fingers, extra digit, fewer digits, cropped, jpeg artifacts, blurry, "
    "mutation, deformed, grayscale, lowres, (asymmetry eyes:1.2), (mismatched pupils), "
    "(cross-eyed), (dilated pupils), (bleeding eyes), (melting eyes), (fused eyes), poorly drawn eyes"
)

# То же плюс подавление "стоковости": вылизанная кожа, студийный свет, ретушь. Без этих слов
# SDXL-финетюны тянут в рекламную картинку, и "shot on iPhone" в промпте их не переубеждает.
LIFESTYLE_NEGATIVE_PROMPT = REALISTIC_NEGATIVE_PROMPT + (
    ", airbrushed, smooth plastic skin, porcelain skin, heavy makeup, beauty filter, "
    "instagram filter, oversaturated, hdr, 3d render, cgi, illustration, painting, "
    "studio lighting, softbox, stock photo, catalog photo, professional retouching, "
    "glamour shot, perfect symmetry, waxy skin"
)

# === Фото: чекпойнты ===
PHOTO_MODEL_CONFIGS = {
    # RealVisXL V5.0 — фотореалистичный финетюн SDXL. Ванильный sdxl-base даёт пластиковую кожу
    # и мыло. Тот же SDXL UNet, поэтому FaceID подключается без правок.
    # DPM++ 2M Karras чище на коже и волосах, чем Euler на том же числе шагов.
    "realvis-xl": {
        "source": "huggingface",
        "hf_repo_id": "SG161222/RealVisXL_V5.0",
        "pipeline_class": StableDiffusionXLPipeline,
        "scheduler_class": DPMSolverMultistepScheduler,
        "scheduler_kwargs": {"use_karras_sigmas": True, "algorithm_type": "dpmsolver++"},
        "torch_dtype": torch.float16,
        "variant": "fp16",
        "negative_prompt": LIFESTYLE_NEGATIVE_PROMPT,
    },
    # Juggernaut XL v9 — альтернатива по вкусу: свет теплее и контрастнее, чем нейтральный
    # у RealVisXL.
    "juggernaut-xl": {
        "source": "huggingface",
        "hf_repo_id": "RunDiffusion/Juggernaut-XL-v9",
        "pipeline_class": StableDiffusionXLPipeline,
        "scheduler_class": DPMSolverMultistepScheduler,
        "scheduler_kwargs": {"use_karras_sigmas": True, "algorithm_type": "dpmsolver++"},
        "torch_dtype": torch.float16,
        "variant": "fp16",
        "negative_prompt": LIFESTYLE_NEGATIVE_PROMPT,
    },
    # Ванильный SDXL 1.0 — оставлен как эталон сравнения и запасной вариант, но НЕ дефолт:
    # FaceID для него validated по документации h94/IP-Adapter-FaceID, зато фотореализм заметно
    # ниже любого специализированного финетюна.
    "sdxl-base": {
        "source": "huggingface",
        "hf_repo_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "pipeline_class": StableDiffusionXLPipeline,
        "scheduler_class": EulerAncestralDiscreteScheduler,
        "torch_dtype": torch.float16,
        "variant": "fp16",
        "negative_prompt": REALISTIC_NEGATIVE_PROMPT,
    },
    # Аниме-режим без Civitai-токена. Лицензия CreativeML Open RAIL++-M, коммерческое
    # использование разрешено.
    "animagine-xl": {
        "source": "huggingface",
        "hf_repo_id": "cagliostrolab/animagine-xl-3.1",
        "pipeline_class": StableDiffusionXLPipeline,
        "scheduler_class": EulerAncestralDiscreteScheduler,
        "torch_dtype": torch.float16,
        # Найдено на реальном прогоне: в этом репозитории нет отдельных fp16-variant файлов (не
        # у каждого HF-репо они есть) — variant="fp16" убран, torch_dtype=float16 всё равно
        # приводит веса к fp16 при загрузке, просто без отдельного набора файлов под этот вариант.
        "negative_prompt": REALISTIC_NEGATIVE_PROMPT,
    },
    # === Чекпойнты без контентных ограничений ===
    # Две независимые оси: цензура в весах и лицензия. Свободное по одной бывает закрыто по
    # другой, поэтому разведено явно.
    #
    # Chroma1-HD свободен по обеим: FLUX.1-schnell без фильтра, apache-2.0, рабочий CFG.
    # Но архитектура DiT, а не SDXL UNet — FaceID не подключается. Отсюда supports_faceid=False
    # и явная ошибка: иначе личность молча не долетала бы до модели.
    "chroma": {
        "source": "huggingface",
        "hf_repo_id": "lodestones/Chroma1-HD",
        "pipeline_class": ChromaPipeline,
        # bfloat16, а не float16: Flux-линейка на fp16 склонна давать NaN в attention.
        "torch_dtype": torch.bfloat16,
        "negative_prompt": REALISTIC_NEGATIVE_PROMPT,
        "supports_faceid": False,
        # ~19 GB весов + T5 — на 24 GB влезает только с оффлоадом, на 48 GB свободно.
        "needs_cpu_offload_below_gb": 40,
    },
    # LUSTIFY v2.0 — выбор по умолчанию: без цензуры, лицензия creativeml-openrail-m, и это
    # SDXL, поэтому identity-стек работает без правок.
    "lustify": {
        "source": "huggingface",
        "hf_repo_id": "John6666/lustify-sdxl-nsfwsfw-v2-sdxl",
        "pipeline_class": StableDiffusionXLPipeline,
        "scheduler_class": DPMSolverMultistepScheduler,
        "scheduler_kwargs": {"use_karras_sigmas": True, "algorithm_type": "dpmsolver++"},
        "torch_dtype": torch.float16,
        "negative_prompt": LIFESTYLE_NEGATIVE_PROMPT,
    },
    # NoobAI-XL 1.1 — аниме-линейка, сильнее Illustrious по охвату тегов и наполнению.
    # Лицензия "other" (fair-ai-public-license) — читать карточку перед коммерческим применением.
    "noobai-xl": {
        "source": "huggingface",
        "hf_repo_id": "Laxhar/noobai-XL-1.1",
        "pipeline_class": StableDiffusionXLPipeline,
        "scheduler_class": EulerAncestralDiscreteScheduler,
        "torch_dtype": torch.float16,
        "negative_prompt": REALISTIC_NEGATIVE_PROMPT,
    },

    # === Uncensored SDXL ===
    # Вместо Flux-uncensored: FLUX — DiT, к нему FaceID не подключается, то есть переход стоил бы
    # всей identity-механики. Эти финетюны дают то же снятие ограничений, оставаясь SDXL.
    # fp16-variant файлов ни у одного нет, поэтому "variant" не указан.
    "pony-realism-uncensored": {
        "source": "huggingface",
        "hf_repo_id": "nesaorg/PonyRealism-Uncensored",
        "pipeline_class": StableDiffusionXLPipeline,
        "scheduler_class": DPMSolverMultistepScheduler,
        "scheduler_kwargs": {"use_karras_sigmas": True, "algorithm_type": "dpmsolver++"},
        "torch_dtype": torch.float16,
        "negative_prompt": LIFESTYLE_NEGATIVE_PROMPT,
        # Pony-линейка обучена с качественными тегами в промпте: без префикса score_9 модель
        # выдаёт заметно более слабый результат. Это особенность обучения Pony, а не наша
        # стилизация, поэтому префикс живёт в конфиге модели, а не в промпт-шаблонах.
        "prompt_prefix": "score_9, score_8_up, score_7_up, ",
    },
    # Тот же Pony Realism v2.3, конвертация обкатаннее — скачиваний заметно больше.
    "pony-realism": {
        "source": "huggingface",
        "hf_repo_id": "John6666/pony-realism-v23-sdxl",
        "pipeline_class": StableDiffusionXLPipeline,
        "scheduler_class": DPMSolverMultistepScheduler,
        "scheduler_kwargs": {"use_karras_sigmas": True, "algorithm_type": "dpmsolver++"},
        "torch_dtype": torch.float16,
        "negative_prompt": LIFESTYLE_NEGATIVE_PROMPT,
        "prompt_prefix": "score_9, score_8_up, score_7_up, ",
    },
    # Illustrious XL Improved Uncensored v3.0 — аниме-линейка, не фотореализм. Прямая замена
    # animagine-xl выше там, где нужен аниме-стиль без ограничений. Для реалистичных людей брать
    # pony-realism-uncensored, не этот.
    "illustrious-uncensored": {
        "source": "huggingface",
        "hf_repo_id": "John6666/illustrious-xl10-improved-uncensored-v30-sdxl",
        "pipeline_class": StableDiffusionXLPipeline,
        "scheduler_class": EulerAncestralDiscreteScheduler,
        "torch_dtype": torch.float16,
        "negative_prompt": REALISTIC_NEGATIVE_PROMPT,
    },
}
DEFAULT_PHOTO_MODEL = os.getenv("DEFAULT_PHOTO_MODEL", "realvis-xl")

# === InstantID ===
# Пробовали как второй канал фиксации лица (ControlNet на 5 точках поверх FaceID). Не заработало:
# community-пайплайн несовместим с текущим diffusers, conditioning не долетал до UNet.
# Конфиг оставлен, код удалён.
INSTANTID_CONFIG = {
    "repo": "InstantX/InstantID",
    "controlnet_subfolder": "ControlNetModel",
    "ip_adapter_weight_name": "ip-adapter.bin",
    "pipeline_source_url": (
        "https://raw.githubusercontent.com/huggingface/diffusers/main/examples/community/"
        "pipeline_stable_diffusion_xl_instantid.py"
    ),
    "face_analysis_model": "buffalo_l",  # оригинальный InstantID использует antelopev2 — на этом
    # поде он ломается при автозагрузке (см. IDENTITY_CONFIG ниже); recognition-модель
    # (w600k_r50, ArcFace) в buffalo_l та же, что в antelopev2, так что embedding-пространство
    # совместимо, а kps (5 точек) берутся из детектора, который есть в обоих паках.
    "default_ip_adapter_scale": 0.8,
    "default_controlnet_conditioning_scale": 0.8,
    "image_emb_dim": 512,
    "num_tokens": 16,
}

# === IP-Adapter FaceID: три режима ===
# Отличаются не только файлом весов, но и кодом инференса (photo_generation.py::_build_face_embeds),
# поэтому режим — запись с флагами, а не строка.
# Веса лежат в корне репозитория, папки image_encoder там нет: subfolder=None и
# image_encoder_folder=None, иначе 404.
IDENTITY_MODES = {
    # --- base: один эмбеддинг, без CLIP. Проверенный рабочий режим этого пайплайна. ---
    "base": {
        "weight_name": "ip-adapter-faceid_sdxl.bin",
        "needs_clip_encoder": False,
        "multi_reference": False,
        # 0.95 вместо 0.8: InstantID не заработал, поэтому давим сильнее базовым FaceID.
        # Плата — чуть меньше вариативности позы между кадрами.
        "default_scale": 0.95,
    },
    # portrait — портретный проектор FaceID.
    #
    # Замерено на поде (A6000, realvis-xl, три сцены, один seed) — ракурсов эталона против
    # похожести лица:  1 → 0.653   2 → 0.463   3 → 0.448   5 → 0.432
    #
    # Рекомендация h94 «берите 5 изображений» у нас вредна: там пять фото живого человека, у нас
    # пять сгенерированных ракурсов одного лица. Проектор усредняет их, и личность уезжает от
    # фронтального эталона, с которым мы же и сравниваем. Появятся настоящие фото — перемерить.
    #
    # На одном ракурсе всё равно обходит base (0.653 против 0.616).
    # CLIP-энкодер не нужен, своей LoRA у этих весов нет.
    "portrait": {
        "weight_name": "ip-adapter-faceid-portrait_sdxl.bin",
        "needs_clip_encoder": False,
        "multi_reference": True,
        "default_reference_count": 1,
        # 0.85 — замерено, а не выбрано: на realvis-xl это лучший результат (0.653), 0.7 и 0.95
        # оба хуже. Полный перебор — benchmarking/refsweep_realvis.json.
        "default_scale": 0.85,
    },
    # plusv2 — FaceID плюс CLIP-эмбеддинг: раздельный контроль лица и структуры.
    # Нужен отдельный CLIP-энкодер и ручная простановка clip_embeds в проекционном слое UNet —
    # без этого режим молча не работает.
    # shortcut=True — это и есть отличие v2 от v1, а не опция на вкус.
    "plusv2": {
        "weight_name": "ip-adapter-faceid-plusv2_sdxl.bin",
        "needs_clip_encoder": True,
        "clip_encoder_repo": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        "shortcut": True,
        "multi_reference": False,
        # У PlusV2 есть парная LoRA в том же репозитории. Без неё веса работают, но авторы
        # рассчитывали на пару — сходство заметно ниже.
        "lora_weight_name": "ip-adapter-faceid-plusv2_sdxl_lora.safetensors",
        # Ниже, чем у base: CLIP-половина добавляет своё влияние поверх face-эмбеддинга, и на
        # 0.95 суммарный лок пережимает промпт (сцена перестаёт слушаться текста).
        "default_scale": 0.7,
    },
}

DEFAULT_IDENTITY_MODE = os.getenv("DEFAULT_IDENTITY_MODE", "portrait")

IDENTITY_CONFIG = {
    "ip_adapter_repo": "h94/IP-Adapter-FaceID",
    "face_analysis_model": "buffalo_l",  # antelopev2 проверен на реальном поде и оказался
    # неполным при автозагрузке (assert 'detection' in self.models — известная проблема с
    # хостингом этого конкретного пака); buffalo_l — тот же тип модели (InsightFace, ArcFace
    # эмбеддинг + возраст), надёжно грузится из официального зоопарка insightface.
    "mode": DEFAULT_IDENTITY_MODE,
    # Оставлены для обратной совместимости с кодом, который читал плоский конфиг до появления
    # IDENTITY_MODES. Значения берутся из выбранного режима, чтобы не разъезжались.
    "ip_adapter_weight_name": IDENTITY_MODES[DEFAULT_IDENTITY_MODE]["weight_name"],
    "default_scale": IDENTITY_MODES[DEFAULT_IDENTITY_MODE]["default_scale"],
}

# === Видео ===
# pipeline_class_name — строка, а не класс: резолвится лениво в video_generation.py. Разные версии
# diffusers по-разному называют видео-пайплайны, и config.py не должен падать при импорте там,
# где видео вообще не нужно.
VIDEO_MODEL_CONFIGS = {
    # Wan2.2-I2V-A14B: 117.5 GB весов (два эксперта по 14B в bf16). Обещание карточки «пойдёт на
    # 4090» относится к квантованной GGUF для ComfyUI, а не к этому чекпойнту. На карту меньше
    # 80 GB и 32 GB RAM не влезает даже с оффлоадом — нужен под пожирнее.
    "wan2.2-i2v-a14b": {
        "source": "huggingface",
        "hf_repo_id": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        "pipeline_class_name": "WanImageToVideoPipeline",
        "torch_dtype": torch.bfloat16,
        # Родное разрешение I2V-A14B — 1280x720. Прежние 624x624 стояли от попытки уместить модель
        # на 24 GB и резали качество вдвое: модель обучена на 720p, и на меньшем размере она даёт
        # рыхлую картинку, а не просто мельче.
        "default_width": 1280,
        "default_height": 720,
        # 81 кадр при 16 fps — те же пять секунд, что 121 при 24. Больше кадров на этой модели
        # упирается в память внимания: латент 720p вчетверо тяжелее, чем у 5B.
        "default_num_frames": 81,
        "default_fps": 16,
        "default_num_inference_steps": 40,
        "default_guidance_scale": 3.5,
        # Оффлоад обязателен: A14B — MoE, эксперты работают на разных стадиях шумоподавления,
        # и оффлоад держит на карте одного (28.6 GB) вместо обоих (57.2 GB). Освободившиеся
        # ~50 GB уходят на внимание, без которого 720p не считается.
        # На плотной 5B тот же оффлоад был чистым вредом — там модель и так помещалась.
        "requires_cpu_offload": True,
        # Порог намеренно выше любой существующей карты: прямая загрузка обоих экспертов
        # (63 GB весов) не оставляет памяти на внимание при 720p даже на 80 GB.
        "min_vram_gb_for_direct": 200,
        # LightX2V — дистилляция числа шагов: 4 вместо 40. Замерено: 61.6 минуты без неё против
        # 13.2 с ней на 81 кадре 720p (оффлоад подменяет экспертов на каждом шаге, шаг тут дорогой).
        # LoRA две, по одной на эксперта: high на ранних шагах, low на поздних. Одна на оба —
        # это чужие веса в чужом эксперте.
        "speed_lora": {
            "repo": "rzgar/Wan2.2_LightX2V_4Step_Uncensored",
            "high": "Wan2.2_LightX2V_high_n54vv.safetensors",
            "low": "Wan2.2_LightX2V_low_n54vv.safetensors",
            "steps": 4,
            # CFG отключается: дистилляция обучена работать без него, и guidance>1 на 4 шагах
            # даёт пережжённую картинку.
            "guidance_scale": 1.0,
        },
    },
    # Wan2.2-TI2V-5B — вариант для одной 4090: плотная 5B, репозиторий 34.2 GB.
    # WanPipeline из карточки модели не принимает image= вообще (TypeError), поэтому грузим тот же
    # чекпойнт через WanImageToVideoPipeline. VAE — отдельно в float32, см. needs_wan_vae_override.
    # «Помещается на 4090» в карточке относится к их скрипту с оффлоадом внутри: прямой вызов дал
    # OOM на одних весах (23.38 из 23.52 GiB). Отсюда requires_cpu_offload.
    "wan2.2-ti2v-5b": {
        "source": "huggingface",
        "hf_repo_id": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "pipeline_class_name": "WanImageToVideoPipeline",
        "needs_wan_vae_override": True,
        "requires_cpu_offload": True,
        # Порог, выше которого оффлоад не нужен: на такой карте пайплайн помещается
        # целиком, и оффлоад только гоняет веса по шине, роняя скорость в разы.
        "min_vram_gb_for_direct": 40,
        "torch_dtype": torch.bfloat16,
        # Родное разрешение модели — 720p@24fps, 480p она не поддерживает. Стоявшие тут 704x704
        # давали ~40% от пикселей, на которые модель обучена, отсюда мыло. Кадр приводится к 16:9
        # кадрированием в video_generation.py, а не растягиванием.
        "default_width": 1280,
        "default_height": 704,
        "default_num_frames": 121,  # ~5.04s @ 24fps; карточка модели рекомендует <=120 кадров
        "default_fps": 24,
        "default_num_inference_steps": 50,  # официальный дефолт модели-карты
        "default_guidance_scale": 5.0,
    },
}
DEFAULT_VIDEO_MODEL = os.getenv("DEFAULT_VIDEO_MODEL", "wan2.2-ti2v-5b")

