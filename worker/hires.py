"""
Hires fix — второй проход по всему кадру для финишной детализации.

Зачем. 1024x1024 это родное разрешение SDXL, но не потолок качества: в кадре такого размера
физически нет пикселей под микротекстуру кожи, отдельные волосы, нити ткани. Стандартный приём
(в ComfyUI/A1111 он так и называется — hires fix): апскейлим готовый кадр до ~1.5x и прогоняем
img2img с низким strength. Модель не меняет композицию (шум добавляется слабый), но дорисовывает
детали уже в новом разрешении.

Порядок в пайплайне важен: сначала hires по всему кадру, потом face-detailer. Тогда детейлер
работает по лицу, которое уже стало в 1.5 раза крупнее в пикселях, и его кроп содержит больше
исходной информации.

Как и FaceDetailer, использует img2img-пайплайн, собранный из УЖЕ ЗАГРУЖЕННЫХ компонентов
основного txt2img — второй чекпойнт в VRAM не грузится, IP-Adapter тот же самый.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from worker.config import DEVICE, HIRES_CONFIG, IDENTITY_CONFIG  # noqa: E402


class HiresFix:
    """Апскейл + img2img-доводка всего кадра. Один экземпляр на прогон, пайплайн строится лениво."""

    def __init__(
        self,
        model_manager,
        *,
        model_name: str,
        ip_adapter_scale: Optional[float] = None,
    ):
        self.model_manager = model_manager
        self.model_name = model_name
        self.ip_adapter_scale = ip_adapter_scale or IDENTITY_CONFIG["default_scale"]
        self._img2img = None

    def _get_pipeline(self):
        if self._img2img is not None:
            return self._img2img

        from diffusers import StableDiffusionXLImg2ImgPipeline

        base = self.model_manager.load_model(self.model_name, with_ip_adapter=True)["pipeline"]
        pipeline = StableDiffusionXLImg2ImgPipeline(**base.components)
        if hasattr(pipeline, "set_progress_bar_config"):
            pipeline.set_progress_bar_config(disable=True)
        pipeline.to(DEVICE)

        # Найдено на реальном прогоне: на 1536x1536 декодирование VAE запрашивает ~2.25GB одним
        # куском и валит генерацию с OutOfMemory на 24GB карте (UNet к этому моменту уже держит
        # ~18.5GB). VAE-тайлинг режет декодирование на плитки и снимает именно этот пик.
        # Важно: трогаем только VAE. enable_attention_slicing() здесь применять НЕЛЬЗЯ — он ломает
        # attention-процессор IP-Adapter (см. комментарий в photo_generation.py), а identity нам
        # нужна и на этом проходе.
        for enable in ("enable_vae_tiling", "enable_vae_slicing"):
            if hasattr(pipeline, enable):
                try:
                    getattr(pipeline, enable)()
                except Exception:
                    pass

        self._img2img = pipeline
        return pipeline

    def refine(
        self,
        image_path: str | Path,
        face_embedding: Sequence[float],
        *,
        prompt: str,
        seed: int,
        scale: float = None,
        strength: float = None,
        num_inference_steps: int = None,
        guidance_scale: float = 6.0,
        output_path: Optional[str | Path] = None,
    ) -> dict:
        """Апскейлит кадр в `scale` раз и проходит по нему img2img с тем же identity-условием.

        Возвращает {applied, resolution, elapsed_ms, output_path}. Кадр перезаписывается на месте,
        если output_path не задан — дальше по пайплайну (face-detailer, safety, метрика) работают
        уже с улучшенной версией.
        """
        import torch
        from PIL import Image

        scale = scale if scale is not None else HIRES_CONFIG["scale"]
        strength = strength if strength is not None else HIRES_CONFIG["strength"]
        steps = num_inference_steps if num_inference_steps is not None else HIRES_CONFIG["num_inference_steps"]

        image_path = Path(image_path)
        image = Image.open(image_path).convert("RGB")

        # Кратность 8 обязательна для латентного пространства SDXL: VAE ужимает в 8 раз, размер
        # не кратный восьми молча округляется и даёт рассинхрон с исходником при вклейке.
        target = (
            int(round(image.width * scale / 8)) * 8,
            int(round(image.height * scale / 8)) * 8,
        )
        upscaled = image.resize(target, Image.LANCZOS)

        pipeline = self._get_pipeline()
        pipeline.set_ip_adapter_scale(self.ip_adapter_scale)

        # Та же форма (2,1,512), что в photo_generation.py и face_detailer.py: [uncond, cond] по
        # батч-оси. И так же без текстового negative_prompt — он конфликтует с
        # ip_adapter_image_embeds на SDXL.
        embedding = torch.from_numpy(np.array(face_embedding, dtype=np.float32))
        cond_embeds = torch.stack([embedding.unsqueeze(0)], dim=0).unsqueeze(0)
        uncond_embeds = torch.zeros_like(cond_embeds)
        id_embeds = torch.cat([uncond_embeds, cond_embeds]).to(dtype=torch.float16, device=DEVICE)

        generator = torch.Generator(device=DEVICE).manual_seed(seed)

        # Промпт длиннее 77 токенов режется текстовым энкодером молча — здесь ровно так же, как в
        # основной генерации. Проход hires переписывает весь кадр заново, и обрезанный промпт
        # означал бы, что доводка идёт по другому описанию, чем сама генерация: одежда и
        # обстановка из хвоста до неё не доехали бы.
        from worker.photo_generation import _long_prompt_embeds

        prompt_kwargs = {"prompt": prompt}
        if len(prompt) > 280:
            embeds = _long_prompt_embeds(pipeline, prompt, None)
            if embeds is not None:
                conditioning, pooled, _, _ = embeds
                prompt_kwargs = {"prompt_embeds": conditioning,
                                 "pooled_prompt_embeds": pooled}

        start = time.monotonic()
        with self.model_manager.get_generation_lock():
            with torch.no_grad():
                output = pipeline(
                    **prompt_kwargs,
                    image=upscaled,
                    strength=strength,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    ip_adapter_image_embeds=[id_embeds],
                )
        elapsed_ms = int((time.monotonic() - start) * 1000)

        # Освобождаем кеш аллокатора сразу: следующий шаг (face-detailer) снова просит несколько
        # гигабайт, а фрагментированный кеш после прохода на 1536x1536 остаётся занятым.
        torch.cuda.empty_cache()

        result = output.images[0]
        output_path = Path(output_path) if output_path else image_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(output_path)

        return {
            "applied": True,
            "resolution": f"{result.width}x{result.height}",
            "elapsed_ms": elapsed_ms,
            "output_path": str(output_path),
        }
