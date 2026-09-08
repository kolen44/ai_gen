"""Второй проход по лицу для кадров, где оно слишком мелкое.

FaceID подмешивает эмбеддинг в внимание всего кадра. Когда лицо занимает меньше 5% площади
(ростовой план), латентных токенов на него приходится слишком мало, и модель рисует «человека
вообще» в нужной композиции — черты уходят от эталона.

Лечится стандартным приёмом: находим лицо, расширяем bbox, масштабируем кроп до 1024, гоняем
img2img с тем же эмбеддингом и умеренным strength, вклеиваем обратно по растушёванной маске.

Пайплайн img2img собирается из уже загруженных компонентов txt2img с подключённым адаптером —
второй SDXL в VRAM не грузится. Компоненты берём именно у варианта с адаптером: у «голого»
применён attention_slicing, который ломает attention-процессор IP-Adapter.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from worker.config import DEVICE, IDENTITY_CONFIG  # noqa: E402
from worker.identity import get_face_analysis  # noqa: E402

# Порог "лицо мелкое": высота bbox лица относительно высоты кадра. 0.12 — эмпирически граница,
# ниже которой на нашем наборе начинался дрейф (портреты и medium shot дают 0.25-0.6 и в
# детейлере не нуждаются).
DEFAULT_MIN_FACE_RATIO = 0.12
# Во сколько раз расширяем bbox лица перед кропом: нужно захватить волосы, подбородок и немного
# фона, иначе img2img дорисовывает обрезанную голову и шов виден при вклейке обратно.
DEFAULT_PADDING = 1.6
# Насколько сильно перерисовываем кроп. 0.45 — компромисс: ниже ~0.35 черты почти не меняются
# (дрейф остаётся), выше ~0.6 модель начинает менять ракурс и освещение, и вклейка перестаёт
# сходиться с остальным кадром.
DEFAULT_STRENGTH = 0.45
DEFAULT_CROP_SIZE = 1024
# Радиус растушёвки маски вклейки в пикселях кропа (до уменьшения обратно).
DEFAULT_FEATHER = 48
# Минимальная высота лица в пикселях, к которой стремимся на выходе. Ниже примерно 300 px лицо
# читается как мыло независимо от модели: не хватает пикселей на текстуру кожи, ресницы и блик в
# глазу. Замерено на реальных кадрах — лицо 120 px размыто, 216 px уже держит текстуру.
DEFAULT_MIN_FACE_PX = 320
# Предел увеличения всего кадра. Больше 2x смысла нет: апскейл не создаёт информации, которой не
# было, а вес файла растёт квадратично.
DEFAULT_MAX_UPSCALE = 2.0

FACE_PROMPT_TEMPLATE = (
    "close-up portrait photo of a {age}-year-old {ethnicity} woman, {hair}, {eyes}, "
    "{distinguishing_features}, photorealistic, high detail skin texture, sharp focus, 85mm lens"
)


class FaceDetailer:
    """Второй проход по лицу. Держит один img2img-пайплайн, построенный поверх компонентов
    основного txt2img — собственных весов не грузит."""

    def __init__(
        self,
        model_manager,
        *,
        model_name: str,
        ip_adapter_scale: Optional[float] = None,
        face_analysis_model: Optional[str] = None,
    ):
        self.model_manager = model_manager
        self.model_name = model_name
        self.ip_adapter_scale = ip_adapter_scale or IDENTITY_CONFIG["default_scale"]
        self.face_analysis_model = face_analysis_model or IDENTITY_CONFIG["face_analysis_model"]
        self._img2img = None

    # --- пайплайн ---

    def _get_pipeline(self):
        if self._img2img is not None:
            return self._img2img

        from diffusers import StableDiffusionXLImg2ImgPipeline

        base = self.model_manager.load_model(self.model_name, with_ip_adapter=True)["pipeline"]
        # Компоненты (включая unet с уже подключённым IP-Adapter) переиспользуются по ссылке —
        # это и экономит ~7GB VRAM, и гарантирует, что детейлер работает тем же адаптером и тем
        # же чекпойнтом, что и основная генерация.
        pipeline = StableDiffusionXLImg2ImgPipeline(**base.components)
        if hasattr(pipeline, "set_progress_bar_config"):
            pipeline.set_progress_bar_config(disable=True)
        pipeline.to(DEVICE)
        self._img2img = pipeline
        return pipeline

    # --- геометрия лица ---

    def find_face_box(self, image) -> Optional[tuple[int, int, int, int]]:
        """bbox самого уверенного лица, либо None.

        Не нашлось — повторяем на кадре с полями: детектор не видит лица крупнее своего большого
        якоря. Координаты возвращаем в систему исходного кадра, иначе кроп уедет.
        """
        app = get_face_analysis(self.face_analysis_model)
        img_arr = np.array(image.convert("RGB"))

        faces = app.get(img_arr)
        if faces:
            face = max(faces, key=lambda f: f.det_score)
            x1, y1, x2, y2 = (int(v) for v in face.bbox)
            return x1, y1, x2, y2

        pad_factor = 0.6
        h, w = img_arr.shape[:2]
        pad_h, pad_w = int(h * pad_factor), int(w * pad_factor)
        padded = np.pad(img_arr, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode="edge")

        faces = app.get(padded)
        if not faces:
            return None

        face = max(faces, key=lambda f: f.det_score)
        x1, y1, x2, y2 = (int(v) for v in face.bbox)
        # Обратно в координаты исходного кадра + зажим в его границы: лицо, доходящее до края,
        # после сдвига может дать отрицательную координату.
        x1, x2 = max(0, x1 - pad_w), min(w, x2 - pad_w)
        y1, y2 = max(0, y1 - pad_h), min(h, y2 - pad_h)
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def face_ratio(self, image, box: tuple[int, int, int, int]) -> float:
        """Высота лица относительно высоты кадра — по ней решаем, нужен ли второй проход."""
        _, y1, _, y2 = box
        return (y2 - y1) / float(image.height)

    def _padded_square(self, image, box: tuple[int, int, int, int], padding: float) -> tuple[int, int, int, int]:
        """Расширенный квадратный кроп вокруг лица, прижатый к границам кадра. Квадрат — потому
        что SDXL ждёт близкое к 1:1 разрешение, а нести в него вытянутый прямоугольник значит
        получить искажённые пропорции лица на выходе."""
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        half = max(x2 - x1, y2 - y1) * padding / 2.0

        # Если запрошенный квадрат больше кадра — ограничиваем стороной кадра, иначе после
        # прижатия к границам получится не квадрат и пропорции поедут.
        half = min(half, image.width / 2.0, image.height / 2.0)
        cx = min(max(cx, half), image.width - half)
        cy = min(max(cy, half), image.height - half)
        return int(cx - half), int(cy - half), int(cx + half), int(cy + half)

    @staticmethod
    def _feather_mask(size: tuple[int, int], feather: int):
        """Маска вклейки: белый прямоугольник с отступом, размытый по Гауссу. Отступ = feather,
        чтобы после размытия края маски гарантированно дошли до нуля на границе кропа и шва
        не было видно."""
        from PIL import Image, ImageDraw, ImageFilter

        mask = Image.new("L", size, 0)
        inset = feather
        ImageDraw.Draw(mask).rectangle(
            [inset, inset, size[0] - inset, size[1] - inset], fill=255
        )
        return mask.filter(ImageFilter.GaussianBlur(radius=feather / 2.0))

    # --- основной проход ---

    def refine(
        self,
        image_path: str | Path,
        face_embedding: Sequence[float],
        *,
        prompt: str,
        seed: int,
        strength: float = DEFAULT_STRENGTH,
        padding: float = DEFAULT_PADDING,
        crop_size: int = DEFAULT_CROP_SIZE,
        feather: int = DEFAULT_FEATHER,
        num_inference_steps: int = 28,
        guidance_scale: float = 6.0,
        min_face_ratio: float = DEFAULT_MIN_FACE_RATIO,
        min_face_px: int = DEFAULT_MIN_FACE_PX,
        max_upscale: float = DEFAULT_MAX_UPSCALE,
        force: bool = False,
        output_path: Optional[str | Path] = None,
    ) -> dict:
        """Перерисовывает лицо на кадре и сохраняет результат.

        applied=False — штатный исход, а не ошибка: лицо крупное и проход не нужен, либо лицо
        не найдено вовсе. Кадр в этом случае остаётся нетронутым.
        """
        import torch
        from PIL import Image

        image_path = Path(image_path)
        image = Image.open(image_path).convert("RGB")

        box = self.find_face_box(image)
        if box is None:
            return {"applied": False, "reason": "лицо не найдено", "face_ratio": None,
                    "elapsed_ms": 0, "output_path": str(image_path)}

        ratio = self.face_ratio(image, box)
        if not force and ratio >= min_face_ratio:
            return {"applied": False, "reason": f"лицо крупное (ratio={ratio:.3f})", "face_ratio": ratio,
                    "elapsed_ms": 0, "output_path": str(image_path)}

        trigger = "по метрике" if force and ratio >= min_face_ratio else f"лицо мелкое (ratio={ratio:.3f})"

        crop_box = self._padded_square(image, box, padding)
        crop = image.crop(crop_box)
        crop_original_size = crop.size
        crop_upscaled = crop.resize((crop_size, crop_size), Image.LANCZOS)

        pipeline = self._get_pipeline()
        pipeline.set_ip_adapter_scale(self.ip_adapter_scale)

        # Та же форма тензора, что в photo_generation.py: (2,1,512) = [uncond=0, cond=emb] по
        # батч-оси для classifier-free guidance. И так же не передаём текстовый negative_prompt —
        # вместе с ip_adapter_image_embeds он ломает attention-процессор на SDXL.
        embedding = torch.from_numpy(np.array(face_embedding, dtype=np.float32))
        cond_embeds = torch.stack([embedding.unsqueeze(0)], dim=0).unsqueeze(0)
        uncond_embeds = torch.zeros_like(cond_embeds)
        id_embeds = torch.cat([uncond_embeds, cond_embeds]).to(dtype=torch.float16, device=DEVICE)

        generator = torch.Generator(device=DEVICE).manual_seed(seed)

        start = time.monotonic()
        with self.model_manager.get_generation_lock():
            with torch.no_grad():
                output = pipeline(
                    prompt=prompt,
                    image=crop_upscaled,
                    strength=strength,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    ip_adapter_image_embeds=[id_embeds],
                )
        elapsed_ms = int((time.monotonic() - start) * 1000)
        torch.cuda.empty_cache()

        # Лицо рендерится в 1024, и раньше результат сжимался обратно до размера кропа — для
        # ростового кадра это ~260 px, то есть вся детализация выбрасывалась в ресайзе. Косинус
        # этого не замечал (личность в низких частотах), а лицо оставалось мылом.
        # Поэтому увеличиваем весь кадр ровно настолько, чтобы отрендеренное лицо влезло без
        # потерь. Фон становится мягче — это осознанный размен.
        face_height = box[3] - box[1]
        upscale = 1.0
        if face_height and face_height < min_face_px:
            upscale = min(max_upscale, min_face_px / face_height)

        if upscale > 1.01:
            image = image.resize(
                (round(image.width * upscale), round(image.height * upscale)), Image.LANCZOS)
            crop_box = tuple(round(v * upscale) for v in crop_box)
            crop_original_size = (crop_box[2] - crop_box[0], crop_box[3] - crop_box[1])
            feather = max(8, round(feather * upscale))

        refined = output.images[0].resize(crop_original_size, Image.LANCZOS)
        mask = self._feather_mask(crop_original_size, feather)

        result = image.copy()
        result.paste(refined, crop_box[:2], mask)

        output_path = Path(output_path) if output_path else image_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(output_path)

        return {
            "applied": True,
            "reason": trigger,
            "face_ratio": ratio,
            "upscale": round(upscale, 2),
            "face_px_after": round(face_height * upscale),
            "elapsed_ms": elapsed_ms,
            "output_path": str(output_path),
        }


def build_face_prompt(character: dict) -> str:
    """Промпт для второго прохода — только описание лица персонажа, без сцены. Сцена уже есть в
    исходном кадре и сохраняется за счёт img2img со средним strength; повторять её в промпте
    вредно (модель начинает перерисовывать фон внутри кропа)."""
    return FACE_PROMPT_TEMPLATE.format(
        age=character["age"],
        ethnicity=character["ethnicity"],
        hair=character["hair"],
        eyes=character["eyes"],
        distinguishing_features=character["distinguishing_features"],
    )
