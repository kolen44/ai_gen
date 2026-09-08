"""Покадровое восстановление лица в готовом ролике.

Зачем: на ростовом плане лицу достаётся ~90x120 пикселей кадра, то есть 6x8 латентных после VAE.
Текстуру кожи и блик в глазу там физически не из чего нарисовать, настройками генерации это не
чинится — информации просто нет в латенте.

  detailer — кроп лица, img2img через SDXL с FaceID, вклейка обратно. Медленно, зато лицо тянется
             к эталону персонажа, а не к абстрактному «красивому лицу».
  gfpgan   — сеть восстановления лиц, один проход на кадр. На порядок быстрее, но про личность не
             знает и слегка усредняет черты.

Оба обрабатывают кадры независимо, отсюда общий риск — мерцание. У детейлера оно гасится
фиксированным seed на весь ролик, у GFPGAN — подмешиванием исходного кадра (blend).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np


def _read_frames(video_path: Path) -> tuple[list, float]:
    import imageio.v3 as iio

    frames = iio.imread(str(video_path), plugin="pyav")
    # fps достаём из метаданных, а не подставляем 24: пересборка с чужой частотой меняет
    # длительность ролика, и это заметно.
    try:
        meta = iio.immeta(str(video_path), plugin="pyav")
        fps = float(meta.get("fps") or 24)
    except Exception:  # noqa: BLE001
        fps = 24.0
    return [np.asarray(f) for f in frames], fps


def _write_frames(frames: list, fps: float, output_path: Path) -> None:
    """Собирает mp4 из кадров.

    Через ffmpeg-плагин, а не pyav: тот падает с "Cannot change width after codec is open", если
    кадр отличается от первого хоть на пиксель, а детейлер вклеивает кроп с округлением.
    Заодно приводим кадры к размеру первого и добиваем стороны до чётных — H.264 нечётные
    не кодирует.
    """
    import imageio

    output_path.parent.mkdir(parents=True, exist_ok=True)

    height, width = frames[0].shape[:2]
    width -= width % 2
    height -= height % 2

    writer = imageio.get_writer(str(output_path), fps=fps, codec="libx264",
                                quality=8, macro_block_size=1)
    try:
        for frame in frames:
            if frame.shape[0] != height or frame.shape[1] != width:
                import cv2

                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
            writer.append_data(np.ascontiguousarray(frame[:height, :width]))
    finally:
        writer.close()


class GFPGANRestorer:
    """Обёртка над GFPGAN. Модель грузится лениво и один раз на процесс."""

    def __init__(self, model_dir: Optional[Path] = None, upscale: int = 1):
        self.model_dir = Path(model_dir) if model_dir else Path("/root/models/gfpgan")
        self.upscale = upscale
        self._restorer = None

    def _get(self):
        if self._restorer is not None:
            return self._restorer

        # basicsr (зависимость GFPGAN) импортирует torchvision.transforms.functional_tensor,
        # который убрали в torchvision 0.17+. Подсовываем псевдомодуль до импорта — это известная
        # несовместимость, а не наша ошибка, и чинится ровно так.
        import sys
        import types

        if "torchvision.transforms.functional_tensor" not in sys.modules:
            import torchvision.transforms.functional as F

            shim = types.ModuleType("torchvision.transforms.functional_tensor")
            shim.rgb_to_grayscale = F.rgb_to_grayscale
            sys.modules["torchvision.transforms.functional_tensor"] = shim

        from gfpgan import GFPGANer

        self.model_dir.mkdir(parents=True, exist_ok=True)
        weights = self.model_dir / "GFPGANv1.4.pth"
        if not weights.exists():
            import urllib.request

            url = ("https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/"
                   "GFPGANv1.4.pth")
            urllib.request.urlretrieve(url, weights)

        self._restorer = GFPGANer(
            model_path=str(weights),
            upscale=self.upscale,
            arch="clean",
            channel_multiplier=2,
            # bg_upsampler=None: фон не трогаем. Real-ESRGAN на фоне даёт «пластиковую» картинку,
            # а нам нужно только лицо — остальное в ролике и так на своём месте.
            bg_upsampler=None,
        )
        return self._restorer

    def restore(self, frame: np.ndarray, blend: float = 1.0) -> np.ndarray:
        """blend < 1 подмешивает исходный кадр к восстановленному.

        GFPGAN обрабатывает кадры независимо, и на соседних, отличающихся на доли пикселя, даёт
        заметно разные лица — на воспроизведении это дрожание. Смешивание с оригиналом его гасит:
        ниже blend — меньше мерцания и меньше резкости.
        """
        import cv2

        restorer = self._get()
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # only_center_face=True: в кадре бывают прохожие, и восстанавливать их лица незачем —
        # это лишнее время и лишний риск подмены не того лица.
        _, _, restored = restorer.enhance(
            bgr, has_aligned=False, only_center_face=True, paste_back=True)
        if restored is None:
            return frame
        result = cv2.cvtColor(restored, cv2.COLOR_BGR2RGB)
        if blend < 0.999:
            # Размеры расходятся при upscale>1 — приводим исходник к размеру результата.
            if result.shape != frame.shape:
                frame = cv2.resize(frame, (result.shape[1], result.shape[0]),
                                   interpolation=cv2.INTER_LANCZOS4)
            result = cv2.addWeighted(result, blend, frame, 1.0 - blend, 0)
        return result


def _run_detailer(frames, detailer, face_embedding, prompt, seed, every) -> int:
    """Перерисовывает лицо на каждом кадре через SDXL-детейлер с IP-Adapter FaceID.

    Тянет лицо К ЭТАЛОНУ персонажа — это единственный из двух методов, который вообще знает, на
    кого лицо должно быть похоже.
    """
    if detailer is None or face_embedding is None:
        raise ValueError("для detailer нужны detailer и face_embedding")

    import tempfile

    from PIL import Image

    processed = 0
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for index in range(len(frames)):
            if index % every:
                continue
            frame_path = tmp_dir / f"{index:04d}.png"
            Image.fromarray(frames[index]).save(frame_path)
            out = detailer.refine(
                frame_path, face_embedding, prompt=prompt,
                # Один seed на весь ролик: с разным seed диффузия даёт своё лицо на каждом кадре,
                # и ролик начинает мерцать.
                seed=seed,
                force=True,
                # Кадр не увеличиваем: размер ролика задан, а сборка mp4 требует одинаковых
                # кадров.
                min_face_px=0,
                output_path=frame_path,
            )
            if out.get("applied"):
                processed += 1
            frames[index] = np.asarray(Image.open(frame_path).convert("RGB"))
    return processed


def enhance_video(
    video_path: str | Path,
    output_path: str | Path,
    *,
    method: str = "gfpgan",
    face_embedding: Optional[list] = None,
    detailer=None,
    prompt: str = "",
    seed: int = 0,
    every: int = 1,
    upscale: int = 1,
    blend: float = 0.75,
) -> dict:
    """Прогоняет ролик через восстановление лица и пересобирает mp4.

    method="both" — сначала детейлер, потом GFPGAN, и порядок важен: пустить GFPGAN первым значит
    причесать лицо к его усреднённому виду, и детейлеру пришлось бы вытягивать личность из уже
    искажённого. В обратном порядке GFPGAN лишь добавляет чёткости готовому лицу.

    upscale > 1 растит кадр целиком, а не только лицо: резкое лицо на мыльном фоне выглядит
    наклейкой.

    every > 1 — только для быстрой пробы: пропуски дают пульсирование чёткости.
    """
    video_path = Path(video_path)
    output_path = Path(output_path)
    started = time.monotonic()

    frames, fps = _read_frames(video_path)
    processed = 0

    steps = ["detailer", "gfpgan"] if method == "both" else [method]
    for step in steps:
        if step == "detailer":
            processed += _run_detailer(frames, detailer, face_embedding, prompt, seed, every)
        elif step == "gfpgan":
            restorer = GFPGANRestorer(upscale=upscale)
            for index in range(len(frames)):
                if index % every:
                    continue
                frames[index] = restorer.restore(frames[index], blend=blend)
                processed += 1
        else:
            raise ValueError(f"неизвестный метод: {method}")

    _write_frames(frames, fps, output_path)
    return {
        "method": method,
        "frames_total": len(frames),
        "frames_processed": processed,
        "elapsed_ms": int((time.monotonic() - started) * 1000),
        "output_path": str(output_path),
    }
