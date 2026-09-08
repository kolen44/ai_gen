"""Seed для всех этапов: один base_seed на персонажа, остальное — производные от него.

Нужно для воспроизводимости: любой кадр можно пересобрать теми же числами.

    plan = SeedPlan("nova", 20260907)
    plan.reference_seed()          # эталонное лицо
    plan.photo_seed(3, attempt=1)  # кадр 3, вторая попытка
    plan.video_seeds(3)            # (первый кадр, движение)
"""

from dataclasses import dataclass

# Шаги подобраны так, чтобы диапазоны разных этапов не пересекались.
PHOTO_STEP = 1000       # между кадрами — чтобы композиция не повторялась
ATTEMPT_STEP = 37       # между попытками одного кадра — иначе отбор лучшей бессмыслен
VIDEO_SOURCE_BASE = 9000
VIDEO_SOURCE_STEP = 100
MOTION_OFFSET = 500_000


@dataclass(frozen=True)
class SeedPlan:
    character_name: str
    base_seed: int

    def reference_seed(self) -> int:
        return self.base_seed

    def photo_seed(self, index: int, attempt: int = 0) -> int:
        """Кадр index, попытка attempt. Личность от seed не зависит — её держит эмбеддинг лица."""
        if index < 1:
            raise ValueError("index начинается с 1")
        return self.base_seed + PHOTO_STEP * index + ATTEMPT_STEP * attempt

    def video_source_seed(self, number: int) -> int:
        """Кадр-исходник под ролик. Своя область, чтобы не совпасть с кадрами фотопака."""
        return self.base_seed + VIDEO_SOURCE_BASE + VIDEO_SOURCE_STEP * number

    def video_seeds(self, photo_index: int) -> tuple[int, int]:
        """Первый кадр и движение сэмплируются раздельно, поэтому seed два."""
        frame_seed = self.photo_seed(photo_index)
        return frame_seed, frame_seed + MOTION_OFFSET


if __name__ == "__main__":
    plan = SeedPlan("example", 20260907)
    print("эталон:", plan.reference_seed())
    for i in range(1, 4):
        print(f"кадр {i}:", plan.photo_seed(i), "| попытка 2:", plan.photo_seed(i, 1))
    print("исходник под ролик 1:", plan.video_source_seed(1))
    print("видео из кадра 3:", plan.video_seeds(3))
