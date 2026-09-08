"""
Считает стоимость одной генерации из двух измеренных чисел: время и тариф GPU.
Ничего не оценивает "на глаз" — вход всегда реальные секунды с пода и реальный $/час
с биллинг-страницы провайдера. Используется для заполнения benchmarking/benchmark_log.csv.

Пример:
    python cost_calculator.py --seconds 12.4 --hourly-rate 0.39
    python cost_calculator.py --seconds 45 --hourly-rate 0.79 --label "video_ltx_5s"
"""

import argparse
from dataclasses import dataclass


@dataclass
class CostResult:
    seconds: float
    hourly_rate_usd: float
    cost_usd: float
    label: str | None = None

    def __str__(self) -> str:
        prefix = f"[{self.label}] " if self.label else ""
        return (
            f"{prefix}{self.seconds:.1f}s @ ${self.hourly_rate_usd:.2f}/hr "
            f"-> ${self.cost_usd:.5f} per generation"
        )


def generation_cost(seconds: float, hourly_rate_usd: float, label: str | None = None) -> CostResult:
    if seconds < 0 or hourly_rate_usd < 0:
        raise ValueError("seconds и hourly_rate_usd не могут быть отрицательными")
    cost = (seconds / 3600.0) * hourly_rate_usd
    return CostResult(seconds=seconds, hourly_rate_usd=hourly_rate_usd, cost_usd=cost, label=label)


def batch_photo_cost(per_photo_seconds: float, hourly_rate_usd: float, count: int = 8) -> dict:
    """Стоимость всей серии из N фото + суммарное время."""
    per_photo = generation_cost(per_photo_seconds, hourly_rate_usd, label="photo")
    return {
        "count": count,
        "per_photo_cost_usd": per_photo.cost_usd,
        "total_cost_usd": per_photo.cost_usd * count,
        "total_seconds": per_photo_seconds * count,
    }


def session_cost(
    cold_start_seconds: float,
    reference_seconds: float,
    photo_seconds_each: float,
    video_seconds: float,
    hourly_rate_usd: float,
    photo_count: int = 8,
) -> dict:
    """
    Полная стоимость сессии "от поднятия пода до готового результата".
    cold_start_seconds — время установки окружения/загрузки весов, тарифицируется по тому же
    GPU-часу, но это не стоимость "одной генерации" — фиксированная стоимость на сессию,
    поэтому считается отдельной строкой, а не размазывается по фото (см. PLAN.md, п.6).
    """
    active_seconds = reference_seconds + photo_seconds_each * photo_count + video_seconds
    total_seconds = cold_start_seconds + active_seconds
    return {
        "cold_start_cost_usd": generation_cost(cold_start_seconds, hourly_rate_usd).cost_usd,
        "reference_cost_usd": generation_cost(reference_seconds, hourly_rate_usd).cost_usd,
        "photos_total_cost_usd": generation_cost(photo_seconds_each * photo_count, hourly_rate_usd).cost_usd,
        "video_cost_usd": generation_cost(video_seconds, hourly_rate_usd).cost_usd,
        "session_total_cost_usd": generation_cost(total_seconds, hourly_rate_usd).cost_usd,
        "session_total_seconds": total_seconds,
    }


def _cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, required=True, help="измеренное время генерации, сек")
    parser.add_argument("--hourly-rate", type=float, required=True, help="тариф GPU, $/час")
    parser.add_argument("--label", type=str, default=None)
    args = parser.parse_args()
    print(generation_cost(args.seconds, args.hourly_rate, args.label))


if __name__ == "__main__":
    _cli()
