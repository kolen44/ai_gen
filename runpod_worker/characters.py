"""Реестр персонажей: эталонный кадр и его эмбеддинг лица.

Зачем отдельный модуль. Личность персонажа — это не промпт, а вектор ArcFace, снятый с
эталонного кадра. Промпт можно переписать, вектор — нет: потеряв его, тот же человек больше не
получится, только похожий. Поэтому он хранится файлом на диске (на RunPod — на сетевом томе),
переживает пересоздание пода и версионируется вместе с эталонным кадром.

Формат: characters/<имя>.json + characters/<имя>.png (эталон).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from runpod_worker.config import CHARACTER_DIR


class CharacterError(RuntimeError):
    pass


def _paths(name: str) -> tuple[Path, Path]:
    CHARACTER_DIR.mkdir(parents=True, exist_ok=True)
    return CHARACTER_DIR / f"{name}.json", CHARACTER_DIR / f"{name}.png"


def exists(name: str) -> bool:
    return _paths(name)[0].exists()


def list_characters() -> List[dict]:
    if not CHARACTER_DIR.exists():
        return []
    out = []
    for meta in sorted(CHARACTER_DIR.glob("*.json")):
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        out.append({
            "name": meta.stem,
            "created_at": data.get("created_at", ""),
            "reference_image": data.get("reference_image"),
            "has_embedding": bool(data.get("embedding")),
            # Сколько ракурсов лежит в эталоне. Наружу отдаём именно число, а не сами векторы:
            # знать, работает ли портретный режим на полную, нужно, а гонять по сети килобайты
            # эмбеддингов на каждый список персонажей — нет.
            "reference_count": len(data.get("embeddings") or []) or (1 if data.get("embedding") else 0),
        })
    return out


def save(name: str, embedding: List[float], reference_image: Optional[Path] = None,
         *, overwrite: bool = False, embeddings: Optional[List[List[float]]] = None) -> dict:
    """Сохраняет персонажа.

    `embedding` — основной вектор (фронтальный эталон). Он же идёт в safety-гейт и в сверку
    похожести, поэтому остаётся одиночным и обязательным.

    `embeddings` — несколько ракурсов того же человека для портретного режима FaceID. Замерено на
    поде: portrait даёт 0.58 против 0.43 у base, причём base обваливается на отдельных сценах до
    0.29, а portrait держит 0.54+. Ради этого режим и существует, поэтому ракурсы хранятся рядом
    с персонажем, а не пересчитываются при каждом запросе.
    """
    meta_path, ref_path = _paths(name)
    if meta_path.exists() and not overwrite:
        raise CharacterError(f"персонаж {name!r} уже есть; передайте overwrite=true, "
                             f"чтобы заменить его эмбеддинг")

    if reference_image is not None:
        from PIL import Image

        Image.open(reference_image).convert("RGB").save(ref_path)

    data = {
        "name": name,
        "embedding": [float(x) for x in embedding],
        "embeddings": [[float(x) for x in vec] for vec in embeddings] if embeddings else None,
        "reference_image": str(ref_path) if reference_image is not None else None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


def load(name: str) -> dict:
    meta_path, _ = _paths(name)
    if not meta_path.exists():
        known = ", ".join(c["name"] for c in list_characters()) or "ни одного"
        raise CharacterError(f"персонаж {name!r} не найден; есть: {known}")
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    if not data.get("embedding"):
        raise CharacterError(f"у персонажа {name!r} нет эмбеддинга — эталон надо пересоздать")
    return data


def delete(name: str) -> bool:
    meta_path, ref_path = _paths(name)
    found = meta_path.exists()
    for p in (meta_path, ref_path):
        if p.exists():
            p.unlink()
    return found
