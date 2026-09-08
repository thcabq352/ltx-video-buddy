"""Persist Imagine corkboard panel records as JSON under a KB directory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_panels(store_dir: Path, records: list[dict[str, Any]]) -> Path:
    store_dir.mkdir(parents=True, exist_ok=True)
    path = store_dir / "corkboard_panels.json"
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return path


def load_panels(store_dir: Path) -> list[dict[str, Any]]:
    path = store_dir / "corkboard_panels.json"
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []
