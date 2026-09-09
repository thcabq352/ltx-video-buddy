"""Priority render queue: cheap, high-confidence scenes first."""

from __future__ import annotations

from typing import Any


def order_scenes(scenes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(item: dict[str, Any]) -> tuple:
        flagged = 1 if item.get("flagged") else 0
        cost = float(item.get("vram_gb") or item.get("cost") or 0.0)
        conf = float(item.get("confidence") or item.get("judge_confidence") or 0.0)
        return (flagged, cost, -conf)

    return sorted(scenes, key=key)
