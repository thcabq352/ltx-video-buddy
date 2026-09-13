"""Estimate VRAM and wall time from variant + frame count before queueing."""

from __future__ import annotations

from typing import Any

# 16GB card budget. Separate fine-tunes, not LoRA stacks.
VARIANT_VRAM_GB = {
    "base": 9.5,
    "eros": 11.0,
    "directors": 13.5,
    "dasiwa": 12.0,
    "lipsync": 10.5,
    "wan22": 11.5,
}
SECONDS_PER_FRAME = {
    "base": 1.8,
    "eros": 2.0,
    "directors": 2.4,
    "dasiwa": 1.6,
    "lipsync": 2.2,
    "wan22": 2.1,
}


def estimate_cost(
    variant: str,
    frames: int,
    *,
    threshold_vram_gb: float = 14.5,
    threshold_time_s: float | None = None,
) -> dict[str, Any]:
    key = (variant or "base").strip().lower()
    try:
        from master_agent.models.vram_policy import expected_vram_gb

        base_vram = VARIANT_VRAM_GB.get(key) or expected_vram_gb(key)
    except Exception:
        base_vram = VARIANT_VRAM_GB.get(key, 10.0)
    # Longer clips add latent cache pressure.
    vram = base_vram + max(0, int(frames) - 81) * 0.012
    time_s = max(1, int(frames)) * SECONDS_PER_FRAME.get(key, 2.0)
    over_vram = vram > float(threshold_vram_gb)
    over_time = False if threshold_time_s is None else time_s > float(threshold_time_s)
    vram_min = round(vram * time_s / 60.0, 4)
    return {
        "variant": key,
        "frames": int(frames),
        "vram_gb": round(vram, 2),
        "time_s": round(time_s, 1),
        "vram_min": vram_min,
        "threshold_vram_gb": float(threshold_vram_gb),
        "flagged": over_vram or over_time,
        "reasons": [r for r, on in (("vram", over_vram), ("time", over_time)) if on],
    }
