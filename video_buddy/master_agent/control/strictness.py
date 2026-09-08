"""Judge strictness dial: 0–1 scales every threshold on all three legs."""

from __future__ import annotations

from dataclasses import dataclass

from master_agent.config import JUDGE_SCORE_THRESHOLD


@dataclass
class Strictness:
    value: float
    project: float | None = None
    scene: float | None = None

    def effective(self) -> float:
        for item in (self.scene, self.project, self.value):
            if item is not None:
                return max(0.0, min(1.0, float(item)))
        return 0.5


def scale_threshold(base: float, strictness: float, *, floor: float = 0.40, ceil: float = 0.95) -> float:
    s = max(0.0, min(1.0, float(strictness)))
    b = float(base)
    if s <= 0.5:
        return round(floor + (b - floor) * (s / 0.5), 4)
    return round(b + (ceil - b) * ((s - 0.5) / 0.5), 4)


def leg_thresholds(strictness: float, *, base: float | None = None) -> dict[str, float]:
    b = float(base if base is not None else JUDGE_SCORE_THRESHOLD)
    combined = scale_threshold(b, strictness)
    return {
        "heuristic": combined,
        "llm": combined,
        "vision": scale_threshold(0.75, strictness, floor=0.45, ceil=0.92),
        "combined": combined,
        "strictness": max(0.0, min(1.0, float(strictness))),
    }


def passes(score: float, strictness: float, *, base: float | None = None) -> bool:
    return float(score) >= leg_thresholds(strictness, base=base)["combined"]
