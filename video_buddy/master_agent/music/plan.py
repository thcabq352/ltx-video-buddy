"""Stable beat-plan schema for music-video mode (Buddy MV).

Library entry: ``build_beat_plan`` / ``build_beat_plan_from_audio`` /
``synthetic_beat_plan``. CLI: ``python -m master_agent mv plan --audio …``.

Schema id: ``buddy.mv.beat_plan/v1``.
Windows carry index, start/end seconds, start/end frames at 30 fps, and
optional energy / section label.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from master_agent.music.beats import (
    BeatMap,
    analyze_audio,
    plan_shot_windows,
    section_at,
)

BEAT_PLAN_SCHEMA = "buddy.mv.beat_plan/v1"
MV_FPS = 30
MIN_WINDOW_FRAMES = 2


class BeatPlanError(ValueError):
    """Beat-plan math or schema failure."""


@dataclass
class BeatWindow:
    index: int
    start_s: float
    end_s: float
    start_frame: int
    end_frame: int
    energy: Optional[float] = None
    label: Optional[str] = None

    @property
    def duration_s(self) -> float:
        return float(self.end_s) - float(self.start_s)

    @property
    def duration_frames(self) -> int:
        return int(self.end_frame) - int(self.start_frame)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "index": int(self.index),
            "start_s": round(float(self.start_s), 6),
            "end_s": round(float(self.end_s), 6),
            "start_frame": int(self.start_frame),
            "end_frame": int(self.end_frame),
        }
        if self.energy is not None:
            payload["energy"] = round(float(self.energy), 4)
        if self.label:
            payload["label"] = str(self.label)
        return payload


@dataclass
class BeatPlan:
    fps: int
    duration_s: float
    duration_frames: int
    bpm: float
    windows: list[BeatWindow]
    audio_path: Optional[str] = None
    schema: str = BEAT_PLAN_SCHEMA
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "fps": int(self.fps),
            "duration_s": round(float(self.duration_s), 6),
            "duration_frames": int(self.duration_frames),
            "bpm": round(float(self.bpm), 2),
            "audio_path": self.audio_path,
            "windows": [w.to_dict() for w in self.windows],
        }
        if self.extras:
            payload.update(self.extras)
        return payload


def sec_to_frame(t: float, fps: int = MV_FPS) -> int:
    """Round seconds onto the integer frame grid."""
    return int(round(float(t) * int(fps)))


def frame_to_sec(frame: int, fps: int = MV_FPS) -> float:
    return float(frame) / float(fps)


def validate_beat_plan(plan: dict[str, Any] | BeatPlan) -> list[str]:
    """Return missing/invalid field names. Empty list = ok."""
    data = plan.to_dict() if isinstance(plan, BeatPlan) else plan
    missing: list[str] = []
    if data.get("schema") != BEAT_PLAN_SCHEMA:
        missing.append("schema")
    for key in ("fps", "duration_s", "duration_frames", "windows"):
        if key not in data:
            missing.append(key)
    fps = int(data.get("fps") or 0)
    if fps != MV_FPS and "fps" not in missing:
        # Allow only the MTV cut grid. Callers may pass fps=30 only.
        if fps <= 0:
            missing.append("fps")
    windows = data.get("windows")
    if not isinstance(windows, list) or not windows:
        missing.append("windows")
        return missing
    prev_end: Optional[int] = None
    for i, win in enumerate(windows):
        if not isinstance(win, dict):
            missing.append(f"windows[{i}]")
            continue
        for key in ("index", "start_s", "end_s", "start_frame", "end_frame"):
            if key not in win:
                missing.append(f"windows[{i}].{key}")
        try:
            start_f = int(win["start_frame"])
            end_f = int(win["end_frame"])
        except (KeyError, TypeError, ValueError):
            continue
        if end_f - start_f < MIN_WINDOW_FRAMES:
            missing.append(f"windows[{i}].still_hold")
        if prev_end is not None and start_f != prev_end:
            missing.append(f"windows[{i}].gap")
        prev_end = end_f
    return missing


def _energy_for(bmap: BeatMap, t: float) -> tuple[Optional[float], Optional[str]]:
    if not bmap.sections:
        return None, None
    sec = section_at(bmap.sections, t)
    energy = sec.get("energy")
    label = sec.get("label")
    try:
        energy_f = float(energy) if energy is not None else None
    except (TypeError, ValueError):
        energy_f = None
    return energy_f, str(label) if label else None


def windows_from_pairs(
    pairs: list[tuple[float, float]],
    *,
    duration_s: float,
    fps: int = MV_FPS,
    bmap: Optional[BeatMap] = None,
) -> list[BeatWindow]:
    """Snap beat-second windows onto a contiguous 30 fps frame grid.

    Seconds stay beat-aligned (from ``plan_shot_windows``). Frames are the
    rounded 30 fps projection, then stitched so they tile ``[0, duration_frames)``.
    A window that collapses below 2 frames is a still-hold and is refused.
    """
    from master_agent.music.still_hold import StillHoldError

    if fps <= 0:
        raise BeatPlanError("fps must be positive")
    duration_s = float(duration_s)
    if duration_s <= 0:
        raise BeatPlanError("duration_s must be positive")
    duration_frames = sec_to_frame(duration_s, fps)
    if duration_frames < MIN_WINDOW_FRAMES:
        raise StillHoldError(
            f"track is a still-hold: {duration_frames} frames @ {fps}fps"
        )
    if not pairs:
        raise StillHoldError("no beat windows — track would be a still-hold")

    starts = [sec_to_frame(s, fps) for s, _ in pairs]
    ends = [sec_to_frame(e, fps) for _, e in pairs]
    for i, (start_f, end_f) in enumerate(zip(starts, ends)):
        if end_f - start_f < MIN_WINDOW_FRAMES:
            raise StillHoldError(
                f"window {i} is a still-hold: {start_f}-{end_f} frames "
                f"({pairs[i][1] - pairs[i][0]:.3f}s @ {fps}fps)"
            )
    starts[0] = 0
    ends[-1] = duration_frames
    for i in range(1, len(starts)):
        starts[i] = ends[i - 1]

    windows: list[BeatWindow] = []
    for i, ((start_s, end_s), start_f, end_f) in enumerate(zip(pairs, starts, ends)):
        span = end_f - start_f
        if span < MIN_WINDOW_FRAMES:
            raise StillHoldError(
                f"window {i} is a still-hold: {start_f}-{end_f} frames "
                f"({end_s - start_s:.3f}s @ {fps}fps)"
            )
        energy, label = _energy_for(bmap, start_s) if bmap is not None else (None, None)
        windows.append(
            BeatWindow(
                index=i,
                start_s=float(start_s),
                end_s=float(end_s),
                start_frame=int(start_f),
                end_frame=int(end_f),
                energy=energy,
                label=label,
            )
        )
    return windows


def build_beat_plan_from_beatmap(
    bmap: BeatMap,
    *,
    fps: int = MV_FPS,
    audio_path: str | Path | None = None,
    min_s: float = 2.0,
    max_s: float = 6.0,
    high_energy_s: float = 2.0,
    low_energy_s: float = 4.0,
) -> BeatPlan:
    """Plan MTV windows from an existing ``BeatMap``."""
    from master_agent.config import MUSIC_DEFAULTS

    pairs = plan_shot_windows(
        bmap,
        min_s=min_s if min_s is not None else float(MUSIC_DEFAULTS["min_shot_s"]),
        max_s=max_s,
        high_energy_s=high_energy_s
        if high_energy_s is not None
        else float(MUSIC_DEFAULTS["high_energy_s"]),
        low_energy_s=low_energy_s
        if low_energy_s is not None
        else float(MUSIC_DEFAULTS["low_energy_s"]),
    )
    windows = windows_from_pairs(pairs, duration_s=bmap.duration_s, fps=fps, bmap=bmap)
    return BeatPlan(
        fps=int(fps),
        duration_s=float(bmap.duration_s),
        duration_frames=sec_to_frame(bmap.duration_s, fps),
        bpm=float(bmap.bpm),
        windows=windows,
        audio_path=str(audio_path) if audio_path else None,
        extras={"beats": [round(b, 4) for b in bmap.beats], "sections": bmap.sections},
    )


def build_beat_plan_from_audio(
    audio_path: str | Path,
    *,
    fps: int = MV_FPS,
    min_s: float = 2.0,
    max_s: float = 6.0,
    high_energy_s: float = 2.0,
    low_energy_s: float = 4.0,
) -> BeatPlan:
    """Analyze ``audio_path`` and return a ``buddy.mv.beat_plan/v1`` plan."""
    path = Path(audio_path)
    bmap = analyze_audio(path)
    return build_beat_plan_from_beatmap(
        bmap,
        fps=fps,
        audio_path=path,
        min_s=min_s,
        max_s=max_s,
        high_energy_s=high_energy_s,
        low_energy_s=low_energy_s,
    )


def synthetic_beatmap(duration_s: float, bpm: float) -> BeatMap:
    """Deterministic BeatMap for tests / dry-run without ffmpeg."""
    import numpy as np

    period = 60.0 / float(bpm) if bpm > 0 else 0.5
    beats = [float(t) for t in np.arange(0.0, float(duration_s), period)]
    mid = float(duration_s) / 2.0
    sections = [
        {"start_s": 0.0, "end_s": mid, "label": "verse", "energy": 0.25},
        {"start_s": mid, "end_s": float(duration_s), "label": "drop", "energy": 0.85},
    ]
    return BeatMap(
        bpm=float(bpm),
        beats=beats,
        downbeats=beats[::4],
        sections=sections,
        duration_s=float(duration_s),
    )


def synthetic_beat_plan(
    duration_s: float,
    bpm: float = 120.0,
    *,
    fps: int = MV_FPS,
    audio_path: str | Path | None = None,
) -> BeatPlan:
    """Beat plan from duration + tempo — no audio decode, no GPU."""
    return build_beat_plan_from_beatmap(
        synthetic_beatmap(duration_s, bpm),
        fps=fps,
        audio_path=audio_path,
    )


def build_beat_plan(
    audio_path: str | Path | None = None,
    *,
    bmap: BeatMap | None = None,
    fps: int = MV_FPS,
    duration_s: float | None = None,
    bpm: float | None = None,
    min_s: float = 2.0,
    max_s: float = 6.0,
    high_energy_s: float = 2.0,
    low_energy_s: float = 4.0,
) -> BeatPlan:
    """Unified library entry for the beat plan.

    Prefer ``audio_path`` (real track). Tests may pass ``bmap`` or
    ``duration_s`` + ``bpm`` (synthetic).
    """
    if bmap is not None:
        return build_beat_plan_from_beatmap(
            bmap,
            fps=fps,
            audio_path=audio_path,
            min_s=min_s,
            max_s=max_s,
            high_energy_s=high_energy_s,
            low_energy_s=low_energy_s,
        )
    if audio_path:
        return build_beat_plan_from_audio(
            audio_path,
            fps=fps,
            min_s=min_s,
            max_s=max_s,
            high_energy_s=high_energy_s,
            low_energy_s=low_energy_s,
        )
    if duration_s is None or bpm is None:
        raise BeatPlanError("need --audio, a BeatMap, or duration_s+bpm")
    return synthetic_beat_plan(duration_s, bpm, fps=fps, audio_path=audio_path)


def write_beat_plan(plan: BeatPlan | dict[str, Any], dest: str | Path) -> Path:
    path = Path(dest)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = plan.to_dict() if isinstance(plan, BeatPlan) else plan
    path.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    return path


def read_beat_plan(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise BeatPlanError("beat plan must be a JSON object")
    missing = validate_beat_plan(data)
    if missing:
        raise BeatPlanError(f"invalid beat plan: {', '.join(missing)}")
    return data


def beat_plan_from_dict(data: dict[str, Any]) -> BeatPlan:
    missing = validate_beat_plan(data)
    if missing:
        raise BeatPlanError(f"invalid beat plan: {', '.join(missing)}")
    windows = [
        BeatWindow(
            index=int(w["index"]),
            start_s=float(w["start_s"]),
            end_s=float(w["end_s"]),
            start_frame=int(w["start_frame"]),
            end_frame=int(w["end_frame"]),
            energy=float(w["energy"]) if w.get("energy") is not None else None,
            label=w.get("label"),
        )
        for w in data["windows"]
    ]
    return BeatPlan(
        fps=int(data["fps"]),
        duration_s=float(data["duration_s"]),
        duration_frames=int(data["duration_frames"]),
        bpm=float(data.get("bpm") or 0.0),
        windows=windows,
        audio_path=data.get("audio_path"),
        schema=str(data.get("schema") or BEAT_PLAN_SCHEMA),
    )
