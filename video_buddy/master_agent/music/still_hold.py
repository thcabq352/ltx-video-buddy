"""Hard-fail still-holds: every MV window must be a real motion clip."""

from __future__ import annotations

from pathlib import Path
from typing import Any

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
STILL_MARKER = b"buddy-mv-still"
MOTION_MARKER = b"buddy-mv-motion-clip"
MIN_MOTION_FRAMES = 2


class StillHoldError(RuntimeError):
    """A window would be (or is) a frozen still instead of a motion clip."""


def window_span_frames(window: dict[str, Any]) -> int:
    try:
        return int(window["end_frame"]) - int(window["start_frame"])
    except (KeyError, TypeError, ValueError):
        return 0


def assert_window_not_still_hold(window: dict[str, Any], *, fps: int = 30) -> None:
    """Refuse a planned window that cannot hold motion (0/1 frame)."""
    idx = window.get("index", "?")
    span = window_span_frames(window)
    try:
        dur = float(window.get("end_s", 0)) - float(window.get("start_s", 0))
    except (TypeError, ValueError):
        dur = 0.0
    if span < MIN_MOTION_FRAMES or dur <= 0:
        raise StillHoldError(
            f"window {idx} is a still-hold: {span} frames / {dur:.3f}s @ {fps}fps"
        )


def assert_clip_not_still_hold(
    clip_path: str | Path,
    window: dict[str, Any] | None = None,
    *,
    dry_run: bool = False,
) -> None:
    """Refuse a burned file that is a still image, 1-frame clip, or hold marker."""
    idx = (window or {}).get("index", "?")
    path = Path(clip_path)
    if path.suffix.lower() in IMAGE_EXTS:
        raise StillHoldError(
            f"window {idx}: still image {path.name} cannot be a motion clip"
        )
    if not path.is_file():
        raise StillHoldError(f"window {idx}: missing clip {path}")
    head = path.read_bytes()[:64]
    if head.startswith(STILL_MARKER):
        raise StillHoldError(f"window {idx}: clip is an explicit still-hold")
    if dry_run:
        if not head.startswith(MOTION_MARKER):
            raise StillHoldError(
                f"window {idx}: dry-run clip is not a motion placeholder"
            )
        return

    from master_agent.judge.probe import (
        DEAD_MOTION_THRESHOLD,
        MIN_FRAMES,
        frame_motion_score,
        probe_video,
    )

    info = probe_video(path)
    frames = info.get("frames")
    if frames is not None and int(frames) < MIN_FRAMES:
        raise StillHoldError(
            f"window {idx}: still-hold — only {int(frames)} frames (min {MIN_FRAMES})"
        )
    motion = frame_motion_score(path)
    if motion is not None and float(motion) < DEAD_MOTION_THRESHOLD:
        raise StillHoldError(
            f"window {idx}: still-hold — dead motion score {motion:.3f}"
        )
