"""Video probing + cheap heuristic quality analysis (ffprobe/ffmpeg).

Trimmed port of LTX Project agent/quality_correction.py probe helpers.
This is the judge's heuristic leg: no LLM, no GPU, just file facts and a
frame-diff motion score.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

TINY_FILE_BYTES = 100_000
MIN_FRAMES = 3
SHORT_DURATION_RATIO = 0.6
# Motion score below this means a frozen / slideshow-like clip
DEAD_MOTION_THRESHOLD = 0.05
HEALTH_ISSUE_CODES = frozenset(
    {"missing_video", "tiny_file", "short_duration", "too_few_frames"}
)


def _which(name: str) -> Optional[str]:
    return shutil.which(name)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def probe_video(path: str | Path | None) -> dict[str, Any]:
    info: dict[str, Any] = {"exists": False, "size_bytes": 0, "duration_s": None, "frames": None}
    p = Path(path) if path else None
    if p is None or not p.is_file():
        return info
    info["exists"] = True
    info["size_bytes"] = p.stat().st_size
    ffprobe = _which("ffprobe")
    if not ffprobe:
        return info
    try:
        out = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_frames,duration,width,height",
                "-show_entries",
                "format=duration,size",
                "-of",
                "json",
                str(p),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        data = json.loads(out.stdout or "{}")
        streams = data.get("streams") or []
        fmt = data.get("format") or {}
        if streams:
            st = streams[0]
            if st.get("duration"):
                info["duration_s"] = float(st["duration"])
            if st.get("nb_frames") and str(st["nb_frames"]).isdigit():
                info["frames"] = int(st["nb_frames"])
            info["width"] = st.get("width")
            info["height"] = st.get("height")
        if info["duration_s"] is None and fmt.get("duration"):
            info["duration_s"] = float(fmt["duration"])
    except Exception as e:
        info["probe_error"] = str(e)
    return info


def frame_motion_score(path: str | Path, samples: int = 6) -> Optional[float]:
    """
    Cheap temporal activity score via ffmpeg frame extracts + pixel diffs.
    Returns 0–1-ish (higher = more motion). None if tools/deps missing.
    """
    ffmpeg = _which("ffmpeg")
    p = Path(path)
    if not ffmpeg or not p.is_file():
        return None
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        return None

    try:
        with tempfile.TemporaryDirectory() as td:
            pattern = str(Path(td) / "f_%03d.jpg")
            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-i",
                    str(p),
                    "-vf",
                    "fps=2,scale=160:-1",
                    "-frames:v",
                    str(samples),
                    pattern,
                ],
                capture_output=True,
                timeout=60,
            )
            frames = sorted(Path(td).glob("f_*.jpg"))
            if len(frames) < 2:
                return 0.0
            diffs = []
            prev = None
            for fp in frames:
                arr = np.asarray(Image.open(fp).convert("L"), dtype=np.float32) / 255.0
                if prev is not None:
                    diffs.append(float(np.mean(np.abs(arr - prev))))
                prev = arr
            if not diffs:
                return 0.0
            # Typical small motion ~0.01–0.08; normalize softly
            mean_diff = sum(diffs) / len(diffs)
            return float(_clamp(mean_diff / 0.05, 0.0, 1.0))
    except Exception:
        return None


def analyze(
    video_path: str | Path | None,
    *,
    expected_duration_s: float | None = None,
) -> tuple[float, list[dict[str, Any]]]:
    """
    Heuristic quality score (0–1) + issue list for one generated clip.
    Issue codes: missing_video, tiny_file, short_duration, too_few_frames, dead_motion.
    """
    issues: list[dict[str, Any]] = []
    p = Path(video_path) if video_path else None
    if p is None or not p.is_file():
        issues.append({"code": "missing_video", "severity": 1.0, "detail": "No output video file"})
        return 0.0, issues

    score = 1.0
    probe = probe_video(p)
    frames = probe.get("frames")
    if frames is not None and int(frames) < MIN_FRAMES:
        issues.append(
            {
                "code": "too_few_frames",
                "severity": 1.0,
                "detail": f"only {int(frames)} frames (min {MIN_FRAMES})",
            }
        )
        score = min(score, 0.3)
    size = int(probe.get("size_bytes") or 0)
    if size < TINY_FILE_BYTES:
        issues.append(
            {"code": "tiny_file", "severity": 1.0, "detail": f"Output file only {size} bytes"}
        )
        score = min(score, 0.3)

    duration = probe.get("duration_s")
    if (
        expected_duration_s
        and duration is not None
        and duration < expected_duration_s * SHORT_DURATION_RATIO
    ):
        issues.append(
            {
                "code": "short_duration",
                "severity": 0.8,
                "detail": f"duration {duration:.2f}s < expected {expected_duration_s:.2f}s",
            }
        )
        score = min(score, 0.4)

    motion = frame_motion_score(p)
    if motion is not None and motion < DEAD_MOTION_THRESHOLD:
        issues.append(
            {
                "code": "dead_motion",
                "severity": 0.5,
                "detail": f"motion_score {motion:.3f} (frozen/slideshow-like)",
            }
        )
        score = min(score, 0.6)

    return round(score, 3), issues


def frame_notes(video_path: str | Path | None, samples: int = 5) -> str:
    """One-line summary fed to the LLM judge."""
    p = Path(video_path) if video_path else None
    if p is None or not p.is_file():
        return "no video file"
    try:
        probe = probe_video(p)
        motion = frame_motion_score(p, samples=samples)
        parts = [
            f"size_bytes={probe.get('size_bytes')}",
            f"duration_s={probe.get('duration_s')}",
            f"WxH={probe.get('width')}x{probe.get('height')}",
            f"motion_score={motion}",
        ]
        return ", ".join(str(x) for x in parts)
    except Exception as e:
        return f"probe_failed: {e}"
