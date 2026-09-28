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
    info: dict[str, Any] = {
        "exists": False,
        "size_bytes": 0,
        "duration_s": None,
        "frames": None,
        "has_audio": False,
    }
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
        info["has_audio"] = _probe_has_audio(ffprobe, p)
    except Exception as e:
        info["probe_error"] = str(e)
    return info


def _probe_has_audio(ffprobe: str, path: Path) -> bool:
    try:
        out = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=index",
                "-of",
                "csv=p=0",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return bool((out.stdout or "").strip())
    except Exception:
        return False


def media_frame_count(path: str | Path) -> int:
    """Decoded frame count, or duration×24 when the container omits nb_frames."""
    info = probe_video(path)
    frames = info.get("frames")
    if isinstance(frames, int) and frames > 0:
        return frames
    duration = info.get("duration_s")
    if duration:
        return max(int(round(float(duration) * 24)), 1)
    return 0


def even_frame_indices(n_frames: int, count: int) -> list[int]:
    """``count`` indices spread across the whole clip, earliest first."""
    n = int(n_frames)
    if n <= 0:
        return []
    want = max(1, min(int(count), n))
    if want == 1:
        return [n // 2]
    return [int(round(i * (n - 1) / (want - 1))) for i in range(want)]


def extract_sampled_jpegs(
    path: str | Path,
    count: int,
    *,
    scale_width: int = 384,
    dest_dir: Path,
) -> list[tuple[Path, float]]:
    """Evenly spaced JPEGs across the whole clip, earliest first.

    ``fps=N, -frames:v count`` only decodes the first ``count/N`` seconds.
    Image-conditioned clips often hold the still at the head, so that sample
    is frozen and the vision model describes the wrong frames. Index select
    covers the full timeline, and ``-fps_mode vfr`` stops the image muxer
    from repeating frame 0.
    """
    ffmpeg = _which("ffmpeg")
    p = Path(path)
    if not ffmpeg or not p.is_file():
        return []
    n_frames = media_frame_count(p)
    indices = even_frame_indices(n_frames, count)
    if not indices:
        return []
    info = probe_video(p)
    duration = float(info.get("duration_s") or 0.0)
    fps = (n_frames / duration) if duration > 0 else 24.0
    expr = "+".join(f"eq(n,{i})" for i in indices)
    width = max(int(scale_width), 16)
    pattern = str(Path(dest_dir) / "f_%03d.jpg")
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(p),
        "-vf",
        f"select='{expr}',scale={width}:-2",
        "-fps_mode",
        "vfr",
        "-frames:v",
        str(len(indices)),
        pattern,
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=60)
    except Exception:
        return []
    frames = sorted(Path(dest_dir).glob("f_*.jpg"))
    timed: list[tuple[Path, float]] = []
    for path_i, index in zip(frames, indices):
        timed.append((path_i, round(index / fps, 4)))
    return timed


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
            sampled = extract_sampled_jpegs(
                p, samples, scale_width=160, dest_dir=Path(td)
            )
            frames = [fp for fp, _t in sampled]
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
