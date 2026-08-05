"""Stitch segment videos into one longer clip (ffmpeg preferred)."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


def find_ffmpeg() -> Optional[str]:
    return shutil.which("ffmpeg")


def concat_videos(paths: list[Path], dest: Path) -> Path:
    """Concatenate videos in order. Returns dest path."""
    paths = [Path(p) for p in paths if Path(p).is_file()]
    if not paths:
        raise FileNotFoundError("No segment videos to concatenate")
    if len(paths) == 1:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(paths[0], dest)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    ff = find_ffmpeg()
    if ff:
        with tempfile.TemporaryDirectory() as td:
            lst = Path(td) / "list.txt"
            # ffmpeg concat demuxer needs escaped paths
            lines = []
            for p in paths:
                ap = p.resolve().as_posix().replace("'", r"'\''")
                lines.append(f"file '{ap}'")
            lst.write_text("\n".join(lines), encoding="utf-8")
            cmd = [
                ff,
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(lst),
                "-c",
                "copy",
                str(dest),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode == 0 and dest.is_file():
                return dest
            # re-encode fallback
            cmd = [
                ff,
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(lst),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                str(dest),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode == 0 and dest.is_file():
                return dest
            raise RuntimeError(f"ffmpeg concat failed: {proc.stderr[-500:]}")

    # No ffmpeg: copy first and warn via name
    dest = dest.with_name(dest.stem + "_seg0_only" + dest.suffix)
    shutil.copy2(paths[0], dest)
    return dest


def mux_audio(video: Path, audio: Path, dest: Path) -> Path:
    """Mux an audio track onto a video (video stream copied, audio -> aac)."""
    ff = find_ffmpeg()
    if not ff:
        raise RuntimeError("ffmpeg not found on PATH; cannot mux audio")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ff,
        "-y",
        "-i",
        str(video),
        "-i",
        str(audio),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(dest),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not dest.is_file():
        raise RuntimeError(f"ffmpeg mux failed: {proc.stderr[-500:]}")
    return dest


def cut_to_windows(
    video: Path,
    windows: list[tuple[float, float]],
    out_dir: Path,
    *,
    prefix: str = "cut",
) -> list[Path]:
    """Trim a clip into (start_s, end_s) windows — one re-encoded mp4 each."""
    ff = find_ffmpeg()
    if not ff:
        raise RuntimeError("ffmpeg not found on PATH; cannot cut windows")
    video = Path(video)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outs: list[Path] = []
    for i, (start, end) in enumerate(windows):
        dest = out_dir / f"{prefix}_{i:03d}.mp4"
        cmd = [
            ff,
            "-y",
            "-ss",
            f"{start:.3f}",
            "-to",
            f"{end:.3f}",
            "-i",
            str(video),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(dest),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not dest.is_file():
            raise RuntimeError(
                f"ffmpeg cut window {i} ({start:.2f}-{end:.2f}) failed: {proc.stderr[-500:]}"
            )
        outs.append(dest)
    return outs
