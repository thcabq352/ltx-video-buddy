"""ffmpeg assembly for a segmented lipdub. No GPU.

Speech and mouth-bridge clips come from Comfy. This module trims them onto
the frame grid, holds a still or a closed-mouth frame across a pause, concats
in timeline order, and muxes the original wav.
"""

from __future__ import annotations

from pathlib import Path

from master_agent.proc import run_media
from master_agent.video_concat import find_ffmpeg


def _run(cmd: list[str]) -> None:
    proc = run_media(cmd, text=True)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "")[-600:]
        raise RuntimeError(f"ffmpeg failed: {err}")


def trim_span(src: Path, dest: Path, start_frame: int, count: int, *, fps: int) -> Path:
    """Keep ``count`` frames starting at ``start_frame`` (0-based). No audio."""
    ff = find_ffmpeg()
    if not ff:
        raise RuntimeError("ffmpeg not found on PATH; cannot trim a lipdub slice")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if count <= 0:
        raise RuntimeError(f"trim count must be positive ({src})")
    end = int(start_frame) + int(count) - 1
    vf = f"select='between(n,{int(start_frame)},{end})',setpts=N/{int(fps)}/TB"
    _run(
        [
            ff, "-y", "-i", str(src),
            "-vf", vf,
            "-an",
            "-r", str(int(fps)),
            "-frames:v", str(int(count)),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(dest),
        ]
    )
    if not dest.is_file() or dest.stat().st_size == 0:
        raise RuntimeError(f"trim produced no file: {dest}")
    return dest


def extract_frame(video: Path, dest: Path, index: int) -> Path:
    ff = find_ffmpeg()
    if not ff:
        raise RuntimeError("ffmpeg not found on PATH; cannot extract a continuity frame")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    vf = f"select=eq(n\\,{int(index)})"
    _run(
        [
            ff, "-y", "-i", str(video),
            "-vf", vf,
            "-vframes", "1",
            str(dest),
        ]
    )
    if not dest.is_file() or dest.stat().st_size == 0:
        raise RuntimeError(f"frame extract produced no file: {dest}")
    return dest


def even_size(width: int, height: int) -> tuple[int, int]:
    w = max(2, int(width) - (int(width) % 2))
    h = max(2, int(height) - (int(height) % 2))
    return w, h


def hold_image(
    image: Path,
    dest: Path,
    *,
    frames: int,
    fps: int,
    size: tuple[int, int] | None = None,
) -> Path:
    """Repeat one image for ``frames`` at ``fps``. Optional even pixel size."""
    ff = find_ffmpeg()
    if not ff:
        raise RuntimeError("ffmpeg not found on PATH; cannot hold a silence plate")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ff, "-y", "-loop", "1", "-i", str(image), "-frames:v", str(int(frames)), "-r", str(int(fps))]
    if size:
        w, h = even_size(*size)
        cmd += ["-vf", f"scale={w}:{h}:flags=lanczos,format=yuv420p"]
    else:
        cmd += ["-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p"]
    cmd += ["-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(dest)]
    _run(cmd)
    if not dest.is_file() or dest.stat().st_size == 0:
        raise RuntimeError(f"hold produced no file: {dest}")
    return dest


def concat_silent(paths: list[Path], dest: Path, *, fps: int) -> Path:
    """Re-encode concat so every piece shares one timebase. Audio is dropped."""
    ff = find_ffmpeg()
    if not ff:
        raise RuntimeError("ffmpeg not found on PATH; cannot concat lipdub slices")
    paths = [Path(p) for p in paths]
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if len(paths) == 1:
        _run(
            [
                ff, "-y", "-i", str(paths[0]),
                "-an", "-r", str(int(fps)),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                str(dest),
            ]
        )
        return dest
    list_path = dest.with_suffix(".concat.txt")
    lines = []
    for path in paths:
        escaped = path.resolve().as_posix().replace("'", r"'\''")
        lines.append(f"file '{escaped}'")
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _run(
        [
            ff, "-y",
            "-f", "concat", "-safe", "0", "-i", str(list_path),
            "-an", "-r", str(int(fps)),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(dest),
        ]
    )
    if not dest.is_file() or dest.stat().st_size == 0:
        raise RuntimeError(f"concat produced no file: {dest}")
    return dest


def mux_original_audio(video: Path, audio: Path, dest: Path) -> Path:
    """Map the full source wav onto the video as AAC. The wav is not sliced."""
    ff = find_ffmpeg()
    if not ff:
        raise RuntimeError("ffmpeg not found on PATH; cannot mux the original wav")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            ff, "-y",
            "-i", str(video),
            "-i", str(audio),
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", "48000",
            str(dest),
        ]
    )
    if not dest.is_file() or dest.stat().st_size == 0:
        raise RuntimeError(f"mux produced no file: {dest}")
    return dest


def video_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    ff = find_ffmpeg()
    if ff:
        probe = __import__("shutil").which("ffprobe")
        if probe:
            proc = run_media(
                [
                    probe, "-v", "error",
                    "-select_streams", "v:0",
                    "-show_entries", "stream=width,height",
                    "-of", "csv=p=0:s=x",
                    str(path),
                ],
                text=True,
            )
            text = (proc.stdout or "").strip()
            if proc.returncode == 0 and "x" in text:
                w, h = text.split("x", 1)
                return even_size(int(w), int(h))
    # Fall back to a decoded frame.
    frame = path.with_name(path.stem + "_size.png")
    extract_frame(path, frame, 0)
    with Image.open(frame) as im:
        return even_size(*im.size)
