"""Fractal pipeline: deep-zoom, inpaint, or outpaint video.

Modes:
  - zoom (default): pure Mandelbrot/Julia deep-zoom
  - inpaint: fill a hole (center ellipse or custom mask) with animated fractal
  - outpaint: expand the canvas around a photo and fill borders with fractal

Optional beat-reactive audio + mux. Emits kind: "fractal" run records.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from master_agent.config import OUTPUTS_DIR, RUNS_DIR
from master_agent.fractal.render import (
    MODES,
    PALETTES,
    TARGETS,
    center_hole_mask,
    ensure_even_size,
    load_mask,
    load_rgb,
    prepare_outpaint,
    render_paint_video,
    render_zoom_video,
    soft_mask_from_gray,
)


def _clamp_int(v, lo: int, hi: int, default: int) -> int:
    try:
        n = int(v)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, n))


def _clamp_float(v, lo: float, hi: float, default: float) -> float:
    try:
        n = float(v)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, n))


def run_fractal(
    brief: str = "",
    *,
    mode: str = "zoom",
    duration_s: float = 20.0,
    fps: int = 24,
    width: int = 768,
    height: int = 512,
    target: str = "seahorse",
    palette: str = "fire",
    seed: int | None = None,
    julia: bool = False,
    audio_path: str | Path | None = None,
    image_path: str | Path | None = None,
    mask_path: str | Path | None = None,
    expand: int = 128,
    cover: float = 0.4,
    feather: int = 28,
    max_iter: int = 256,
    log=print,
) -> dict:
    """Render a fractal video (zoom / inpaint / outpaint)."""
    mode = (mode or "zoom").strip().lower()
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

    duration_s = _clamp_float(duration_s, 0.5, 600.0, 20.0)
    fps = _clamp_int(fps, 1, 60, 24)
    # H.264 / yuv420p requires even dimensions
    width = max(2, _clamp_int(width, 64, 3840, 768) & ~1)
    height = max(2, _clamp_int(height, 64, 2160, 512) & ~1)
    expand = _clamp_int(expand, 0, 1024, 128)
    cover = _clamp_float(cover, 0.05, 0.95, 0.4)
    feather = _clamp_int(feather, 0, 256, 28)
    max_iter = _clamp_int(max_iter, 32, 2048, 256)
    if target not in TARGETS:
        target = "seahorse"
    if palette not in PALETTES:
        palette = "fire"

    run_id = uuid.uuid4().hex[:12]
    out_dir = OUTPUTS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    beat_map = None
    if audio_path:
        from master_agent.music.beats import analyze_audio

        beat_map = analyze_audio(audio_path)
        log(
            f"beat map: {beat_map.bpm:.0f} BPM, {len(beat_map.beats)} beats, "
            f"{len(beat_map.sections)} sections"
        )
        duration_s = max(duration_s, float(beat_map.duration_s))

    dest = out_dir / f"fractal_{run_id}.mp4"

    if mode == "zoom":
        log(f"fractal mode=zoom {width}x{height} {duration_s:.1f}s")
        render_zoom_video(
            dest,
            duration_s=duration_s,
            fps=fps,
            width=width,
            height=height,
            target=target,
            palette=palette,
            max_iter=max_iter,
            beat_map=beat_map,
            julia=julia,
            seed=seed,
            log=log,
        )
        out_w, out_h = width, height
    else:
        if not image_path:
            raise ValueError(f"mode={mode} requires image_path")
        img_path = Path(image_path)
        if not img_path.is_file():
            raise FileNotFoundError(f"image not found: {img_path}")

        base = ensure_even_size(load_rgb(img_path))
        if mode == "outpaint":
            base, mask = prepare_outpaint(base, expand=expand, feather=feather)
            base = ensure_even_size(base)
            mask = ensure_even_size(mask)
            log(
                f"fractal mode=outpaint expand={expand}px "
                f"canvas={base.shape[1]}x{base.shape[0]}"
            )
        else:  # inpaint
            if mask_path:
                mpath = Path(mask_path)
                if not mpath.is_file():
                    raise FileNotFoundError(f"mask not found: {mpath}")
                gray = load_mask(mpath, base.shape[:2])
                mask = soft_mask_from_gray(gray, feather=feather)
                log(f"fractal mode=inpaint custom mask from {mpath.name}")
            else:
                mask = center_hole_mask(
                    base.shape[0],
                    base.shape[1],
                    cover=cover,
                    feather=feather,
                )
                log(f"fractal mode=inpaint center hole cover={cover}")
            mask = ensure_even_size(mask)
        out_h, out_w = base.shape[:2]
        render_paint_video(
            dest,
            base_rgb=base,
            mask=mask,
            duration_s=duration_s,
            fps=fps,
            target=target,
            palette=palette,
            max_iter=max_iter,
            beat_map=beat_map,
            julia=julia,
            seed=seed,
            log=log,
        )

    video_path = dest
    if audio_path:
        from master_agent.video_concat import mux_audio

        muxed = out_dir / f"fractal_{run_id}_mux.mp4"
        mux_audio(dest, Path(audio_path), muxed)
        log(f"muxed audio: {muxed}")
        video_path = muxed

    record = {
        "run_id": run_id,
        "kind": "fractal",
        "request": brief,
        "status": "done",
        "video_path": str(video_path.resolve()),
        "params": {
            "mode": mode,
            "duration_s": duration_s,
            "fps": fps,
            "width": out_w,
            "height": out_h,
            "target": target,
            "palette": palette,
            "julia": bool(julia),
            "seed": seed,
            "max_iter": max_iter,
            "expand": expand if mode == "outpaint" else None,
            "cover": cover if mode == "inpaint" and not mask_path else None,
            "feather": feather if mode != "zoom" else None,
        },
        "beat_map": beat_map.to_dict() if beat_map else None,
        "audio_path": str(audio_path) if audio_path else None,
        "image_path": str(image_path) if image_path else None,
        "mask_path": str(mask_path) if mask_path else None,
    }
    _write_record(run_id, record, log=log)
    return record


def _write_record(run_id: str, record: dict, *, log=print) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RUNS_DIR / f"{ts}_{run_id}_fractal.json"
    try:
        path.write_text(
            json.dumps(record, indent=1, default=str) + "\n", encoding="utf-8"
        )
        log(f"run record: {path}")
    except OSError as e:
        log(f"could not write run record: {e}")
    try:
        from master_agent.kb.ingest import ingest_run_record

        if ingest_run_record(record):
            log("kb: run record ingested")
    except Exception:
        pass


__all__ = ["run_fractal", "TARGETS", "PALETTES", "MODES"]
