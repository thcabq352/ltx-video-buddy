"""Fractal pipeline: render deep-zoom video, optional beat-reactive + audio mux.

Emits the same run-record shape as the other pipelines (state/runs/) so the
knowledge base and future job queue treat it uniformly (kind: "fractal").
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from master_agent.config import OUTPUTS_DIR, RUNS_DIR
from master_agent.fractal.render import PALETTES, TARGETS, render_zoom_video


def run_fractal(
    brief: str = "",
    *,
    duration_s: float = 20.0,
    fps: int = 24,
    width: int = 768,
    height: int = 512,
    target: str = "seahorse",
    palette: str = "fire",
    seed: int | None = None,
    julia: bool = False,
    audio_path: str | Path | None = None,
    max_iter: int = 256,
    log=print,
) -> dict:
    """Render a fractal zoom video; if audio is given, beat-react and mux."""
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
        duration_s = max(duration_s, beat_map.duration_s)

    dest = out_dir / f"fractal_{run_id}.mp4"
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
            "duration_s": duration_s,
            "fps": fps,
            "width": width,
            "height": height,
            "target": target,
            "palette": palette,
            "julia": julia,
            "seed": seed,
            "max_iter": max_iter,
        },
        "beat_map": beat_map.to_dict() if beat_map else None,
        "audio_path": str(audio_path) if audio_path else None,
    }
    _write_record(run_id, record, log=log)
    return record


def _write_record(run_id: str, record: dict, *, log=print) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RUNS_DIR / f"{ts}_{run_id}_fractal.json"
    try:
        path.write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
        log(f"run record: {path}")
    except OSError as e:
        log(f"could not write run record: {e}")
    try:
        from master_agent.kb.ingest import ingest_run_record

        if ingest_run_record(record):
            log("kb: run record ingested")
    except Exception:
        pass


__all__ = ["run_fractal", "TARGETS", "PALETTES"]
