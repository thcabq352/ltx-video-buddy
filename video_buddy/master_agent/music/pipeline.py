"""Beat-synced music video pipeline.

Flow: BeatMap -> beat-snapped shot windows -> LLM storyboard with beat context
-> per-shot Orchestrator runs (validate + judge loop) -> trim clips to exact
beat windows -> concat -> mux audio -> full-video judge -> run record
(kind: "music_video").

visual="fractal" skips ComfyUI entirely and renders a beat-reactive fractal
for the whole track (fast, CPU-only).
"""

from __future__ import annotations

import json
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from master_agent.comfy.client import ComfyClient
from master_agent.config import (
    JUDGE_ENABLED,
    JUDGE_SCORE_THRESHOLD,
    MAX_JUDGE_ROUNDS,
    OUTPUTS_DIR,
    RUNS_DIR,
    get_quality_profile,
)
from master_agent.music.beats import BeatMap, analyze_audio, plan_shot_windows
from master_agent.video_concat import concat_videos, cut_to_windows, mux_audio

MUSIC_KEYWORDS = ("music video", "song", "beat", "track", "mv")


def _beat_context(bmap: BeatMap, windows: list[tuple[float, float]]) -> str:
    lines = [
        f"Beat map: {bmap.bpm:.0f} BPM, {len(bmap.beats)} beats.",
        "Shots land on beat boundaries; section per shot:",
    ]
    for i, (s, e) in enumerate(windows):
        sec = next(
            (x for x in bmap.sections if x["start_s"] <= s < x["end_s"]),
            bmap.sections[-1] if bmap.sections else {"label": "verse"},
        )
        lines.append(f"  shot {i + 1}: {s:.1f}-{e:.1f}s ({sec['label']})")
    lines.append("Cut on the beat; match motion energy to the section energy.")
    return "\n".join(lines)


def run_music_video(
    request: str,
    audio_path: str | Path,
    *,
    visual: str = "shots",
    variant: Optional[str] = None,
    quality: Optional[str] = None,
    seed: Optional[int] = None,
    width: int = 768,
    height: int = 512,
    judge_enabled: Optional[bool] = None,
    max_judge_rounds: int = MAX_JUDGE_ROUNDS,
    judge_threshold: float = JUDGE_SCORE_THRESHOLD,
    llm_panel: Optional[str] = None,
    panel_judge: Optional[str] = None,
    upscale: Optional[str] = None,
    client: Optional[ComfyClient] = None,
    log=print,
) -> dict:
    audio_path = Path(audio_path)
    run_id = uuid.uuid4().hex[:12]
    out_dir = OUTPUTS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    j_enabled = JUDGE_ENABLED if judge_enabled is None else judge_enabled
    base_seed = seed if seed is not None else random.randint(0, 2**32 - 1)

    log(f"analyzing audio: {audio_path}")
    bmap = analyze_audio(audio_path)
    log(
        f"beat map: {bmap.bpm:.0f} BPM, {len(bmap.beats)} beats, "
        f"sections: {[s['label'] for s in bmap.sections]}"
    )

    record: dict = {
        "run_id": run_id,
        "kind": "music_video",
        "request": request,
        "status": "started",
        "audio_path": str(audio_path),
        "visual": visual,
        "beat_map": bmap.to_dict(),
        "seed": base_seed,
    }

    if visual == "fractal":
        final = _fractal_visual(
            run_id, out_dir, bmap, width=width, height=height,
            seed=base_seed, audio_path=audio_path, log=log,
        )
        if final is None:
            record["status"] = "error"
            record["error"] = "fractal render failed"
            _write_record(run_id, record, log=log)
            return record
        record["segment_paths"] = []
        record["segment_scores"] = []
    else:
        final = _shots_visual(
            request, run_id, out_dir, bmap, audio_path,
            variant=variant, quality=quality, base_seed=base_seed,
            width=width, height=height, j_enabled=j_enabled,
            max_judge_rounds=max_judge_rounds, llm_panel=llm_panel,
            panel_judge=panel_judge, client=client, log=log, record=record,
        )
        if final is None:
            _write_record(run_id, record, log=log)
            return record

    # Optional upscale post-stage
    if upscale:
        try:
            from master_agent.upscale import upscale_video

            up = upscale_video(final, method=upscale, run_id=run_id, client=client, log=log)
            record["upscaled_path"] = str(up)
            final = up
        except Exception as e:
            log(f"upscale failed ({e}); keeping original")
            record["upscale_error"] = str(e)

    # Full-video judge
    record["video_path"] = str(Path(final).resolve())
    record["music_bed_attached"] = True
    record["kind"] = "music_video"
    if j_enabled and visual != "fractal":
        from master_agent.judge.judge import judge_full_video

        fj = judge_full_video(
            user_request=request,
            storyboard=record.get("storyboard", []),
            video_path=str(final),
            segment_scores=record.get("segment_scores", []),
            threshold=judge_threshold,
        )
        record["full_judge_score"] = fj.combined_score
        record["full_judge_pass"] = fj.pass_
        record["full_judge_notes"] = fj.reason
        log(f"full judge: score={fj.combined_score:.2f} pass={fj.pass_} — {fj.reason}")
        record["status"] = "done" if fj.pass_ else "done_with_warnings"
    else:
        record["status"] = "done"

    log(f"final: {record['video_path']} ({record['status']})")
    _write_record(run_id, record, log=log)
    return record


def _fractal_visual(run_id, out_dir, bmap, *, width, height, seed, audio_path, log):
    from master_agent.fractal.render import render_zoom_video

    dest = out_dir / f"music_fractal_{run_id}.mp4"
    try:
        render_zoom_video(
            dest,
            duration_s=bmap.duration_s,
            width=width,
            height=height,
            beat_map=bmap,
            seed=seed,
            log=log,
        )
        muxed = out_dir / f"music_video_{run_id}.mp4"
        mux_audio(dest, audio_path, muxed)
        log(f"muxed audio: {muxed}")
        return muxed
    except Exception as e:
        log(f"fractal visual failed: {e}")
        return None


def _shots_visual(
    request, run_id, out_dir, bmap, audio_path, *,
    variant, quality, base_seed, width, height, j_enabled,
    max_judge_rounds, llm_panel, panel_judge, client, log, record,
):
    from master_agent.orchestrator.machine import Orchestrator
    from master_agent.orchestrator.pipeline import _plan_storyboard, _synthetic_cards
    from master_agent.storyboard.storyboard import storyboard_to_markdown

    profile = get_quality_profile(quality)
    from master_agent.config import MUSIC_DEFAULTS

    windows = plan_shot_windows(
        bmap,
        min_s=float(MUSIC_DEFAULTS["min_shot_s"]),
        max_s=float(profile["segment_max_s"]),
        high_energy_s=float(MUSIC_DEFAULTS["high_energy_s"]),
        low_energy_s=float(MUSIC_DEFAULTS["low_energy_s"]),
    )
    durations = [round(e - s, 3) for s, e in windows]
    record["shot_windows"] = [[round(s, 3), round(e, 3)] for s, e in windows]
    log(f"shot plan: {len(windows)} beat-snapped windows {durations}")

    context_request = f"{request}\n\n{_beat_context(bmap, windows)}"
    try:
        cards, global_style, panel_meta = _plan_storyboard(
            context_request,
            durations,
            variant=variant,
            quality=quality,
            llm_panel=llm_panel,
            panel_judge=panel_judge,
            log=log,
        )
        record["panel_meta"] = panel_meta
        record["storyboard_md"] = storyboard_to_markdown(cards, global_style=global_style)
        log(f"storyboard ({len(cards)} shots):\n{record['storyboard_md']}")
    except Exception as e:
        log(f"storyboard failed ({e}); using synthetic cards")
        cards = _synthetic_cards(request, durations)
    record["storyboard"] = [c.to_dict() for c in cards]

    orch = Orchestrator(client=client)
    clip_paths: list[str] = []
    scores: list[float] = []
    for i, card in enumerate(cards):
        s, e = windows[i]
        log(f"--- shot {i + 1}/{len(cards)} ({s:.1f}-{e:.1f}s, {durations[i]}s) ---")
        seg_seed = (base_seed + card.seed_offset) & 0xFFFFFFFF
        st = orch.run(
            request,
            prompt=card.ltx_prompt or request,
            shot=card.to_dict(),
            variant=variant,
            duration_s=durations[i],
            quality=quality,
            seed=seg_seed,
            width=width,
            height=height,
            judge_enabled=j_enabled,
            max_judge_rounds=max_judge_rounds,
            kind="music_video",
            music_bed_attached=True,
            audio_path=str(audio_path),
        )
        if st.state != "DONE" or not st.video_path:
            record["status"] = "error"
            record["error"] = f"shot {i + 1} failed: {st.error}"
            return None
        scores.append(st.judge_score)
        # Trim the clip to the exact beat window (drop generator tail)
        trimmed = cut_to_windows(
            Path(st.video_path), [(0.0, e - s)], out_dir / "trimmed",
            prefix=f"shot_{i:03d}",
        )[0]
        clip_paths.append(str(trimmed))
        try:
            orch.client.free_memory()
        except Exception:
            pass

    record["segment_paths"] = clip_paths
    record["segment_scores"] = scores

    silent = out_dir / f"music_video_{run_id}_silent.mp4"
    try:
        concat_videos([Path(p) for p in clip_paths], silent)
        final = out_dir / f"music_video_{run_id}.mp4"
        mux_audio(silent, audio_path, final)
        log(f"assembled + muxed: {final}")
        try:
            from master_agent.provenance import inherit_clip_provenance

            inherit_clip_provenance(
                clip_paths[0],
                final,
                revise_notes="music mux: source track as music bed",
                extra={"prompt": request},
            )
        except Exception as e:
            log(f"provenance mux skipped: {e}")
        return final
    except Exception as e:
        record["status"] = "done_with_warnings"
        record["error"] = f"assembly failed: {e}"
        log(f"assembly failed ({e}); keeping first clip")
        return Path(clip_paths[0])


def _write_record(run_id: str, record: dict, *, log=print) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RUNS_DIR / f"{ts}_{run_id}_music.json"
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
