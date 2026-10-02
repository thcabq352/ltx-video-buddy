"""Music video, Remotion MTV, and HeartMuLa commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from master_agent.cli.runtime import comfy_client
from master_agent.cli.common import _maybe_interview

def cmd_mv(args: argparse.Namespace) -> int:
    """Comfy/LTX music-video mode: plan → unique burns → Remotion."""
    from master_agent.music.mv import render_music_video
    from master_agent.music.plan import (
        BeatPlanError,
        build_beat_plan,
        write_beat_plan,
    )
    from master_agent.music.still_hold import StillHoldError
    from master_agent.music.unique import DuplicateClipError

    action = getattr(args, "mv_command", None) or "render"
    audio = getattr(args, "audio", None)
    heartmula_block = None
    if action == "plan":
        if not audio:
            print("FAIL  mv plan needs --audio <file>")
            return 2
        try:
            plan = build_beat_plan(audio, fps=int(getattr(args, "fps", 30) or 30))
        except Exception as e:
            print(f"FAIL  beat plan: {e}")
            return 1
        dest = Path(args.out) if args.out else Path("out") / "beat_plan.json"
        write_beat_plan(plan, dest)
        if getattr(args, "json", False):
            print(json.dumps(plan.to_dict(), indent=1))
        else:
            print(f"OK    {dest}")
            print(f"      {plan.bpm:.0f} BPM, {len(plan.windows)} windows, {plan.duration_s:.2f}s @ {plan.fps}fps")
        return 0

    out = args.out or str(Path("out") / "MV-FIXED.mp4")
    if not audio or getattr(args, "heartmula_lyrics", None) or getattr(args, "heartmula_tags", None):
        from master_agent.heartmula.config import HeartMuLaConfigError
        from master_agent.heartmula.generate import (
            HeartMuLaUnavailable,
            MissingHeartMuLaWeights,
        )
        from master_agent.heartmula.wire import materialize_track

        work = getattr(args, "work_dir", None) or str(Path(out).parent / "heartmula-track")
        try:
            source = materialize_track(
                audio=audio,
                lyrics=getattr(args, "heartmula_lyrics", None),
                tags=getattr(args, "heartmula_tags", None),
                out_dir=work,
                dry_run=bool(getattr(args, "dry_run", False)),
                duration_s=getattr(args, "heartmula_duration", None),
                seed=getattr(args, "heartmula_seed", None),
            )
        except (HeartMuLaConfigError, HeartMuLaUnavailable, MissingHeartMuLaWeights) as exc:
            if not audio:
                print(f"FAIL  {exc}")
                return 2
            print(f"FAIL  {exc}")
            return 1
        if source.note:
            print(f"note: {source.note}")
        audio = str(source.audio)
        heartmula_block = source.heartmula
    if not audio:
        print("FAIL  mv render needs --audio, or --heartmula-lyrics and --heartmula-tags")
        return 2
    brief = (getattr(args, "prompt", None) or getattr(args, "request", None) or "music video").strip()
    plan_arg = getattr(args, "plan", None)
    try:
        rec = render_music_video(
            audio,
            out=out,
            prompt=brief,
            image=getattr(args, "image", None),
            variant=getattr(args, "variant", None),
            dry_run=bool(getattr(args, "dry_run", False)),
            plan=plan_arg,
            seed=getattr(args, "seed", None),
            width=int(getattr(args, "width", 768) or 768),
            height=int(getattr(args, "height", 512) or 512),
            fps=int(getattr(args, "fps", 30) or 30),
            work_dir=getattr(args, "work_dir", None),
            heartmula=heartmula_block,
        )
    except DuplicateClipError as e:
        print(f"FAIL  uniqueness gate: {e}")
        return 2
    except StillHoldError as e:
        print(f"FAIL  still-hold: {e}")
        return 2
    except BeatPlanError as e:
        print(f"FAIL  beat plan: {e}")
        return 2
    except Exception as e:
        print(f"FAIL  {e}")
        return 1
    if getattr(args, "json", False):
        print(json.dumps(rec, indent=1, default=str))
    else:
        print()
        mode = "DRY-RUN" if rec.get("dry_run") else "DONE"
        print(f"{mode}  plan: {rec.get('plan_path')}")
        print(f"       windows: {len((rec.get('beat_plan') or {}).get('windows') or [])}")
        rem = rec.get("remotion") or {}
        print(f"       remotion props: {rem.get('props_path')}")
        print(f"       remotion cmd: {' '.join(rem.get('command') or [])}")
        if not rec.get("dry_run"):
            print(f"       out: {rec.get('out')}")
    return 0 if rec.get("ok") else 1


def cmd_music(args: argparse.Namespace) -> int:
    from master_agent.config import JUDGE_ENABLED, MAX_JUDGE_ROUNDS
    from master_agent.music.pipeline import run_music_video

    audio = args.audio
    heartmula_block = None
    if not audio or getattr(args, "heartmula_lyrics", None) or getattr(args, "heartmula_tags", None):
        from master_agent.heartmula.config import HeartMuLaConfigError
        from master_agent.heartmula.generate import (
            HeartMuLaUnavailable,
            MissingHeartMuLaWeights,
        )
        from master_agent.heartmula.wire import materialize_track

        try:
            source = materialize_track(
                audio=audio,
                lyrics=getattr(args, "heartmula_lyrics", None),
                tags=getattr(args, "heartmula_tags", None),
                out_dir=Path("out") / "heartmula-track",
                dry_run=False,
                duration_s=getattr(args, "heartmula_duration", None),
                seed=getattr(args, "seed", None),
            )
        except (HeartMuLaConfigError, HeartMuLaUnavailable, MissingHeartMuLaWeights) as exc:
            print(f"FAIL  {exc}")
            return 2 if not audio else 1
        if source.note:
            print(f"note: {source.note}")
        audio = str(source.audio)
        heartmula_block = source.heartmula
    if not audio:
        print("FAIL  music needs --audio, or --heartmula-lyrics and --heartmula-tags")
        return 2
    args.request = _maybe_interview(args.request, no_interview=args.no_interview)
    client = comfy_client()
    if args.visual == "shots" and not client.is_up():
        print("FAIL  ComfyUI is not reachable. Start: ComfyUI_windows_portable\\run_api_8188.bat")
        print("      (or use --visual fractal — CPU only, no ComfyUI needed)")
        return 1
    rec = run_music_video(
        args.request,
        audio,
        heartmula=heartmula_block,
        visual=args.visual,
        variant=args.variant,
        quality=args.quality,
        seed=args.seed,
        width=args.width,
        height=args.height,
        judge_enabled=False if args.no_judge else JUDGE_ENABLED,
        revise_enabled=not args.no_judge,
        max_judge_rounds=args.max_judge_rounds or MAX_JUDGE_ROUNDS,
        llm_panel=args.llm_panel,
        panel_judge=args.panel_judge,
        upscale=args.upscale,
        client=client,
    )
    print()
    if rec.get("status") in ("done", "done_with_warnings"):
        print(f"{rec['status'].upper()}   video: {rec.get('video_path')}")
        bmap = rec.get("beat_map") or {}
        print(f"       {bmap.get('bpm')} BPM, {len(rec.get('shot_windows') or [])} shots")
        if rec.get("full_judge_score") is not None:
            print(f"       full judge: score={rec.get('full_judge_score', 0):.2f} pass={rec.get('full_judge_pass')}")
        return 0 if rec["status"] == "done" else 2
    print(f"ERROR  {rec.get('error')}")
    return 1


def cmd_heartmula(args: argparse.Namespace) -> int:
    """``heartmula generate|transcribe``. Dry-run never imports heartlib."""
    from master_agent.heartmula.config import HeartMuLaConfigError
    from master_agent.heartmula.generate import (
        HeartMuLaUnavailable,
        MissingHeartMuLaWeights,
        format_plan,
        generate_track,
        plan_generate,
    )
    from master_agent.heartmula.transcribe import (
        format_transcribe_plan,
        plan_transcribe,
        transcribe_audio,
    )

    action = getattr(args, "heartmula_command", None)
    dry = bool(getattr(args, "dry_run", False))
    try:
        if action == "generate":
            plan = plan_generate(
                lyrics=args.lyrics,
                tags=args.tags,
                out=args.out,
                duration_s=args.duration,
                seed=args.seed,
                topk=args.topk,
                temperature=args.temperature,
                cfg_scale=args.cfg_scale,
                dry_run=dry,
            )
            print(format_plan(plan))
            if dry:
                return 0
            generate_track(
                lyrics=args.lyrics,
                tags=args.tags,
                out=args.out,
                duration_s=args.duration,
                seed=args.seed,
                topk=args.topk,
                temperature=args.temperature,
                cfg_scale=args.cfg_scale,
            )
            print(f"OK    {args.out}")
            return 0
        if action == "transcribe":
            plan = plan_transcribe(audio=args.audio, out=args.out, dry_run=dry)
            print(format_transcribe_plan(plan))
            if dry:
                return 0
            result = transcribe_audio(audio=args.audio, out=args.out)
            print(f"OK    {result.path} ({len(result.words)} word(s))")
            return 0
    except (HeartMuLaConfigError, HeartMuLaUnavailable, MissingHeartMuLaWeights) as exc:
        print(f"FAIL  {exc}")
        return 1
    print("FAIL  heartmula needs generate or transcribe")
    return 2

