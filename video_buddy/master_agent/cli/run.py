"""Orchestrated run and interview-only brief commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from master_agent.cli.runtime import comfy_client
from master_agent.cli.common import _inoutpaint_plan, _reframe_flag, _music_intent, _maybe_interview

def cmd_run(args: argparse.Namespace) -> int:
    from master_agent.config import (
        JUDGE_ENABLED,
        MAX_FULL_JUDGE_ROUNDS,
        MAX_JUDGE_ROUNDS,
    )
    from master_agent.orchestrator.pipeline import (
        dry_run_pipeline,
        duration_from_request,
        run_pipeline,
    )

    brief_applied = False
    parsed_duration = duration_from_request(args.request)
    if (
        parsed_duration
        and args.duration == 5.0
        and not getattr(args, "duration_set", False)
    ):
        args.duration = parsed_duration
        brief_applied = True

    from master_agent.orchestrator.director_presets import (
        apply_rainey1_namespace,
        format_preset_line,
        mentions_rainey1,
        reapply_rainey1_prompt,
    )

    if getattr(args, "preset", None) == "rainey1" and not mentions_rainey1(
        getattr(args, "request", "")
    ):
        args.request = f"rainey1 {args.request}".strip()

    rainey_plan = apply_rainey1_namespace(args)
    if rainey_plan is not None:
        print(format_preset_line(rainey_plan))
        if rainey_plan.delivery:
            print(f"preset-delivery: {rainey_plan.delivery}")

    from master_agent.comfy.partner_pointers import route_pack_c

    routed, refusal = route_pack_c(getattr(args, "request", ""), getattr(args, "variant", None))
    if refusal:
        print(f"FAIL  {refusal}")
        return 2
    if routed:
        args.variant = routed

    if args.variant:
        from master_agent.comfy.catalog import default_variant_ids, is_known_variant

        if not is_known_variant(args.variant):
            print(f"FAIL  unknown variant {args.variant!r}")
            print("known default catalog:")
            for vid in default_variant_ids():
                print(f"  {vid}")
            return 2

        from master_agent.models.selector import job_version_block

        blocked = job_version_block(args.variant)
        if blocked:
            print(f"FAIL  {blocked}")
            return 2

    attach_recipe = None
    attach_loaded = None
    if getattr(args, "attach", None):
        from master_agent.comfy.attach import AttachError, load_attach_recipe

        try:
            attach_loaded = load_attach_recipe(args.attach)
            attach_recipe = attach_loaded.raw
        except AttachError as e:
            print(f"FAIL  attach recipe: {e}")
            return 1

    from master_agent.orchestrator.talking import (
        duration_following_audio,
        h3_r2v_audio_warning,
        is_audio_driven,
        is_h3_voice_route,
        media_route_error,
        preview_media_variant,
        skip_music_autoroute,
    )

    has_image = bool(getattr(args, "image", None))
    has_audio = bool(getattr(args, "audio", None))
    has_video = bool(getattr(args, "video", None))
    spoken_line = (getattr(args, "line", None) or "").strip()
    voice_sample = None
    preview = preview_media_variant(
        args.request,
        variant=args.variant,
        has_image=has_image,
        has_audio=has_audio,
        has_video=has_video,
    )
    route_err = media_route_error(
        preview,
        has_image=has_image,
        has_audio=has_audio,
        has_video=has_video,
    )
    if route_err:
        print(f"FAIL  {route_err}")
        return 1
    from master_agent.orchestrator.h3_voice import VoiceSampleError, h3_voice_preflight

    if is_h3_voice_route(
        args.request,
        variant=args.variant,
        has_image=has_image,
        has_audio=has_audio,
        has_video=has_video,
    ):
        try:
            pre = h3_voice_preflight(
                request=args.request,
                variant=args.variant,
                audio_path=args.audio,
                has_image=has_image,
                has_video=has_video,
                line=spoken_line or None,
            )
        except VoiceSampleError as exc:
            print(f"FAIL  {exc}")
            return 1
        args.audio = pre.audio_path
        voice_sample = pre.voice_sample
        spoken_line = pre.spoken_line or ""
        if pre.trimmed_note:
            print(f"warn: {pre.trimmed_note}")
        if pre.line_warning:
            print(f"warn: {pre.line_warning}")
    if (
        has_audio
        and not getattr(args, "duration_set", False)
        and not brief_applied
        and is_audio_driven(
            preview,
            has_image=has_image,
            has_audio=has_audio,
            has_video=has_video,
            request=args.request,
        )
    ):
        seconds, note = duration_following_audio(args.audio)
        args.duration = seconds
        if note:
            print(f"warn: {note}")
    if has_image and has_audio and not has_video:
        print(f"route: photo + voice → {preview}")
        voice_warn = h3_r2v_audio_warning(
            args.request,
            variant=args.variant,
            has_image=has_image,
            has_audio=has_audio,
            has_video=has_video,
        )
        if voice_warn:
            print(f"warn: {voice_warn}")

    words = None
    heartmula_block = None
    if getattr(args, "words", None):
        from master_agent.orchestrator.lipdub import load_words

        words_path = Path(args.words)
        if not words_path.is_file():
            print(f"FAIL  --words file not found: {args.words}")
            return 1
        try:
            words = load_words(str(words_path))
        except Exception as exc:
            print(f"FAIL  --words: {exc}")
            return 1
        from master_agent.heartmula.transcribe import transcribe_requested

        if transcribe_requested(bool(getattr(args, "heartmula_transcribe", False))):
            print("note: --words set; HeartTranscriptor skipped (faster-whisper file kept)")
    else:
        from master_agent.heartmula.transcribe import transcribe_requested

        if transcribe_requested(bool(getattr(args, "heartmula_transcribe", False))):
            if not getattr(args, "audio", None):
                print("FAIL  --heartmula-transcribe needs --audio")
                return 2
            from master_agent.heartmula.config import HeartMuLaConfigError
            from master_agent.heartmula.generate import (
                HeartMuLaUnavailable,
                MissingHeartMuLaWeights,
            )
            from master_agent.heartmula.transcribe import (
                format_transcribe_plan,
                plan_transcribe,
                transcribe_audio,
            )

            words_out = Path(args.audio).with_suffix(".heartmula.words.json")
            if getattr(args, "dry_run", False) or getattr(args, "self_improve_dry", False):
                heartmula_block = plan_transcribe(
                    audio=args.audio, out=words_out, dry_run=True
                )
                print(format_transcribe_plan(heartmula_block))
            else:
                try:
                    transcribed = transcribe_audio(audio=args.audio, out=words_out)
                except (HeartMuLaUnavailable, MissingHeartMuLaWeights, HeartMuLaConfigError) as exc:
                    print(f"FAIL  {exc}")
                    return 1
                from master_agent.orchestrator.lipdub import load_words

                words = load_words(str(transcribed.path))
                heartmula_block = transcribed.provenance
                print(f"heartmula words: {transcribed.path} ({len(words)} word(s))")

    if getattr(args, "self_improve_dry", False):
        from master_agent.orchestrator.machine import Orchestrator

        args.request = _maybe_interview(args.request, no_interview=args.no_interview)
        reapply_rainey1_prompt(args)
        image_name = Path(args.image).name if getattr(args, "image", None) else None
        audio_name = Path(args.audio).name if getattr(args, "audio", None) else None
        st = Orchestrator().run(
            args.request,
            variant=args.variant,
            duration_s=args.duration,
            quality=args.quality,
            seed=args.seed,
            width=args.width,
            height=args.height,
            image_name=image_name,
            mask_name=Path(args.mask).name if getattr(args, "mask", None) else None,
            audio_name=audio_name,
            video_name=Path(args.video).name if getattr(args, "video", None) else None,
            audio_path=args.audio if getattr(args, "audio", None) else None,
            judge_enabled=False if args.no_judge else JUDGE_ENABLED,
            revise_enabled=not args.no_judge,
            spoken_line=spoken_line or None,
            voice_sample=voice_sample,
            heartmula=heartmula_block,
            max_judge_rounds=args.max_judge_rounds or MAX_JUDGE_ROUNDS,
            attach_recipe=attach_recipe,
            dry_run=True,
            negative_prompt=getattr(args, "negative_prompt", None),
            frames=getattr(args, "frames", None),
            control_pack_present=bool(attach_loaded and attach_loaded.control_pack_present),
            previs_source=(attach_loaded.previs_source if attach_loaded else ""),
        )
        print()
        print(f"loop_status: {st.loop_status}")
        print(f"attempts:    {st.attempt}/{st.max_judge_rounds}")
        print(f"decision:    {st.judge_decision}")
        fails = (st.quality_bar or {}).get("fails") or []
        if fails:
            print("quality_bar: " + ", ".join(f"{f.get('id')}:{f.get('code')}" for f in fails))
        if st.revise_history:
            print(f"revises:     {len(st.revise_history)}")
        if st.provenance:
            lin = st.provenance.get("lineage") or {}
            judge = st.provenance.get("judge") or {}
            print(
                f"provenance:  schema={st.provenance.get('schema')} "
                f"attempt_id={lin.get('attempt_id')} "
                f"score={judge.get('score')}"
            )
        if st.provenance_sidecar:
            print(f"sidecar:     {st.provenance_sidecar}")
        print(f"dry-run     no Comfy queue, no GPU")
        if st.loop_status == "passed":
            return 0
        if st.loop_status in ("exhausted", "human_veto"):
            return 2
        return 1

    inoutpaint = None
    try:
        outpaint_plan = _inoutpaint_plan(args)
    except (ValueError, RuntimeError) as exc:
        print(f"FAIL  {exc}")
        return 1
    if outpaint_plan:
        # Mask bytes are uploaded before queue. Keep only the pads on run
        # state so the JSON record does not embed the PNG.
        inoutpaint = {
            "mode": "outpaint",
            "pad": outpaint_plan["pad"],
        }
        args.width = outpaint_plan["width"]
        args.height = outpaint_plan["height"]
        print(
            "outpaint canvas "
            f"{outpaint_plan['canvas_w']}x{outpaint_plan['canvas_h']} "
            f"pad L{outpaint_plan['pad']['left']} T{outpaint_plan['pad']['top']} "
            f"R{outpaint_plan['pad']['right']} B{outpaint_plan['pad']['bottom']} "
            f"stage {outpaint_plan['width']}x{outpaint_plan['height']}"
        )
    elif has_video:
        from master_agent.comfy.inoutpaint import requests_inoutpaint

        if requests_inoutpaint(args.request):
            print(
                "note: routed as inpaint. Pass --outpaint --aspect 9:16 "
                "(or --width and --height) to extend the canvas."
            )

    client = comfy_client()
    if args.dry_run:
        args.request = _maybe_interview(args.request, no_interview=args.no_interview)
        reapply_rainey1_prompt(args)
        return dry_run_pipeline(
            args.request,
            variant=args.variant,
            duration_s=args.duration,
            quality=args.quality,
            width=args.width,
            height=args.height,
            storyboard_mode=args.storyboard,
            llm_panel=args.llm_panel,
            panel_judge=args.panel_judge,
            attach_recipe=attach_recipe,
            client=client,
            kind="run",
            image_name=Path(args.image).name if getattr(args, "image", None) else None,
            mask_name=Path(args.mask).name if getattr(args, "mask", None) else None,
            audio_name=Path(args.audio).name if getattr(args, "audio", None) else None,
            video_name=Path(args.video).name if getattr(args, "video", None) else None,
            audio_path=args.audio if getattr(args, "audio", None) else None,
            spoken_line=spoken_line or None,
            seed=args.seed,
            tripod=bool(getattr(args, "tripod", False)),
            words=words,
            heartmula=heartmula_block,
            lipdub_max_s=getattr(args, "lipdub_max_s", None),
            silence_min_s=getattr(args, "silence_min_s", None),
            lipdub_overlap=getattr(args, "lipdub_overlap", None),
            silence_mode=getattr(args, "silence_mode", None),
            anchor=getattr(args, "anchor", None),
            reframe=_reframe_flag(args),
            max_piece_s=getattr(args, "max_piece_seconds", None),
            pause_reset_strength=getattr(args, "pause_reset_strength", None),
            pause_reset_min_s=getattr(args, "pause_reset_min_s", None),
            inoutpaint=inoutpaint,
            latent_frames=getattr(args, "frames", None),
            negative_prompt=getattr(args, "negative_prompt", None),
        )

    if not client.is_up():
        print(f"FAIL  ComfyUI is not reachable. Start: ComfyUI_windows_portable\\run_api_8188.bat")
        return 1

    args.request = _maybe_interview(args.request, no_interview=args.no_interview)
    reapply_rainey1_prompt(args)

    # Auto-route music videos to the beat-synced pipeline (explicit --variant wins)
    if (
        args.audio
        and not args.variant
        and not skip_music_autoroute(args.request, has_image=has_image)
        and _music_intent(args.request, args.audio, args.quality)
    ):
        from master_agent.music.pipeline import run_music_video

        print("auto-route: music video detected -> beat-synced music pipeline")
        rec = run_music_video(
            args.request,
            args.audio,
            visual="shots",
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
            return 0 if rec["status"] == "done" else 2
        print(f"ERROR  {rec.get('error')}")
        return 1

    # Upload input media into ComfyUI's input dir first
    video_name = image_name = audio_name = mask_name = None
    try:
        if args.video:
            video_name = client.upload_image(Path(args.video))
            print(f"uploaded video: {video_name}")
        if args.image:
            image_name = client.upload_image(Path(args.image))
            print(f"uploaded image: {image_name}")
        mask_path = None
        if outpaint_plan is not None:
            import tempfile

            handle = tempfile.NamedTemporaryFile(
                prefix="ltx23-outpaint-", suffix=".png", delete=False
            )
            handle.write(outpaint_plan["mask_png"])
            handle.close()
            mask_path = Path(handle.name)
        elif getattr(args, "mask", None):
            mask_path = Path(args.mask)
        if mask_path is not None:
            mask_name = client.upload_image(mask_path)
            print(f"uploaded mask: {mask_name}")
            if outpaint_plan is not None:
                mask_path.unlink(missing_ok=True)
        if args.audio:
            audio_name = client.upload_audio(Path(args.audio))
            print(f"uploaded audio: {audio_name}")
    except Exception as e:
        print(f"FAIL  upload: {e}")
        return 1

    result = run_pipeline(
        args.request,
        variant=args.variant,
        duration_s=args.duration,
        quality=args.quality,
        seed=args.seed,
        width=args.width,
        height=args.height,
        video_name=video_name,
        image_name=image_name,
        image_path=args.image if getattr(args, "image", None) else None,
        mask_name=mask_name,
        audio_name=audio_name,
        audio_path=args.audio if getattr(args, "audio", None) else None,
        tripod=bool(getattr(args, "tripod", False)),
        words=words,
        lipdub_max_s=getattr(args, "lipdub_max_s", None),
        silence_min_s=getattr(args, "silence_min_s", None),
        lipdub_overlap=getattr(args, "lipdub_overlap", None),
        silence_mode=getattr(args, "silence_mode", None),
        anchor=getattr(args, "anchor", None),
        reframe=_reframe_flag(args),
        max_piece_s=getattr(args, "max_piece_seconds", None),
        pause_reset_strength=getattr(args, "pause_reset_strength", None),
        pause_reset_min_s=getattr(args, "pause_reset_min_s", None),
        storyboard_mode=args.storyboard,
        judge_enabled=False if args.no_judge else JUDGE_ENABLED,
        revise_enabled=not args.no_judge,
        spoken_line=spoken_line or None,
        voice_sample=voice_sample,
        heartmula=heartmula_block,
        max_judge_rounds=args.max_judge_rounds or MAX_JUDGE_ROUNDS,
        max_full_judge_rounds=args.max_full_judge_rounds or MAX_FULL_JUDGE_ROUNDS,
        llm_panel=args.llm_panel,
        panel_judge=args.panel_judge,
        power_mode=True if getattr(args, "power_mode", False) else (
            False if getattr(args, "no_power_mode", False) else None
        ),
        attach_recipe=attach_recipe,
        client=client,
        inoutpaint=inoutpaint,
        latent_frames=getattr(args, "frames", None),
        negative_prompt=getattr(args, "negative_prompt", None),
    )
    print()
    if result.status in ("done", "done_with_warnings"):
        print(f"{result.status.upper()}   video: {result.video_path}")
        print(f"       segments: {len(result.segment_paths)} scores={[round(s, 2) for s in result.segment_scores]}")
        print(f"       full judge: score={result.full_judge_score:.2f} pass={result.full_judge_pass}")
        if result.full_judge_notes:
            print(f"       notes: {result.full_judge_notes}")
        if result.provenance:
            lin = result.provenance.get("lineage") or {}
            print(
                f"       provenance: schema={result.provenance.get('schema')} "
                f"attempt_id={lin.get('attempt_id')} "
                f"hash={(result.provenance.get('hash') or '')[:12]}"
            )
        if result.provenance_sidecar:
            print(f"       sidecar: {result.provenance_sidecar}")
        return 0 if result.status == "done" else 2
    print(f"ERROR  {result.error}")
    return 1


def cmd_brief(args: argparse.Namespace) -> int:
    from master_agent.persona.intake import run_interview_cli

    brief = run_interview_cli(args.request)
    if not args.go:
        return 0

    client = comfy_client()
    if not client.is_up():
        print("FAIL  ComfyUI is not reachable. Start: ComfyUI_windows_portable\\run_api_8188.bat")
        return 1
    from master_agent.orchestrator.pipeline import run_pipeline

    duration = brief.duration_s or args.duration
    print(f"generating: {duration}s @ {args.quality or 'default'} quality")
    result = run_pipeline(
        brief.to_request(),
        duration_s=duration,
        quality=args.quality,
        client=client,
    )
    print()
    if result.status in ("done", "done_with_warnings"):
        print(f"{result.status.upper()}   video: {result.video_path}")
        return 0 if result.status == "done" else 2
    print(f"ERROR  {result.error}")
    return 1

