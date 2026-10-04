"""Run a segmented ltx25_a2v plan: render, stitch, one ClipProvenance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from master_agent.config import OUTPUTS_DIR
from master_agent.orchestrator.lipdub import (
    ANCHOR_HYBRID,
    ANCHOR_PAUSE_RESET,
    ANCHOR_PREVIOUS,
    ANCHOR_SOURCE,
    CONTINUITY_HYBRID,
    CONTINUITY_PREV,
    CONTINUITY_SOURCE,
    CONTINUITY_STILL,
    KIND_PLATE,
    LipdubPiece,
    LipdubPlan,
    attach_lipdub_params,
    format_lipdub_plan,
    lipdub_param_block,
    resolve_pause_reset,
)
from master_agent.orchestrator.lipdub_guide import probe_pause_reset_guide
from master_agent.orchestrator.lipdub_media import (
    concat_silent,
    extract_frame,
    hold_image,
    mux_original_audio,
    trim_span,
    video_size,
)
from master_agent.orchestrator.state import RunState
from master_agent.provenance import (
    CLIP_PROVENANCE_SCHEMA,
    build_clip_provenance,
    sha256_file,
    write_clip_provenance,
)


def _upload_image(client, path: Path, *, dry_run: bool) -> str:
    if dry_run:
        return path.name
    return client.upload_image(path)


def run_segmented_lipdub(
    result,
    plan: LipdubPlan,
    *,
    orch,
    request: str,
    variant: Optional[str],
    seed: Optional[int],
    width: int,
    height: int,
    image_name: Optional[str],
    image_path: Optional[str],
    audio_name: Optional[str],
    audio_path: Optional[str],
    spoken_line: Optional[str],
    quality: Optional[str],
    judge_enabled: bool,
    revise_enabled: Optional[bool],
    max_judge_rounds: int,
    dry_run: bool,
    tripod: bool,
    heartmula: Optional[dict] = None,
) -> Any:
    """Render each Comfy piece, idle the pauses, mux the original wav."""
    from master_agent.orchestrator.pipeline import _budget_admit, _write_record

    result.log(format_lipdub_plan(plan))
    if plan.warning:
        result.log(f"warn: {plan.warning}")
    if dry_run:
        result.status = "done"
        result.log("dry-run: lipdub plan only, nothing queued")
        _write_record(result)
        return result

    out_dir = OUTPUTS_DIR / result.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    still_path = Path(image_path) if image_path else None
    needs_still = (
        plan.reframe
        or plan.anchor in (ANCHOR_SOURCE, ANCHOR_HYBRID, ANCHOR_PAUSE_RESET)
        or any(p.kind == KIND_PLATE or p.continuity == CONTINUITY_STILL for p in plan.pieces)
    )
    if needs_still and (still_path is None or not still_path.is_file()):
        result.status = "error"
        result.error = "lipdub anchor/reframe needs the source still path (--image)"
        _write_record(result)
        return result
    _probe_pause_reset(plan, orch.client, log=result.log)
    records: list[dict] = []
    clip_paths: list[Optional[Path]] = [None] * len(plan.pieces)
    last_frame: Optional[Path] = None
    plate_blend_from: dict[int, Path] = {}
    size: Optional[tuple[int, int]] = None
    base_seed = seed if seed is not None else 0

    for piece in plan.pieces:
        record = piece.to_record()
        record["seed"] = base_seed
        if piece.kind == KIND_PLATE:
            if (
                plan.anchor == ANCHOR_PAUSE_RESET
                and piece.silence_crossfade_frames
                and last_frame is not None
            ):
                plate_blend_from[piece.index] = last_frame
                if still_path is not None:
                    last_frame = still_path
            records.append(record)
            continue
        seg_image, source_note = _conditioning_image(
            piece,
            plan,
            orch=orch,
            image_name=image_name,
            still_path=still_path,
            last_frame=last_frame,
            out_dir=out_dir,
        )
        if seg_image is None:
            result.status = "error"
            result.error = source_note or f"lipdub piece {piece.index} has no conditioning frame"
            _write_record(result)
            return result
        record["source_frame"] = source_note
        if still_path is not None and (
            plan.anchor in (ANCHOR_SOURCE, ANCHOR_HYBRID, ANCHOR_PAUSE_RESET)
            or piece.end_keyframe
            or piece.silence_crossfade_frames
        ):
            record["identity_frame"] = str(still_path)
        end_guide = None
        if piece.end_keyframe == "source_still" and image_name:
            end_guide = image_name
            record["guide_node"] = "LTXVAddGuide"
            record["guide_frame_idx"] = piece.guide_frame_idx
            record["guide_strength"] = piece.guide_strength
        scene_id = f"{result.run_id}:lip{piece.index}"
        gate = _budget_admit(result, scene_id, variant, max(piece.audio_duration_s, 0.5))
        if gate["decision"] == "hold":
            result.budget_held.append({"id": scene_id, **gate})
            result.status = "paused"
            result.error = (
                f"render budget paused on lipdub piece {piece.index} "
                f"(used {gate['used_after']}/{gate['cap']} VRAM-min)"
            )
            _write_record(result)
            return result

        result.log(
            f"--- lipdub {piece.index + 1}/{len(plan.pieces)} {piece.kind} "
            f"{piece.start_s:.3f}-{piece.end_s:.3f}s "
            f"frames={piece.render_frames} source={piece.continuity} "
            f"anchor={plan.anchor} reframe={'on' if plan.reframe else 'off'} ---"
        )
        st = orch.run(
            request,
            prompt=piece.prompt or request,
            variant=variant,
            duration_s=piece.audio_duration_s,
            quality=quality,
            seed=base_seed,
            width=width,
            height=height,
            image_name=seg_image,
            audio_name=audio_name,
            audio_path=audio_path,
            audio_start_s=piece.audio_start_s,
            negative_prompt=piece.negative,
            frames=piece.render_frames,
            i2v_strength=piece.i2v_strength,
            end_guide_image=end_guide,
            end_guide_frame_idx=piece.guide_frame_idx,
            end_guide_strength=piece.guide_strength,
            judge_enabled=judge_enabled,
            heartmula=heartmula,
            revise_enabled=False if not judge_enabled else revise_enabled,
            duration_cap_s=float(piece.audio_duration_s),
            # max-piece bounds a revise only when this piece already fits it.
            # An unsplittable word is longer than max_piece on purpose; snapping
            # that piece down would cut the word. A one-pass clip never reaches
            # this function, so a 3.86s take is not forced to 3.0s.
            max_piece_s=(
                float(plan.max_piece_s)
                if plan.segmented
                and float(piece.audio_duration_s) <= float(plan.max_piece_s) + 0.05
                else None
            ),
            spoken_line=piece.spoken_line or None,
            max_judge_rounds=max_judge_rounds,
            dry_run=False,
            shot_index=piece.index + 1,
        )
        result.messages.extend(st.messages)
        record["attempt"] = int(getattr(st, "attempt", 1) or 1)
        record["seed"] = st.seed if st.seed is not None else base_seed
        record["prompt_id"] = st.prompt_id
        if st.state != "DONE" or not st.video_path:
            result.status = "error"
            result.error = f"lipdub piece {piece.index} ({piece.kind}) failed: {st.error}"
            _write_record(result)
            return result
        src = Path(st.video_path)
        try:
            trimmed, overlap_path = _fit_piece(
                src,
                piece,
                plan,
                out_dir=out_dir,
                still_path=still_path,
                record=record,
            )
            if piece.silence_crossfade_frames and last_frame is not None:
                from master_agent.orchestrator.lipdub_reframe import crossfade_head

                blended = crossfade_head(
                    trimmed,
                    last_frame,
                    out_dir / f"piece_{piece.index:02d}_sxfade.mp4",
                    piece.silence_crossfade_frames,
                    fps=plan.fps,
                )
                trimmed = blended
                record["silence_crossfade_frames"] = int(piece.silence_crossfade_frames)
            if size is None:
                size = video_size(trimmed)
            if piece.hold_frames:
                frame = extract_frame(
                    trimmed,
                    out_dir / f"piece_{piece.index:02d}_hold.png",
                    piece.generated_keep - 1,
                )
                held = hold_image(
                    frame,
                    out_dir / f"piece_{piece.index:02d}_hold.mp4",
                    frames=piece.hold_frames,
                    fps=plan.fps,
                    size=size,
                )
                trimmed = concat_silent(
                    [trimmed, held],
                    out_dir / f"piece_{piece.index:02d}.mp4",
                    fps=plan.fps,
                )
            faded = _crossfade_previous(
                piece,
                plan,
                overlap_path=overlap_path,
                clip_paths=clip_paths,
                records=records,
                out_dir=out_dir,
            )
            if faded is not None:
                old_path, new_path = faded
                for i, stored in enumerate(result.segment_paths):
                    if stored == str(old_path):
                        result.segment_paths[i] = str(new_path)
                        break
            last_frame = extract_frame(
                trimmed,
                out_dir / f"piece_{piece.index:02d}_last.png",
                max(piece.keep_frames - 1, 0),
            )
        except Exception as exc:
            result.status = "error"
            result.error = f"lipdub piece {piece.index} trim failed: {exc}"
            _write_record(result)
            return result
        clip_paths[piece.index] = trimmed
        record["output_path"] = str(trimmed)
        record["hash"] = sha256_file(trimmed)
        records.append(record)
        result.segment_paths.append(str(trimmed))
        result.segment_scores.append(float(st.judge_score or 0.0))
        try:
            orch.client.free_memory()
        except Exception:
            pass

    if size is None and still_path is not None:
        from PIL import Image

        from master_agent.orchestrator.lipdub_media import even_size

        with Image.open(still_path) as im:
            size = even_size(*im.size)
    if size is None:
        result.status = "error"
        result.error = "lipdub stitch has no pixel size (no render and no still)"
        _write_record(result)
        return result
    if any(p.kind == KIND_PLATE for p in plan.pieces) and (
        still_path is None or not still_path.is_file()
    ):
        result.status = "error"
        result.error = "lipdub silence plate needs the source still path (--image)"
        _write_record(result)
        return result

    ordered: list[Path] = []
    for piece, record in zip(plan.pieces, records):
        if piece.kind != KIND_PLATE:
            path = clip_paths[piece.index]
            if path is None:
                result.status = "error"
                result.error = f"lipdub missing clip for piece {piece.index}"
                _write_record(result)
                return result
            ordered.append(path)
            continue
        try:
            plate = hold_image(
                still_path,
                out_dir / f"piece_{piece.index:02d}.mp4",
                frames=piece.hold_frames,
                fps=plan.fps,
                size=size,
            )
            blend_from = plate_blend_from.get(piece.index)
            if blend_from is not None and piece.silence_crossfade_frames:
                from master_agent.orchestrator.lipdub_reframe import crossfade_head

                plate = crossfade_head(
                    plate,
                    blend_from,
                    out_dir / f"piece_{piece.index:02d}_sxfade.mp4",
                    piece.silence_crossfade_frames,
                    fps=plan.fps,
                )
                record["silence_crossfade_frames"] = int(piece.silence_crossfade_frames)
        except Exception as exc:
            result.status = "error"
            result.error = f"lipdub plate {piece.index} failed: {exc}"
            _write_record(result)
            return result
        record["source_frame"] = str(still_path)
        record["output_path"] = str(plate)
        record["hash"] = sha256_file(plate)
        record["seed"] = base_seed
        ordered.append(plate)
        result.segment_paths.append(str(plate))

    silent = out_dir / "lipdub_silent.mp4"
    final = out_dir / "shot-1.mp4"
    try:
        concat_silent(ordered, silent, fps=plan.fps)
        if not audio_path:
            raise RuntimeError("original wav path is missing; cannot mux")
        mux_original_audio(silent, Path(audio_path), final)
    except Exception as exc:
        result.status = "error"
        result.error = f"lipdub stitch failed: {exc}"
        _write_record(result)
        return result

    positive = request
    negative = ""
    if tripod:
        from master_agent.orchestrator.lipdub import tripod_conditioning

        positive, negative, _strength = tripod_conditioning(request)
    judge_reason = ""
    judge_score = 0.0
    if not judge_enabled:
        judge_reason = "judge skipped (--no-judge): one render, no quality_bar revise"
    elif result.segment_scores:
        judge_score = sum(result.segment_scores) / len(result.segment_scores)
    st = RunState(
        request=request,
        prompt=positive,
        negative_prompt=negative,
        variant=variant or "ltx25_a2v",
        seed=base_seed,
        width=width,
        height=height,
        duration_s=plan.duration_s,
        fps=plan.fps,
        attempt=1,
        shot_index=1,
        image_name=image_name,
        audio_name=audio_name,
        audio_path=audio_path,
        spoken_line=(spoken_line or "").strip(),
        judge_score=judge_score,
        judge_reason=judge_reason,
        judge_decision="skipped" if not judge_enabled else "",
        video_path=str(final),
        planned_clip=str(final),
        output_dir=str(out_dir),
    )
    if judge_reason:
        st.judge_issues = [judge_reason]
    payload = build_clip_provenance(
        st,
        path=final,
        revise_notes=(
            f"lipdub stitch continuity={plan.continuity_method} "
            f"silence={plan.silence_handling} mode={plan.silence_mode} "
            f"anchor={plan.anchor} reframe={bool(plan.reframe)} "
            f"join={plan.join_method} tripod={bool(tripod)} "
            f"full_audio_mux overlap_frames={plan.overlap_frames}"
        ),
    )
    block = lipdub_param_block(
        plan,
        audio_sha256=sha256_file(audio_path) if audio_path else None,
        segments=records,
    )
    # Per-segment source frames were filled above; keep those, not the plan default.
    block["segments"] = records
    attach_lipdub_params(payload, block)
    if payload.get("schema") != CLIP_PROVENANCE_SCHEMA:
        payload["schema"] = CLIP_PROVENANCE_SCHEMA
    try:
        sidecar = write_clip_provenance(final, payload)
        result.provenance_sidecar = str(sidecar)
    except OSError as exc:
        result.log(f"provenance write failed: {exc}")
    result.provenance = payload
    result.provenance_history = [payload]
    result.video_path = str(final.resolve())
    result.segment_durations = [round(p.end_s - p.start_s, 4) for p in plan.pieces]
    result.full_judge_score = judge_score
    result.full_judge_pass = not judge_enabled or judge_score >= 0
    result.status = "done"
    result.log(f"final: {result.video_path} lipdub pieces={len(plan.pieces)}")
    _write_record(result)
    return result


def dry_run_lipdub_validate(
    plan: LipdubPlan,
    *,
    request: str,
    variant: str,
    width: int,
    height: int,
    image_name: Optional[str],
    audio_name: Optional[str],
    video_name: Optional[str],
    seed: Optional[int],
    client,
) -> int:
    """Print the plan and patch/validate each Comfy piece. No queue."""
    from master_agent.comfy.validator import format_report, validate_workflow
    from master_agent.comfy.workflow_patcher import load_and_patch_workflow, media_wiring_error
    from master_agent.config import OBJECT_INFO_CACHE
    from master_agent.orchestrator.pipeline import _is_topology_error

    if not plan.segmented:
        print(format_lipdub_plan(plan))
        return 0
    try:
        object_info, source = client.load_object_info(prefer_live=True)
    except Exception as exc:
        if OBJECT_INFO_CACHE.is_file():
            object_info = json.loads(OBJECT_INFO_CACHE.read_text(encoding="utf-8"))
            source = f"cache:{OBJECT_INFO_CACHE}"
            print(f"object_info: {source} (live unavailable: {exc})")
        else:
            print(format_lipdub_plan(plan))
            print(f"FAIL  cannot load /object_info: {exc}")
            print("dry-run plan printed above; workflow validate skipped")
            return 1
    else:
        print(f"object_info: {source}")

    if plan.anchor == ANCHOR_PAUSE_RESET:
        _probe_pause_reset(plan, client, log=print, object_info=object_info)
    print(format_lipdub_plan(plan))

    failures = 0
    for piece in plan.pieces:
        if piece.kind == KIND_PLATE:
            print(
                f"segment {piece.index + 1}/{len(plan.pieces)} silence_plate "
                f"hold={piece.hold_frames} frames source=still (no Comfy) "
                f"anchor={plan.anchor} reframe={'on' if plan.reframe else 'off'}"
            )
            continue
        image = _dry_image_name(piece, plan, image_name)
        try:
            wf, meta = load_and_patch_workflow(
                variant,
                prompt=piece.prompt or request,
                negative_prompt=piece.negative,
                duration_s=piece.audio_duration_s,
                width=width,
                height=height,
                seed=seed if seed is not None else 0,
                image_name=image,
                audio_name=audio_name,
                video_name=video_name,
                audio_start_s=piece.audio_start_s,
                frames=piece.render_frames,
                i2v_strength=piece.i2v_strength,
            )
            if piece.end_keyframe == "source_still":
                from master_agent.orchestrator.lipdub_guide import apply_last_frame_guide

                wf, guide_patch = apply_last_frame_guide(
                    wf,
                    image_name=image_name or "still.png",
                    object_info=object_info,
                    frame_idx=-1 if piece.guide_frame_idx is None else int(piece.guide_frame_idx),
                    strength=1.0 if piece.guide_strength is None else float(piece.guide_strength),
                )
                print(
                    f"  end_keyframe={piece.end_keyframe} guide={guide_patch.status} "
                    f"{guide_patch.message}"
                )
                if not guide_patch.applied:
                    print(f"FAIL  segment {piece.index + 1}: last-frame guide was not patched")
                    failures += 1
            elif piece.silence_crossfade_frames:
                print(
                    f"  silence crossfade {piece.silence_crossfade_frames} frames "
                    "inside this pause (no LTXVAddGuide)"
                )
        except Exception as exc:
            print(f"FAIL  segment {piece.index + 1} patch: {exc}")
            failures += 1
            continue
        print(
            f"segment {piece.index + 1}/{len(plan.pieces)} {piece.kind} "
            f"duration={piece.audio_duration_s:.3f}s frames={meta.get('frames')} "
            f"audio_start={piece.audio_start_s:.3f} image_name={image!r} "
            f"continuity={piece.continuity} anchor={plan.anchor} "
            f"reframe={'on' if plan.reframe else 'off'} "
            f"crossfade={piece.crossfade_frames} i2v_strength={piece.i2v_strength}"
        )
        wiring = media_wiring_error(
            wf, image_name=image, audio_name=audio_name, video_name=video_name
        )
        if wiring:
            print(f"FAIL  segment {piece.index + 1}: {wiring}")
            failures += 1
        report = validate_workflow(
            wf, object_info, file_label=f"segment:{piece.index + 1}", object_info_source=source
        )
        print(format_report(report))
        topology = [err for err in report.errors if _is_topology_error(err)]
        if topology:
            print(f"FAIL  segment {piece.index + 1} topology: {topology}")
            failures += 1
        elif report.errors:
            print(
                f"note  segment {piece.index + 1}: inventory/combo errors ignored for "
                "Comfy-less dry-run (plan + wiring still proven)"
            )
    print(f"\ndry-run {'OK' if not failures else f'FAILED ({failures})'} — nothing queued")
    return 0 if not failures else 1


def _dry_image_name(piece: LipdubPiece, plan: LipdubPlan, image_name: Optional[str]) -> str:
    if piece.continuity in (CONTINUITY_STILL, CONTINUITY_SOURCE) or plan.anchor == ANCHOR_SOURCE:
        return image_name or "still.png"
    if piece.continuity == CONTINUITY_HYBRID:
        return f"lipdub_hybrid_{piece.index}.png"
    if piece.continuity == CONTINUITY_PREV or plan.anchor == ANCHOR_PREVIOUS:
        return f"lipdub_last_{piece.index}.png"
    return image_name or "still.png"


def _conditioning_image(
    piece: LipdubPiece,
    plan: LipdubPlan,
    *,
    orch,
    image_name: Optional[str],
    still_path: Optional[Path],
    last_frame: Optional[Path],
    out_dir: Path,
) -> tuple[Optional[str], str]:
    """Comfy image name plus the provenance source_frame note."""
    use_still = piece.continuity in (CONTINUITY_STILL, CONTINUITY_SOURCE) or (
        plan.anchor == ANCHOR_SOURCE and piece.continuity != CONTINUITY_PREV
    )
    if plan.anchor == ANCHOR_HYBRID and piece.continuity == CONTINUITY_HYBRID:
        if last_frame is None or still_path is None:
            return None, f"lipdub piece {piece.index} hybrid guide needs the still and the previous frame"
        from master_agent.orchestrator.lipdub_reframe import compose_hybrid_guide_file

        guide = out_dir / f"piece_{piece.index:02d}_guide.png"
        try:
            compose_hybrid_guide_file(still_path, last_frame, guide)
            name = _upload_image(orch.client, guide, dry_run=False)
        except Exception as exc:
            return None, f"lipdub hybrid guide failed: {exc}"
        return name, str(guide)
    if use_still or plan.anchor == ANCHOR_SOURCE:
        return image_name, image_name or (str(still_path) if still_path else "still")
    if last_frame is None:
        return None, f"lipdub piece {piece.index} has no previous frame to continue from"
    try:
        name = _upload_image(orch.client, last_frame, dry_run=False)
    except Exception as exc:
        return None, f"lipdub continuity upload failed: {exc}"
    return name, str(last_frame)


def _fit_piece(
    src: Path,
    piece: LipdubPiece,
    plan: LipdubPlan,
    *,
    out_dir: Path,
    still_path: Optional[Path],
    record: dict,
) -> tuple[Path, Optional[Path]]:
    """Trim onto the grid, optionally reframe, and split off the overlap head."""
    from master_agent.orchestrator.lipdub_reframe import reframe_video

    span = piece.drop_leading + piece.generated_keep
    full = trim_span(src, out_dir / f"piece_{piece.index:02d}_span.mp4", 0, span, fps=plan.fps)
    if plan.reframe and still_path is not None and still_path.is_file():
        reframed, summary = reframe_video(
            full,
            still_path,
            out_dir / f"piece_{piece.index:02d}_reframe.mp4",
            fps=plan.fps,
            method="auto",
        )
        full = reframed
        drift = {"index": piece.index, **summary.as_dict(), "applied": bool(summary.worth_applying)}
        plan.scale_drift.append(drift)
        record["scale"] = drift["scale"]
        record["dx"] = drift["dx"]
        record["dy"] = drift["dy"]
    overlap_path = None
    if piece.drop_leading:
        overlap_path = trim_span(
            full,
            out_dir / f"piece_{piece.index:02d}_overlap.mp4",
            0,
            piece.drop_leading,
            fps=plan.fps,
        )
        body = trim_span(
            full,
            out_dir / f"piece_{piece.index:02d}_gen.mp4",
            piece.drop_leading,
            piece.generated_keep,
            fps=plan.fps,
        )
    else:
        body = out_dir / f"piece_{piece.index:02d}_gen.mp4"
        if full != body:
            full.replace(body)
    return body, overlap_path


def _crossfade_previous(
    piece: LipdubPiece,
    plan: LipdubPlan,
    *,
    overlap_path: Optional[Path],
    clip_paths: list,
    records: list,
    out_dir: Path,
) -> Optional[tuple[Path, Path]]:
    if piece.crossfade_frames <= 0 or overlap_path is None:
        return None
    prev_index = piece.index - 1
    while prev_index >= 0 and clip_paths[prev_index] is None:
        prev_index -= 1
    if prev_index < 0:
        return None
    from master_agent.orchestrator.lipdub_reframe import crossfade_tail

    faded = crossfade_tail(
        clip_paths[prev_index],
        overlap_path,
        out_dir / f"piece_{prev_index:02d}_xfade.mp4",
        piece.crossfade_frames,
        fps=plan.fps,
    )
    old = clip_paths[prev_index]
    clip_paths[prev_index] = faded
    if 0 <= prev_index < len(records):
        records[prev_index]["output_path"] = str(faded)
        records[prev_index]["hash"] = sha256_file(faded)
        records[prev_index]["crossfaded_frames"] = int(piece.crossfade_frames)
    return old, faded


def _probe_pause_reset(plan: LipdubPlan, client, *, log, object_info=None) -> None:
    """Ask object_info whether the last-frame keyframe can be wired, then lock the plan."""
    if plan.anchor != ANCHOR_PAUSE_RESET:
        return
    info = object_info
    if info is None:
        from master_agent.config import OBJECT_INFO_CACHE

        try:
            info, _source = client.load_object_info(prefer_live=True)
        except Exception:
            if OBJECT_INFO_CACHE.is_file():
                info = json.loads(OBJECT_INFO_CACHE.read_text(encoding="utf-8"))
            else:
                info = {}
    available, message = probe_pause_reset_guide(info if isinstance(info, dict) else {})
    resolve_pause_reset(plan, guide_available=available)
    log(message)
