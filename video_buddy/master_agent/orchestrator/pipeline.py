"""Multi-segment pipeline — task decomposition above the Orchestrator.

Flow: plan segment durations → storyboard (LLM/heuristic shot cards) →
per-segment Orchestrator runs (each with its own validate + judge loop) →
ffmpeg stitch → full-video judge → selective weak-shot re-gen → re-stitch.

Single-segment requests delegate straight to Orchestrator.run.
"""

from __future__ import annotations

import json
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from master_agent.comfy.client import ComfyClient
from master_agent.config import (
    JUDGE_ENABLED,
    JUDGE_SCORE_THRESHOLD,
    MAX_FULL_JUDGE_ROUNDS,
    MAX_JUDGE_ROUNDS,
    OUTPUTS_DIR,
    RUNS_DIR,
    STORYBOARD_MODE,
    plan_segment_durations,
)
from master_agent.hands import Hands, default_hands
from master_agent.judge.judge import judge_full_video, parse_weak_shot_indices
from master_agent.judge.probe import probe_video
from master_agent.orchestrator.machine import Orchestrator
from master_agent.orchestrator.state import RunState
from master_agent.storyboard.storyboard import (
    ShotCard,
    build_storyboard,
    build_storyboard_panel,
    should_storyboard,
    storyboard_to_markdown,
)


def _plan_storyboard(
    request: str,
    segs: list[float],
    *,
    variant: Optional[str],
    quality: Optional[str],
    llm_panel: Optional[str],
    panel_judge: Optional[str],
    log,
) -> tuple[list[ShotCard], str, Optional[dict[str, Any]]]:
    """Storyboard via the LLM panel when available, else the single-LLM path."""
    from master_agent.llm_panel import resolve_panel

    rag_context = ""
    try:
        from master_agent.kb.recall import recall_similar_runs, recall_workflows

        blocks = [recall_similar_runs(request), recall_workflows(request)]
        rag_context = "\n\n".join(b for b in blocks if b)
        if rag_context:
            log("kb recall: injecting similar past runs into storyboard prompt")
    except Exception:
        rag_context = ""

    res = resolve_panel(llm_panel)
    for spec, why in res.skipped:
        log(f"panel skip {spec}: {why}")
    if res.members:
        cards, style, meta = build_storyboard_panel(
            request,
            segs,
            members=res.members,
            judge_provider=panel_judge,
            variant=variant,
            quality=quality,
            rag_context=rag_context,
        )
        for c in meta.get("candidates", []):
            status = "ok" if c.get("ok") else f"error: {c.get('error')}"
            log(f"panel candidate {c.get('provider')}: {c.get('latency_s')}s {status}")
        if meta.get("judge"):
            log(
                f"panel judge ({meta['judge']}) winner: {meta.get('winner')}"
                + (f" — {meta.get('judge_reason')}" if meta.get("judge_reason") else "")
            )
        elif meta.get("winner"):
            log(f"panel winner: {meta['winner']} ({meta.get('judge_reason')})")
        if meta.get("fallback"):
            log(f"panel fallback: {meta['fallback']}")
        return cards, style, meta
    cards, style = build_storyboard(
        request, segs, variant=variant, quality=quality, rag_context=rag_context
    )
    return cards, style, None


class PipelineResult:
    def __init__(self, run_id: str, request: str = ""):
        self.run_id = run_id
        self.request = request
        self.status = "started"  # done | done_with_warnings | error
        self.error: Optional[str] = None
        self.video_path: Optional[str] = None
        self.segment_paths: list[str] = []
        self.segment_scores: list[float] = []
        self.segment_durations: list[float] = []
        self.storyboard: list[dict[str, Any]] = []
        self.storyboard_md: str = ""
        self.panel_meta: Optional[dict[str, Any]] = None
        self.full_judge_score: float = 0.0
        self.full_judge_pass: bool = False
        self.full_judge_notes: str = ""
        self.full_judge_history: list[dict[str, Any]] = []
        self.budget: dict[str, Any] = {}
        self.budget_held: list[dict[str, Any]] = []
        self.budget_decisions: list[dict[str, Any]] = []
        self.messages: list[str] = []
        self.previs_source: Optional[str] = None
        self.control_pack_present: bool = False
        self.control_pack_used: dict[str, bool] = {}
        self.loop_status: str = ""
        self.quality_bar: dict[str, Any] = {}
        self.revise_history: list[dict[str, Any]] = []
        self.attempt: int = 0
        self.provenance: dict[str, Any] = {}
        self.provenance_history: list[dict[str, Any]] = []
        self.provenance_sidecar: str = ""

    def log(self, msg: str) -> None:
        self.messages.append(msg)
        print(f"[pipeline] {msg}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "request": self.request,
            "status": self.status,
            "error": self.error,
            "video_path": self.video_path,
            "segment_paths": self.segment_paths,
            "segment_scores": self.segment_scores,
            "segment_durations": self.segment_durations,
            "storyboard": self.storyboard,
            "storyboard_md": self.storyboard_md,
            "panel_meta": self.panel_meta,
            "full_judge_score": self.full_judge_score,
            "full_judge_pass": self.full_judge_pass,
            "full_judge_notes": self.full_judge_notes,
            "full_judge_history": self.full_judge_history,
            "budget": self.budget,
            "budget_held": self.budget_held,
            "budget_decisions": self.budget_decisions,
            "messages": self.messages,
            "previs_source": self.previs_source,
            "control_pack_present": self.control_pack_present,
            "control_pack_used": self.control_pack_used,
            "loop_status": self.loop_status,
            "quality_bar": self.quality_bar,
            "revise_history": self.revise_history,
            "attempt": self.attempt,
            "provenance": self.provenance,
            "provenance_history": self.provenance_history,
            "provenance_sidecar": self.provenance_sidecar,
        }


def _write_record(result: PipelineResult) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    record = RUNS_DIR / f"{ts}_{result.run_id}_pipeline.json"
    try:
        record.write_text(
            json.dumps(result.to_dict(), indent=1, default=str) + "\n", encoding="utf-8"
        )
        result.log(f"pipeline record: {record}")
    except OSError as e:
        result.log(f"could not write pipeline record: {e}")
    try:
        from master_agent.kb.ingest import ingest_run_record

        if ingest_run_record(result.to_dict()):
            result.log("kb: run record ingested")
    except Exception:
        pass


def _budget_admit(result: PipelineResult, scene_id: str, variant: Optional[str], duration_s: float) -> dict[str, Any]:
    from master_agent.control.budget import admit_scene

    row = admit_scene(scene_id, variant=variant, duration_s=duration_s)
    result.budget_decisions.append(row)
    result.budget = {
        "used": row["used_after"],
        "cap": row["cap"],
        "paused": row["paused"],
    }
    result.log(
        f"budget {row['decision']} {scene_id} +{row['cost_vram_min']} "
        f"used={row['used_after']}/{row['cap']}"
    )
    return row


_MUSIC_KINDS = frozenset({"music", "music_video", "mv", "music_video_mv"})


def plan_story_segments(
    duration_s: float,
    *,
    kind: str = "",
    quality: Optional[str] = None,
    hands: Hands | None = None,
) -> list[float]:
    """Split a story. Hands owns the 8s last-frame chain; music keeps beats/quality."""
    kind_l = (kind or "").strip().lower()
    if kind_l in _MUSIC_KINDS:
        return plan_segment_durations(duration_s, quality=quality)
    checker = hands if hands is not None else default_hands()
    chain = checker.plan_last_frame_chain(float(duration_s))
    return [float(c.duration_s) for c in chain.clips]


def _synthetic_cards(request: str, segs: list[float]) -> list[ShotCard]:
    return [
        ShotCard(
            index=i,
            title=f"Shot {i + 1}",
            duration_s=d,
            ltx_prompt=request,
            seed_offset=i * 17,
        )
        for i, d in enumerate(segs)
    ]


def run_pipeline(
    request: str,
    *,
    variant: Optional[str] = None,
    duration_s: float = 5.0,
    quality: Optional[str] = None,
    seed: Optional[int] = None,
    width: int = 768,
    height: int = 512,
    video_name: Optional[str] = None,
    image_name: Optional[str] = None,
    audio_name: Optional[str] = None,
    storyboard_mode: Optional[str] = None,
    judge_enabled: Optional[bool] = None,
    max_judge_rounds: int = MAX_JUDGE_ROUNDS,
    max_full_judge_rounds: int = MAX_FULL_JUDGE_ROUNDS,
    judge_threshold: float = JUDGE_SCORE_THRESHOLD,
    llm_panel: Optional[str] = None,
    panel_judge: Optional[str] = None,
    power_mode: Optional[bool] = None,
    attach_recipe: Optional[dict[str, Any]] = None,
    client: Optional[ComfyClient] = None,
    dry_run: bool = False,
    kind: str = "",
    music_bed_attached: bool = False,
) -> PipelineResult:
    run_id = uuid.uuid4().hex[:12]
    result = PipelineResult(run_id, request=request)
    j_enabled = JUDGE_ENABLED if judge_enabled is None else judge_enabled
    sb_mode = (storyboard_mode or STORYBOARD_MODE).strip().lower()
    base_seed = seed if seed is not None else random.randint(0, 2**32 - 1)
    orch = Orchestrator(client=client)

    segs = plan_story_segments(duration_s, kind=kind, quality=quality)
    result.segment_durations = segs
    result.log(f"plan: {duration_s}s -> {len(segs)} segment(s) {segs} (judge={j_enabled})")

    # Single segment → plain orchestrator run (no storyboard/stitch overhead)
    if len(segs) == 1:
        gate = _budget_admit(result, f"{run_id}:seg0", variant, segs[0])
        if gate["decision"] == "hold":
            result.status = "paused"
            result.budget_held.append({"id": f"{run_id}:seg0", **gate})
            result.error = (
                f"render budget paused {run_id}:seg0 "
                f"(used {gate['used_after']}/{gate['cap']} VRAM-min)"
            )
            _write_record(result)
            return result
        st = orch.run(
            request,
            variant=variant,
            duration_s=segs[0],
            quality=quality,
            seed=base_seed,
            width=width,
            height=height,
            video_name=video_name,
            image_name=image_name,
            audio_name=audio_name,
            judge_enabled=j_enabled,
            max_judge_rounds=max_judge_rounds,
            power_mode=power_mode,
            attach_recipe=attach_recipe,
            dry_run=dry_run,
            kind=kind,
            music_bed_attached=music_bed_attached,
            shot_index=1,
        )
        result.messages.extend(st.messages)
        result.status = "done" if st.state == "DONE" else "error"
        result.error = st.error
        result.video_path = st.video_path
        result.previs_source = st.previs_source
        result.control_pack_present = st.control_pack_present
        result.control_pack_used = st.control_pack_used
        result.loop_status = st.loop_status
        result.quality_bar = st.quality_bar
        result.revise_history = st.revise_history
        result.attempt = st.attempt
        result.provenance = st.provenance
        result.provenance_history = st.provenance_history
        result.provenance_sidecar = st.provenance_sidecar
        if st.video_path:
            result.segment_paths = [st.video_path]
            result.segment_scores = [st.judge_score]
        result.full_judge_score = st.judge_score
        result.full_judge_pass = st.loop_status == "passed" or st.judge_decision == "accept"
        _write_record(result)
        return result

    # Storyboard once for the whole piece
    use_board = should_storyboard(
        sb_mode, segment_count=len(segs), user_request=request, variant=variant
    )
    if use_board:
        cards, global_style, panel_meta = _plan_storyboard(
            request,
            segs,
            variant=variant,
            quality=quality,
            llm_panel=llm_panel,
            panel_judge=panel_judge,
            log=result.log,
        )
        result.panel_meta = panel_meta
        result.storyboard_md = storyboard_to_markdown(cards, global_style=global_style)
        result.log(f"storyboard ({len(cards)} shots):\n{result.storyboard_md}")
    else:
        cards = _synthetic_cards(request, segs)
    result.storyboard = [c.to_dict() for c in cards]

    def _gen_segment(i: int, seed_bump: int = 0) -> RunState:
        card = cards[i]
        seg_seed = (base_seed + seed_bump + card.seed_offset) & 0xFFFFFFFF
        seg_image = image_name
        if i > 0 and result.segment_paths:
            from master_agent.hands import extract_last_frame

            dest = OUTPUTS_DIR / result.run_id / f"chain_last_{i}.png"
            frame = extract_last_frame(result.segment_paths[i - 1], dest)
            if frame is not None:
                seg_image = str(frame)
                result.log(f"hands last-frame chain: clip {i + 1} from {frame.name}")
        return orch.run(
            request,
            prompt=card.ltx_prompt or request,
            shot=card.to_dict(),
            variant=variant,
            duration_s=segs[i],
            quality=quality,
            seed=seg_seed,
            width=width,
            height=height,
            video_name=video_name,
            image_name=seg_image,
            audio_name=audio_name,
            judge_enabled=j_enabled,
            max_judge_rounds=max_judge_rounds,
            power_mode=power_mode,
            attach_recipe=attach_recipe,
            dry_run=dry_run,
            kind=kind,
            music_bed_attached=music_bed_attached,
            shot_index=i + 1,
        )

    # Per-segment generation (budget can pause the remaining queue)
    for i in range(len(segs)):
        scene_id = f"{run_id}:seg{i}"
        gate = _budget_admit(result, scene_id, variant, segs[i])
        if gate["decision"] == "hold":
            result.budget_held.append({"id": scene_id, **gate})
            for j in range(i + 1, len(segs)):
                extra_id = f"{run_id}:seg{j}"
                extra = _budget_admit(result, extra_id, variant, segs[j])
                result.budget_held.append({"id": extra_id, **extra})
            result.log(f"budget pause: {len(result.budget_held)} scene(s) held for review")
            break
        result.log(f"--- segment {i + 1}/{len(segs)} ({segs[i]}s) ---")
        st = _gen_segment(i)
        result.messages.extend(st.messages)
        if st.state != "DONE" or not st.video_path:
            result.status = "error"
            result.error = f"segment {i + 1} failed: {st.error}"
            _write_record(result)
            return result
        result.segment_paths.append(st.video_path)
        result.segment_scores.append(st.judge_score)
        try:
            orch.client.free_memory()
        except Exception:
            pass

    if result.budget_held and not result.segment_paths:
        result.status = "paused"
        result.error = (
            f"render budget paused; {len(result.budget_held)} scene(s) held for review "
            f"(used {result.budget.get('used')}/{result.budget.get('cap')} VRAM-min)"
        )
        _write_record(result)
        return result

    # Stitch
    final = _stitch(result, result.segment_paths, suffix="")
    if final is None:
        _write_record(result)
        return result

    if result.budget_held:
        result.video_path = str(Path(final).resolve())
        result.status = "paused"
        result.error = (
            f"render budget paused after {len(result.segment_paths)} scene(s); "
            f"{len(result.budget_held)} held for review"
        )
        _write_record(result)
        return result

    # Outer full-video judge + selective weak-shot re-gen
    if j_enabled:
        for fr in range(max_full_judge_rounds):
            fj = judge_full_video(
                user_request=request,
                storyboard=result.storyboard,
                video_path=str(final),
                segment_scores=result.segment_scores,
                threshold=judge_threshold,
            )
            result.full_judge_score = fj.combined_score
            result.full_judge_pass = fj.pass_
            result.full_judge_notes = fj.reason
            result.full_judge_history.append(fj.to_dict())
            result.log(
                f"full judge round {fr + 1}/{max_full_judge_rounds}: "
                f"combined={fj.combined_score:.2f} pass={fj.pass_} — {fj.reason}"
            )
            if fj.pass_:
                break
            weak = parse_weak_shot_indices(fj.issues, len(segs))
            if not weak:
                break
            result.log(f"re-generating weak shots: {[w + 1 for w in weak]}")
            if fj.prompt_rewrite and weak[0] < len(cards):
                cards[weak[0]].ltx_prompt = fj.prompt_rewrite
                result.storyboard = [c.to_dict() for c in cards]
            for wi in weak:
                if wi < 0 or wi >= len(segs):
                    continue
                regen_id = f"{run_id}:seg{wi}:r{fr + 1}"
                regen_gate = _budget_admit(result, regen_id, variant, segs[wi])
                if regen_gate["decision"] == "hold":
                    result.budget_held.append({"id": regen_id, **regen_gate})
                    result.log(f"budget hold skip regen {regen_id}")
                    continue
                regen = _gen_segment(wi, seed_bump=777 * (fr + 1) + wi)
                result.messages.extend(regen.messages)
                if regen.state == "DONE" and regen.video_path:
                    result.segment_paths[wi] = regen.video_path
                    if wi < len(result.segment_scores):
                        result.segment_scores[wi] = regen.judge_score
                try:
                    orch.client.free_memory()
                except Exception:
                    pass
            final = _stitch(result, result.segment_paths, suffix=f"_r{fr + 1}")
            if final is None:
                _write_record(result)
                return result
    else:
        result.full_judge_score = (
            sum(result.segment_scores) / len(result.segment_scores)
            if result.segment_scores
            else 0.0
        )
        result.full_judge_pass = result.full_judge_score >= judge_threshold

    result.video_path = str(Path(final).resolve())
    result.status = "done" if result.full_judge_pass or not j_enabled else "done_with_warnings"
    probe = probe_video(result.video_path)
    result.log(
        f"final: {result.video_path} "
        f"({probe.get('duration_s')}s, full_score={result.full_judge_score:.2f}, {result.status})"
    )
    _write_record(result)
    return result


def _stitch(result: PipelineResult, segment_paths: list[str], *, suffix: str) -> Optional[Path]:
    """Concat segments; on failure keep first segment and mark warnings."""
    from master_agent.video_concat import concat_videos

    paths = [Path(p) for p in segment_paths]
    if len(paths) == 1:
        return paths[0]
    out_dir = OUTPUTS_DIR / result.run_id
    dest = out_dir / f"master_agent_full_{result.run_id}{suffix}.mp4"
    try:
        final = concat_videos(paths, dest)
        result.log(f"stitched {len(paths)} segments -> {final}")
        try:
            from master_agent.provenance import inherit_clip_provenance

            payload = inherit_clip_provenance(
                paths[0],
                final,
                revise_notes="pipeline stitch",
                extra={"brief": result.request, "prompt": result.request},
            )
            result.provenance = payload
            result.provenance_sidecar = str(
                Path(final).with_name(Path(final).stem + ".buddy.json")
            )
        except Exception as e:
            result.log(f"provenance stitch skipped: {e}")
        return final
    except Exception as e:
        result.log(f"stitch failed ({e}); keeping first segment")
        result.status = "done_with_warnings"
        result.video_path = segment_paths[0]
        result.error = f"stitch failed: {e}"
        return None


def dry_run_pipeline(
    request: str,
    *,
    variant: Optional[str] = None,
    duration_s: float = 5.0,
    quality: Optional[str] = None,
    width: int = 768,
    height: int = 512,
    storyboard_mode: Optional[str] = None,
    llm_panel: Optional[str] = None,
    panel_judge: Optional[str] = None,
    attach_recipe: Optional[dict[str, Any]] = None,
    client: Optional[ComfyClient] = None,
) -> int:
    """Storyboard + patch + validate every segment without queueing. CLI exit code."""
    from master_agent.comfy.validator import format_report, validate_workflow
    from master_agent.comfy.workflow_patcher import load_and_patch_workflow

    sb_mode = (storyboard_mode or STORYBOARD_MODE).strip().lower()
    segs = plan_story_segments(duration_s, quality=quality)
    print(f"plan: {duration_s}s -> {len(segs)} segment(s) {segs}")

    client = client or ComfyClient()
    orch = Orchestrator(client=client)
    probe_state = RunState(request=request, attach_recipe=attach_recipe)
    variant_sel = orch._select_variant(probe_state, variant)
    print(f"variant: {variant_sel}")

    use_board = should_storyboard(
        sb_mode, segment_count=len(segs), user_request=request, variant=variant_sel
    )
    if use_board and len(segs) > 1:
        cards, global_style, _meta = _plan_storyboard(
            request,
            segs,
            variant=variant_sel,
            quality=quality,
            llm_panel=llm_panel,
            panel_judge=panel_judge,
            log=lambda m: print(f"[panel] {m}"),
        )
        print(storyboard_to_markdown(cards, global_style=global_style))
    else:
        cards = _synthetic_cards(request, segs)

    try:
        object_info, source = client.load_object_info(prefer_live=True)
    except Exception as e:
        print(f"FAIL  cannot load /object_info: {e}")
        return 1
    print(f"object_info: {source}")

    failures = 0
    for i, card in enumerate(cards):
        try:
            wf, meta = load_and_patch_workflow(
                variant_sel,
                prompt=card.ltx_prompt or request,
                duration_s=segs[i],
                width=width,
                height=height,
                seed=card.seed_offset,
            )
        except Exception as e:
            print(f"FAIL  segment {i + 1} patch: {e}")
            failures += 1
            continue
        if attach_recipe:
            try:
                from master_agent.comfy.attach import apply_attach_recipe

                attached = apply_attach_recipe(wf, attach_recipe, object_info=object_info)
                wf = attached.workflow
                print(
                    f"attach: previs_source={attached.previs_source!r} "
                    f"used={attached.control_pack_used}"
                )
            except Exception as e:
                print(f"FAIL  segment {i + 1} attach: {e}")
                failures += 1
                continue
        report = validate_workflow(
            wf, object_info, file_label=f"segment:{i + 1}", object_info_source=source
        )
        print(format_report(report))
        failures += 0 if report.ok else 1
    print(f"\ndry-run {'OK' if not failures else f'FAILED ({failures})'} — nothing queued")
    return 0 if not failures else 1
