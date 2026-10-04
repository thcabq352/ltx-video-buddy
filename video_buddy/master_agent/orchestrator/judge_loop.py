"""Judge revise / re-judge loop.

The while-True cycle lives here. Orchestrator.run calls ``judge_loop`` after
a successful resolve (or immediately on a self-improve dry run).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from master_agent.judge.judge import judge_segment
from master_agent.judge.quality_bar import (
    RevisePlan,
    apply_revise_plan,
    build_revise_plan,
    context_from_state,
)
from master_agent.orchestrator.state import (
    LOOP_ERROR,
    LOOP_EXHAUSTED,
    LOOP_HUMAN_VETO,
    LOOP_PASSED,
    RunState,
)
from master_agent.orchestrator.talking import (
    clip_duration_cap,
    enforce_audio_duration,
    scrub_overlong_duration,
)
from master_agent.provenance import (
    apply_sidecar_to_state,
    attempt_id_of,
    latest_revise_notes,
    persist_clip_provenance,
    read_sidecar_for_state,
    shot_id_of,
)

# Param hints the judge may tune (whitelist; anything else is ignored)
RETUNE_ALLOWED = {"steps", "cfg", "stg_scale", "stg_blocks", "sampler_name", "seed"}
RETUNE_RANGES = {
    "steps": (1, 80),
    "cfg": (1.0, 10.0),
    "stg_scale": (0.0, 2.0),
}

def judge_loop(orch, st: RunState) -> None:
    """JUDGE state. Fail → revise plan → re-run (or dry re-judge) → stop."""
    from pathlib import Path

    while True:
        prior = read_sidecar_for_state(st)
        if prior:
            lin = prior.get("lineage") or {}
            st.log(
                f"provenance read attempt_id={lin.get('attempt_id')} "
                f"hash={(prior.get('hash') or '')[:12]}"
            )
        if st.dry_run and not (st.video_path and Path(st.video_path).is_file()):
            heuristic, issues = 1.0, []
        else:
            expected = st.duration_s
            cap = clip_duration_cap(st)
            if cap is not None:
                expected = min(float(expected or cap), float(cap))
            import master_agent.orchestrator.machine as machine

            # Tests patch master_agent.orchestrator.machine.analyze.
            heuristic, issues = machine.analyze(
                st.video_path, expected_duration_s=expected
            )
            try:
                from master_agent.judge.probe import probe_video

                st.has_audio = bool(probe_video(st.video_path).get("has_audio"))
            except Exception:
                pass
        ctx = context_from_state(st)
        result = judge_segment(
            user_request=st.request,
            ltx_prompt=st.prompt,
            shot=st.shot,
            video_path=st.video_path,
            heuristic_score=heuristic,
            heuristic_issues=issues,
            judge_retries=st.attempt - 1,
            max_rounds=st.max_judge_rounds,
            judge_enabled=st.judge_enabled,
            context=ctx,
        )
        st.judge_score = result.combined_score
        st.judge_decision = result.decision
        st.judge_issues = result.issues
        st.judge_reason = result.reason
        st.quality_bar = result.quality_bar or {}
        st.judge_round = st.attempt - 1
        st.judge_history.append(result.to_dict())
        st.log(
            f"judge attempt {st.attempt}/{st.max_judge_rounds}: "
            f"combined={result.combined_score:.2f} decision={result.decision} "
            f"reason={result.reason}"
        )
        persist_clip_provenance(st, revise_notes=latest_revise_notes(st))

        if result.decision == "human_veto":
            st.loop_status = LOOP_HUMAN_VETO
            st.transition("DONE")
            return
        if result.decision == "accept" and result.pass_:
            st.loop_status = LOOP_PASSED
            st.transition("DONE")
            return
        if result.decision == "exhausted" or st.attempt >= st.max_judge_rounds:
            st.loop_status = LOOP_EXHAUSTED
            st.judge_decision = "exhausted"
            st.log(
                f"loop stop: exhausted after {st.attempt}/{st.max_judge_rounds} attempts"
            )
            st.transition("DONE")
            return

        judged_attempt = int(st.attempt or 1)
        judged_shot = st.shot_id or shot_id_of(st)
        prior = read_sidecar_for_state(st)
        if prior:
            apply_sidecar_to_state(st, prior)
            st.log(
                f"provenance source-of-truth "
                f"{(prior.get('lineage') or {}).get('attempt_id')}"
            )
        # Parent link is the attempt just judged, not whatever the sidecar
        # already says (that record can be the child shell).
        st.parent_shot_id = judged_shot
        st.parent_attempt_id = attempt_id_of(judged_shot, judged_attempt)
        plan = revise_plan(st, result)
        if not plan.actionable:
            st.loop_status = LOOP_EXHAUSTED
            st.judge_decision = "exhausted"
            st.log("loop stop: no actionable revise plan")
            st.transition("DONE")
            return

        apply_full_revise(orch, st, plan, result)
        st.revise_history.append(
            {"attempt": judged_attempt, **plan.to_dict()}
        )
        # The child has not been judged and does not own the parent file yet.
        st.judge_score = 0.0
        st.judge_issues = []
        st.judge_reason = ""
        st.judge_decision = ""
        st.quality_bar = {}
        st.attempt = judged_attempt + 1
        st.judge_round = st.attempt - 1
        persist_clip_provenance(
            st, revise_notes=latest_revise_notes(st), omit_hash=True
        )

        if st.dry_run:
            st.transition("JUDGE")
            continue

        st.transition("PATCH")
        if not (orch._patch(st) and orch._validate(st)):
            st.loop_status = LOOP_ERROR
            return
        if not orch._submit_and_poll(st):
            st.loop_status = LOOP_ERROR
            return
        st.transition("RESOLVE")
        if not orch._resolve(st):
            st.loop_status = LOOP_ERROR
            return
        st.transition("JUDGE")


def revise_plan(st: RunState, result) -> RevisePlan:
    fails = list((result.quality_bar or {}).get("fails") or [])
    plan = build_revise_plan(
        fails,
        context_from_state(st),
        base_prompt=st.prompt,
        steps=st.steps,
    )
    if result.prompt_rewrite and result.prompt_rewrite.strip():
        rewrite = result.prompt_rewrite.strip()
        if rewrite not in plan.prompt_deltas and rewrite != (st.prompt or "").strip():
            plan.prompt_deltas.insert(0, rewrite)
            # Full rewrite wins over additive deltas when the judge supplied one.
            if rewrite:
                plan.prompt_deltas = [rewrite]
    if result.param_hints:
        for key, value in result.param_hints.items():
            plan.param_deltas.setdefault(key, value)
    if result.revise_plan and not plan.actionable:
        raw = result.revise_plan
        plan = RevisePlan(
            fail_ids=list(raw.get("fail_ids") or []),
            prompt_deltas=list(raw.get("prompt_deltas") or []),
            param_deltas=dict(raw.get("param_deltas") or {}),
            shot_patch=dict(raw.get("shot_patch") or {}),
            control_pack_used=dict(raw.get("control_pack_used") or {}),
            music_bed_attached=bool(raw.get("music_bed_attached")),
            reason=str(raw.get("reason") or ""),
        )
    return plan


def apply_full_revise(orch, st: RunState, plan: RevisePlan, result) -> None:
    applied = apply_revise_plan(st, plan)
    if result.decision == "rewrite" and result.prompt_rewrite.strip():
        st.prompt = result.prompt_rewrite.strip()
        if "prompt" not in applied:
            applied.append("prompt")
        st.log("judge requested prompt rewrite")
    if plan.param_deltas or result.param_hints:
        hints = dict(result.param_hints or {})
        hints.update(plan.param_deltas)
        apply_retune(st, hints)
        st.log(f"revise params: {hints}")
    cap = enforce_audio_duration(st)
    if cap and st.prompt:
        st.prompt = scrub_overlong_duration(st.prompt, cap)
    if applied:
        st.log(f"revise applied: {applied} ({plan.reason})")


def apply_retune(st: RunState, hints: dict[str, Any]) -> None:
    for key, value in hints.items():
        if key not in RETUNE_ALLOWED:
            continue
        if key in RETUNE_RANGES:
            lo, hi = RETUNE_RANGES[key]
            try:
                value = max(lo, min(hi, float(value)))
            except (TypeError, ValueError):
                continue
            value = int(value) if key == "steps" else value
        if key == "seed":
            try:
                value = int(value)
            except (TypeError, ValueError):
                continue
        setattr(st, key, value)

