"""Orchestrator — the Director. State machine over the ComfyUI bridge.

Flow: SELECT_VARIANT → PATCH → VALIDATE → SUBMIT → POLL → RESOLVE → JUDGE
      → DONE | ERROR   (with PLAN_OOM_RETRY and judge rewrite/retune loops)

Design rules:
- The orchestrator executes; the judge (master_agent.judge) only evaluates.
- Nothing is queued before the validator passes.
- OOM walks config.DOWNSCALE_LADDER instead of dying.
- Every run leaves a JSON record in state/runs/ (seed of the knowledge base).
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from master_agent.comfy.client import ComfyClient, ComfyClientError
from master_agent.comfy.linter import LintBlocked, hard_gate, lint_workflow
from master_agent.comfy.workflow_patcher import load_and_patch_workflow, media_wiring_error
from master_agent.config import (
    DOWNSCALE_LADDER,
    JUDGE_ENABLED,
    MAX_JUDGE_ROUNDS,
    MAX_RETRIES,
    POWER_MODE,
    POWER_MODE_MAX_OPS,
    POWER_MODE_REPAIR,
    RUNS_DIR,
    SEGMENT_MAX_S,
)
from master_agent.judge.judge import judge_segment
from master_agent.judge.probe import analyze
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
    LOOP_STARTED,
    RunState,
)
from master_agent.provenance import (
    apply_sidecar_to_state,
    latest_revise_notes,
    persist_clip_provenance,
    plan_clip_paths,
    planned_clip_path,
    read_sidecar_for_state,
)

log = logging.getLogger(__name__)

# Param hints the judge may tune (whitelist; anything else is ignored)
RETUNE_ALLOWED = {"steps", "cfg", "stg_scale", "stg_blocks", "sampler_name", "seed"}
RETUNE_RANGES = {
    "steps": (1, 80),
    "cfg": (1.0, 10.0),
    "stg_scale": (0.0, 2.0),
}

_OOM_PATTERN = re.compile(
    r"out of memory|cuda oom|CUDAOOM|allocation on device|OOM", re.IGNORECASE
)


def _looks_oom(exc: Exception) -> bool:
    return bool(_OOM_PATTERN.search(str(exc) or ""))


class Orchestrator:
    def __init__(self, client: Optional[ComfyClient] = None):
        self.client = client or ComfyClient()

    # ── states ────────────────────────────────────────────

    def _select_variant(self, st: RunState, force_variant: Optional[str]) -> str:
        from master_agent.hands import LiveHands
        from master_agent.orchestrator.director import choose_variant

        variant, source = choose_variant(
            st.request or "",
            has_video=bool(st.video_name),
            has_image=bool(st.image_name),
            has_audio=bool(st.audio_name),
            force=force_variant,
            attach_recipe=st.attach_recipe,
            hands=LiveHands(client=self.client),
        )
        if source not in ("forced", "input"):
            st.log(f"director routed variant={variant} ({source})")
        return variant

    def _patch(self, st: RunState) -> bool:
        """(Re)build the workflow from current params. Returns success."""
        self._inject_spoken_line(st)
        try:
            workflow, meta = load_and_patch_workflow(
                st.variant or "base",
                prompt=st.prompt,
                negative_prompt=st.negative_prompt,
                width=st.width,
                height=st.height,
                duration_s=st.duration_s,
                seed=st.seed,
                steps=st.steps,
                cfg=st.cfg,
                image_name=st.image_name,
                audio_name=st.audio_name,
                video_name=st.video_name,
                audio_start_s=st.audio_start_s,
                filename_prefix="master_agent",
                stg_scale=st.stg_scale,
                stg_blocks=st.stg_blocks,
                sampler_name=st.sampler_name,
                spoken_line=st.spoken_line or None,
            )
        except Exception as e:
            st.fail(f"patch failed: {e}")
            return False
        st.workflow_meta = meta
        st.seed = meta.get("seed")
        self._workflow = workflow
        if st.attach_recipe:
            try:
                from master_agent.comfy.attach import apply_attach_recipe

                object_info = None
                try:
                    object_info, _src = self.client.load_object_info(prefer_live=True)
                except ComfyClientError:
                    object_info = None
                attached = apply_attach_recipe(
                    self._workflow, st.attach_recipe, object_info=object_info
                )
                self._workflow = attached.workflow
                st.previs_source = attached.previs_source
                st.control_pack_present = attached.control_pack_present
                st.control_pack_used = attached.control_pack_used
                st.log(
                    f"attach recipe applied previs_source={attached.previs_source!r} "
                    f"used={attached.control_pack_used}"
                )
            except Exception as e:
                st.fail(f"attach failed: {e}")
                return False
        if st.power_mode or POWER_MODE:
            self._power_mode(st)
        wiring = media_wiring_error(
            self._workflow,
            image_name=st.image_name,
            audio_name=st.audio_name,
            video_name=st.video_name,
        )
        if wiring:
            st.fail(wiring)
            return False
        return True

    @staticmethod
    def _resolve_h3_line(st: RunState) -> None:
        """Fill a missing H3 line from the brief or a local transcriber, else warn."""
        from master_agent.orchestrator.h3_voice import (
            H3_MISSING_LINE_WARNING,
            resolve_spoken_line,
        )
        from master_agent.orchestrator.talking import H3_AUDIO_VARIANTS, canonical_variant

        if canonical_variant(st.variant) not in H3_AUDIO_VARIANTS:
            return
        if not st.audio_name and not st.audio_path:
            return
        if not (st.spoken_line or "").strip():
            st.spoken_line = resolve_spoken_line("", st.request, st.audio_path)
        if not (st.spoken_line or "").strip():
            st.log(f"warn: {H3_MISSING_LINE_WARNING}")

    @staticmethod
    def _inject_spoken_line(st: RunState) -> None:
        """Keep the H3 spoken line in the prompt across revise re-patches."""
        if not (st.spoken_line or "").strip():
            return
        from master_agent.orchestrator.h3_voice import inject_spoken_line
        from master_agent.orchestrator.talking import H3_AUDIO_VARIANTS, canonical_variant

        if canonical_variant(st.variant) not in H3_AUDIO_VARIANTS:
            return
        st.prompt = inject_spoken_line(st.prompt or st.request, st.spoken_line)

    def _skip_judge(self, st: RunState) -> None:
        """--no-judge: one render. No judge call, no quality-bar revise, no a2."""
        st.judge_decision = "skipped"
        st.judge_reason = "judge skipped (--no-judge): one render, no quality_bar revise"
        st.loop_status = LOOP_PASSED
        st.quality_bar = {}
        st.log("judge skipped: one render, no quality_bar revise")
        persist_clip_provenance(st, revise_notes=latest_revise_notes(st))
        st.transition("DONE")

    def _power_mode(self, st: RunState) -> None:
        """LLM graph ops on the patched workflow; keep base graph if invalid."""
        try:
            from master_agent.comfy.power_mode import power_tune

            object_info, _src = self.client.load_object_info(prefer_live=True)
            pm = power_tune(
                self._workflow,
                request=st.request or st.prompt or "",
                object_info=object_info,
                repair=POWER_MODE_REPAIR,
                max_ops=POWER_MODE_MAX_OPS,
                log=st.log,
            )
            st.power_meta = pm.to_dict()
            if pm.valid and pm.applied:
                self._workflow = pm.workflow
                st.log(f"power-mode applied {len(pm.applied)} op(s)")
            elif pm.error:
                st.log(f"power-mode skipped: {pm.error}")
            else:
                st.log("power-mode: no graph changes")
        except Exception as e:
            st.log(f"power-mode failed (continuing with heuristic patch): {e}")

    def _validate(self, st: RunState) -> bool:
        try:
            object_info, source = self.client.load_object_info(prefer_live=True)
        except ComfyClientError as e:
            st.fail(f"cannot load /object_info: {e}")
            return False
        report = lint_workflow(
            self._workflow, object_info, file_label=f"run:{st.run_id}"
        )
        try:
            hard_gate(report)
        except LintBlocked:
            details = "; ".join(str(i) for i in report.errors[:5])
            st.fail(f"workflow validation failed ({len(report.errors)} errors): {details}")
            return False
        for warning in report.warnings[:5]:
            st.log(f"validator warning: {warning}")
        return True

    def _submit_and_poll(self, st: RunState) -> bool:
        """SUBMIT + POLL states. True on success; on OOM prepares retry."""
        try:
            from master_agent.models.weights import MissingWeightsError, require_weights

            require_weights(st.variant or "base")
        except MissingWeightsError as e:
            st.fail(str(e))
            return False
        try:
            self.client.free_memory()
            st.prompt_id = self.client.queue_prompt(self._workflow)
            st.log(f"queued prompt_id={st.prompt_id}")
        except ComfyClientError as e:
            return self._handle_job_error(st, e, phase="submit")
        st.transition("POLL")
        try:
            entry = self.client.wait_for_prompt(st.prompt_id)
        except ComfyClientError as e:
            return self._handle_job_error(st, e, phase="poll")
        files = ComfyClient.extract_video_files(entry)
        if not files:
            st.fail("job completed but produced no video files")
            return False
        self._output_files = files
        return True

    def _handle_job_error(self, st: RunState, exc: Exception, *, phase: str) -> bool:
        if _looks_oom(exc):
            st.retries += 1
            try:
                from master_agent.models.vram_policy import downscale_ladder_for

                ladder = downscale_ladder_for(st.variant or "")
            except Exception:
                ladder = DOWNSCALE_LADDER
            if st.retries > MAX_RETRIES or st.downscale_level + 1 >= len(ladder):
                st.fail(f"{phase} OOM after {st.retries} retries: {exc}")
                return False
            st.transition("PLAN_OOM_RETRY")
            st.downscale_level += 1
            w, h, frames = ladder[st.downscale_level]
            st.width, st.height = w, h
            st.log(f"OOM at {phase}: downscaling to {w}x{h} ({frames}f), retry {st.retries}")
            return self._patch(st) and self._validate(st) and self._submit_and_poll(st)
        st.fail(f"{phase} failed: {exc}")
        return False

    def _resolve(self, st: RunState) -> bool:
        dst = planned_clip_path(st)
        dst.parent.mkdir(parents=True, exist_ok=True)
        for info in self._output_files:
            src = ComfyClient.resolve_output_path(info)
            if not src.is_file():
                continue
            try:
                shutil.copy2(src, dst)
                st.video_path = str(dst)
                persist_clip_provenance(
                    st, revise_notes=latest_revise_notes(st), path=dst
                )
                st.log(f"output: {dst}")
                if st.provenance_sidecar:
                    st.log(f"provenance: {st.provenance_sidecar}")
                return True
            except OSError as e:
                st.fail(f"could not copy output {src}: {e}")
                return False
        st.fail(f"output files missing on disk: {self._output_files}")
        return False

    def _judge(self, st: RunState) -> None:
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
                heuristic, issues = analyze(
                    st.video_path, expected_duration_s=st.duration_s
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

            prior = read_sidecar_for_state(st)
            if prior:
                apply_sidecar_to_state(st, prior)
                st.log(
                    f"provenance source-of-truth "
                    f"{(prior.get('lineage') or {}).get('attempt_id')}"
                )
            plan = self._revise_plan(st, result)
            if not plan.actionable:
                st.loop_status = LOOP_EXHAUSTED
                st.judge_decision = "exhausted"
                st.log("loop stop: no actionable revise plan")
                st.transition("DONE")
                return

            self._apply_full_revise(st, plan, result)
            st.revise_history.append(
                {"attempt": st.attempt, **plan.to_dict()}
            )
            st.attempt += 1
            st.judge_round = st.attempt - 1
            persist_clip_provenance(st, revise_notes=latest_revise_notes(st))

            if st.dry_run:
                st.transition("JUDGE")
                continue

            st.transition("PATCH")
            if not (self._patch(st) and self._validate(st)):
                st.loop_status = LOOP_ERROR
                return
            if not self._submit_and_poll(st):
                st.loop_status = LOOP_ERROR
                return
            st.transition("RESOLVE")
            if not self._resolve(st):
                st.loop_status = LOOP_ERROR
                return
            st.transition("JUDGE")

    @staticmethod
    def _revise_plan(st: RunState, result) -> RevisePlan:
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

    def _apply_full_revise(self, st: RunState, plan: RevisePlan, result) -> None:
        applied = apply_revise_plan(st, plan)
        if result.decision == "rewrite" and result.prompt_rewrite.strip():
            st.prompt = result.prompt_rewrite.strip()
            if "prompt" not in applied:
                applied.append("prompt")
            st.log("judge requested prompt rewrite")
        if plan.param_deltas or result.param_hints:
            hints = dict(result.param_hints or {})
            hints.update(plan.param_deltas)
            self._apply_retune(st, hints)
            st.log(f"revise params: {hints}")
        if applied:
            st.log(f"revise applied: {applied} ({plan.reason})")

    @staticmethod
    def _apply_retune(st: RunState, hints: dict[str, Any]) -> None:
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

    # ── entry point ───────────────────────────────────────

    def run(
        self,
        request: str,
        *,
        prompt: Optional[str] = None,
        shot: Optional[dict] = None,
        variant: Optional[str] = None,
        duration_s: float = 5.0,
        quality: Optional[str] = None,
        seed: Optional[int] = None,
        width: int = 768,
        height: int = 512,
        video_name: Optional[str] = None,
        image_name: Optional[str] = None,
        audio_name: Optional[str] = None,
        judge_enabled: Optional[bool] = None,
        revise_enabled: Optional[bool] = None,
        spoken_line: Optional[str] = None,
        voice_sample: Optional[dict[str, Any]] = None,
        max_judge_rounds: int = MAX_JUDGE_ROUNDS,
        power_mode: Optional[bool] = None,
        attach_recipe: Optional[dict[str, Any]] = None,
        dry_run: bool = False,
        kind: str = "",
        music_bed_attached: bool = False,
        audio_path: Optional[str] = None,
        audio_start_s: float = 0.0,
        control_pack_present: bool = False,
        control_pack_used: Optional[dict[str, bool]] = None,
        previs_source: str = "",
        shot_index: int = 1,
    ) -> RunState:
        run_id = uuid.uuid4().hex[:12]
        st = RunState(
            request=request,
            run_id=run_id,
            prompt=prompt or request,
            shot=shot,
            shot_index=max(1, int(shot_index or 1)),
            duration_s=duration_s,
            quality=quality,
            seed=seed,
            width=width,
            height=height,
            video_name=video_name,
            image_name=image_name,
            audio_name=audio_name,
            audio_path=audio_path,
            audio_start_s=float(audio_start_s or 0.0),
            kind=kind,
            music_bed_attached=music_bed_attached,
            judge_enabled=JUDGE_ENABLED if judge_enabled is None else judge_enabled,
            revise_enabled=True if revise_enabled is None else bool(revise_enabled),
            spoken_line=(spoken_line or "").strip(),
            voice_sample=dict(voice_sample or {}),
            max_judge_rounds=max_judge_rounds,
            power_mode=POWER_MODE if power_mode is None else bool(power_mode),
            attach_recipe=attach_recipe,
            dry_run=bool(dry_run),
            attempt=1,
            loop_status=LOOP_STARTED,
            control_pack_present=control_pack_present,
            control_pack_used=dict(control_pack_used or {}),
            previs_source=previs_source,
        )
        try:
            st.variant = self._select_variant(st, variant)
            from master_agent.orchestrator.talking import h3_r2v_audio_warning, media_route_error

            route_err = media_route_error(
                st.variant,
                has_image=bool(st.image_name),
                has_audio=bool(st.audio_name),
                has_video=bool(st.video_name),
            )
            if route_err:
                st.fail(route_err)
                return self._finish(st)
            if st.image_name and st.audio_name and not st.video_name:
                st.log(f"route: photo + voice → {st.variant}")
                voice_warn = h3_r2v_audio_warning(
                    st.request,
                    variant=st.variant,
                    has_image=True,
                    has_audio=True,
                    has_video=False,
                )
                if voice_warn:
                    st.log(f"warn: {voice_warn}")
                self._resolve_h3_line(st)
            st.log(f"variant={st.variant} duration={duration_s}s quality={quality or 'default'}")
            plan_clip_paths(st)
            persist_clip_provenance(st, revise_notes=latest_revise_notes(st))

            if st.dry_run:
                if not st.revise_enabled:
                    st.log("dry-run: --no-judge is one pass, no quality_bar revise")
                    self._skip_judge(st)
                    return self._finish(st)
                st.log("self-improve dry-run: skip Comfy queue, close judge→revise→rejudge")
                st.transition("JUDGE")
                self._judge(st)
                return self._finish(st)

            st.transition("PATCH")
            if not self._patch(st):
                return self._finish(st)
            st.transition("VALIDATE")
            if not self._validate(st):
                return self._finish(st)
            st.transition("SUBMIT")
            if not self._submit_and_poll(st):
                return self._finish(st)
            st.transition("RESOLVE")
            if not self._resolve(st):
                return self._finish(st)
            if not st.revise_enabled:
                self._skip_judge(st)
            else:
                st.transition("JUDGE")
                self._judge(st)
        except Exception as e:
            st.fail(f"unhandled orchestrator error: {e}")
        return self._finish(st)

    def _finish(self, st: RunState) -> RunState:
        if st.state == "ERROR" and st.loop_status in ("", LOOP_STARTED):
            st.loop_status = LOOP_ERROR
        elif st.state == "DONE" and st.loop_status in ("", LOOP_STARTED):
            st.loop_status = LOOP_PASSED if st.judge_decision == "accept" else LOOP_EXHAUSTED
        if not st.provenance:
            persist_clip_provenance(st, revise_notes=latest_revise_notes(st))
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        record = RUNS_DIR / f"{ts}_{st.run_id}.json"
        try:
            record.write_text(
                json.dumps(st.to_dict(), indent=1, default=str) + "\n", encoding="utf-8"
            )
            st.log(f"run record: {record}")
        except OSError as e:
            st.log(f"could not write run record: {e}")
        try:
            from master_agent.kb.ingest import ingest_run_record

            if ingest_run_record(st.to_dict()):
                st.log("kb: run record ingested")
        except Exception:
            log.debug("kb ingest failed for run %s", st.run_id, exc_info=True)
        return st
