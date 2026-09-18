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
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from master_agent.comfy.client import ComfyClient, ComfyClientError
from master_agent.comfy.linter import LintBlocked, hard_gate, lint_workflow
from master_agent.comfy.workflow_patcher import load_and_patch_workflow
from master_agent.config import (
    DOWNSCALE_LADDER,
    JUDGE_ENABLED,
    MAX_JUDGE_ROUNDS,
    MAX_RETRIES,
    OUTPUTS_DIR,
    POWER_MODE,
    POWER_MODE_MAX_OPS,
    POWER_MODE_REPAIR,
    RUNS_DIR,
    SEGMENT_MAX_S,
)
from master_agent.judge.judge import judge_segment
from master_agent.judge.probe import analyze
from master_agent.orchestrator.state import RunState

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
        from master_agent.orchestrator.director import choose_variant

        variant, source = choose_variant(
            st.request or "",
            has_video=bool(st.video_name),
            force=force_variant,
            attach_recipe=st.attach_recipe,
        )
        if source not in ("forced", "input"):
            st.log(f"director routed variant={variant} ({source})")
        return variant

    def _patch(self, st: RunState) -> bool:
        """(Re)build the workflow from current params. Returns success."""
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
                filename_prefix="master_agent",
                stg_scale=st.stg_scale,
                stg_blocks=st.stg_blocks,
                sampler_name=st.sampler_name,
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
        return True

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
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        for info in self._output_files:
            src = ComfyClient.resolve_output_path(info)
            if not src.is_file():
                continue
            dst = OUTPUTS_DIR / f"{st.run_id}_{src.name}"
            try:
                shutil.copy2(src, dst)
                st.video_path = str(dst)
                st.log(f"output: {dst}")
                return True
            except OSError as e:
                st.fail(f"could not copy output {src}: {e}")
                return False
        st.fail(f"output files missing on disk: {self._output_files}")
        return False

    def _judge(self, st: RunState) -> None:
        """JUDGE state. On rewrite/retune, loops back through PATCH."""
        while True:
            heuristic, issues = analyze(
                st.video_path, expected_duration_s=st.duration_s
            )
            result = judge_segment(
                user_request=st.request,
                ltx_prompt=st.prompt,
                shot=st.shot,
                video_path=st.video_path,
                heuristic_score=heuristic,
                heuristic_issues=issues,
                judge_retries=st.judge_round,
                max_rounds=st.max_judge_rounds,
                judge_enabled=st.judge_enabled,
            )
            st.judge_score = result.combined_score
            st.judge_decision = result.decision
            st.judge_issues = result.issues
            st.judge_reason = result.reason
            st.judge_history.append(result.to_dict())
            st.log(
                f"judge round {st.judge_round}: combined={result.combined_score:.2f} "
                f"decision={result.decision} reason={result.reason}"
            )

            if result.decision == "accept" or st.judge_round >= st.max_judge_rounds:
                st.transition("DONE")
                return

            st.judge_round += 1
            if result.decision == "rewrite" and result.prompt_rewrite.strip():
                st.prompt = result.prompt_rewrite.strip()
                st.log("judge requested prompt rewrite")
            elif result.decision == "retune" and result.param_hints:
                self._apply_retune(st, result.param_hints)
                st.log(f"judge retune: {result.param_hints}")
            else:
                # No actionable feedback — accept what we have
                st.transition("DONE")
                return

            st.transition("PATCH")
            if not (self._patch(st) and self._validate(st)):
                return
            if not self._submit_and_poll(st):
                return
            st.transition("RESOLVE")
            if not self._resolve(st):
                return
            st.transition("JUDGE")

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
        max_judge_rounds: int = MAX_JUDGE_ROUNDS,
        power_mode: Optional[bool] = None,
        attach_recipe: Optional[dict[str, Any]] = None,
    ) -> RunState:
        run_id = uuid.uuid4().hex[:12]
        st = RunState(
            request=request,
            run_id=run_id,
            prompt=prompt or request,
            shot=shot,
            duration_s=duration_s,
            quality=quality,
            seed=seed,
            width=width,
            height=height,
            video_name=video_name,
            image_name=image_name,
            audio_name=audio_name,
            judge_enabled=JUDGE_ENABLED if judge_enabled is None else judge_enabled,
            max_judge_rounds=max_judge_rounds,
            power_mode=POWER_MODE if power_mode is None else bool(power_mode),
            attach_recipe=attach_recipe,
        )
        try:
            st.variant = self._select_variant(st, variant)
            st.log(f"variant={st.variant} duration={duration_s}s quality={quality or 'default'}")

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
            st.transition("JUDGE")
            self._judge(st)
        except Exception as e:
            st.fail(f"unhandled orchestrator error: {e}")
        return self._finish(st)

    def _finish(self, st: RunState) -> RunState:
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
            pass
        return st
