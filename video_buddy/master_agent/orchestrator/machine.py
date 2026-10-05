"""Orchestrator — the Director. State machine over the ComfyUI bridge.

Flow: SELECT_VARIANT → PATCH → VALIDATE → SUBMIT → POLL → RESOLVE → JUDGE
      → DONE | ERROR   (with PLAN_OOM_RETRY and judge rewrite/retune loops)

The class coordinates. The judge revise cycle lives in ``judge_loop`` and the
OOM downscale ladder lives in ``oom``.

Design rules:
- The orchestrator executes; the judge (master_agent.judge) only evaluates.
- Nothing is queued before the validator passes.
- OOM walks config.DOWNSCALE_LADDER instead of dying.
- Every run leaves a JSON record in state/runs/ (seed of the knowledge base).
"""

from __future__ import annotations

import json
import logging
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from master_agent.comfy.client import ComfyClient, ComfyClientError
from master_agent.judge.probe import analyze
from master_agent.comfy.linter import LintBlocked, hard_gate, lint_workflow
from master_agent.comfy.workflow_patcher import load_and_patch_workflow, media_wiring_error
from master_agent.config import (
    JUDGE_ENABLED,
    MAX_JUDGE_ROUNDS,
    POWER_MODE,
    POWER_MODE_MAX_OPS,
    POWER_MODE_REPAIR,
    RUNS_DIR,
    SEGMENT_MAX_S,
)
from master_agent.orchestrator.state import (
    LOOP_ERROR,
    LOOP_EXHAUSTED,
    LOOP_HUMAN_VETO,
    LOOP_PASSED,
    LOOP_STARTED,
    RunState,
)
from master_agent.orchestrator.talking import enforce_audio_duration
from master_agent.provenance import (
    latest_revise_notes,
    persist_clip_provenance,
    plan_clip_paths,
    planned_clip_path,
)

log = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, client: Optional[ComfyClient] = None):
        self.client = client or ComfyClient()

    def _select_variant(self, st: RunState, force_variant: Optional[str]) -> str:
        from master_agent.comfy.partner_pointers import PartnerPointerError, route_pack_c
        from master_agent.hands import LiveHands
        from master_agent.orchestrator.director import choose_variant

        force_variant, refusal = route_pack_c(st.request or "", force_variant)
        if refusal:
            raise PartnerPointerError(refusal)

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
        enforce_audio_duration(st)
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
                mask_name=st.mask_name,
                audio_name=st.audio_name,
                video_name=st.video_name,
                audio_start_s=st.audio_start_s,
                inoutpaint=st.inoutpaint,
                filename_prefix="master_agent",
                stg_scale=st.stg_scale,
                stg_blocks=st.stg_blocks,
                sampler_name=st.sampler_name,
                spoken_line=st.spoken_line or None,
                frames=st.frames,
                i2v_strength=st.i2v_strength,
            )
        except Exception as e:
            st.fail(f"patch failed: {e}")
            return False
        if (st.variant or "") in ("ltx23_inoutpaint", "ltx25_inoutpaint"):
            st.width = int(meta.get("width") or st.width)
            st.height = int(meta.get("height") or st.height)
            if meta.get("inoutpaint_default_length") and st.frames is None:
                st.frames = int(meta["frames"])
                st.duration_s = float(meta["duration_s"])
                st.fps = 24
        if st.end_guide_image:
            from master_agent.orchestrator.lipdub_guide import apply_last_frame_guide

            object_info = None
            try:
                object_info, _src = self.client.load_object_info(prefer_live=True)
            except Exception:
                object_info = None
            if not isinstance(object_info, dict):
                from master_agent.config import OBJECT_INFO_CACHE

                if OBJECT_INFO_CACHE.is_file():
                    object_info = json.loads(OBJECT_INFO_CACHE.read_text(encoding="utf-8"))
            workflow, guide_patch = apply_last_frame_guide(
                workflow,
                image_name=st.end_guide_image,
                object_info=object_info,
                frame_idx=-1 if st.end_guide_frame_idx is None else int(st.end_guide_frame_idx),
                strength=1.0 if st.end_guide_strength is None else float(st.end_guide_strength),
            )
            meta = dict(meta)
            meta["end_guide"] = guide_patch.status
            st.log(guide_patch.message)
            if not guide_patch.applied:
                st.fail(
                    "pause-reset last-frame guide was not patched "
                    f"({guide_patch.status}). {guide_patch.message}"
                )
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
        if not st.dry_run:
            self._ensure_fun_inpaint_mask(st)
        return True

    def _ensure_fun_inpaint_mask(self, st: RunState) -> None:
        """Upload fun_inpaint_mask.png before live /object_info validation."""
        from master_agent.comfy.fun_inpaint import ensure_fun_inpaint_mask

        try:
            uploaded = ensure_fun_inpaint_mask(self._workflow, self.client.upload_image)
        except Exception as exc:
            st.log(f"warn: fun inpaint mask upload failed: {exc}")
            return
        if uploaded:
            st.log(f"fun inpaint mask uploaded: {uploaded}")
        from master_agent.comfy.inoutpaint import SourceVideoTrimError, prepare_queue_inputs

        try:
            uploaded_io = prepare_queue_inputs(self._workflow, self.client.upload_image)
        except SourceVideoTrimError:
            raise
        except Exception as exc:
            st.log(f"warn: ltx in/outpaint mask upload failed: {exc}")
            return
        if uploaded_io:
            st.log(f"ltx in/outpaint mask uploaded: {uploaded_io}")

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
        self._output_files = self._order_final_output(st, files)
        return True

    def _handle_job_error(self, st: RunState, exc: Exception, *, phase: str) -> bool:
        """OOM downscale ladder. See orchestrator.oom."""
        from master_agent.orchestrator.oom import handle_job_error

        return handle_job_error(self, st, exc, phase=phase)

    def _order_final_output(self, st: RunState, files: list[dict]) -> list[dict]:
        """Prefer the manifest's final node, else the clip with audio at this length."""
        from master_agent.comfy.workflow_patcher import variant_final_node_id

        def _probe(info: dict) -> dict:
            try:
                from master_agent.judge.probe import probe_video

                return probe_video(ComfyClient.resolve_output_path(info))
            except Exception:
                return {}

        return ComfyClient.choose_final_output(
            files,
            final_node_id=variant_final_node_id(st.variant or ""),
            duration_s=st.duration_s,
            frames=st.frames,
            probe=_probe,
        )

    def _resolve(self, st: RunState) -> bool:
        dst = planned_clip_path(st)
        dst.parent.mkdir(parents=True, exist_ok=True)
        ordered = self._order_final_output(st, self._output_files)
        for info in ordered:
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
        """JUDGE state. See orchestrator.judge_loop."""
        from master_agent.orchestrator.judge_loop import judge_loop

        judge_loop(self, st)

    @staticmethod
    def _revise_plan(st: RunState, result):
        from master_agent.orchestrator.judge_loop import revise_plan

        return revise_plan(st, result)

    def _apply_full_revise(self, st: RunState, plan, result) -> None:
        from master_agent.orchestrator.judge_loop import apply_full_revise

        apply_full_revise(self, st, plan, result)

    @staticmethod
    def _apply_retune(st: RunState, hints: dict[str, Any]) -> None:
        from master_agent.orchestrator.judge_loop import apply_retune

        apply_retune(st, hints)

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
        steps: Optional[int] = None,
        cfg: Optional[float] = None,
        width: int = 768,
        height: int = 512,
        video_name: Optional[str] = None,
        image_name: Optional[str] = None,
        mask_name: Optional[str] = None,
        audio_name: Optional[str] = None,
        judge_enabled: Optional[bool] = None,
        revise_enabled: Optional[bool] = None,
        spoken_line: Optional[str] = None,
        voice_sample: Optional[dict[str, Any]] = None,
        heartmula: Optional[dict[str, Any]] = None,
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
        negative_prompt: Optional[str] = None,
        frames: Optional[int] = None,
        i2v_strength: Optional[float] = None,
        end_guide_image: Optional[str] = None,
        end_guide_frame_idx: Optional[int] = None,
        end_guide_strength: Optional[float] = None,
        inoutpaint: Optional[dict[str, Any]] = None,
        duration_cap_s: Optional[float] = None,
        max_piece_s: Optional[float] = None,
        downscale_level: int = 0,
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
            steps=steps,
            cfg=cfg,
            width=width,
            height=height,
            video_name=video_name,
            image_name=image_name,
            mask_name=mask_name,
            audio_name=audio_name,
            audio_path=audio_path,
            audio_start_s=float(audio_start_s or 0.0),
            kind=kind,
            music_bed_attached=music_bed_attached,
            judge_enabled=JUDGE_ENABLED if judge_enabled is None else judge_enabled,
            revise_enabled=True if revise_enabled is None else bool(revise_enabled),
            spoken_line=(spoken_line or "").strip(),
            voice_sample=dict(voice_sample or {}),
            heartmula=dict(heartmula or {}),
            max_judge_rounds=max_judge_rounds,
            power_mode=POWER_MODE if power_mode is None else bool(power_mode),
            attach_recipe=attach_recipe,
            dry_run=bool(dry_run),
            attempt=1,
            loop_status=LOOP_STARTED,
            control_pack_present=control_pack_present,
            control_pack_used=dict(control_pack_used or {}),
            previs_source=previs_source,
            negative_prompt=negative_prompt or "",
            frames=int(frames) if frames is not None else None,
            i2v_strength=float(i2v_strength) if i2v_strength is not None else None,
            end_guide_image=end_guide_image or None,
            end_guide_frame_idx=(
                int(end_guide_frame_idx) if end_guide_frame_idx is not None else None
            ),
            end_guide_strength=(
                float(end_guide_strength) if end_guide_strength is not None else None
            ),
            inoutpaint=inoutpaint,
            duration_cap_s=(
                float(duration_cap_s) if duration_cap_s is not None else None
            ),
            max_piece_s=float(max_piece_s) if max_piece_s is not None else None,
        )
        st.downscale_level = max(0, int(downscale_level))
        from master_agent.comfy.partner_pointers import PartnerPointerError

        try:
            st.variant = self._select_variant(st, variant)
        except PartnerPointerError as exc:
            st.fail(str(exc))
            return st
        try:
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
            enforce_audio_duration(st)
            if st.duration_s != duration_s:
                st.log(
                    f"audio cap: duration {duration_s}s -> {st.duration_s}s "
                    f"frames={st.frames}"
                )
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

