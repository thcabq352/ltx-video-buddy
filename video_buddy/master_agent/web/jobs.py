"""Job runner for the web UI — threads + in-memory registry.

Single-workstation model: one GPU-bound `run` job at a time; `dry-run`
jobs may overlap. stdout/stderr of each worker is teed into the job log.
"""

from __future__ import annotations

import io
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for st in self._streams:
            try:
                st.write(s)
            except Exception:
                pass
        return len(s)

    def flush(self):
        for st in self._streams:
            try:
                st.flush()
            except Exception:
                pass


@dataclass
class Job:
    id: str
    kind: str  # run | dry-run | fractal | music | comfy-run
    request: str
    params: dict[str, Any]
    status: str = "queued"  # queued | running | done | error
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None
    result: Optional[dict[str, Any]] = None
    _log: io.StringIO = field(default_factory=io.StringIO, repr=False)

    def log_tail(self, n: int = 200) -> list[str]:
        lines = self._log.getvalue().splitlines()
        return lines[-n:]

    def to_dict(self, *, with_log: bool = False) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
            "request": self.request,
            "params": {k: v for k, v in self.params.items() if v is not None},
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "result": self.result,
        }
        if with_log:
            d["log"] = self.log_tail()
        return d


class JobManager:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._run_gate = threading.Lock()  # one GPU run at a time

    def submit(self, kind: str, request: str, **params) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, request=request, params=params)
        with self._lock:
            self._jobs[job.id] = job
            # bound registry size
            if len(self._jobs) > 100:
                for old in sorted(self._jobs.values(), key=lambda j: j.created_at)[:20]:
                    if old.status in ("done", "error"):
                        self._jobs.pop(old.id, None)
        threading.Thread(target=self._work, args=(job,), daemon=True).start()
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def list(self, limit: int = 50) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: -j.created_at)[:limit]

    def gpu_lock(self) -> threading.Lock:
        """One-GPU-at-a-time lock shared with A2A / Hermes facade submits."""
        return self._run_gate

    # ── workers ───────────────────────────────────────────

    def _work(self, job: Job) -> None:
        tee = _Tee(job._log, sys.stdout)
        needs_gpu = job.kind in ("run", "comfy-run") or (
            job.kind == "music" and job.params.get("visual") != "fractal"
        )
        if needs_gpu:
            with self._run_gate:
                self._execute(job, tee)
        else:
            self._execute(job, tee)

    def _execute(self, job: Job, tee: _Tee) -> None:
        job.status = "running"
        job.started_at = time.time()
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = tee
        try:
            if job.kind == "run":
                self._do_run(job)
            elif job.kind == "dry-run":
                self._do_dry_run(job)
            elif job.kind == "fractal":
                self._do_fractal(job)
            elif job.kind == "music":
                self._do_music(job)
            elif job.kind == "comfy-run":
                self._do_comfy_run(job)
            else:
                raise ValueError(f"unknown job kind {job.kind!r}")
            if job.status == "paused":
                pass
            else:
                job.status = "done" if job.error is None else "error"
        except Exception as e:
            job.status = "error"
            job.error = f"{type(e).__name__}: {e}"
            print(f"job error: {job.error}")
        finally:
            sys.stdout, sys.stderr = old_out, old_err
            job.finished_at = time.time()

    def _guard_media(self, job: Job) -> bool:
        """Fail fast when the chosen graph cannot consume the attached files."""
        from master_agent.orchestrator.talking import (
            h3_r2v_audio_warning,
            media_route_error,
            preview_media_variant,
        )

        p = job.params
        preview = preview_media_variant(
            job.request,
            variant=p.get("variant"),
            has_image=bool(p.get("image_path")),
            has_audio=bool(p.get("audio_path")),
            has_video=bool(p.get("video_path")),
        )
        err = media_route_error(
            preview,
            has_image=bool(p.get("image_path")),
            has_audio=bool(p.get("audio_path")),
            has_video=bool(p.get("video_path")),
        )
        if err:
            print(err)
            job.error = err
            return False
        note = p.get("audio_duration_note")
        if note:
            print(f"warn: {note}")
        if p.get("image_path") and p.get("audio_path") and not p.get("video_path"):
            print(f"route: photo + voice → {preview}")
            voice_warn = h3_r2v_audio_warning(
                job.request,
                variant=p.get("variant"),
                has_image=True,
                has_audio=True,
                has_video=False,
            )
            if voice_warn:
                print(f"warn: {voice_warn}")
            from master_agent.orchestrator.h3_voice import (
                h3_missing_line_warning,
                resolve_spoken_line,
            )
            from master_agent.orchestrator.talking import is_h3_voice_route

            if is_h3_voice_route(
                job.request,
                variant=p.get("variant"),
                has_image=True,
                has_audio=True,
                has_video=False,
            ):
                spoken = resolve_spoken_line(p.get("line"), job.request, None)
                p["spoken_line"] = spoken or None
                missing = h3_missing_line_warning(spoken, h3_voice=True)
                if missing:
                    print(f"warn: {missing}")
        return True

    def _do_run(self, job: Job) -> None:
        from master_agent.comfy.client import ComfyClient
        from master_agent.orchestrator.pipeline import run_pipeline

        p = job.params
        if not self._guard_media(job):
            return
        audio_path = p.get("audio_path")
        if audio_path:
            from master_agent.orchestrator.h3_voice import VoiceSampleError, h3_voice_preflight
            from master_agent.orchestrator.talking import is_h3_voice_route

            if is_h3_voice_route(
                job.request,
                variant=p.get("variant"),
                has_image=bool(p.get("image_path")),
                has_audio=True,
                has_video=bool(p.get("video_path")),
            ):
                try:
                    pre = h3_voice_preflight(
                        request=job.request,
                        variant=p.get("variant"),
                        audio_path=audio_path,
                        has_image=bool(p.get("image_path")),
                        has_video=bool(p.get("video_path")),
                        line=p.get("line") or p.get("spoken_line"),
                    )
                except VoiceSampleError as exc:
                    print(f"FAIL  {exc}")
                    job.error = str(exc)
                    return
                audio_path = pre.audio_path or audio_path
                p["audio_path"] = audio_path
                p["voice_sample"] = pre.voice_sample
                p["spoken_line"] = pre.spoken_line or None
                if pre.trimmed_note:
                    print(f"warn: {pre.trimmed_note}")
        client = ComfyClient()
        image_name = audio_name = video_name = None
        if p.get("image_path"):
            image_name = client.upload_image(Path(p["image_path"]))
            print(f"uploaded image -> ComfyUI: {image_name}")
        if audio_path:
            audio_name = client.upload_audio(Path(audio_path))
            print(f"uploaded audio -> ComfyUI: {audio_name}")
        if p.get("video_path"):
            # Comfy treats source video like an image upload into input/
            video_name = client.upload_image(Path(p["video_path"]))
            print(f"uploaded video -> ComfyUI: {video_name}")

        result = run_pipeline(
            job.request,
            variant=p.get("variant"),
            duration_s=float(p.get("duration_s") or 5.0),
            quality=p.get("quality"),
            seed=p.get("seed"),
            video_name=video_name,
            image_name=image_name,
            audio_name=audio_name,
            audio_path=p.get("audio_path"),
            spoken_line=p.get("spoken_line") or p.get("line"),
            voice_sample=p.get("voice_sample"),
            storyboard_mode=p.get("storyboard"),
            llm_panel=p.get("llm_panel"),
            client=client,
        )
        d = result.to_dict()
        d.pop("messages", None)
        video = d.get("video_path")
        if p.get("upscale") and video and result.status != "error":
            from master_agent.upscale import upscale_video

            print(f"upscaling via {p['upscale']}…")
            up_path = str(upscale_video(video, method=p["upscale"], run_id=result.run_id))
            d["upscaled_path"] = up_path
            video = up_path
        d["video_url"] = _outputs_url(video)
        if image_name or audio_name or video_name:
            d["media"] = {
                "image_name": image_name,
                "audio_name": audio_name,
                "video_name": video_name,
            }
        job.result = d
        if result.status == "error":
            job.error = result.error
        elif result.status == "paused":
            job.status = "paused"
            job.error = result.error

    def _do_dry_run(self, job: Job) -> None:
        from master_agent.orchestrator.pipeline import _plan_storyboard, plan_story_segments
        from master_agent.storyboard.storyboard import storyboard_to_markdown

        if not self._guard_media(job):
            return
        quality = job.params.get("quality")
        segs = plan_story_segments(
            float(job.params.get("duration_s") or 8.0),
            kind=job.kind,
            quality=quality,
        )
        logs: list[str] = []
        cards, style, meta = _plan_storyboard(
            job.request,
            segs,
            variant=job.params.get("variant"),
            quality=quality,
            llm_panel=job.params.get("llm_panel"),
            panel_judge=None,
            log=lambda m: (logs.append(m), print(f"[panel] {m}")),
        )
        print(storyboard_to_markdown(cards, global_style=style))
        job.result = {
            "segments": segs,
            "global_style": style,
            "shots": [c.to_dict() for c in cards],
            "panel_meta": meta,
            "log": logs,
        }

    def _do_fractal(self, job: Job) -> None:
        from master_agent.fractal.pipeline import run_fractal

        p = job.params
        rec = run_fractal(
            job.request,
            mode=p.get("mode") or "zoom",
            duration_s=float(p.get("duration_s") or 20.0),
            fps=int(p.get("fps") or 24),
            width=int(p.get("width") or 768),
            height=int(p.get("height") or 512),
            target=p.get("target") or "seahorse",
            palette=p.get("palette") or "fire",
            seed=p.get("seed"),
            julia=bool(p.get("julia")),
            audio_path=p.get("audio_path"),
            image_path=p.get("image_path"),
            mask_path=p.get("mask_path"),
            expand=int(p.get("expand") or 128),
            cover=float(p.get("cover") if p.get("cover") is not None else 0.4),
            feather=int(p.get("feather") or 28),
        )
        if p.get("upscale"):
            from master_agent.upscale import upscale_video

            rec["upscaled_path"] = str(
                upscale_video(rec["video_path"], method=p["upscale"], run_id=rec["run_id"])
            )
        rec["video_url"] = _outputs_url(rec.get("upscaled_path") or rec.get("video_path"))
        job.result = rec

    def _do_music(self, job: Job) -> None:
        from master_agent.music.pipeline import run_music_video

        p = job.params
        rec = run_music_video(
            job.request,
            p["audio_path"],
            visual=p.get("visual") or "shots",
            quality=p.get("quality"),
            seed=p.get("seed"),
            upscale=p.get("upscale"),
        )
        rec["video_url"] = _outputs_url(rec.get("upscaled_path") or rec.get("video_path"))
        rec["shots"] = rec.get("storyboard")  # reuse the storyboard renderer
        job.result = rec
        if rec.get("status") == "error":
            job.error = rec.get("error")

    def _do_comfy_run(self, job: Job) -> None:
        from master_agent.comfy.cli_run import execute_prepared

        workflow = job.params.get("workflow")
        if not isinstance(workflow, dict):
            raise ValueError("comfy-run requires a workflow dict")
        rec = execute_prepared(workflow, run_id=job.id)
        rec["video_url"] = _outputs_url(rec.get("video_path"))
        job.result = rec


def _outputs_url(video_path) -> Optional[str]:
    from master_agent.config import OUTPUTS_DIR

    if not video_path:
        return None
    try:
        rel = Path(video_path).resolve().relative_to(OUTPUTS_DIR.resolve())
    except (ValueError, OSError):
        return None
    return f"/videos/{rel.as_posix()}"


MANAGER = JobManager()
