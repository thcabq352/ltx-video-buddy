"""Browser dashboard over ``master_agent.agent_api``: type a prompt, get a video.

From ``video_buddy/``::

    pip install -r requirements.txt -r requirements-ui.txt
    streamlit run streamlit_app.py --server.address 127.0.0.1

Every action is one ``agent_api`` call; the dashboard has no render logic of
its own. ``create_video`` blocks for the whole render and returns no job id,
so the dashboard starts a detached OS process (``python -m
master_agent.detached_job``) and polls that process. Closing the terminal
does not kill the job. The helpers above ``main`` import neither Streamlit
nor the GPU stack, so tests run without the UI extra.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from master_agent import agent_api

DURATION_MIN_S = 1.0
DURATION_MAX_S = 20.0
DURATION_DEFAULT_S = 5.0
POLL_INTERVAL_S = 2.0
HEALTH_TTL_S = 30
AUTO = "auto"


def build_create_args(
    prompt: str,
    *,
    duration_s: float,
    quality: str,
    variant: str = "",
    storyboard: str = AUTO,
    upscale: str = "none",
    seed: Optional[int] = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Map form values to ``agent_api.create_video`` keyword arguments.

    Blank ``variant`` and ``storyboard="auto"`` become ``None`` so the director
    picks; ``upscale="none"`` becomes ``None``. A negative seed means random.
    """
    return {
        "request": (prompt or "").strip(),
        "duration_s": float(duration_s),
        "quality": quality,
        "variant": (variant or "").strip() or None,
        "storyboard": None if storyboard == AUTO else storyboard,
        "upscale": None if upscale == "none" else upscale,
        "seed": None if seed is None or seed < 0 else int(seed),
        "dry_run": bool(dry_run),
    }


class _ProcFuture:
    """``done`` / ``result`` / ``exception`` for a detached Buddy process."""

    def __init__(self, proc: Any, result_path: Path):
        self._proc = proc
        self._result_path = result_path
        self._cached: dict[str, Any] | None = None

    def done(self) -> bool:
        return self._proc.poll() is not None

    def _load(self) -> dict[str, Any]:
        if self._cached is not None:
            return self._cached
        if self._result_path.is_file():
            import json

            self._cached = json.loads(self._result_path.read_text(encoding="utf-8"))
        else:
            code = self._proc.poll()
            self._cached = {
                "status": "error",
                "error": f"detached job exited {code} without a result file",
            }
        return self._cached

    def result(self, timeout: float | None = None) -> dict[str, Any]:
        self._proc.wait(timeout=timeout)
        return self._load()

    def exception(self, timeout: float | None = None) -> BaseException | None:
        del timeout
        if not self.done():
            return None
        payload = self._load()
        if payload.get("status") == "error" and payload.get("error"):
            return RuntimeError(str(payload["error"]))
        return None


def _detached_spawn(fn: Callable[..., dict], args: dict[str, Any]) -> _ProcFuture:
    """Production handoff. ``fn`` is the in-process fallback tests replace."""
    del fn
    from master_agent.config import STATE_DIR
    from master_agent.detached_job import write_brief
    from master_agent.process_detach import buddy_python_argv, spawn_detached

    job_dir = Path(STATE_DIR) / "jobs"
    brief = job_dir / f"{uuid.uuid4().hex}.json"
    result = brief.with_suffix(".result.json")
    write_brief(brief, args, result=result)
    proc = spawn_detached(
        buddy_python_argv("-m", "master_agent.detached_job", str(brief)),
        log_path=brief.with_suffix(".log"),
    )
    return _ProcFuture(proc, result)


@dataclass
class Job:
    """One ``create_video`` handoff. The default future is a detached process."""

    id: str
    args: dict[str, Any]
    future: Any
    started_at: float = field(default_factory=time.monotonic)

    def elapsed_s(self) -> float:
        return time.monotonic() - self.started_at


class JobBoard:
    """Runs one render at a time as a detached OS process.

    Shared across browser sessions, so a page refresh finds the job again.
    ``submit`` refuses while a job runs; the GPU lock inside ``create_video``
    would answer ``busy`` anyway. Tests pass ``spawn`` to avoid a real process.
    """

    def __init__(self, spawn: Callable[..., Any] | None = None) -> None:
        self._spawn = spawn or _detached_spawn
        self._jobs: dict[str, Job] = {}
        self._latest: Optional[str] = None

    def submit(self, fn: Callable[..., dict], args: dict[str, Any]) -> Optional[str]:
        """Start the handoff; returns the job id, or ``None`` when a job is still running."""
        if self.running() is not None:
            return None
        job_id = uuid.uuid4().hex[:8]
        self._jobs[job_id] = Job(job_id, dict(args), self._spawn(fn, dict(args)))
        self._latest = job_id
        return job_id

    def get(self, job_id: Optional[str]) -> Optional[Job]:
        return self._jobs.get(job_id) if job_id else None

    def latest(self) -> Optional[Job]:
        return self.get(self._latest)

    def running(self) -> Optional[Job]:
        job = self.latest()
        return job if job is not None and not job.future.done() else None


@dataclass
class Outcome:
    """What the page shows for a finished job."""

    kind: str  # success | warning | paused | busy | planned | error
    message: str
    video: Optional[Path] = None
    result: dict[str, Any] = field(default_factory=dict)


def resolve_video_path(result: dict[str, Any], roots: Iterable[Path] = ()) -> Optional[Path]:
    """First existing file among ``upscaled_path`` then ``video_path``.

    Absolute paths are used as-is; relative ones are tried against each root.
    """
    for key in ("upscaled_path", "video_path"):
        raw = result.get(key)
        if not raw:
            continue
        path = Path(raw)
        candidates = [path] if path.is_absolute() else [Path(root) / path for root in roots] + [path]
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
    return None


def _default_roots() -> list[Path]:
    from master_agent.config import OUTPUTS_DIR, PROJECT_ROOT

    return [OUTPUTS_DIR, PROJECT_ROOT]


def interpret_job(job: Job, roots: Optional[Iterable[Path]] = None) -> Optional[Outcome]:
    """``None`` while the job runs, else the :class:`Outcome` for its result or exception."""
    if not job.future.done():
        return None
    exc = job.future.exception()
    if exc is not None:
        return Outcome("error", f"create_video raised {type(exc).__name__}: {exc}")
    result = job.future.result()
    return interpret_result(result if isinstance(result, dict) else {}, roots)


def interpret_result(result: dict[str, Any], roots: Optional[Iterable[Path]] = None) -> Outcome:
    """Map a ``create_video`` return value to what the page shows.

    ``paused`` is a budget HOLD and ``busy`` is another render on the GPU;
    neither is a failure. A finished render with no file on disk is an error.
    """
    status = str(result.get("status") or "")
    error = result.get("error") or ""
    if status == "paused":
        run = f" Run `{result['run_id']}` keeps its finished segments." if result.get("run_id") else ""
        return Outcome(
            "paused",
            "Paused: this shift's render budget is used up (HOLD). Nothing failed."
            f"{run} It continues when an operator starts a new shift "
            "(`python -m master_agent budget reset-shift`).",
            result=result,
        )
    if status == "busy":
        return Outcome("busy", "Another render is using the GPU. Try again when it finishes.", result=result)
    if status == "dry-run":
        return Outcome("planned", "Plan checked: no GPU was used and no video was made.", result=result)
    if status in ("done", "done_with_warnings"):
        video = resolve_video_path(result, _default_roots() if roots is None else roots)
        if video is None:
            return Outcome("error", "The render finished but its video file was not found.", result=result)
        if status == "done_with_warnings":
            notes = result.get("full_judge_notes") or "the judge did not pass it"
            return Outcome("warning", f"Video made, with warnings: {notes}", video, result)
        return Outcome("success", "Video made.", video, result)
    return Outcome("error", error or f"Unexpected status {status!r}.", result=result)


def summarize_health(health: dict[str, Any]) -> list[tuple[str, bool, str]]:
    """``(label, ok, text)`` rows for the header: ComfyUI, local LLM, budget."""
    if health.get("error") and "comfyui" not in health:
        return [("health", False, str(health["error"]))]
    rows: list[tuple[str, bool, str]] = []
    comfy = health.get("comfyui") or {}
    if comfy.get("up"):
        gpus = ", ".join(
            f"{g.get('name')} ({g.get('vram_free_gb')}/{g.get('vram_total_gb')} GB free)"
            for g in comfy.get("gpus") or []
        )
        rows.append(("ComfyUI", True, f"up{': ' + gpus if gpus else ''}"))
    else:
        rows.append(("ComfyUI", False, f"down: {comfy.get('error') or 'not reachable'}"))
    llm = health.get("local_llm") or {}
    active = llm.get("active")
    rows.append(("LLM", bool(active), f"active: {active}" if active else "no local LLM answering"))
    budget = health.get("budget") or {}
    if budget:
        rows.append(("Budget", not budget.get("paused"), budget.get("line") or ""))
    return rows


def _call_safely(fn: Callable[[], dict]) -> dict:
    try:
        return fn()
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _health() -> dict:
    return _call_safely(agent_api.health)


def main() -> None:
    """Render the page. Streamlit runs this file top to bottom on every interaction."""
    import streamlit as st

    st.set_page_config(page_title="VIDEO BUDDY", page_icon="🎬", layout="centered")
    board = st.cache_resource(JobBoard)()
    cached_health = st.cache_data(ttl=HEALTH_TTL_S, show_spinner="Checking health...")(_health)

    st.title("VIDEO BUDDY")
    health = cached_health()
    for label, ok, text in summarize_health(health):
        (st.success if ok else st.warning)(f"**{label}** {text}")
    left, right = st.columns(2)
    if left.button("Refresh health"):
        cached_health.clear()
        st.rerun()
    if not (health.get("comfyui") or {}).get("up") and right.button("Start ComfyUI"):
        with st.spinner("Starting ComfyUI (up to 3 minutes)..."):
            started = _call_safely(agent_api.comfy_start)
        if started.get("ready"):
            cached_health.clear()
            st.rerun()
        message = started.get("error") or started.get("next") or f"ComfyUI status: {started.get('status')}"
        (st.info if started.get("status") == "starting" else st.error)(message)

    running = board.running()
    with st.form("generate"):
        prompt = st.text_area("Prompt", placeholder="cinematic close-up of rain on a window")
        duration_s = st.slider("Duration (seconds)", DURATION_MIN_S, DURATION_MAX_S, DURATION_DEFAULT_S, 0.5)
        quality = st.select_slider("Quality", options=list(agent_api.QUALITIES), value="draft")
        with st.expander("Advanced"):
            variant = st.text_input("Model variant", help="Catalog id such as ltx25_t2v_i2v. Blank lets the director pick.")
            storyboard = st.selectbox("Storyboard", [AUTO, *[m for m in agent_api.STORYBOARD_MODES if m]])
            upscale = st.selectbox("Upscale", ["none", *[m for m in agent_api.UPSCALE_METHODS if m]])
            seed = st.number_input("Seed (-1 = random)", min_value=-1, value=-1, step=1)
            dry_run = st.checkbox("Plan only (no GPU)")
        submitted = st.form_submit_button("Generate", disabled=running is not None, type="primary")
    if submitted:
        args = build_create_args(
            prompt,
            duration_s=duration_s,
            quality=quality,
            variant=variant,
            storyboard=storyboard,
            upscale=upscale,
            seed=int(seed),
            dry_run=dry_run,
        )
        if not args["request"]:
            st.error("Type a prompt first.")
        else:
            st.session_state["job_id"] = board.submit(agent_api.create_video, args)
            st.rerun()

    job = board.get(st.session_state.get("job_id")) or board.latest()
    if job is None:
        return
    outcome = interpret_job(job)
    if outcome is None:
        with st.spinner(f"Rendering \"{job.args['request'][:60]}\"... {job.elapsed_s():.0f} s"):
            time.sleep(POLL_INTERVAL_S)
        st.rerun()
    show = {
        "success": st.success,
        "warning": st.warning,
        "paused": st.info,
        "busy": st.warning,
        "planned": st.info,
        "error": st.error,
    }[outcome.kind]
    show(outcome.message)
    if outcome.video is not None:
        st.video(str(outcome.video))
        st.caption(str(outcome.video))
    if outcome.result:
        with st.expander("Details"):
            st.json(outcome.result)


if __name__ == "__main__":
    main()
