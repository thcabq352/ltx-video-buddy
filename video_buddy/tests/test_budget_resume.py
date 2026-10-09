"""Reset used continues a generate that paused on the render budget.

No GPU. Records stay in a temp runs dir.

Run: python -m pytest tests/test_budget_resume.py -q
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from master_agent.control.budget import RenderBudget
from master_agent.orchestrator.state import RunState
from master_agent.web.jobs import Job, JobManager


class _Orch:
    calls: list[dict] = []

    def __init__(self, client=None):
        self.client = self

    def free_memory(self):
        return None

    def upload_image(self, path):
        return "frame.png"

    def run(self, request, **kwargs):
        _Orch.calls.append({"request": request, **kwargs})
        state = RunState(request=request)
        state.state = "DONE"
        state.video_path = "/tmp/budget-resume-clip.mp4"
        state.judge_score = 0.9
        state.judge_decision = "accept"
        state.loop_status = "passed"
        return state


@pytest.fixture
def budget_box(tmp_path, monkeypatch):
    import master_agent.control.budget as budget_mod
    from master_agent.orchestrator import pipeline as pipeline_mod

    saved = budget_mod._BUDGET
    store = RenderBudget(tmp_path / "budget.json", cap=80, used=0)
    budget_mod._BUDGET = store
    monkeypatch.setattr(pipeline_mod, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(pipeline_mod, "Orchestrator", _Orch)
    monkeypatch.setattr(pipeline_mod, "_stitch", lambda result, paths, suffix="": Path(paths[-1]))
    _Orch.calls = []
    try:
        yield store
    finally:
        budget_mod._BUDGET = saved
        _Orch.calls = []


def test_single_clip_hold_resumes_after_reset_used(budget_box):
    from master_agent.orchestrator.pipeline import resume_after_budget_clear, run_pipeline

    budget_box.used = 80
    budget_box.paused = True
    budget_box.persist()
    held = run_pipeline(
        "rain on a window",
        variant="base",
        duration_s=5,
        judge_enabled=False,
        storyboard_mode="off",
    )
    assert held.status == "paused"
    assert held.resume and held.resume["start_index"] == 0
    assert held.resume["run_id"] == held.run_id
    assert _Orch.calls == []

    budget_box.reset_used()
    continued = resume_after_budget_clear(background=False)
    assert held.run_id in continued
    assert len(_Orch.calls) == 1


def test_multi_segment_resumes_from_held_shot(budget_box, monkeypatch):
    from master_agent.orchestrator import pipeline as pipeline_mod

    def _boom(*_args, **_kwargs):
        raise AssertionError("storyboard should be restored")

    held = pipeline_mod.run_pipeline(
        "rain on a window",
        variant="base",
        duration_s=16,
        judge_enabled=False,
        storyboard_mode="off",
    )
    assert held.status == "paused"
    assert held.resume["start_index"] == 1
    assert len(held.segment_paths) == 1
    assert len(_Orch.calls) == 1

    record = held.to_dict()
    record["resume"]["storyboard"][1]["ltx_prompt"] = "RESTORED SHOT"
    monkeypatch.setattr(pipeline_mod, "_plan_storyboard", _boom)
    budget_box.reset_used()
    done = pipeline_mod.continue_budget_paused(record)
    assert done.status == "done"
    assert len(_Orch.calls) == 2
    assert _Orch.calls[1]["prompt"] == "RESTORED SHOT"


def test_reset_shift_clears_pending_and_cli_resumes(budget_box, monkeypatch, capsys):
    from master_agent.__main__ import cmd_budget

    budget_box.pending = [{"id": "held", "vram_min": 3}]
    seen: dict = {}

    def _resume(*, client=None, background=None):
        seen["background"] = background
        seen["client"] = client
        return ["run-from-cli"]

    monkeypatch.setattr(
        "master_agent.orchestrator.pipeline.resume_after_budget_clear",
        _resume,
    )
    rc = cmd_budget(Namespace(budget_command="reset-shift", json=False))
    assert rc == 0
    assert budget_box.pending == []
    assert budget_box.used == 0
    assert seen["background"] is False
    assert "continuing paused generate: run-from-cli" in capsys.readouterr().out


def test_job_manager_resume_requeues_same_job(monkeypatch):
    started: list = []

    class _Thread:
        def __init__(self, target=None, args=(), daemon=None):
            self.target = target
            self.args = args

        def start(self):
            started.append(self.args)

    monkeypatch.setattr("master_agent.web.jobs.threading.Thread", _Thread)
    mgr = JobManager()
    job = Job(
        id="job1",
        kind="run",
        request="rain on a window",
        params={"variant": "base", "duration_s": 5},
        status="paused",
    )
    job.result = {"status": "paused", "resume": {"run_id": "pipe1", "request": "rain on a window"}}
    job.error = "render budget paused"
    mgr._jobs[job.id] = job
    assert mgr.resume_paused() == ["pipe1"]
    assert job.status == "queued"
    assert job.error is None
    assert job.params["_resume"]["run_id"] == "pipe1"
    assert started and started[0][0] is job


def test_resume_works_without_the_web_package(budget_box, monkeypatch):
    import builtins

    from master_agent.orchestrator import pipeline as pipeline_mod

    real_import = builtins.__import__

    def no_web(name, *args, **kwargs):
        if name.startswith("master_agent.web"):
            raise ImportError("web package removed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_web)
    assert pipeline_mod.resume_after_budget_clear(background=False) == []
