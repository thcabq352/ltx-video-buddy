"""Importable agent API + `python -m master_agent agent` JSON front end. No GPU.

Run: python -m pytest tests/test_agent_api.py -q
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from master_agent import agent_api
from master_agent.cli.parser import main

VIDEO_BUDDY = Path(__file__).resolve().parents[1]


def _json_stdout(capsys) -> object:
    return json.loads(capsys.readouterr().out)


def test_tool_registry_covers_every_capability():
    expected = {
        "about",
        "health",
        "create_video",
        "plan_storyboard",
        "judge_asset",
        "search_workflows",
        "search_runs",
        "search_knowledge",
        "kb_ingest",
        "list_runs",
        "list_models",
        "validate_workflow",
        "create_character",
        "train_lora",
        "control_get",
        "control_set",
        "budget_status",
        "budget_reset_shift",
    }
    assert set(agent_api.TOOLS) == expected
    for fn in agent_api.TOOLS.values():
        assert (fn.__doc__ or "").strip()


def test_agent_list_prints_only_json(capsys):
    assert main(["agent", "list"]) == 0
    rows = _json_stdout(capsys)
    names = {row["tool"] for row in rows}
    assert "create_video" in names and "control_set" in names


def test_agent_unknown_tool_and_bad_args(capsys):
    assert main(["agent", "nope"]) == 2
    assert "unknown tool" in _json_stdout(capsys)["error"]
    assert main(["agent", "budget_status", "--args", "[1]"]) == 2
    assert "bad --args" in _json_stdout(capsys)["error"]
    assert main(["agent", "budget_status", "--args", '{"bogus": 1}']) == 2
    assert "budget_status" in _json_stdout(capsys)["error"]


def test_agent_args_from_file_and_logs_stay_off_stdout(capsys, tmp_path, monkeypatch):
    def noisy(request: str) -> dict:
        print("pipeline chatter")
        return {"status": "done", "echo": request}

    monkeypatch.setitem(agent_api.TOOLS, "noisy", noisy)
    args = tmp_path / "args.json"
    args.write_text(json.dumps({"request": "a fox"}), encoding="utf-8")
    assert main(["agent", "noisy", "--args", f"@{args}"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"status": "done", "echo": "a fox"}
    assert "pipeline chatter" in captured.err


def test_error_status_exits_nonzero(capsys):
    assert main(["agent", "create_video", "--args", '{"request": "  "}']) == 1
    assert _json_stdout(capsys)["status"] == "error"


def test_control_set_round_trip_and_budget_reset(monkeypatch):
    monkeypatch.setattr(
        "master_agent.orchestrator.pipeline.resume_after_budget_clear",
        lambda background=None, client=None: [],
    )
    before = agent_api.control_get()
    out = agent_api.control_set(judge_score_threshold=0.61, session="test")
    assert out["judge_score_threshold"] == pytest.approx(0.61)
    assert out["hash"] != before["hash"]
    reset = agent_api.control_set(reset_budget=True)
    assert reset["render_budget_used_vram_min"] == 0
    shift = agent_api.budget_reset_shift()
    assert shift["budget"]["used"] == 0
    assert shift["reset"]["event"] == "shift_reset"
    assert agent_api.budget_status()["shift_id"] == shift["budget"]["shift_id"]


def test_list_runs_reads_newest_records(tmp_path, monkeypatch):
    import master_agent.config as cfg

    monkeypatch.setattr(cfg, "RUNS_DIR", tmp_path)
    (tmp_path / "20260101_a.json").write_text(
        json.dumps({"request": "old", "status": "done", "segment_paths": []}), encoding="utf-8"
    )
    (tmp_path / "20260102_b.json").write_text(
        json.dumps({"request": "new", "full_judge_score": 0.8, "full_judge_pass": True}),
        encoding="utf-8",
    )
    (tmp_path / "broken.json").write_text("{", encoding="utf-8")
    import os

    os.utime(tmp_path / "20260101_a.json", (1, 1))
    rows = agent_api.list_runs(limit=5)
    assert [r["request"] for r in rows] == ["new", "old"]
    assert rows[0]["passed"] is True
    assert rows[1]["kind"] == "pipeline"


def test_create_video_validates_before_gpu():
    assert "quality" in agent_api.create_video("x", quality="ultra")["error"]
    assert "upscale" in agent_api.create_video("x", upscale="magic")["error"]
    assert "storyboard" in agent_api.create_video("x", storyboard="maybe")["error"]


def test_create_video_returns_busy_when_gpu_held(monkeypatch, tmp_path):
    from master_agent.fileutil import file_lock

    monkeypatch.setattr(agent_api, "_gpu_lock_path", lambda: tmp_path / "gpu")
    rendered = []
    monkeypatch.setattr(agent_api, "_render", lambda *a, **k: rendered.append(1) or {"status": "done"})
    child = (
        "from pathlib import Path; from master_agent import agent_api as api; "
        f"api._gpu_lock_path = lambda: Path({str(tmp_path)!r}) / 'gpu'; "
        "api._render = lambda *a, **k: {'status': 'done'}; "
        "print(api.create_video('a fox runs')['status'])"
    )
    with file_lock(tmp_path / "gpu"):
        proc = subprocess.run(
            [sys.executable, "-c", child],
            cwd=str(VIDEO_BUDDY),
            capture_output=True,
            text=True,
            timeout=120,
        )
    assert proc.stdout.strip().splitlines()[-1] == "busy", proc.stderr
    assert agent_api.create_video("a fox runs")["status"] == "done"
    assert rendered == [1]


def test_create_video_forwards_seed_storyboard_and_upscale(monkeypatch, tmp_path):
    monkeypatch.setattr(agent_api, "_gpu_lock_path", lambda: tmp_path / "gpu")
    captured: dict = {}

    class _Result:
        run_id = "r1"
        status = "done"
        video_path = str(tmp_path / "out.mp4")
        segment_paths: list = []
        segment_scores: list = []
        full_judge_score = 0.9
        full_judge_pass = True
        full_judge_notes = ""
        storyboard: list = []
        panel_meta: dict = {}
        error = None

    monkeypatch.setattr(
        "master_agent.orchestrator.pipeline.run_pipeline",
        lambda *_a, **kw: captured.update(kw) or _Result(),
    )
    monkeypatch.setattr("master_agent.comfy.client.ComfyClient", lambda *a, **k: object())
    monkeypatch.setattr(
        "master_agent.upscale.upscale_video",
        lambda path, method, run_id: Path(path).with_suffix(f".{method}.mp4"),
    )
    out = agent_api.create_video("a fox runs", seed=7, storyboard="off", upscale="rtx")
    assert out["status"] == "done" and out["run_id"] == "r1"
    assert captured["seed"] == 7
    assert captured["storyboard_mode"] == "off"
    assert out["upscaled_path"].endswith(".rtx.mp4")
