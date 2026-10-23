"""Streamlit dashboard: pure helpers, plus the page under AppTest. No GPU, agent_api mocked.

The page tests skip when ``requirements-ui.txt`` is not installed.

Run: python -m pytest tests/test_streamlit_app.py -q
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from concurrent.futures import Future
from pathlib import Path

import pytest

import streamlit_app as app

VIDEO_BUDDY = Path(__file__).resolve().parents[1]


def _job(result=None, exc=None, done=True) -> app.Job:
    fut: Future = Future()
    if done:
        if exc is not None:
            fut.set_exception(exc)
        else:
            fut.set_result(result)
    return app.Job("abc", {"request": "x"}, fut)


def test_build_create_args_defaults_let_the_director_pick():
    args = app.build_create_args("  neon rain  ", duration_s=5, quality="draft")
    assert args == {
        "request": "neon rain",
        "duration_s": 5.0,
        "quality": "draft",
        "variant": None,
        "storyboard": None,
        "upscale": None,
        "seed": None,
        "dry_run": False,
    }


def test_build_create_args_passes_explicit_choices():
    args = app.build_create_args(
        "x", duration_s=8.5, quality="quality", variant=" ltx25_t2v_i2v ",
        storyboard="always", upscale="rtx", seed=42, dry_run=True,
    )
    assert args["variant"] == "ltx25_t2v_i2v"
    assert args["storyboard"] == "always" and args["upscale"] == "rtx"
    assert args["seed"] == 42 and args["dry_run"] is True


def test_build_create_args_negative_seed_is_random():
    assert app.build_create_args("x", duration_s=5, quality="draft", seed=-1)["seed"] is None


def test_build_create_args_keys_match_create_video_signature():
    import inspect

    from master_agent import agent_api

    params = inspect.signature(agent_api.create_video).parameters
    assert set(app.build_create_args("x", duration_s=5, quality="draft")) <= set(params)


def test_form_choices_come_from_agent_api():
    from master_agent import agent_api

    for mode in agent_api.STORYBOARD_MODES:
        assert app.build_create_args("x", duration_s=5, quality="draft", storyboard=mode or app.AUTO)[
            "storyboard"
        ] == mode


def test_resolve_video_path_prefers_upscaled(tmp_path):
    base = tmp_path / "final.mp4"
    up = tmp_path / "final_up.mp4"
    base.write_bytes(b"v")
    up.write_bytes(b"v")
    assert app.resolve_video_path({"video_path": str(base), "upscaled_path": str(up)}) == up.resolve()


def test_resolve_video_path_falls_back_when_upscale_missing(tmp_path):
    base = tmp_path / "final.mp4"
    base.write_bytes(b"v")
    result = {"video_path": str(base), "upscaled_path": str(tmp_path / "gone.mp4")}
    assert app.resolve_video_path(result) == base.resolve()


def test_resolve_video_path_relative_against_roots(tmp_path):
    (tmp_path / "runs").mkdir()
    clip = tmp_path / "runs" / "a.mp4"
    clip.write_bytes(b"v")
    assert app.resolve_video_path({"video_path": "runs/a.mp4"}, roots=[tmp_path / "nope", tmp_path]) == clip.resolve()


def test_resolve_video_path_none_when_absent(tmp_path):
    assert app.resolve_video_path({"video_path": str(tmp_path / "missing.mp4")}) is None
    assert app.resolve_video_path({"video_path": None}) is None


def test_done_shows_video(tmp_path):
    clip = tmp_path / "out.mp4"
    clip.write_bytes(b"v")
    out = app.interpret_result({"status": "done", "video_path": str(clip)}, roots=[])
    assert out.kind == "success" and out.video == clip.resolve()


def test_done_with_warnings_keeps_video_and_notes(tmp_path):
    clip = tmp_path / "out.mp4"
    clip.write_bytes(b"v")
    out = app.interpret_result(
        {"status": "done_with_warnings", "video_path": str(clip), "full_judge_notes": "soft focus"}, roots=[]
    )
    assert out.kind == "warning" and out.video == clip.resolve() and "soft focus" in out.message


def test_done_without_file_is_an_error(tmp_path):
    out = app.interpret_result({"status": "done", "video_path": str(tmp_path / "x.mp4")}, roots=[])
    assert out.kind == "error" and out.video is None


def test_budget_hold_is_paused_not_error():
    out = app.interpret_result({"status": "paused", "run_id": "r1", "error": "budget cap reached"})
    assert out.kind == "paused"
    assert "HOLD" in out.message and "r1" in out.message
    assert "budget reset-shift" in out.message


def test_busy_and_dry_run_are_not_errors():
    assert app.interpret_result({"status": "busy", "error": "gpu"}).kind == "busy"
    assert app.interpret_result({"status": "dry-run", "ok": True}).kind == "planned"


def test_error_carries_agent_api_message():
    out = app.interpret_result({"status": "error", "error": "request must not be empty"})
    assert out.kind == "error" and out.message == "request must not be empty"
    assert app.interpret_result({"status": "weird"}).message == "Unexpected status 'weird'."


def test_interpret_job_running_is_none():
    assert app.interpret_job(_job(done=False)) is None


def test_interpret_job_exception_is_error():
    out = app.interpret_job(_job(exc=RuntimeError("comfy gone")))
    assert out.kind == "error" and "RuntimeError" in out.message and "comfy gone" in out.message


def test_interpret_job_result():
    out = app.interpret_job(_job({"status": "paused"}), roots=[])
    assert out.kind == "paused"


def test_job_board_runs_one_job_at_a_time():
    gate = threading.Event()
    calls = []

    def fake_create_video(**kwargs):
        calls.append(kwargs)
        gate.wait(5)
        return {"status": "busy"}

    from concurrent.futures import ThreadPoolExecutor

    pool = ThreadPoolExecutor(max_workers=1)

    def thread_spawn(fn, args):
        return pool.submit(fn, **args)

    board = app.JobBoard(spawn=thread_spawn)
    first = board.submit(fake_create_video, {"request": "a"})
    assert first and board.running() is board.get(first)
    assert board.submit(fake_create_video, {"request": "b"}) is None
    gate.set()
    board.get(first).future.result(timeout=5)
    assert board.running() is None and board.latest() is board.get(first)
    second = board.submit(fake_create_video, {"request": "c"})
    board.get(second).future.result(timeout=5)
    assert [c["request"] for c in calls] == ["a", "c"]
    assert board.latest().id == second


def test_job_board_unknown_id():
    board = app.JobBoard()
    assert board.get(None) is None and board.get("nope") is None and board.latest() is None


def test_summarize_health_up():
    rows = app.summarize_health(
        {
            "comfyui": {"up": True, "gpus": [{"name": "RTX", "vram_free_gb": 10.0, "vram_total_gb": 16.0}]},
            "local_llm": {"active": "llamacpp"},
            "budget": {"paused": False, "line": "budget: 1.0 / 80.0 VRAM-min used"},
        }
    )
    assert rows[0] == ("ComfyUI", True, "up: RTX (10.0/16.0 GB free)")
    assert rows[1] == ("LLM", True, "active: llamacpp")
    assert rows[2] == ("Budget", True, "budget: 1.0 / 80.0 VRAM-min used")


def test_summarize_health_down_and_hold():
    rows = dict((label, (ok, text)) for label, ok, text in app.summarize_health(
        {"comfyui": {"up": False, "error": "refused"}, "local_llm": {}, "budget": {"paused": True, "line": "HOLD"}}
    ))
    assert rows["ComfyUI"] == (False, "down: refused")
    assert rows["LLM"][0] is False
    assert rows["Budget"] == (False, "HOLD")


def test_health_exception_becomes_a_row(monkeypatch):
    def boom():
        raise OSError("chroma locked")

    monkeypatch.setattr(app.agent_api, "health", boom)
    health = app._health()
    assert health == {"error": "OSError: chroma locked"}
    assert app.summarize_health(health) == [("health", False, "OSError: chroma locked")]


def test_module_import_does_not_load_streamlit():
    code = "import sys, streamlit_app; assert 'streamlit' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], cwd=VIDEO_BUDDY, check=True)


@pytest.fixture
def page(monkeypatch):
    """The real page under Streamlit's AppTest, with agent_api health/create_video mocked."""
    testing = pytest.importorskip("streamlit.testing.v1")
    import streamlit as st

    from master_agent import agent_api

    st.cache_resource.clear()
    st.cache_data.clear()
    monkeypatch.setattr(agent_api, "health", lambda: {
        "comfyui": {"up": True, "gpus": []},
        "local_llm": {"active": "llamacpp"},
        "budget": {"paused": False, "line": "budget: 0.0 / 80.0 VRAM-min used"},
    })
    yield testing.AppTest.from_file(str(VIDEO_BUDDY / "streamlit_app.py"), default_timeout=15)
    st.cache_resource.clear()
    st.cache_data.clear()


def _generate(at, prompt: str):
    at.run()
    at.text_area[0].input(prompt)
    at.button[-1].click()
    at.run()
    for _ in range(20):
        if at.exception or at.success[1:] or at.info or at.error or at.warning:
            break
        at.run()
    return at


def test_page_shows_health_and_renders_video(page, monkeypatch, tmp_path):
    from master_agent import agent_api

    clip = tmp_path / "out.mp4"
    clip.write_bytes(b"\x00")
    calls = []

    def fake_create_video(**kwargs):
        calls.append(kwargs)
        time.sleep(0.5)
        return {"status": "done", "video_path": str(clip)}

    monkeypatch.setattr(agent_api, "create_video", fake_create_video)
    at = _generate(page, "neon rain")
    assert not at.exception
    assert "ComfyUI" in at.success[0].value
    assert any(s.value == "Video made." for s in at.success)
    assert calls[0]["request"] == "neon rain" and calls[0]["quality"] == "draft"
    assert calls[0]["duration_s"] == app.DURATION_DEFAULT_S


def test_page_budget_hold_is_info_not_error(page, monkeypatch):
    from master_agent import agent_api

    monkeypatch.setattr(agent_api, "create_video", lambda **_: {"status": "paused", "run_id": "r9"})
    at = _generate(page, "neon rain")
    assert not at.exception and not at.error
    assert "HOLD" in at.info[0].value


def test_page_empty_prompt_never_calls_create_video(page, monkeypatch):
    from master_agent import agent_api

    monkeypatch.setattr(agent_api, "create_video", lambda **_: pytest.fail("called"))
    page.run()
    page.button[-1].click()
    page.run()
    assert page.error[0].value == "Type a prompt first."
