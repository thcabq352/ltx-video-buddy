"""Proofs for the judge / revise loop bugs seen on ltx23 lip-sync runs.

Vision and Comfy are mocked. The frame-sampling proof uses ffmpeg on a
synthetic clip (no GPU). Skip that case when ffmpeg is not installed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from master_agent.config import frames_for_duration
from master_agent.judge.judge import (
    brand_required,
    judge_segment,
    judge_system_prompt,
)
from master_agent.judge.probe import extract_sampled_jpegs, frame_motion_score
from master_agent.judge.vision import extract_frames_timed, vision_review
from master_agent.orchestrator.machine import Orchestrator
from master_agent.orchestrator.state import LOOP_PASSED, RunState
from master_agent.orchestrator.talking import (
    enforce_audio_duration,
    is_audio_driven,
    scrub_overlong_duration,
)
from master_agent.provenance import plan_clip_paths


def _black_then_bars(dest: Path) -> None:
    """8s 24fps clip: 4s black, then 4s of moving color bars."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=64x64:r=24:d=4",
            "-f",
            "lavfi",
            "-i",
            "testsrc=s=64x64:r=24:d=4",
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0",
            "-pix_fmt",
            "yuv420p",
            str(dest),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )


def _mean_luma(path: Path) -> float:
    from PIL import Image
    import numpy as np

    arr = np.asarray(Image.open(path).convert("L"), dtype="float32")
    return float(arr.mean())


def test_sampled_frames_cover_the_moving_tail(tmp_path, monkeypatch):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg required")
    clip = tmp_path / "talk.mp4"
    _black_then_bars(clip)

    calls: list[list] = []
    real_run = subprocess.run

    def _spy(cmd, *args, **kwargs):
        if isinstance(cmd, (list, tuple)):
            calls.append(list(cmd))
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr("master_agent.judge.probe.subprocess.run", _spy)
    timed = extract_frames_timed(clip, 4)
    assert len(timed) == 4
    times = [t for _p, t in timed]
    assert times == sorted(times)
    assert times[0] < 1.0
    assert times[-1] > 6.0
    assert _mean_luma(timed[0][0]) < 20
    assert _mean_luma(timed[-1][0]) > 40
    select_cmds = [c for c in calls if "select=" in " ".join(c)]
    assert select_cmds
    argv = select_cmds[0]
    assert "-fps_mode" in argv and argv[argv.index("-fps_mode") + 1] == "vfr"
    vf = argv[argv.index("-vf") + 1]
    assert "fps=1" not in vf
    motion = frame_motion_score(clip, samples=4)
    assert motion is not None and motion > 0.2

    captured: dict = {}

    def _chat(backend, system, user_text, images):
        captured["text"] = user_text
        captured["n"] = len(images)
        return json.dumps(
            {"score": 0.9, "pass": True, "issues": [], "reason": "mouth and head move"}
        )

    monkeypatch.setattr("master_agent.judge.vision.VISION_ENABLED", True)
    monkeypatch.setattr(
        "master_agent.judge.vision.active_local_backend", lambda: "llamacpp"
    )
    monkeypatch.setattr("master_agent.judge.vision._vision_chat", _chat)
    review = vision_review(clip, user_request="lip-sync the talking head")
    assert review and review["pass"] is True
    brief = json.loads(captured["text"].split("\n", 1)[1])
    assert brief["frame_order"] == "temporal, earliest first"
    seen = [row["time_s"] for row in brief["frames"]]
    assert seen == sorted(seen)
    assert seen[-1] > 6.0
    assert captured["n"] == len(seen)


def test_brand_rules_only_when_the_brief_asks(monkeypatch):
    assert brand_required("lip-sync the talking head") is False
    assert brand_required("cinematic brand film with an emblem") is True
    plain = judge_system_prompt(user_request="lip-sync the talking head")
    assert "The brief asked for branding" not in plain
    assert "Do not fail the clip" in plain
    branded = judge_system_prompt(user_request="cinematic brand film with an emblem")
    assert "The brief asked for branding" in branded

    monkeypatch.setattr("master_agent.judge.judge._vision_review_safe", lambda *a, **k: None)
    monkeypatch.setattr(
        "master_agent.judge.judge._llm_judge",
        lambda **k: {
            "pass": False,
            "score": 0.91,
            "brief_adherence": 0.2,
            "issues": ["missing emblem/logo"],
            "prompt_rewrite": "",
            "param_hints": {},
            "reason": "no logo",
        },
    )
    cleared = judge_segment(
        user_request="lip-sync the talking head",
        ltx_prompt="a person speaking to camera",
        video_path=None,
        heuristic_score=0.95,
        heuristic_issues=[],
        judge_enabled=True,
    )
    assert cleared.pass_ is True
    assert cleared.decision != "human_veto"
    assert cleared.brief_adherence == 0.9
    assert not any("logo" in i.lower() or "emblem" in i.lower() for i in cleared.issues)

    kept = judge_segment(
        user_request="cinematic brand film with an emblem",
        ltx_prompt="hero product and logo lockup",
        video_path=None,
        heuristic_score=0.95,
        heuristic_issues=[],
        judge_enabled=True,
    )
    assert kept.pass_ is False
    assert any("logo" in i.lower() or "emblem" in i.lower() for i in kept.issues)


def test_revise_respects_audio_length_and_frame_law(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "master_agent.orchestrator.talking.probed_audio_seconds", lambda _p: 3.86
    )
    assert is_audio_driven(
        "ltx23_lipsync_v08", has_image=True, has_audio=True, has_video=True
    )
    st = RunState(
        request="lip-sync the talking head",
        variant="ltx23_lipsync_v08",
        duration_s=5.0,
        image_name="face.png",
        audio_name="line.wav",
        audio_path="/tmp/line.wav",
        video_name="guide.mp4",
    )
    cap = enforce_audio_duration(st)
    assert cap == pytest.approx(3.86)
    assert st.duration_s <= 3.86
    assert st.frames == 89
    assert st.frames % 8 == 1
    assert st.frames / 24 <= 3.86 + 1e-6

    piece = RunState(
        request="lip-sync the talking head",
        variant="ltx23_lipsync_v08",
        duration_s=5.0,
        image_name="face.png",
        audio_name="line.wav",
        audio_path="/tmp/line.wav",
        duration_cap_s=3.86,
        max_piece_s=3.0,
    )
    enforce_audio_duration(piece)
    assert piece.frames == 65
    assert piece.frames % 8 == 1
    assert piece.duration_s <= 3.0
    assert frames_for_duration(3.0) == 73

    rewritten = scrub_overlong_duration("a 5 second talking head", 3.86)
    assert "3.86" in rewritten
    assert "5 second" not in rewritten

    calls = {"n": 0}

    def _llm(**_k):
        calls["n"] += 1
        if calls["n"] == 1:
            return {
                "pass": False,
                "score": 0.4,
                "brief_adherence": 0.85,
                "issues": ["hold longer"],
                "prompt_rewrite": "make it a 5 second talking head with more mouth motion",
                "param_hints": {},
                "reason": "asked for a longer clip",
            }
        return {
            "pass": True,
            "score": 0.93,
            "brief_adherence": 0.9,
            "issues": [],
            "prompt_rewrite": "",
            "reason": "matches the voice",
        }

    n = {"i": 0}

    def _resolve(state):
        n["i"] += 1
        plan_clip_paths(state)
        clip = Path(state.planned_clip)
        clip.parent.mkdir(parents=True, exist_ok=True)
        clip.write_bytes(f"clip-{n['i']}-{state.frames}".encode())
        state.video_path = str(clip)
        return True

    orch = Orchestrator()
    monkeypatch.setattr("master_agent.judge.judge._llm_judge", _llm)
    monkeypatch.setattr("master_agent.judge.judge._vision_review_safe", lambda *a, **k: None)
    monkeypatch.setattr(
        "master_agent.orchestrator.machine.analyze", lambda *a, **k: (0.95, [])
    )
    monkeypatch.setattr("master_agent.orchestrator.director.DIRECTOR_LLM", False)
    monkeypatch.setattr("master_agent.orchestrator.machine.RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr("master_agent.provenance.OUTPUTS_DIR", tmp_path / "outputs")
    monkeypatch.setattr(orch, "_patch", lambda _st: True)
    monkeypatch.setattr(orch, "_validate", lambda _st: True)
    monkeypatch.setattr(orch, "_submit_and_poll", lambda _st: True)
    monkeypatch.setattr(orch, "_resolve", _resolve)
    done = orch.run(
        "lip-sync the talking head",
        variant="ltx23_lipsync_v08",
        duration_s=5.0,
        image_name="face.png",
        audio_name="line.wav",
        audio_path="/tmp/line.wav",
        video_name="guide.mp4",
        shot={"camera": "locked medium", "action": "speaks the line"},
        judge_enabled=True,
        seed=3,
        max_judge_rounds=2,
    )
    assert done.loop_status == LOOP_PASSED
    assert done.frames == 89
    assert done.duration_s <= 3.86
    assert "5 second" not in (done.prompt or "")
    assert "3.86" in (done.prompt or "")


def test_attempt_lineage_does_not_mix(monkeypatch, tmp_path):
    calls = {"n": 0}

    def _llm(**_k):
        calls["n"] += 1
        if calls["n"] == 1:
            return {
                "pass": False,
                "score": 0.35,
                "brief_adherence": 0.8,
                "issues": ["mouth barely moves"],
                "prompt_rewrite": "clearer mouth motion on the talking head",
                "param_hints": {"seed": 77},
                "reason": "needs more mouth motion",
            }
        return {
            "pass": True,
            "score": 0.94,
            "brief_adherence": 0.92,
            "issues": [],
            "prompt_rewrite": "",
            "reason": "looks alive",
        }

    n = {"i": 0}

    def _resolve(state):
        n["i"] += 1
        plan_clip_paths(state)
        clip = Path(state.planned_clip)
        clip.parent.mkdir(parents=True, exist_ok=True)
        clip.write_bytes(f"bytes-attempt-{state.attempt}-{n['i']}".encode())
        state.video_path = str(clip)
        from master_agent.provenance import latest_revise_notes, persist_clip_provenance

        persist_clip_provenance(state, revise_notes=latest_revise_notes(state), path=clip)
        return True

    orch = Orchestrator()
    monkeypatch.setattr("master_agent.judge.judge._llm_judge", _llm)
    monkeypatch.setattr("master_agent.judge.judge._vision_review_safe", lambda *a, **k: None)
    monkeypatch.setattr(
        "master_agent.orchestrator.machine.analyze", lambda *a, **k: (0.95, [])
    )
    monkeypatch.setattr("master_agent.orchestrator.director.DIRECTOR_LLM", False)
    monkeypatch.setattr("master_agent.orchestrator.machine.RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr("master_agent.provenance.OUTPUTS_DIR", tmp_path / "outputs")
    monkeypatch.setattr(orch, "_patch", lambda _st: True)
    monkeypatch.setattr(orch, "_validate", lambda _st: True)
    monkeypatch.setattr(orch, "_submit_and_poll", lambda _st: True)
    monkeypatch.setattr(orch, "_resolve", _resolve)
    st = orch.run(
        "lip-sync the talking head",
        variant="base",
        judge_enabled=True,
        dry_run=False,
        seed=3,
        max_judge_rounds=2,
    )
    assert st.loop_status == LOOP_PASSED
    assert st.attempt == 2
    grouped: dict[str, list] = {}
    for rec in st.provenance_history:
        assert rec["schema"] == "buddy.clip.provenance/v1"
        grouped.setdefault(rec["lineage"]["attempt_id"], []).append(rec)
    a1 = grouped["shot-1.a1"]
    a2 = grouped["shot-1.a2"]
    assert {r["prompts"]["positive"] for r in a1} == {"lip-sync the talking head"}
    assert {r["params"]["seed"] for r in a1} == {3}
    assert all(not r["lineage"].get("parent_attempt_id") for r in a1)
    assert {r["prompts"]["positive"] for r in a2} == {
        "clearer mouth motion on the talking head"
    }
    assert {r["params"]["seed"] for r in a2} == {77}
    assert {r["lineage"]["parent_attempt_id"] for r in a2} == {"shot-1.a1"}
    assert a2[0]["hash"] is None
    assert a2[0]["judge"]["score"] == 0.0
    a1_hash = a1[-1]["hash"]
    a2_hash = a2[-1]["hash"]
    assert a1_hash and a2_hash and a1_hash != a2_hash
    assert a1[-1]["judge"]["score"] > 0
    assert a2[-1]["judge"]["score"] != a1[-1]["judge"]["score"]
    assert a2[-1]["judge"]["score"] > a1[-1]["judge"]["score"]
    assert "parent shot-1.a1" in (a2[-1].get("revise_notes") or "")


def test_no_judge_renders_once_and_does_not_requeue(monkeypatch, tmp_path):
    submits = {"n": 0}

    def _submit(_st):
        submits["n"] += 1
        return True

    def _resolve(state):
        plan_clip_paths(state)
        clip = Path(state.planned_clip)
        clip.parent.mkdir(parents=True, exist_ok=True)
        clip.write_bytes(b"one-render")
        state.video_path = str(clip)
        return True

    orch = Orchestrator()
    monkeypatch.setattr("master_agent.orchestrator.director.DIRECTOR_LLM", False)
    monkeypatch.setattr("master_agent.orchestrator.machine.RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr("master_agent.provenance.OUTPUTS_DIR", tmp_path / "outputs")
    monkeypatch.setattr(orch, "_patch", lambda _st: True)
    monkeypatch.setattr(orch, "_validate", lambda _st: True)
    monkeypatch.setattr(orch, "_submit_and_poll", _submit)
    monkeypatch.setattr(orch, "_resolve", _resolve)
    st = orch.run(
        "music video for a synthwave track",
        kind="music_video",
        variant="base",
        judge_enabled=False,
        revise_enabled=False,
        seed=1,
        max_judge_rounds=3,
    )
    assert submits["n"] == 1
    assert st.attempt == 1
    assert st.revise_history == []
    assert st.judge_decision == "skipped"
    assert st.loop_status == LOOP_PASSED
    reasons = (st.provenance.get("judge") or {}).get("fail_reasons")
    assert reasons == [{"kind": "judge", "detail": "skipped"}]
    assert not any(r.get("kind") == "cpu_fail_rules" for r in reasons)

    from argparse import Namespace

    from master_agent.__main__ import cmd_run

    args = Namespace(
        request="music video for a synthwave track",
        variant="base",
        duration=3.0,
        quality="draft",
        seed=1,
        width=768,
        height=512,
        video=None,
        image=None,
        audio=None,
        mask=None,
        no_judge=True,
        max_judge_rounds=3,
        storyboard="off",
        llm_panel=None,
        panel_judge=None,
        max_full_judge_rounds=None,
        dry_run=False,
        self_improve_dry=True,
        power_mode=False,
        no_power_mode=False,
        upscale=None,
        no_interview=True,
        attach=None,
        words=None,
    )
    rc = cmd_run(args)
    assert rc == 0
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (tmp_path / "runs").glob("*.json")
    ]
    dry = [rec for rec in records if rec.get("duration_s") == 3.0]
    assert len(dry) == 1
    assert dry[0]["attempt"] == 1
    assert dry[0]["judge_decision"] == "skipped"
    assert dry[0]["revise_history"] == []
