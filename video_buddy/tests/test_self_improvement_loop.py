"""Closed judge → revise → re-judge loop. No GPU, no live Comfy.

Run: python -m pytest tests/test_self_improvement_loop.py -q
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from master_agent.judge.judge import decide_action, judge_segment
from master_agent.judge.quality_bar import (
    RULE_A,
    RULE_B,
    RULE_C,
    RULE_D,
    apply_revise_plan,
    attach_rule_passes,
    build_revise_plan,
    evaluate_quality_bar,
)
from master_agent.orchestrator.machine import Orchestrator
from master_agent.orchestrator.state import (
    LOOP_EXHAUSTED,
    LOOP_PASSED,
    RunState,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "self_improve"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_rule_a_missing_music_bed_from_fixture():
    qb = evaluate_quality_bar(_load("mv_no_bed.json"))
    assert qb["pass"] is False
    ids = [f["id"] for f in qb["fails"]]
    assert RULE_A in ids
    assert RULE_B not in ids
    assert any(s["id"] == RULE_B for s in qb["skipped"])


def test_rule_c_thin_still_i2v_from_fixture():
    qb = evaluate_quality_bar(_load("still_i2v_thin.json"))
    ids = [f["id"] for f in qb["fails"]]
    assert RULE_C in ids
    assert qb["fails"][0]["code"] == "thin_still_i2v"


def test_rule_d_unused_control_pack_from_fixture():
    qb = evaluate_quality_bar(_load("unused_control_pack.json"))
    ids = [f["id"] for f in qb["fails"]]
    assert RULE_D in ids
    attach = attach_rule_passes(_load("unused_control_pack.json"))
    assert attach["c"] is True
    assert attach["d"] is False


def test_rule_b_never_invents_face_scores():
    qb = evaluate_quality_bar({"request": "close-up of Ada"})
    assert qb["pass"] is True
    assert qb["fails"] == []
    skipped = {s["id"]: s for s in qb["skipped"]}
    assert RULE_B in skipped
    assert "not ported" in skipped[RULE_B]["reason"]


def test_high_look_still_fails_quality_bar():
    result = judge_segment(
        user_request="music video for a synthwave track",
        ltx_prompt="neon rain",
        video_path=None,
        heuristic_score=0.95,
        heuristic_issues=[],
        judge_enabled=False,
        context=_load("mv_no_bed.json"),
    )
    assert result.decision == "rewrite"
    assert result.pass_ is False
    assert any("quality_bar.a" in i for i in result.issues)
    assert result.revise_plan.get("music_bed_attached") is True
    assert result.prompt_rewrite


def test_decide_action_exhausted_is_not_accept():
    action = decide_action(
        combined=0.2,
        llm_pass=None,
        prompt_rewrite="try again",
        param_hints={},
        judge_retries=3,
        max_rounds=3,
        critical=False,
        threshold=0.78,
        look_score=0.2,
    )
    assert action == "exhausted"


def _dry_orch(tmp_path=None, **kwargs) -> RunState:
    runs = tmp_path / "runs" if tmp_path is not None else Path("/tmp/buddy-runs")
    outs = tmp_path / "outputs" if tmp_path is not None else Path("/tmp/buddy-outputs")
    with patch("master_agent.orchestrator.director.DIRECTOR_LLM", False), patch(
        "master_agent.orchestrator.machine.RUNS_DIR", runs
    ), patch(
        "master_agent.provenance.OUTPUTS_DIR", outs
    ):
        return Orchestrator().run(dry_run=True, judge_enabled=False, **kwargs)


def test_dry_loop_closes_rule_a_then_passes(tmp_path):
    st = _dry_orch(
        tmp_path,
        request="music video for a synthwave track",
        kind="music_video",
        max_judge_rounds=3,
        variant="base",
    )
    assert st.loop_status == LOOP_PASSED
    assert st.state == "DONE"
    assert st.music_bed_attached is True
    assert st.revise_history
    assert "a" in st.revise_history[0]["fail_ids"]
    assert st.attempt >= 2
    last = st.judge_history[-1]
    assert last["decision"] == "accept"
    assert last["quality_bar"]["pass"] is True


def test_dry_loop_closes_rule_c_still_i2v(tmp_path):
    st = _dry_orch(
        tmp_path,
        request="i2v from this still, neon alley at night",
        image_name="alley_still.png",
        variant="ltx25_t2v_i2v",
        max_judge_rounds=3,
    )
    assert st.loop_status == LOOP_PASSED
    assert st.shot
    assert st.shot.get("camera")
    assert st.shot.get("action")
    assert any("c" in (r.get("fail_ids") or []) for r in st.revise_history)


def test_dry_loop_closes_rule_d_unused_pack(tmp_path):
    st = _dry_orch(
        tmp_path,
        request="rain on a window",
        variant="base",
        attach_recipe={"schema": "buddy.comfy.attach/v1"},
        previs_source="previs/shots/alley_01.export-patch.json",
        control_pack_present=True,
        control_pack_used={},
        max_judge_rounds=3,
    )
    assert st.loop_status == LOOP_PASSED
    assert any(st.control_pack_used.values())
    assert any("d" in (r.get("fail_ids") or []) for r in st.revise_history)


def test_dry_loop_never_queues_comfy(tmp_path):
    queued = []

    class BoomClient:
        def load_object_info(self, prefer_live=True):
            raise AssertionError("dry-run must not load object_info")

        def queue_prompt(self, workflow):
            queued.append(workflow)
            raise AssertionError("dry-run must not POST /prompt")

        def free_memory(self):
            raise AssertionError("dry-run must not free_memory")

    with patch("master_agent.orchestrator.director.DIRECTOR_LLM", False), patch(
        "master_agent.orchestrator.machine.RUNS_DIR", tmp_path / "runs"
    ), patch(
        "master_agent.provenance.OUTPUTS_DIR", tmp_path / "outputs"
    ):
        st = Orchestrator(client=BoomClient()).run(
            "music video for a synthwave track",
            kind="music_video",
            dry_run=True,
            judge_enabled=False,
            variant="base",
            max_judge_rounds=3,
        )
    assert queued == []
    assert st.loop_status == LOOP_PASSED


def test_max_attempts_sets_exhausted_terminal_state(tmp_path):
    """A plan that cannot clear the fail still stops with exhausted, not accept."""

    def sticky_fail(context=None):
        return {
            "evaluated": ["a", "c", "d"],
            "skipped": [{"id": "b", "code": "face_similarity", "reason": "skip"}],
            "fails": [
                {
                    "id": "a",
                    "code": "missing_music_bed",
                    "detail": "sticky",
                    "skipped": False,
                }
            ],
            "pass": False,
        }

    with patch(
        "master_agent.orchestrator.director.DIRECTOR_LLM", False
    ), patch(
        "master_agent.judge.judge.evaluate_quality_bar", side_effect=lambda ctx=None: sticky_fail(ctx)
    ), patch(
        "master_agent.judge.quality_bar.evaluate_quality_bar",
        side_effect=lambda ctx=None: sticky_fail(ctx),
    ), patch(
        "master_agent.orchestrator.machine.RUNS_DIR", tmp_path / "runs"
    ), patch(
        "master_agent.provenance.OUTPUTS_DIR", tmp_path / "outputs"
    ):
        st = Orchestrator().run(
            "music video for a synthwave track",
            kind="music_video",
            dry_run=True,
            judge_enabled=False,
            variant="base",
            max_judge_rounds=2,
        )
    assert st.loop_status == LOOP_EXHAUSTED
    assert st.judge_decision == "exhausted"
    assert st.state == "DONE"
    assert st.attempt == 2
    payload = st.to_dict()
    assert payload["loop_status"] == "exhausted"
    assert payload["judge_decision"] != "accept"


def test_live_rerun_path_repatches_without_real_comfy(tmp_path):
    submits = {"n": 0}

    orch = Orchestrator()

    def ok_patch(st):
        return True

    def ok_validate(st):
        return True

    def ok_submit(st):
        submits["n"] += 1
        return True

    def ok_resolve(st):
        st.video_path = None
        return True

    with patch("master_agent.orchestrator.director.DIRECTOR_LLM", False), patch(
        "master_agent.orchestrator.machine.analyze", return_value=(0.9, [])
    ), patch.object(orch, "_patch", ok_patch), patch.object(
        orch, "_validate", ok_validate
    ), patch.object(orch, "_submit_and_poll", ok_submit), patch.object(
        orch, "_resolve", ok_resolve
    ), patch(
        "master_agent.orchestrator.machine.RUNS_DIR", tmp_path / "runs"
    ), patch(
        "master_agent.provenance.OUTPUTS_DIR", tmp_path / "outputs"
    ):
        st = orch.run(
            "i2v from this still, neon alley",
            image_name="still.png",
            variant="base",
            judge_enabled=False,
            dry_run=False,
            max_judge_rounds=3,
        )
    assert submits["n"] >= 2  # first generate + at least one revise re-run
    assert st.loop_status == LOOP_PASSED
    assert st.shot and st.shot.get("camera")


def test_apply_revise_plan_mutates_state():
    st = RunState(request="mv", prompt="neon")
    plan = build_revise_plan(
        [{"id": "a", "code": "missing_music_bed", "detail": "x"}],
        {"request": "music video for x"},
        base_prompt="neon",
    )
    applied = apply_revise_plan(st, plan)
    assert "prompt" in applied
    assert st.music_bed_attached is True
    assert "music bed" in st.prompt


def test_cmd_self_improve_dry_exit(capsys, monkeypatch, tmp_path):
    from argparse import Namespace

    from master_agent.__main__ import cmd_run

    monkeypatch.setattr(
        "master_agent.orchestrator.director.DIRECTOR_LLM", False
    )
    monkeypatch.setattr(
        "master_agent.orchestrator.machine.RUNS_DIR", tmp_path / "runs"
    )
    monkeypatch.setattr(
        "master_agent.provenance.OUTPUTS_DIR", tmp_path / "outputs"
    )
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
    )
    rc = cmd_run(args)
    out = capsys.readouterr().out
    assert rc == 0
    assert "loop_status: passed" in out
    assert "no Comfy" in out
