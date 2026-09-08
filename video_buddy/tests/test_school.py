"""Lot school rubrics ported to Python. Never blocks export.

Run: .venv/Scripts/python.exe -m pytest tests/test_school.py -q
"""

from __future__ import annotations

from master_agent.school.score import (
    axis_pass,
    continuity_check,
    export_allowed,
    feasibility_check,
    school_exam,
    school_score,
    tutor_fields,
    want_pass,
)
from master_agent.school.tutor import tutor_storyboard
from master_agent.storyboard.storyboard import ShotCard


def test_no_want_fixture_fails_want_rubric():
    report = school_score(fixture="no-want")
    want = next(s for s in report["scores"] if s["id"] == "want-vs-need")
    assert want["pass"] is False
    assert want["rule"]
    assert want["counter_example"]
    assert export_allowed(report) is True


def test_axis_fail_fixture_fails_axis_and_keeps_want():
    report = school_score(fixture="axis-fail")
    axis = next(s for s in report["scores"] if s["id"] == "axis")
    want = next(s for s in report["scores"] if s["id"] == "want-vs-need")
    assert axis["pass"] is False
    assert want["pass"] is True
    exam = school_exam(fixture="axis-fail")
    assert exam["passed"] is False
    assert export_allowed(exam) is True


def test_want_pass_string_match():
    assert want_pass({"synopsis": "Ada wants the coat", "text": "", "characters": []}) is True
    assert want_pass({"synopsis": "Rain.", "text": "Nobody wants anything.", "characters": []}) is False


def test_axis_pass_eyeline_tracking():
    shots = [
        {"desc": "Ada looks camera left", "angle": "eye"},
        {"desc": "Ben also looks camera left", "angle": "reverse"},
    ]
    assert axis_pass(shots) is False
    shots[1]["desc"] = "Ben looks camera right"
    assert axis_pass(shots) is True


def test_continuity_and_feasibility():
    cont = continuity_check(
        [
            {"desc": "looks camera left", "angle": "eye", "title": "A"},
            {"desc": "looks camera left", "angle": "reverse", "title": "B"},
        ]
    )
    assert cont["pass"] is False
    feas = feasibility_check("tiny 8pt readable text overlay saying BUY NOW")
    assert feas["pass"] is False
    assert feas["rewrite"]


def test_tutor_fields_and_storyboard_wiring():
    show = {
        "school": {"enabled": True, "help": "nudge", "help_types": ["theory"]},
        "phase": "writer",
        "scenes": [{"num": "1", "slug": "tent"}],
        "shots": [],
    }
    fields = tutor_fields(show)
    assert fields["theory"]["id"] == "want-vs-need"
    muted = tutor_fields({**show, "school": {"enabled": True, "help": "mute"}})
    assert muted == {}

    cards = [
        ShotCard(index=0, title="Rain", duration_s=3, ltx_prompt="rain on canvas, nobody wants anything"),
    ]
    scene = {"synopsis": "Rain.", "text": "Nobody wants anything.", "characters": []}
    tutored = tutor_storyboard(cards, scene=scene, school={"enabled": True, "help": "nudge"})
    assert tutored[0].ltx_prompt
    assert "want" in (tutored[0].continuity or tutored[0].ltx_prompt).lower()
    assert export_allowed({"passed": False}) is True
