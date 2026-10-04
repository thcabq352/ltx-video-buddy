"""Rainey1 judge rubric. No GPU.

Run: python -m pytest tests/test_rainey1_rubric.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from master_agent.judge.judge import judge_segment, judge_system_prompt
from master_agent.judge.rubric import (
    RAINEY1,
    judge_rubric_debug,
    load_rubric_text,
    resolve_judge_rubric,
    verdict_from_frame_description,
)
from master_agent.judge.vision import vision_system_prompt

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "rainey1" / "frames.json"


def _clear_rubric(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PANEL_JUDGE_RUBRIC", raising=False)
    monkeypatch.setattr("master_agent.config.PANEL_JUDGE_RUBRIC", "")


def _silence_models(monkeypatch: pytest.MonkeyPatch, payload: dict | None) -> None:
    monkeypatch.setattr(
        "master_agent.judge.judge._vision_review_safe", lambda *a, **k: None
    )

    def _llm(**_kwargs):
        return payload

    monkeypatch.setattr("master_agent.judge.judge._llm_judge", _llm)


def test_rainey1_prompt_loads():
    text = load_rubric_text(RAINEY1, kind="judge")
    assert "identity_lock" in text
    assert "density_escalation" in text
    assert "emissive_lighting" in text
    assert "anti_slop" in text
    assert "pacing_hold" in text
    assert "brief_adherence" in text
    assert "human veto" in text.lower()
    assert "album" in text.lower()
    assert "identity_morph" in text
    assert "filesize_junk" in text
    assert "too_few_frames" in text
    vision = load_rubric_text(RAINEY1, kind="vision")
    assert "identity_morph" in vision
    assert "human veto" in vision.lower()
    assert vision_system_prompt(user_request="rainey1 mesa") == vision


def test_selecting_rainey1_changes_instructions(monkeypatch):
    _clear_rubric(monkeypatch)
    plain = judge_system_prompt(user_request="garden proof")
    assert "density_escalation" not in plain
    assert resolve_judge_rubric(user_request="garden proof") is None

    by_keyword = judge_system_prompt(user_request="rainey1 mesa breach")
    assert "density_escalation" in by_keyword
    assert "Admiral" in by_keyword
    assert by_keyword != plain

    by_recipe = judge_system_prompt(
        user_request="garden proof",
        context={"recipe": "rainey1"},
    )
    assert "identity_lock" in by_recipe

    by_shot = judge_system_prompt(
        user_request="garden proof",
        shot={"title": "desk", "rainey1": True},
    )
    assert "pacing_hold" in by_shot

    monkeypatch.setenv("PANEL_JUDGE_RUBRIC", "rainey1")
    by_env = judge_system_prompt(user_request="garden proof")
    assert "emissive_lighting" in by_env
    debug = judge_rubric_debug(user_request="garden proof")
    assert debug["rubric"] == RAINEY1
    assert debug["prompt_file"] == "rainey1.md"
    assert debug["loaded"] is True

    monkeypatch.setenv("PANEL_JUDGE_RUBRIC", "default")
    pinned = judge_system_prompt(user_request="rainey1 mesa breach")
    assert "density_escalation" not in pinned
    off = judge_rubric_debug(user_request="rainey1 mesa breach")
    assert off["rubric"] == ""
    assert off["prompt_file"] == "judge.md"
    assert off["instruction_sha256"] != debug["instruction_sha256"]


def test_default_judge_unchanged_when_rubric_unset(monkeypatch):
    _clear_rubric(monkeypatch)
    _silence_models(
        monkeypatch,
        {
            "pass": True,
            "score": 0.93,
            "look_score": 0.93,
            "brief_adherence": 0.9,
            "issues": ["identity morph, twin faces"],
            "identity_morph": True,
            "prompt_rewrite": "",
            "param_hints": {},
            "reason": "mentions morph but rubric is off",
        },
    )
    result = judge_segment(
        user_request="garden proof",
        ltx_prompt="a garden",
        video_path=None,
        heuristic_score=0.95,
        heuristic_issues=[],
        judge_enabled=True,
    )
    payload = result.to_dict()
    assert result.decision == "accept"
    assert result.pass_ is True
    assert result.album_lock is False
    assert "rubric" not in payload
    assert "hard_fails" not in payload
    assert "rubric_scores" not in payload

    tiny = judge_segment(
        user_request="garden proof",
        ltx_prompt="a garden",
        video_path=None,
        heuristic_score=0.4,
        heuristic_issues=[{"code": "tiny_file", "severity": 1.0, "detail": "50 bytes"}],
        judge_enabled=False,
    )
    assert tiny.decision == "accept"
    assert "hard_fails" not in tiny.to_dict()


def test_rainey1_segment_retries_morph_and_junk(monkeypatch):
    _clear_rubric(monkeypatch)
    _silence_models(
        monkeypatch,
        {
            "pass": True,
            "score": 0.93,
            "look_score": 0.93,
            "identity_lock": 0.1,
            "density_escalation": 0.95,
            "emissive_lighting": 0.95,
            "anti_slop": 0.95,
            "pacing_hold": 0.95,
            "brief_adherence": 0.91,
            "identity_morph": True,
            "issues": ["identity morph, twin faces"],
            "prompt_rewrite": "",
            "param_hints": {},
            "reason": "face morphs into a twin",
        },
    )
    morphed = judge_segment(
        user_request="rainey1 lock test",
        ltx_prompt="locked operator at the desk",
        video_path=None,
        heuristic_score=0.95,
        heuristic_issues=[],
        judge_enabled=True,
    )
    assert morphed.pass_ is False
    assert morphed.decision == "rewrite"
    assert morphed.album_lock is False
    assert "identity_morph" in morphed.hard_fails
    assert morphed.prompt_rewrite
    assert morphed.look_score == pytest.approx(0.78, abs=0.02)
    payload = morphed.to_dict()
    assert payload["rubric"] == RAINEY1
    assert payload["album_lock"] is False
    assert "look_score" in payload
    assert "brief_adherence" in payload
    assert payload["rubric_scores"]["identity_lock"] == pytest.approx(0.1)

    junk = judge_segment(
        user_request="rainey1",
        ltx_prompt="locked operator",
        video_path=None,
        heuristic_score=0.4,
        heuristic_issues=[{"code": "tiny_file", "severity": 1.0, "detail": "50 bytes"}],
        judge_enabled=False,
    )
    assert junk.decision == "rewrite"
    assert "filesize_junk" in junk.hard_fails
    assert junk.pass_ is False

    short = judge_segment(
        user_request="use the rainey1 rubric",
        ltx_prompt="locked operator",
        video_path=None,
        heuristic_score=0.4,
        heuristic_issues=[
            {"code": "too_few_frames", "severity": 1.0, "detail": "only 1 frames (min 3)"}
        ],
        judge_enabled=False,
    )
    assert "too_few_frames" in short.hard_fails
    assert short.decision == "rewrite"
    assert short.album_lock is False


def test_rainey1_frame_fixtures():
    fixtures = json.loads(FIXTURES.read_text(encoding="utf-8"))
    assert len(fixtures) >= 2
    by_id = {row["id"]: row for row in fixtures}

    failed = verdict_from_frame_description(by_id["morph_junk"]["description"])
    assert failed["pass"] is False
    assert failed["album_lock"] is False
    for code in by_id["morph_junk"]["expect_hard_fails"]:
        assert code in failed["hard_fails"]
    assert failed["decision"] == "rewrite"
    assert {"pass", "score", "look_score", "brief_adherence", "issues", "prompt_rewrite", "reason"} <= set(
        failed
    )

    passed = verdict_from_frame_description(by_id["music_ready_pass"]["description"])
    assert passed["pass"] is True
    assert passed["decision"] == "accept"
    assert passed["look_score"] >= 0.75
    assert passed["hard_fails"] == []
    assert passed["album_lock"] is False
    assert passed["human_veto"] is False

    veto = verdict_from_frame_description(by_id["brief_miss_human"]["description"])
    assert veto["pass"] is False
    assert veto["decision"] == "human_veto"
    assert veto["HUMAN_VETO"] is True
    assert veto["album_lock"] is False
    assert veto["look_score"] >= 0.7
    assert veto["brief_adherence"] < 0.5
    assert veto["hard_fails"] == []


def test_negated_morph_language_is_not_a_hard_fail():
    verdict = verdict_from_frame_description(
        "Same locked silhouette, identity stable, no morph, no twin faces. "
        "Flora, cables, and craft. UFO beam and cyan mushrooms, deep blacks. "
        "No watermark. Music-video hold, readable motion, nine frames."
    )
    assert "identity_morph" not in verdict["hard_fails"]
    assert verdict["decision"] == "accept"
