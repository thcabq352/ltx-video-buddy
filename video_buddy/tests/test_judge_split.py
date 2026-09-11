"""Judge look vs health split. No GPU.

Run: python -m pytest tests/test_judge_split.py -q
"""

from __future__ import annotations

from master_agent.judge.judge import (
    HUMAN_VETO,
    decide_action,
    health_score_from_issues,
    judge_segment,
    look_score_from_heuristic,
)
from master_agent.judge.probe import MIN_FRAMES, analyze


def test_payload_has_look_and_health():
    result = judge_segment(
        user_request="garden proof",
        ltx_prompt="a garden",
        video_path=None,
        heuristic_score=0.4,
        heuristic_issues=[{"code": "tiny_file", "severity": 1.0, "detail": "50 bytes"}],
        judge_enabled=False,
    )
    payload = result.to_dict()
    assert "look_score" in payload
    assert "health_score" in payload
    assert "combined_score" in payload
    assert payload["health_score"] < 0.5
    assert payload["look_score"] >= payload["health_score"]


def test_retry_ladder_uses_look_only():
    # High look, poor health: do not rewrite/retune just because the file is small.
    action = decide_action(
        combined=0.2,
        llm_pass=None,
        prompt_rewrite="please rewrite this whole prompt",
        param_hints={"steps": 20},
        judge_retries=0,
        max_rounds=3,
        critical=True,
        threshold=0.78,
        look_score=0.92,
    )
    assert action == "accept"

    # Low look still retries
    action_low = decide_action(
        combined=0.2,
        llm_pass=None,
        prompt_rewrite="try a different angle",
        param_hints={},
        judge_retries=0,
        max_rounds=3,
        critical=False,
        threshold=0.78,
        look_score=0.3,
    )
    assert action_low == "rewrite"


def test_brief_miss_and_high_look_is_human_veto():
    action = decide_action(
        combined=0.9,
        llm_pass=True,
        prompt_rewrite="",
        param_hints={},
        judge_retries=0,
        max_rounds=3,
        critical=False,
        threshold=0.78,
        look_score=0.91,
        brief_adherence=0.2,
    )
    assert action == HUMAN_VETO

    result = judge_segment(
        user_request="a red car",
        ltx_prompt="a red car",
        video_path=None,
        heuristic_score=0.95,
        heuristic_issues=[],
        judge_enabled=False,
    )
    # no brief_adherence from heuristic-only path; album stays lockable unless veto
    assert result.decision != HUMAN_VETO
    assert result.album_lock is True

    # Inject LLM brief miss
    import master_agent.judge.judge as judge_mod

    def fake_llm(**_kwargs):
        return {
            "pass": True,
            "score": 0.93,
            "brief_adherence": 0.15,
            "issues": ["wrong subject"],
            "prompt_rewrite": "",
            "param_hints": {},
            "reason": "looks great, wrong brief",
        }

    original_llm = judge_mod._llm_judge
    original_vision = judge_mod._vision_review_safe
    judge_mod._llm_judge = fake_llm
    judge_mod._vision_review_safe = lambda *a, **k: None
    try:
        vetoed = judge_segment(
            user_request="a red car",
            ltx_prompt="a red car",
            video_path=None,
            heuristic_score=0.95,
            heuristic_issues=[],
            judge_enabled=True,
        )
    finally:
        judge_mod._llm_judge = original_llm
        judge_mod._vision_review_safe = original_vision
    payload = vetoed.to_dict()
    assert vetoed.decision == HUMAN_VETO
    assert payload["HUMAN_VETO"] is True
    assert payload["human_veto"] is True
    assert payload["album_lock"] is False
    assert vetoed.look_score >= 0.78


def test_probe_min_frames_is_three(tmp_path, monkeypatch):
    assert MIN_FRAMES == 3
    video = tmp_path / "short.mp4"
    video.write_bytes(b"x" * 200_000)

    def fake_probe(_path):
        return {
            "exists": True,
            "size_bytes": 200_000,
            "duration_s": 0.04,
            "frames": 2,
        }

    monkeypatch.setattr("master_agent.judge.probe.probe_video", fake_probe)
    monkeypatch.setattr("master_agent.judge.probe.frame_motion_score", lambda *_a, **_k: 0.5)
    score, issues = analyze(video, expected_duration_s=1.0)
    assert any(i["code"] == "too_few_frames" for i in issues)
    assert score <= 0.4
    assert health_score_from_issues(issues) < 0.5
    assert look_score_from_heuristic(score, issues) >= health_score_from_issues(issues)
