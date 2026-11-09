"""Readiness rejects non-finite, ready-only, and ambiguous judge JSON.

No network and no weights. These call the verdict parser directly.
"""

from __future__ import annotations

import json

import pytest

from master_agent.judge.local_look import _visual_verdict

_CUP = {
    "reason": "soft window light on a ceramic cup sitting on a wood table",
    "artifacts": "none visible",
    "subject_lock": "the ceramic cup stays one object",
}


def _body(payload: dict) -> str:
    return json.dumps(payload)


def test_substantive_boolean_verdict_is_accepted():
    accepted, reason, score = _visual_verdict(
        _body({"score": 0.91, "pass": True, "issues": [], **_CUP})
    )
    assert accepted is True
    assert score == pytest.approx(0.91)
    assert "ceramic cup" in reason


def test_real_issue_text_can_carry_the_description():
    accepted, reason, score = _visual_verdict(
        _body(
            {
                "score": 0.88,
                "pass": True,
                "issues": ["ready", "the cup rim is slightly clipped"],
            }
        )
    )
    assert accepted is True
    assert score == pytest.approx(0.88)
    assert "clipped" in reason


def test_percent_score_with_boolean_pass_stays_valid():
    accepted, _reason, score = _visual_verdict(
        _body({"score": 90, "pass": True, **_CUP})
    )
    assert accepted is True
    assert score == pytest.approx(0.9)


@pytest.mark.parametrize(
    "score",
    ["NaN", "nan", "NAN", "+nan", "Infinity", "inf", "-Infinity", "-inf", "1e309"],
)
def test_nonfinite_score_encodings_do_not_pass(score):
    accepted, reason, _score = _visual_verdict(
        _body({"score": score, "pass": True, **_CUP})
    )
    assert accepted is False
    assert reason


def test_bare_nan_token_is_not_a_verdict():
    accepted, _reason, _score = _visual_verdict(
        '{"score": NaN, "pass": true, "reason": "a ceramic cup on a wood table"}'
    )
    assert accepted is False


@pytest.mark.parametrize("score", [True, False, "yes", "high", None, [], {}])
def test_non_numeric_score_does_not_pass(score):
    accepted, _reason, _parsed = _visual_verdict(
        _body({"score": score, "pass": True, **_CUP})
    )
    assert accepted is False


@pytest.mark.parametrize("score", [-0.2, -5, 250])
def test_out_of_range_score_does_not_pass(score):
    accepted, _reason, _parsed = _visual_verdict(
        _body({"score": score, "pass": True, **_CUP})
    )
    assert accepted is False


@pytest.mark.parametrize(
    "token",
    [
        "ready,",
        "ready?",
        "ready!",
        "(ready)",
        "ready:",
        "ready;",
        "'ready'",
        '"ready"',
        "ready…",
        "ready？",
        "ready，",
        "  Ready?  ",
        "ready???",
        "…ready…",
        "ready—",
        "\nready\n",
        "«ready»",
        "ｒｅａｄｙ",
        "ready\u0301",
        "???",
        "...",
    ],
)
@pytest.mark.parametrize(
    "field",
    ["reason", "artifacts", "subject_lock", "temporal_consistency", "temporal"],
)
def test_punctuated_ready_is_not_a_description(field, token):
    accepted, _reason, _score = _visual_verdict(
        _body({"score": 0.99, "pass": True, field: token})
    )
    assert accepted is False


@pytest.mark.parametrize(
    "issues",
    [
        ["ready,"],
        ["ready?"],
        ["ready!", "(ready)"],
        [" READY？ "],
        ["«ready»", "ready…"],
        ["???"],
    ],
)
def test_punctuated_ready_issues_are_not_a_description(issues):
    accepted, _reason, _score = _visual_verdict(
        _body({"score": 0.99, "pass": True, "issues": issues})
    )
    assert accepted is False


def test_sentence_that_contains_ready_still_counts():
    accepted, reason, score = _visual_verdict(
        _body(
            {
                "score": 0.9,
                "pass": True,
                "reason": "the ceramic cup looks ready to photograph on the wood table",
                "issues": ["ready?"],
            }
        )
    )
    assert accepted is True
    assert score == pytest.approx(0.9)
    assert "wood table" in reason


@pytest.mark.parametrize(
    "issues",
    [["ready"], ["Ready."], ["READY"], ["ready", "ready."], ["  ready  "]],
)
def test_ready_only_issues_are_not_a_description(issues):
    accepted, reason, _score = _visual_verdict(
        _body({"score": 0.99, "pass": True, "issues": issues})
    )
    assert accepted is False
    assert "ready" in reason.lower() or "describe" in reason.lower()


def test_missing_pass_does_not_pass():
    accepted, reason, _score = _visual_verdict(_body({"score": 0.91, **_CUP}))
    assert accepted is False
    assert "pass" in reason.lower()


@pytest.mark.parametrize("flag", ["true", "false", "yes", 1, 0, None])
def test_non_boolean_pass_does_not_pass(flag):
    payload = {"score": 0.91, **_CUP}
    if flag is not None:
        payload["pass"] = flag
    accepted, reason, _score = _visual_verdict(_body(payload))
    assert accepted is False
    assert "pass" in reason.lower()


def test_explicit_rejection_does_not_pass():
    accepted, reason, _score = _visual_verdict(
        _body(
            {
                "score": 0.92,
                "pass": False,
                "reason": "the cup is melted into the table",
                "artifacts": "melted rim",
            }
        )
    )
    assert accepted is False
    assert "rejected" in reason.lower()
