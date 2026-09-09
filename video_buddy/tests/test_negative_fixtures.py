"""Negative-example fixtures. Run: .venv/Scripts/python.exe -m pytest tests/test_negative_fixtures.py -q"""

from master_agent.school.negatives import all_negatives, negative_example


def test_three_negatives_have_rewrite_and_never_block():
    rows = all_negatives()
    ids = {r["id"] for r in rows}
    assert ids == {"no-want", "axis-fail", "text-on-screen"}
    no_want = negative_example("no-want")
    assert no_want["want_pass"] is False
    assert "want" in no_want["rewrite"].lower()
    axis = negative_example("axis-fail")
    assert axis["axis_pass"] is False
    assert axis["rewrite"]
    text = negative_example("text-on-screen")
    assert text["feasibility_pass"] is False
    assert "type" in text["rewrite"].lower() or "text" in text["rewrite"].lower()
    assert all(r["export_allowed"] for r in rows)
