"""Judge strictness dial. Run: .venv/Scripts/python.exe -m pytest tests/test_strictness.py -q"""

from master_agent.control.strictness import Strictness, leg_thresholds, passes


def test_strictness_raises_all_three_legs():
    loose = leg_thresholds(0.0)
    mid = leg_thresholds(0.5)
    tight = leg_thresholds(1.0)
    for key in ("heuristic", "llm", "vision", "combined"):
        assert loose[key] < mid[key] < tight[key]
    assert passes(0.70, 0.0) is True
    assert passes(0.70, 1.0) is False
    dial = Strictness(value=0.4, project=0.6, scene=0.9)
    assert dial.effective() == 0.9
