"""VRAM/time cost gate. Run: .venv/Scripts/python.exe -m pytest tests/test_cost_gate.py -q"""

from master_agent.control.cost import estimate_cost


def test_flags_high_vram_directors_long_take():
    ok = estimate_cost("base", frames=81, threshold_vram_gb=15.0)
    assert ok["flagged"] is False
    assert ok["vram_gb"] > 0
    assert ok["time_s"] > 0
    hot = estimate_cost("directors", frames=241, threshold_vram_gb=12.0)
    assert hot["flagged"] is True
    assert hot["vram_gb"] >= 12.0
