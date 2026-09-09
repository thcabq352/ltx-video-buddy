"""Checkpoint hot-swap cache. Run: .venv/Scripts/python.exe -m pytest tests/test_hotswap.py -q"""

from master_agent.control.hotswap import VariantCache


def test_keeps_last_two_variants():
    cache = VariantCache(2)
    cache.touch("base")
    cache.touch("eros")
    cache.touch("directors")
    assert cache.warm() == ["eros", "directors"]
    assert cache.evicted("base") is True
    cache.touch("eros")
    assert cache.warm() == ["directors", "eros"]
