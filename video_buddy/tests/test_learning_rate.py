"""Learning-rate knob. Run: .venv/Scripts/python.exe -m pytest tests/test_learning_rate.py -q"""

from pathlib import Path

from master_agent.learn.prompts import PromptStore, needed_gain, should_promote


def test_high_rate_promotes_small_gains(tmp_path: Path):
    assert needed_gain(0.0) > needed_gain(1.0)
    assert should_promote(0.80, 0.78, learning_rate=1.0) is True
    assert should_promote(0.80, 0.78, learning_rate=0.0) is False
    store = PromptStore(tmp_path)
    rec = store.write_candidate("image_prompt", "v2 stills")
    store.path_for("image_prompt", 1).parent.mkdir(parents=True, exist_ok=True)
    store.path_for("image_prompt", 1).write_text("v1\n", encoding="utf-8")
    store.promote("image_prompt", rec.version)
    assert store.current_version("image_prompt") == rec.version
