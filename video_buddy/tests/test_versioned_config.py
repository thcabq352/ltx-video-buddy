"""Versioned config. Run: .venv/Scripts/python.exe -m pytest tests/test_versioned_config.py -q"""

import master_agent.config as cfg
from master_agent.control.versioned_config import VersionedConfig, announce_config, config_hash


def test_knob_change_is_logged_and_hash_shifts(tmp_path):
    saved = (
        cfg.JUDGE_STRICTNESS,
        cfg.LEARNING_RATE,
        cfg.COST_VRAM_THRESHOLD_GB,
        cfg.RENDER_BUDGET_CAP_VRAM_MIN,
        cfg.JUDGE_SCORE_THRESHOLD,
        cfg.RENDER_BUDGET_USED_VRAM_MIN,
        cfg.PERSONA,
        cfg.SOUL,
    )
    try:
        store = VersionedConfig(tmp_path / "config.json", tmp_path / "history.jsonl")
        before = store.snapshot()
        assert len(before["hash"]) == 16
        assert "render_budget_cap_vram_min" in before["values"]
        assert "render_budget_used_vram_min" in before["values"]
        changes = store.set_values(
            {"judge_strictness": 0.9, "learning_rate": 0.1},
            session="studio-test",
        )
        assert len(changes) == 2
        for row in changes:
            assert row["ts"]
            assert row["session"] == "studio-test"
            assert row["old"] != row["new"]
            assert row["key"] in {"judge_strictness", "learning_rate"}
        after = store.snapshot()
        assert after["hash"] != before["hash"]
        assert after["hash"] == config_hash(after["values"])
        last = store.history(10)
        assert len(last) == 2
        assert last[-1]["key"] == "learning_rate"
        line = announce_config(store)
        assert f"config_hash={after['hash']}" in line
        assert "judge_strictness=0.9" in line
    finally:
        (
            cfg.JUDGE_STRICTNESS,
            cfg.LEARNING_RATE,
            cfg.COST_VRAM_THRESHOLD_GB,
            cfg.RENDER_BUDGET_CAP_VRAM_MIN,
            cfg.JUDGE_SCORE_THRESHOLD,
            cfg.RENDER_BUDGET_USED_VRAM_MIN,
            cfg.PERSONA,
            cfg.SOUL,
        ) = saved
