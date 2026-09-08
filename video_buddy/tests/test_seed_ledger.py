"""Seed ledger. Run: .venv/Scripts/python.exe -m pytest tests/test_seed_ledger.py -q"""

from pathlib import Path

from master_agent.control.seeds import record_seed, replay


def test_record_and_replay(tmp_path: Path):
    ledger = tmp_path / "seeds.jsonl"
    record_seed(ledger, seed=42, prompt_version="image_prompt:2", score=0.91, variant="eros")
    record_seed(ledger, seed=99, prompt_version="image_prompt:2", score=0.4, variant="base")
    hit = replay(ledger, 42)
    assert hit["score"] == 0.91
    assert hit["variant"] == "eros"
    assert hit["prompt_version"] == "image_prompt:2"
    assert replay(ledger, 7) is None


def test_replay_skips_corrupt_lines(tmp_path: Path):
    from master_agent.control.seeds import load_ledger

    ledger = tmp_path / "seeds.jsonl"
    ledger.write_text(
        '{"seed": 1, "prompt_version": "p", "score": 0.5, "variant": "base"}\n'
        "not-json\n"
        '{"seed": 2, "prompt_version": "p", "score": 0.8, "variant": "base"}\n',
        encoding="utf-8",
    )
    rows = load_ledger(ledger)
    assert [row["seed"] for row in rows] == [1, 2]
    assert replay(ledger, 2)["score"] == 0.8
