"""Grok Imagine corkboard: parallel panels, vision gate, KB records.

Run: .venv/Scripts/python.exe -m pytest tests/test_imagine_corkboard.py -q
"""

from __future__ import annotations

from pathlib import Path

from master_agent.imagine.corkboard import run_corkboard
from master_agent.kb.corkboard import load_panels
from master_agent.storyboard.storyboard import ShotCard


def _shots():
    return [
        ShotCard(index=0, title="Open", duration_s=3, ltx_prompt="wide rain window"),
        ShotCard(index=1, title="Turn", duration_s=3, ltx_prompt="tight brass light"),
        ShotCard(index=2, title="Close", duration_s=3, ltx_prompt="hold on the latch"),
    ]


def test_corkboard_promotes_only_passing_panels(tmp_path: Path):
    calls: list[str] = []

    def generate(prompt: str, dest: Path, **_kw):
        calls.append(prompt)
        dest.write_bytes(b"png")
        return {"bytes": 3, "path": str(dest)}

    def vision(path, **kwargs):
        prompt = kwargs.get("ltx_prompt") or ""
        ok = "brass" not in prompt
        return {
            "score": 0.9 if ok else 0.2,
            "pass": ok,
            "reason": "" if ok else "muddy mid-tone",
            "issues": [] if ok else ["muddy"],
        }

    records = run_corkboard(
        _shots(),
        dest_dir=tmp_path / "frames",
        generate_fn=generate,
        vision_fn=vision,
        store_dir=tmp_path / "kb",
        prompt_version="image_prompt:1",
    )
    assert len(calls) == 3
    queued = [r for r in records if r.queued]
    assert [r.index for r in queued] == [0, 2]
    failed = next(r for r in records if r.index == 1)
    assert failed.queued is False
    assert failed.failure_reason == "muddy mid-tone"
    assert failed.score == 0.2
    assert failed.prompt_version == "image_prompt:1"
    stored = load_panels(tmp_path / "kb")
    assert len(stored) == 3
    assert stored[1]["failure_reason"] == "muddy mid-tone"
    assert stored[1]["queued"] is False
