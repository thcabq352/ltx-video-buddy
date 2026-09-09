"""Full storyboard dry-run, zero renders.

Run: .venv/Scripts/python.exe -m pytest tests/test_dry_run_board.py -q
"""

from master_agent.control.budget import RenderBudget
from master_agent.control.dry_run import dry_run_storyboard
from master_agent.storyboard.storyboard import ShotCard

_BUDGET = RenderBudget(cap=1000, used=0, ephemeral=True)


def test_dry_run_validates_and_never_queues():
    shots = [
        ShotCard(index=0, title="Open", duration_s=3, ltx_prompt="Ada wants the coat, looks camera left", camera="MCU"),
        ShotCard(index=1, title="Reverse", duration_s=3, ltx_prompt="Ben looks camera right", camera="reverse"),
    ]
    report = dry_run_storyboard(
        shots,
        variant="base",
        scene={"synopsis": "Ada wants the coat", "text": "Ada wants it.", "characters": ["ADA"]},
        frames_per_shot=81,
        budget=_BUDGET,
    )
    assert report["queued"] is False
    assert report["renders"] == 0
    assert report["ok"] is True
    assert len(report["costs"]) == 2
    assert report["school"]["ok"] is True


def test_dry_run_flags_but_still_does_not_queue():
    shots = [
        ShotCard(index=0, title="Card", duration_s=3, ltx_prompt="tiny 8pt readable text overlay saying BUY NOW"),
    ]
    report = dry_run_storyboard(
        shots,
        variant="directors",
        scene={"synopsis": "Rain. Nobody wants anything.", "text": "Nobody wants anything.", "characters": []},
        frames_per_shot=241,
        threshold_vram_gb=8.0,
        budget=_BUDGET,
    )
    assert report["queued"] is False
    assert report["renders"] == 0
    assert report["ok"] is False
    assert report["issues"]
