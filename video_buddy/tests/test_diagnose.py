"""Diagnose hull preset. No GPU.

Run: python -m pytest tests/test_diagnose.py -q
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from master_agent.comfy.diagnose import (
    DiagnoseFailed,
    ScaleRefused,
    diagnose_preset,
    hull_has_sec_per_step,
    refuse_scale_until_hull,
    run_diagnose,
)
from master_agent.config import (
    DIAGNOSE_FRAMES,
    DIAGNOSE_SEED,
    DIAGNOSE_STEPS_MAX,
    DIAGNOSE_STEPS_MIN,
)
from master_agent.control.budget import RenderBudget


OBJECT_INFO = {
    "EmptyLTXVLatentVideo": {
        "input": {
            "required": {
                "width": ["INT", {}],
                "height": ["INT", {}],
                "length": ["INT", {}],
            }
        },
        "output": ["LATENT"],
    },
    "KSampler": {
        "input": {
            "required": {
                "seed": ["INT", {}],
                "steps": ["INT", {}],
            }
        },
        "output": ["LATENT"],
    },
}


def _prepare(_variant, **kwargs):
    steps = int(kwargs["steps"])
    frames = int(kwargs["frames"])
    seed = int(kwargs["seed"])
    wf = {
        "1": {
            "class_type": "EmptyLTXVLatentVideo",
            "inputs": {"width": 768, "height": 512, "length": frames},
        },
        "2": {"class_type": "KSampler", "inputs": {"seed": seed, "steps": steps}},
    }
    return wf, {"frames": frames, "steps": steps, "seed": seed, "variant": _variant}


def test_preset_is_nine_frames_and_short_steps():
    preset = diagnose_preset(prompt="garden proof", variant="base")
    assert preset["frames"] == 9 == DIAGNOSE_FRAMES
    assert DIAGNOSE_STEPS_MIN <= preset["steps"] <= DIAGNOSE_STEPS_MAX
    assert preset["seed"] == DIAGNOSE_SEED


def test_prepare_only_lints_and_skips_fire(tmp_path: Path):
    def boom(_wf):
        raise AssertionError("diagnose --prepare must not fire")

    rec = run_diagnose(
        prompt="garden proof",
        variant="base",
        prepare_only=True,
        object_info=OBJECT_INFO,
        fire=boom,
        prepare_fn=_prepare,
        budget=RenderBudget(tmp_path / "budget.json", cap=80, used=4.0),
        hull_file=tmp_path / "hull.json",
        log=lambda *_: None,
    )
    assert rec["ok"] is True
    assert rec["status"] == "prepared"
    assert rec["charged_budget"] is False
    assert rec["used_delta"] == 0.0
    assert not hull_has_sec_per_step(path=tmp_path / "hull.json")


def test_fire_prints_sec_per_step_and_does_not_spend_budget(tmp_path: Path, capsys):
    video = tmp_path / "diag.mp4"
    video.write_bytes(b"x" * 120_000)
    budget = RenderBudget(tmp_path / "budget.json", cap=80, used=3.5)

    def fire(wf):
        assert wf["1"]["inputs"]["length"] == 9
        assert 6 <= wf["2"]["inputs"]["steps"] <= 8
        assert wf["2"]["inputs"]["seed"] == DIAGNOSE_SEED
        return {"video_path": str(video), "wall_s": 8.0}

    rec = run_diagnose(
        prompt="garden proof",
        variant="base",
        object_info=OBJECT_INFO,
        fire=fire,
        probe_fn=lambda _p: {"exists": True, "size_bytes": 120_000, "frames": 9},
        prepare_fn=_prepare,
        budget=budget,
        hull_file=tmp_path / "hull.json",
        copy_dir=tmp_path / "outputs",
    )
    assert rec["ok"] is True
    assert rec["sec_per_step"] == 1.0
    assert rec["charged_budget"] is False
    assert rec["used_delta"] == 0.0
    assert budget.used == 3.5
    assert hull_has_sec_per_step(path=tmp_path / "hull.json")
    out = capsys.readouterr().out
    assert "sec/step=" in out
    assert "wall_s=" in out


def test_junk_under_100kb_or_three_frames_fails(tmp_path: Path):
    video = tmp_path / "tiny.mp4"
    video.write_bytes(b"nope")
    with pytest.raises(DiagnoseFailed):
        run_diagnose(
            prompt="garden proof",
            object_info=OBJECT_INFO,
            fire=lambda _wf: {"video_path": str(video), "wall_s": 1.0},
            probe_fn=lambda _p: {"exists": True, "size_bytes": 50, "frames": 1},
            prepare_fn=_prepare,
            hull_file=tmp_path / "hull.json",
            log=lambda *_: None,
        )
    assert not hull_has_sec_per_step(path=tmp_path / "hull.json")


def test_refuse_scale_until_hull(tmp_path: Path):
    with pytest.raises(ScaleRefused):
        refuse_scale_until_hull(width=1280, height=720, frames=25, hull_file=tmp_path / "missing.json")
    # hull-sized diagnose request is allowed
    refuse_scale_until_hull(width=768, height=512, frames=9, hull_file=tmp_path / "missing.json")


def test_cmd_diagnose_prepare(monkeypatch, tmp_path: Path):
    from master_agent.__main__ import cmd_diagnose

    seen = {}

    def fake_run(**kwargs):
        seen.update(kwargs)
        return {"ok": True, "status": "prepared"}

    monkeypatch.setattr("master_agent.comfy.diagnose.run_diagnose", fake_run)
    rc = cmd_diagnose(
        Namespace(
            prompt="garden proof",
            variant="base",
            steps=7,
            seed=None,
            prepare=True,
            width=None,
            height=None,
            json=False,
        )
    )
    assert rc == 0
    assert seen["prepare_only"] is True
    assert seen["prompt"] == "garden proof"
    assert seen["variant"] == "base"
