"""Shift budget ledger. No GPU.

Run: python -m pytest tests/test_shift_budget.py -q
"""

from __future__ import annotations

from argparse import Namespace

from master_agent.a2a.protocol import a2a_task_state
from master_agent.comfy.diagnose import run_diagnose
from master_agent.control.budget import RenderBudget
from master_agent.control.cost import estimate_cost
from master_agent.control.dry_run import dry_run_storyboard
from master_agent.storyboard.storyboard import ShotCard


def test_reset_shift_archives_ledger_and_zeros_used(tmp_path):
    budget = RenderBudget(tmp_path / "budget.json", cap=40, used=12.5)
    budget.paused = True
    first_id = budget.shift_id
    budget.log.append({"event": "admit", "scene_id": "keep-me"})
    prior_len = len(budget.log)

    row = budget.reset_shift()
    assert row["event"] == "shift_reset"
    assert row["previous_used"] == 12.5
    assert row["previous_shift_id"] == first_id
    assert budget.used == 0.0
    assert budget.paused is False
    assert budget.shift_id != first_id
    assert budget.shift_started_at
    # history kept
    assert len(budget.log) == prior_len + 1
    assert any(item.get("event") == "shift_reset" for item in budget.log)
    assert any(item.get("scene_id") == "keep-me" for item in budget.log)
    snap = json_snapshot(budget)
    assert snap["shift_id"] == budget.shift_id
    assert snap["log"][-1]["event"] == "shift_reset"


def json_snapshot(budget: RenderBudget) -> dict:
    return budget.snapshot()


def test_diagnose_and_dry_run_do_not_increment_used(tmp_path):
    budget = RenderBudget(tmp_path / "budget.json", cap=80, used=6.0)
    object_info = {
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
            "input": {"required": {"seed": ["INT", {}], "steps": ["INT", {}]}},
            "output": ["LATENT"],
        },
    }

    def prepare(_variant, **kwargs):
        wf = {
            "1": {
                "class_type": "EmptyLTXVLatentVideo",
                "inputs": {"width": 768, "height": 512, "length": kwargs["frames"]},
            },
            "2": {
                "class_type": "KSampler",
                "inputs": {"seed": kwargs["seed"], "steps": kwargs["steps"]},
            },
        }
        return wf, {
            "frames": kwargs["frames"],
            "steps": kwargs["steps"],
            "seed": kwargs["seed"],
            "variant": _variant,
        }

    video = tmp_path / "d.mp4"
    video.write_bytes(b"x" * 120_000)
    rec = run_diagnose(
        prompt="garden proof",
        object_info=object_info,
        fire=lambda _wf: {"video_path": str(video), "wall_s": 4.0},
        probe_fn=lambda _p: {"exists": True, "size_bytes": 120_000, "frames": 9},
        prepare_fn=prepare,
        budget=budget,
        hull_file=tmp_path / "hull.json",
        copy_dir=tmp_path / "out",
        log=lambda *_: None,
    )
    assert rec["charged_budget"] is False
    assert budget.used == 6.0

    shots = [
        ShotCard(index=0, title="Open", duration_s=3, ltx_prompt="Ada looks left", camera="MCU"),
    ]
    report = dry_run_storyboard(shots, variant="base", budget=budget, frames_per_shot=9)
    assert report["queued"] is False
    assert budget.used == 6.0
    cheap = estimate_cost("base", frames=9)
    charged = budget.consider("real-shot", cheap, charge=True)
    assert charged["charged"] is True
    assert budget.used > 6.0


def test_hold_is_input_required_not_failed():
    assert a2a_task_state("hold") == "input-required"
    assert a2a_task_state("held") == "input-required"
    assert a2a_task_state("paused") == "input-required"
    assert a2a_task_state("error") == "failed"


def test_cmd_budget_status_and_reset(monkeypatch, tmp_path, capsys):
    from master_agent.__main__ import cmd_budget
    from master_agent.control import budget as budget_mod

    store = RenderBudget(tmp_path / "budget.json", cap=20, used=5.0)
    monkeypatch.setattr(budget_mod, "get_project_budget", lambda **_k: store)
    monkeypatch.setattr(
        "master_agent.control.budget.get_project_budget", lambda **_k: store
    )

    rc = cmd_budget(Namespace(budget_command="status", json=False))
    assert rc == 0
    out = capsys.readouterr().out
    assert "shift_id=" in out
    assert "used=5" in out

    rc = cmd_budget(Namespace(budget_command="reset-shift", json=True))
    assert rc == 0
    assert store.used == 0.0
    assert any(item.get("event") == "shift_reset" for item in store.log)
    printed = capsys.readouterr().out
    assert "shift_reset" in printed or "previous_used" in printed
