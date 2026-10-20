"""One-line shift budget in health / agent health / doctor. No GPU, no network.

Run: python -m pytest tests/test_health_budget.py -q
"""

from __future__ import annotations

import json
import os
from argparse import Namespace

import pytest

from master_agent.control.budget import RenderBudget, format_shift_summary, shift_summary


@pytest.fixture
def box(tmp_path, monkeypatch):
    import master_agent.config as cfg
    from master_agent.orchestrator import pipeline as pipeline_mod

    monkeypatch.setattr(cfg, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(cfg, "RENDER_BUDGET_CAP_VRAM_MIN", 80.0)
    monkeypatch.setattr(cfg, "RENDER_BUDGET_USED_VRAM_MIN", 0.0)
    monkeypatch.setattr(pipeline_mod, "RUNS_DIR", tmp_path / "runs")
    return tmp_path


def _ledger(box):
    return box / "state" / "control" / "budget.json"


def _run_record(box, run_id: str, status: str, *, ts: str, mtime: float) -> None:
    runs = box / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    path = runs / f"{ts}_{run_id}_pipeline.json"
    path.write_text(json.dumps({"run_id": run_id, "status": status}), encoding="utf-8")
    os.utime(path, (mtime, mtime))


def test_missing_ledger_reads_zero_without_writing(box):
    summary = shift_summary()
    assert summary["ledger"] == "missing"
    assert (summary["used"], summary["cap"], summary["remaining"]) == (0.0, 80.0, 80.0)
    assert summary["paused"] is False
    assert summary["paused_run_ids"] == []
    assert format_shift_summary(summary) == (
        "budget: 0.0 / 80.0 VRAM-min used, 80.0 remaining; paused runs: 0"
    )
    assert not _ledger(box).exists()


def test_empty_ledger_file_is_zero_not_unreadable(box):
    _ledger(box).parent.mkdir(parents=True)
    _ledger(box).write_text("  \n", encoding="utf-8")
    summary = shift_summary()
    assert summary["ledger"] == "empty"
    assert summary["used"] == 0.0
    assert summary["paused"] is False
    assert "unreadable" not in format_shift_summary(summary)
    assert list(_ledger(box).parent.iterdir()) == [_ledger(box)]


def test_partial_use_reads_current_shift(box):
    store = RenderBudget(_ledger(box), cap=80, used=34.2)
    summary = shift_summary()
    assert summary["ledger"] == "ok"
    assert summary["used"] == 34.2
    assert summary["remaining"] == 45.8
    assert summary["shift_id"] == store.shift_id
    assert summary["shift_started_at"] == store.shift_started_at
    assert format_shift_summary(summary) == (
        "budget: 34.2 / 80.0 VRAM-min used, 45.8 remaining; paused runs: 0"
    )

    store.reset_shift()
    after = shift_summary()
    assert after["used"] == 0.0
    assert after["shift_id"] == store.shift_id != summary["shift_id"]


def test_over_cap_hold_lists_paused_runs(box):
    store = RenderBudget(_ledger(box), cap=80, used=78.0)
    row = store.consider("run_cd34:seg1", {"vram_min": 6.0})
    assert row["decision"] == "hold"
    _run_record(box, "run_ab12", "paused", ts="20261019T100000", mtime=1_000)
    _run_record(box, "run_cd34", "paused", ts="20261019T100500", mtime=1_005)
    _run_record(box, "run_ef56", "paused", ts="20261019T090000", mtime=900)
    _run_record(box, "run_ef56", "done", ts="20261019T091000", mtime=910)
    _run_record(box, "run_gh78", "error", ts="20261019T092000", mtime=920)
    before = _ledger(box).read_bytes()

    summary = shift_summary()
    assert summary["paused"] is True
    assert summary["used"] == 78.0
    assert summary["remaining"] == 2.0
    assert summary["paused_run_ids"] == ["run_ab12", "run_cd34"]
    line = format_shift_summary(summary)
    assert line == (
        "budget: 78.0 / 80.0 VRAM-min used, 2.0 remaining (HOLD); "
        "paused runs: 2 (run_ab12, run_cd34)"
    )
    assert _ledger(box).read_bytes() == before

    store.used = 95.5
    store.persist()
    over = shift_summary()
    assert over["remaining"] == 0.0
    assert "95.5 / 80.0 VRAM-min used, 0.0 remaining (HOLD)" in format_shift_summary(over)


def test_many_paused_runs_stay_on_one_line(box):
    for i in range(8):
        _run_record(box, f"run_{i:02d}", "paused", ts=f"20261019T10000{i}", mtime=1_000 + i)
    line = format_shift_summary(shift_summary())
    assert "\n" not in line
    assert line.endswith("paused runs: 8 (run_00, run_01, run_02, run_03, run_04, +3 more)")


def test_unreadable_ledger_is_reported_not_quarantined(box):
    _ledger(box).parent.mkdir(parents=True)
    _ledger(box).write_text("{not json", encoding="utf-8")
    summary = shift_summary()
    assert summary["ledger"] == "unreadable"
    assert summary["paused"] is True
    assert summary["used"] == 0.0
    assert format_shift_summary(summary).endswith("; ledger unreadable")
    assert list(_ledger(box).parent.iterdir()) == [_ledger(box)]


class _FakeConfig:
    def snapshot(self):
        return {"hash": "h", "values": {}}


def test_agent_health_json_has_budget_fields(box, monkeypatch):
    from master_agent import agent_api

    def _down(*_a, **_k):
        raise RuntimeError("comfy down")

    monkeypatch.setattr("master_agent.comfy.client.ComfyClient.health", _down)
    monkeypatch.setattr(
        "master_agent.control.versioned_config.get_versioned_config", lambda: _FakeConfig()
    )
    monkeypatch.setattr("master_agent.llm.attach_llm_health", lambda out: out)
    monkeypatch.setattr("master_agent.kb.store.collection_count", lambda _c: 0)
    monkeypatch.setattr(agent_api, "_managed_comfy_summary", lambda: {})

    RenderBudget(_ledger(box), cap=80, used=81.0).consider("held", {"vram_min": 1.0})
    _run_record(box, "run_ab12", "paused", ts="20261019T100000", mtime=1_000)

    out = json.loads(json.dumps(agent_api.health()))
    budget = out["budget"]
    for key in ("used", "cap", "remaining", "paused_run_ids", "paused", "shift_id", "line"):
        assert key in budget
    assert budget["used"] == 81.0
    assert budget["cap"] == 80.0
    assert budget["remaining"] == 0.0
    assert budget["paused"] is True
    assert budget["paused_run_ids"] == ["run_ab12"]
    assert budget["line"].startswith("budget: 81.0 / 80.0 VRAM-min used")
    assert out["comfyui"]["up"] is False


def test_cli_health_prints_one_budget_line_when_comfy_down(box, monkeypatch, capsys):
    from master_agent.cli.doctor import cmd_health
    from master_agent.comfy.client import ComfyClientError

    class _Down:
        base_url = "http://127.0.0.1:8188"

        def health(self):
            raise ComfyClientError("connection refused")

    monkeypatch.setattr("master_agent.__main__.ComfyClient", lambda *a, **k: _Down())
    monkeypatch.setattr("master_agent.llm.format_local_llm_health", lambda: "")
    RenderBudget(_ledger(box), cap=80, used=34.2)

    assert cmd_health(Namespace()) == 1
    lines = [ln for ln in capsys.readouterr().out.splitlines() if "budget:" in ln]
    assert lines == ["      budget: 34.2 / 80.0 VRAM-min used, 45.8 remaining; paused runs: 0"]


def test_doctor_budget_row_never_fails_on_hold(box, capsys):
    from master_agent.setup import check_budget, print_report

    RenderBudget(_ledger(box), cap=80, used=80.0).consider("held", {"vram_min": 4.0})
    _run_record(box, "run_ab12", "paused", ts="20261019T100000", mtime=1_000)
    row = check_budget()
    assert row["name"] == "budget"
    assert row["ok"] is True
    assert row["detail"] == (
        "80.0 / 80.0 VRAM-min used, 0.0 remaining (HOLD); paused runs: 1 (run_ab12)"
    )
    assert print_report([row]) == 0
    out = capsys.readouterr().out
    budget_lines = [ln for ln in out.splitlines() if "VRAM-min" in ln]
    assert budget_lines == [f"  OK    budget             {row['detail']}"]
