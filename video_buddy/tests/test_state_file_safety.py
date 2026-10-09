"""Shared state files survive concurrent writers and corrupt reads (U4)."""

from __future__ import annotations

import json
import multiprocessing as mp

from master_agent.control.budget import RenderBudget
from master_agent.control.versioned_config import VersionedConfig
from master_agent.fileutil import atomic_write_text, file_lock


def test_atomic_write_leaves_no_temp_files(tmp_path):
    target = tmp_path / "x.json"
    atomic_write_text(target, "one")
    atomic_write_text(target, "two")
    assert target.read_text() == "two"
    assert [p.name for p in tmp_path.iterdir()] == ["x.json"]


def test_nonblocking_lock_reports_contention(tmp_path):
    target = tmp_path / "budget.json"
    with file_lock(target) as outer:
        assert outer is True
        with file_lock(target, blocking=False) as inner:
            assert inner is False
    with file_lock(target, blocking=False) as again:
        assert again is True


def test_corrupt_budget_is_kept_and_pauses_instead_of_wiping(tmp_path):
    path = tmp_path / "budget.json"
    path.write_text('{"cap": 40, "used": 33.0, "log": [{"scene_id": "a"', encoding="utf-8")
    budget = RenderBudget(path)
    assert budget.paused is True
    backups = list(tmp_path.glob("budget.json.corrupt-*"))
    assert len(backups) == 1
    assert backups[0].read_text().startswith('{"cap": 40, "used": 33.0')
    assert budget.log[-1]["event"] == "ledger_unreadable"


def test_two_instances_do_not_lose_each_others_charges(tmp_path):
    path = tmp_path / "budget.json"
    cost = {"vram_min": 1.0}
    a = RenderBudget(path, cap=100, used=0)
    b = RenderBudget(path)
    a.consider("a1", cost)
    b.consider("b1", cost)
    a.consider("a2", cost)
    data = json.loads(path.read_text())
    assert data["used"] == 3.0
    assert [r["scene_id"] for r in data["log"]] == ["a1", "b1", "a2"]


def _charge_many(path: str, tag: str, n: int) -> None:
    budget = RenderBudget(path)
    for i in range(n):
        budget.consider(f"{tag}-{i}", {"vram_min": 0.5})


def test_concurrent_processes_keep_every_charge(tmp_path):
    path = tmp_path / "budget.json"
    RenderBudget(path, cap=10_000, used=0)
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_charge_many, args=(str(path), f"p{k}", 15)) for k in range(3)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
        assert p.exitcode == 0
    data = json.loads(path.read_text())
    assert data["used"] == 22.5
    assert len(data["log"]) == 45


def _pin_live_config(monkeypatch):
    import master_agent.config as cfg

    for name in (
        "JUDGE_STRICTNESS",
        "LEARNING_RATE",
        "COST_VRAM_THRESHOLD_GB",
        "RENDER_BUDGET_CAP_VRAM_MIN",
        "JUDGE_SCORE_THRESHOLD",
        "RENDER_BUDGET_USED_VRAM_MIN",
        "PERSONA",
        "SOUL",
    ):
        monkeypatch.setattr(cfg, name, getattr(cfg, name))


def test_versioned_config_reloads_other_writers(tmp_path, monkeypatch):
    _pin_live_config(monkeypatch)
    a = VersionedConfig(tmp_path / "config.json", tmp_path / "history.jsonl")
    b = VersionedConfig(tmp_path / "config.json", tmp_path / "history.jsonl")
    a.set_values({"judge_strictness": 0.9}, session="a")
    b.set_values({"learning_rate": 0.42}, session="b")
    values = json.loads((tmp_path / "config.json").read_text())["values"]
    assert values["judge_strictness"] == 0.9
    assert values["learning_rate"] == 0.42


def test_versioned_config_keeps_unreadable_snapshot(tmp_path, monkeypatch):
    _pin_live_config(monkeypatch)
    path = tmp_path / "config.json"
    path.write_text("{not json", encoding="utf-8")
    VersionedConfig(path, tmp_path / "history.jsonl")
    assert len(list(tmp_path.glob("config.json.corrupt-*"))) == 1
