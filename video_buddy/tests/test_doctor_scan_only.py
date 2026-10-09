"""doctor --fix --scan-only reports and never installs or downloads."""

from __future__ import annotations

from master_agent import setup


def test_scan_only_never_runs_fix(monkeypatch, capsys):
    def boom(*_a, **_k):
        raise AssertionError("fix() ran under --scan-only")

    monkeypatch.setattr(setup, "fix", boom)
    monkeypatch.setattr(setup, "print_inventory_preamble", lambda: None)
    monkeypatch.setattr(setup, "snapshot", lambda: {})
    monkeypatch.setattr(setup, "print_report", lambda snap, **_k: 0)
    rc = setup.cmd_setup(do_fix=True, mode="scan", yes=True)
    assert rc == 0
    assert "--fix ignored" in capsys.readouterr().out
