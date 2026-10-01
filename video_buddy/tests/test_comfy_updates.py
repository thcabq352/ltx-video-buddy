"""Opt-in Comfy updates. No live comfy-cli, no pip, no weight download.

Run: python -m pytest tests/test_comfy_updates.py -q
"""

from __future__ import annotations

import inspect
import json
from argparse import Namespace
from pathlib import Path

import pytest

from master_agent.comfy import tower, updates
from master_agent.comfy.tower import TowerError, cmd_tower
from master_agent.comfy.updates import (
    SNAPSHOT_NOTE,
    UpdateError,
    apply_update,
    execute_plan,
    plan_actions,
    read_git_head,
    scan_packs,
)
from master_agent.setup import check_pack_pins, fix, print_report, snapshot


def _git(repo: Path, sha: str) -> None:
    git = repo / ".git"
    (git / "refs" / "heads").mkdir(parents=True, exist_ok=True)
    (git / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (git / "refs" / "heads" / "main").write_text(sha + "\n", encoding="utf-8")


def _workspace(tmp_path: Path, *, node_sha: str | None = None) -> Path:
    ws = tmp_path / "ComfyUI"
    ws.mkdir()
    (ws / "main.py").write_text("# comfy\n", encoding="utf-8")
    _git(ws, "abc1234abc1234abc1234abc1234abc1234abc12")
    if node_sha is not None:
        node = ws / "custom_nodes" / "HeartMuLa_ComfyUI"
        node.mkdir(parents=True)
        _git(node, node_sha)
    weight = ws / "models" / "checkpoints" / "keep.safetensors"
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"weight")
    return ws


@pytest.fixture
def quiet_cli(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(updates, "installed_comfy_cli_version", lambda: "1.20.0")
    monkeypatch.delenv("COMFY_MODE", raising=False)
    monkeypatch.delenv("VRAM_GB", raising=False)


def test_read_git_head_follows_ref_and_packed_refs(tmp_path: Path):
    repo = tmp_path / "repo"
    _git(repo, "fdb53c4aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    assert read_git_head(repo).startswith("fdb53c4")
    packed = tmp_path / "packed"
    git = packed / ".git"
    git.mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (git / "packed-refs").write_text(
        "# pack-refs\n0123456789abcdef0123456789abcdef01234567 refs/heads/main\n",
        encoding="utf-8",
    )
    assert read_git_head(packed) == "0123456789abcdef0123456789abcdef01234567"


def test_optional_node_missing_is_not_stale(tmp_path: Path, quiet_cli, monkeypatch):
    ws = _workspace(tmp_path)
    report = scan_packs(ws)
    names = {item["name"]: item for item in report["items"]}
    assert names["comfy-cli"]["stale"] is False
    assert names["ComfyUI"]["stale"] is False
    assert "no core pin" in names["ComfyUI"]["detail"]
    assert names["HeartMuLa_ComfyUI"]["stale"] is False
    assert names["HeartMuLa_ComfyUI"]["optional"] is True
    assert report["stale"] == []

    node = ws / "custom_nodes" / "HeartMuLa_ComfyUI"
    node.mkdir(parents=True)
    _git(node, "fdb53c4aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    matched = scan_packs(ws)
    heart = next(item for item in matched["items"] if item["name"] == "HeartMuLa_ComfyUI")
    assert heart["stale"] is False

    _git(node, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    drifted = scan_packs(ws)
    heart = next(item for item in drifted["items"] if item["name"] == "HeartMuLa_ComfyUI")
    assert heart["stale"] is True
    plan = plan_actions(drifted)
    assert plan["actions"] == []
    assert any("--nodes" in note for note in plan["notes"])


def test_missing_cli_is_stale_and_bare_yes_only_pips_the_pin(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(updates, "installed_comfy_cli_version", lambda: None)
    monkeypatch.setattr(tower, "STATE_DIR", tmp_path / "state")
    ws = _workspace(tmp_path)
    pip_cmds: list[list[str]] = []

    def fake_pip(cmd, **kwargs):
        pip_cmds.append(list(cmd))
        return type("P", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    def forbid_comfy(*_a, **_k):
        raise AssertionError("cli pin install must not call comfy-cli")

    monkeypatch.setattr(updates.subprocess, "run", fake_pip)
    monkeypatch.setattr(tower, "_run_comfy", forbid_comfy)
    refused = apply_update(yes=False, workspace=ws, confirm=lambda _prompt: False)
    assert refused["updated"] is False
    assert refused["reason"] == "no-opt-in"
    assert pip_cmds == []

    done = apply_update(yes=True, workspace=ws, confirm=lambda _prompt: False)
    assert done["updated"] is True
    assert done["snapshot"] == ""
    assert pip_cmds
    assert any(part == "comfy-cli==1.20.0" for part in pip_cmds[0])
    assert ["update", "cli"] not in done["commands"]
    assert (ws / "models" / "checkpoints" / "keep.safetensors").read_bytes() == b"weight"


def test_commit_pin_yes_does_not_update_until_nodes_flag(tmp_path: Path, quiet_cli, monkeypatch):
    monkeypatch.setattr(tower, "STATE_DIR", tmp_path / "state")
    ws = _workspace(tmp_path, node_sha="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
    calls: list[list[str]] = []

    def fake_comfy(args, **kwargs):
        calls.append(list(args))
        if "save-snapshot" in args:
            Path(args[args.index("--output") + 1]).write_text("{}\n", encoding="utf-8")
        return type("P", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(tower, "_run_comfy", fake_comfy)
    monkeypatch.setattr(updates.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("pip")))
    held = apply_update(yes=True, workspace=ws)
    assert held["updated"] is False
    assert calls == []
    assert (ws / "models" / "checkpoints" / "keep.safetensors").read_bytes() == b"weight"

    moved = apply_update(yes=True, nodes=True, workspace=ws)
    assert moved["updated"] is True
    assert calls[0][:2] == ["node", "save-snapshot"]
    assert calls[1] == ["node", "update", "all"]
    assert Path(moved["snapshot"]).name.startswith("pre-update-")
    assert Path(moved["snapshot"]).is_file()
    assert SNAPSHOT_NOTE in "\n".join(moved["lines"])
    assert (ws / "models" / "checkpoints" / "keep.safetensors").read_bytes() == b"weight"
    assert "wipe_version" not in Path(updates.__file__).read_text(encoding="utf-8")


def test_snapshot_failure_skips_the_update(tmp_path: Path, quiet_cli, monkeypatch):
    monkeypatch.setattr(tower, "STATE_DIR", tmp_path / "state")
    ws = _workspace(tmp_path)
    (ws / "pyproject.toml").write_text('version = "0.1.0"\n', encoding="utf-8")
    monkeypatch.setattr(updates, "load_comfyui_pin", lambda: "0.3.10")
    calls: list[list[str]] = []

    def boom(args, **kwargs):
        calls.append(list(args))
        raise TowerError("snapshot failed")

    monkeypatch.setattr(tower, "_run_comfy", boom)
    with pytest.raises(UpdateError, match="snapshot failed"):
        apply_update(yes=True, workspace=ws)
    assert calls
    assert all("update" not in cmd for cmd in calls)


def test_missing_snapshot_file_aborts_before_update(tmp_path: Path, quiet_cli, monkeypatch):
    monkeypatch.setattr(tower, "STATE_DIR", tmp_path / "state")
    ws = _workspace(tmp_path)
    calls: list[list[str]] = []

    def silent(args, **kwargs):
        calls.append(list(args))
        return type("P", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(tower, "_run_comfy", silent)
    with pytest.raises(UpdateError, match="did not write"):
        execute_plan(
            {"actions": [{"kind": "nodes", "argv": ["node", "update", "all"], "label": "nodes"}]},
            workspace=ws,
        )
    assert len(calls) == 1
    assert "save-snapshot" in calls[0]


def test_semver_core_snapshots_then_passes_version(tmp_path: Path, quiet_cli, monkeypatch):
    monkeypatch.setattr(tower, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(updates, "load_comfyui_pin", lambda: "0.3.10")
    ws = _workspace(tmp_path)
    (ws / "pyproject.toml").write_text('[project]\nversion = "0.3.0"\n', encoding="utf-8")
    calls: list[list[str]] = []

    def fake_comfy(args, **kwargs):
        calls.append(list(args))
        if "save-snapshot" in args:
            Path(args[args.index("--output") + 1]).write_text("{}\n", encoding="utf-8")
        return None

    monkeypatch.setattr(tower, "_run_comfy", fake_comfy)
    result = apply_update(yes=True, workspace=ws)
    assert result["updated"] is True
    assert calls[0][:2] == ["node", "save-snapshot"]
    assert calls[1] == ["update", "comfy", "--version", "0.3.10"]
    assert (ws / "models" / "checkpoints" / "keep.safetensors").read_bytes() == b"weight"


def test_external_and_missing_workspace_do_not_install(tmp_path: Path, quiet_cli, monkeypatch):
    monkeypatch.setenv("COMFY_MODE", "external")
    ws = _workspace(tmp_path)
    monkeypatch.setattr(tower, "_run_comfy", lambda *a, **k: (_ for _ in ()).throw(AssertionError("comfy")))
    with pytest.raises(UpdateError, match="comfy_mode=external"):
        apply_update(yes=True, core=True, workspace=ws)

    monkeypatch.delenv("COMFY_MODE", raising=False)
    empty = tmp_path / "empty-comfy"
    empty.mkdir()
    with pytest.raises(UpdateError, match="does not run comfy install"):
        apply_update(yes=True, core=True, workspace=empty)


def test_refuses_to_float_comfy_cli(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(tower, "STATE_DIR", tmp_path / "state")
    with pytest.raises(UpdateError, match="requirements pin"):
        execute_plan({"actions": [{"kind": "cli", "pin": "9.9.9", "label": "nope"}]}, workspace=tmp_path)


def test_doctor_reports_stale_pins_without_updating(monkeypatch, capsys):
    monkeypatch.setattr(updates, "installed_comfy_cli_version", lambda: "0.0.1")

    def boom(*_a, **_k):
        raise AssertionError("doctor must not update")

    monkeypatch.setattr(updates, "apply_update", boom)
    monkeypatch.setattr(updates, "execute_plan", boom)
    assert "apply_update" not in inspect.getsource(fix)
    row = check_pack_pins()
    assert row["name"] == "pack-pins"
    assert row["ok"] is False
    assert "comfy-cli" in row["detail"]
    assert "No update ran." in row["detail"]
    assert "--yes" in row["fix"]
    assert "pack-pins" in {item["name"] for item in snapshot()}
    assert print_report([row]) == 0
    assert "NEED" in capsys.readouterr().out


def test_start_prints_the_report_and_does_not_update(tmp_path: Path, monkeypatch, capsys):
    state = tmp_path / "state"
    state.mkdir()
    workspace = tmp_path / "ComfyUI"
    monkeypatch.setattr(tower, "STATE_DIR", state)
    monkeypatch.setattr(tower, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("MANAGED_COMFY_ROOT", str(workspace))
    monkeypatch.delenv("COMFY_MODE", raising=False)
    calls: list[list[str]] = []
    probes = {"n": 0}

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        return type("P", (), {"returncode": 0, "stdout": "", "stderr": "", "args": cmd})()

    def fake_ready(*_a, **_k):
        probes["n"] += 1
        if probes["n"] == 1:
            return False, "down"
        return True, "/system_stats ok"

    monkeypatch.setattr(tower.subprocess, "run", fake_run)
    monkeypatch.setattr(tower, "http_ready", fake_ready)
    monkeypatch.setattr(updates, "installed_comfy_cli_version", lambda: "1.20.0")
    rc = cmd_tower(
        Namespace(
            comfy_command="start",
            workspace=None,
            port=None,
            base_url="http://127.0.0.1:8188",
            no_wait=True,
            no_watch=True,
            extra_model_paths=None,
            as_json=False,
        )
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Hardware:" in out
    assert "No update ran." in out
    assert "comfy start:" in out
    joined = " ".join(" ".join(cmd) for cmd in calls)
    assert "save-snapshot" not in joined
    assert " update " not in f" {joined} "


def test_cli_update_without_yes_and_client_stays_unauthenticated(tmp_path: Path, monkeypatch, capsys):
    from master_agent.__main__ import cmd_comfy

    monkeypatch.setattr(updates, "installed_comfy_cli_version", lambda: None)
    monkeypatch.setattr(updates.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("pip")))
    monkeypatch.setattr("master_agent.setup.confirm_prompt", lambda _prompt: False)
    rc = cmd_comfy(
        Namespace(
            comfy_command="update",
            yes=False,
            update_core=False,
            update_nodes=False,
            update_cli=False,
            workspace=str(tmp_path),
            as_json=True,
        )
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["updated"] is False
    assert "comfy-cli" in payload["stale"]
    client = Path(__file__).resolve().parents[1].joinpath("master_agent", "comfy", "client.py")
    text = client.read_text(encoding="utf-8")
    assert "Authorization" not in text
    assert "COMFYUI_HTTP_HEADERS" not in text
