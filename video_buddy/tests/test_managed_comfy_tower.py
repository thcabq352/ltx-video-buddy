"""Managed Comfy lifecycle. No live Comfy, no GPU, subprocess mocked.

Run: python -m pytest tests/test_managed_comfy_tower.py -q
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from urllib.error import URLError

import pytest

from master_agent.comfy import tower
from master_agent.comfy.tower import (
    ManagedComfyTower,
    TowerError,
    TowerState,
    cmd_tower,
    comfy_cli_present,
    http_ready,
    load_state,
    resolve_comfy_cli,
    run_watchdog_forever,
    save_state,
    wait_until_ready,
    watchdog_command,
    watchdog_tick,
)


class _Proc:
    def __init__(self, cmd, returncode=0, stdout="", stderr=""):
        self.cmd = cmd
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.args = cmd


class _Popen:
    def __init__(self, cmd, **kwargs):
        self.cmd = cmd
        self.kwargs = kwargs
        self.pid = 424242


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    workspace = tmp_path / "ComfyUI"
    monkeypatch.setattr(tower, "STATE_DIR", state_dir)
    monkeypatch.setattr(tower, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("MANAGED_COMFY_ROOT", str(workspace))
    monkeypatch.delenv("COMFY_MODE", raising=False)
    monkeypatch.delenv("COMFY_CLI", raising=False)

    calls: list[list[str]] = []
    popens: list[_Popen] = []
    kills: list[tuple[int, int]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        assert kwargs.get("env", {}).get("COMFY_WHERE") == "local"
        return _Proc(cmd)

    def fake_popen(cmd, **kwargs):
        proc = _Popen(cmd, **kwargs)
        popens.append(proc)
        return proc

    def fake_kill(pid, sig):
        kills.append((int(pid), int(sig)))

    monkeypatch.setattr(tower.subprocess, "run", fake_run)
    monkeypatch.setattr(tower.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(tower.os, "kill", fake_kill)
    probes = {"n": 0, "script": [(False, "down"), (True, "/system_stats ok")]}

    def fake_ready(url, timeout=5.0):
        script = probes["script"]
        item = script[min(probes["n"], len(script) - 1)]
        probes["n"] += 1
        return item

    monkeypatch.setattr(tower, "http_ready", fake_ready)
    monkeypatch.setattr(tower, "sageattention_available", lambda: True)
    return {
        "root": tmp_path,
        "state_dir": state_dir,
        "workspace": workspace,
        "calls": calls,
        "popens": popens,
        "kills": kills,
        "probes": probes,
    }


def _launch_tail(cmd: list[str]) -> list[str]:
    assert "--" in cmd
    return cmd[cmd.index("--") + 1 :]


def test_launch_argv_is_local_loopback_and_hides_browser(isolated):
    tower_obj = ManagedComfyTower(ready_timeout_s=1)
    result = tower_obj.start(watch=True)
    assert result["status"] == "started"
    assert result["ready"] is True
    assert isolated["calls"], "expected comfy launch"
    cmd = isolated["calls"][0]
    assert cmd[cmd.index("--where") + 1] == "local"
    assert any(part.startswith("--workspace=") for part in cmd)
    assert str(isolated["workspace"]) in " ".join(cmd)
    assert "--skip-prompt" in cmd
    assert "launch" in cmd
    assert "--background" in cmd
    tail = _launch_tail(cmd)
    assert tail[:5] == [
        "--disable-auto-launch",
        "--port",
        "8188",
        "--listen",
        "127.0.0.1",
    ]
    yaml_path = (isolated["state_dir"] / "extra_model_paths.yaml").resolve()
    assert yaml_path.is_file()
    assert tail[-3:] == [
        "--extra-model-paths-config",
        str(yaml_path),
        "--use-sage-attention",
    ]
    assert not (isolated["workspace"] / "extra_model_paths.yaml").exists()
    assert "is_default: true" in yaml_path.read_text(encoding="utf-8")
    assert "0.0.0.0" not in cmd
    assert "cloud" not in cmd
    assert isolated["popens"], "start should spawn a detached watchdog"
    assert isolated["popens"][0].cmd == watchdog_command()
    assert isolated["popens"][0].kwargs.get("start_new_session") is True
    saved = json.loads((isolated["state_dir"] / "managed_comfy.json").read_text(encoding="utf-8"))
    assert saved["want_running"] is True
    assert saved["comfy_mode"] == "managed"
    assert saved["watchdog_pid"] == 424242


def test_refuses_to_adopt_foreign_server(isolated):
    isolated["probes"]["script"] = [(True, "/system_stats ok")]
    with pytest.raises(TowerError, match="foreign"):
        ManagedComfyTower().start()
    assert isolated["calls"] == []
    assert isolated["popens"] == []


def test_already_up_does_not_relaunch(isolated):
    st = load_state()
    st.want_running = True
    st.mode = "managed"
    save_state(st)
    isolated["probes"]["script"] = [(True, "/system_stats ok")]
    result = ManagedComfyTower().start()
    assert result["status"] == "already_up"
    assert isolated["calls"] == []
    assert isolated["popens"]


def test_external_mode_refuses_start_stop_restart(isolated):
    st = load_state()
    st.mode = "external"
    save_state(st)
    tower_obj = ManagedComfyTower()
    for action in (
        lambda: tower_obj.start(),
        lambda: tower_obj.stop(),
        lambda: tower_obj.restart(),
    ):
        with pytest.raises(TowerError, match="comfy_mode=external"):
            action()
    assert isolated["calls"] == []
    assert isolated["kills"] == []


def test_external_start_writes_buddy_yaml_not_attached_tree(isolated, monkeypatch):
    external = isolated["root"] / "external-comfy"
    (external / "models" / "checkpoints").mkdir(parents=True)
    sentinel = external / "models" / "checkpoints" / "keep.safetensors"
    sentinel.write_bytes(b"user-weight")
    monkeypatch.setenv("EXTERNAL_COMFY_ROOT", str(external))
    monkeypatch.setenv("COMFY_MODE", "external")
    with pytest.raises(TowerError, match="comfy_mode=external"):
        ManagedComfyTower().start()
    assert isolated["calls"] == []
    assert not (external / "extra_model_paths.yaml").exists()
    assert sentinel.read_bytes() == b"user-weight"
    buddy = isolated["state_dir"] / "extra_model_paths.yaml"
    assert buddy.is_file()
    text = buddy.read_text(encoding="utf-8")
    assert "is_default: true" in text
    assert external.joinpath("models").resolve().as_posix() in text


def test_env_external_overrides_managed_state(isolated, monkeypatch):
    monkeypatch.setenv("COMFY_MODE", "external")
    with pytest.raises(TowerError, match="comfy_mode=external"):
        ManagedComfyTower().stop()
    assert isolated["calls"] == []


def test_stop_uses_comfy_stop_and_does_not_kill_port(isolated):
    isolated["probes"]["script"] = [(False, "down")]
    st = load_state()
    st.want_running = True
    st.watchdog_pid = 424242
    save_state(st)
    result = ManagedComfyTower().stop()
    assert result["status"] == "stopped"
    cmd = isolated["calls"][0]
    assert cmd[-1] == "stop"
    assert "launch" not in cmd
    assert all("fuser" not in part and "taskkill" not in part for part in cmd)
    assert load_state().want_running is False
    assert any(sig != 0 for _pid, sig in isolated["kills"])


def test_stop_reports_still_reachable_without_adopting_kill(isolated):
    isolated["probes"]["script"] = [(True, "/system_stats ok")]
    result = ManagedComfyTower().stop()
    assert result["status"] == "stop_issued_still_reachable"
    assert isolated["calls"][0][-1] == "stop"


def test_restart_increments_count(isolated):
    isolated["probes"]["script"] = [
        (False, "down"),
        (False, "down"),
        (True, "/system_stats ok"),
    ]
    result = ManagedComfyTower(ready_timeout_s=1).restart()
    assert result["status"] == "restarted"
    assert result["restart_count"] == 1
    verbs = [cmd[cmd.index("stop")] if "stop" in cmd else "launch" for cmd in isolated["calls"]]
    assert "stop" in verbs
    assert any("launch" in cmd for cmd in isolated["calls"])


def test_watchdog_restarts_after_grace(isolated):
    st = load_state()
    st.want_running = True
    st.mode = "managed"
    st.last_start_at = 1.0
    save_state(st)
    isolated["probes"]["script"] = [(False, "down"), (True, "/system_stats ok")]
    tower_obj = ManagedComfyTower(ready_timeout_s=5)
    assert watchdog_tick(tower_obj) == "restarted"
    assert load_state().restart_count == 1
    assert any("launch" in cmd for cmd in isolated["calls"])


def test_watchdog_waits_out_startup_grace(isolated):
    st = load_state()
    st.want_running = True
    st.last_start_at = tower.time.time()
    save_state(st)
    isolated["probes"]["script"] = [(False, "down")]
    assert watchdog_tick(ManagedComfyTower(ready_timeout_s=180)) == "starting"
    assert isolated["calls"] == []


def test_watchdog_loop_stops_when_not_wanted(isolated, monkeypatch):
    assert watchdog_tick(ManagedComfyTower()) == "exit"
    calls = {"n": 0}

    def _exit(_tower=None):
        calls["n"] += 1
        return "exit"

    def _boom(_seconds):
        raise AssertionError("watchdog should return before sleeping")

    monkeypatch.setattr(tower, "watchdog_tick", _exit)
    monkeypatch.setattr(tower.time, "sleep", _boom)
    run_watchdog_forever(interval_s=0.01)
    assert calls["n"] == 1


def test_http_ready_falls_back_to_object_info(monkeypatch):
    seen: list[str] = []

    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def getcode(self):
            return 200

    def fake_urlopen(req, timeout=0):
        seen.append(req.full_url)
        if req.full_url.endswith("/system_stats"):
            raise URLError("refused")
        return _Resp()

    monkeypatch.setattr(tower.urllib.request, "urlopen", fake_urlopen)
    ok, detail = http_ready("http://127.0.0.1:9")
    assert ok is True
    assert detail == "/object_info ok"
    assert seen[0].endswith("/system_stats")
    assert seen[1].endswith("/object_info")


def test_wait_until_ready_times_out(monkeypatch):
    monkeypatch.setattr(tower, "http_ready", lambda *_a, **_k: (False, "down"))
    clock = {"now": 0.0}

    def monotonic():
        return clock["now"]

    def sleep(seconds):
        clock["now"] += seconds

    monkeypatch.setattr(tower.time, "monotonic", monotonic)
    monkeypatch.setattr(tower.time, "sleep", sleep)
    with pytest.raises(TowerError, match="did not become ready"):
        wait_until_ready("http://127.0.0.1:9", timeout_s=1, poll_s=0.5)


def test_missing_cli_is_a_tower_error(isolated, monkeypatch):
    def missing(*_a, **_k):
        raise FileNotFoundError("comfy")

    monkeypatch.setattr(tower.subprocess, "run", missing)
    isolated["probes"]["script"] = [(False, "down")]
    with pytest.raises(TowerError, match="comfy-cli not found"):
        ManagedComfyTower().start(watch=False)
    assert load_state().want_running is False


def test_resolve_comfy_cli_and_presence(monkeypatch):
    monkeypatch.setenv("COMFY_CLI", "/opt/bin/comfy --json")
    assert resolve_comfy_cli() == ["/opt/bin/comfy", "--json"]
    assert comfy_cli_present() is True
    monkeypatch.delenv("COMFY_CLI", raising=False)
    monkeypatch.setattr(tower.shutil, "which", lambda name: None)
    monkeypatch.setattr(tower.importlib.util, "find_spec", lambda name: None)
    prefix = resolve_comfy_cli()
    assert prefix[0] == tower.sys.executable
    assert prefix[-2:] == ["-m", "comfy_cli"]
    assert comfy_cli_present() is False


def test_extra_model_paths_after_double_dash(isolated, tmp_path: Path):
    yaml_path = tmp_path / "extra.yaml"
    yaml_path.write_text("comfyui:\n", encoding="utf-8")
    ManagedComfyTower(ready_timeout_s=1).start(extra_model_paths=yaml_path, watch=False)
    tail = _launch_tail(isolated["calls"][0])
    assert tail[-3:] == [
        "--extra-model-paths-config",
        str(yaml_path),
        "--use-sage-attention",
    ]
    assert isolated["popens"] == []


def test_launch_args_always_use_sage_attention(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(tower, "sageattention_available", lambda: True)
    tower_obj = ManagedComfyTower()
    without = tower_obj._launch_args(None)
    assert "--use-sage-attention" in without
    assert without.index("--use-sage-attention") > without.index("--")
    assert without[-1] == "--use-sage-attention"
    assert "--extra-model-paths-config" not in without

    yaml_path = tmp_path / "extra.yaml"
    yaml_path.write_text("comfyui:\n", encoding="utf-8")
    with_yaml = tower_obj._launch_args(yaml_path)
    assert "--use-sage-attention" in with_yaml
    assert with_yaml.index("--use-sage-attention") > with_yaml.index("--")
    assert with_yaml[-1] == "--use-sage-attention"
    config_at = with_yaml.index("--extra-model-paths-config")
    assert config_at > with_yaml.index("--")
    assert with_yaml[config_at + 1] == str(yaml_path)
    assert with_yaml.index("--use-sage-attention") > config_at


def test_launch_args_omit_sage_attention_when_missing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(tower, "sageattention_available", lambda: False)
    warnings: list[str] = []
    monkeypatch.setattr(tower.log, "warning", lambda msg, *args: warnings.append(msg % args if args else msg))
    args = ManagedComfyTower()._launch_args(None)
    assert "--use-sage-attention" not in args
    assert args[-1] == "127.0.0.1"
    assert any("sageattention" in item for item in warnings)


def test_sageattention_available_follows_find_spec(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(tower.importlib.util, "find_spec", lambda name: object() if name == "sageattention" else None)
    assert tower.sageattention_available() is True
    monkeypatch.setattr(tower.importlib.util, "find_spec", lambda name: None)
    assert tower.sageattention_available() is False
    def broken(name):
        raise ValueError(name)
    monkeypatch.setattr(tower.importlib.util, "find_spec", broken)
    assert tower.sageattention_available() is False


def test_state_accepts_comfy_mode_alias(isolated):
    path = isolated["state_dir"] / "managed_comfy.json"
    path.write_text(json.dumps({"comfy_mode": "external", "workspace": "/tmp/x"}) + "\n", encoding="utf-8")
    st = load_state()
    assert st.mode == "external"
    assert isinstance(st, TowerState)


def test_cmd_comfy_routes_lifecycle_and_keeps_run(isolated, monkeypatch, capsys, tmp_path: Path):
    from master_agent.__main__ import cmd_comfy

    isolated["probes"]["script"] = [(False, "down")]
    rc = cmd_comfy(
        Namespace(
            comfy_command="status",
            as_json=True,
            workspace=None,
            port=None,
            base_url="http://127.0.0.1:8188",
            no_wait=False,
            no_watch=False,
            extra_model_paths=None,
        )
    )
    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["comfy_mode"] == "managed"

    def boom(*_a, **_k):
        raise AssertionError("comfy run must not enter the tower")

    monkeypatch.setattr(tower, "cmd_tower", boom)
    path = tmp_path / "w.json"
    path.write_text('{"12": {"class_type": "KSampler", "inputs": {"steps": 4}}}', encoding="utf-8")

    def fake_exec(workflow, **kwargs):
        return {"status": "done", "prompt_id": "abc", "outputs": [], "nodes": 1}

    monkeypatch.setattr("master_agent.comfy.cli_run.execute_prepared", fake_exec)
    rc = cmd_comfy(
        Namespace(
            comfy_command="run",
            set=[],
            workflow_json=str(path),
            mode="raw",
            template=None,
            variant="base",
            prompt="",
            negative_prompt=None,
            video=None,
            mask=None,
            out=None,
            prepare=False,
            width=None,
            height=None,
            duration=None,
            frames=None,
            seed=None,
        )
    )
    assert rc == 0


def test_comfy_help_lists_lifecycle_without_dropping_run(capsys):
    from master_agent.__main__ import main

    with pytest.raises(SystemExit) as exc:
        main(["comfy", "--help"])
    assert exc.value.code == 0
    text = capsys.readouterr().out
    for token in (
        "run",
        "attach",
        "ingest",
        "learn",
        "dry-run",
        "--ingested",
        "--slug",
        "start",
        "stop",
        "status",
        "restart",
        "update",
        "--as-json",
        "--no-watch",
        "--write-yaml-into-external",
    ):
        assert token in text


def test_requirements_pin_comfy_cli():
    text = Path(__file__).resolve().parents[1].joinpath("requirements.txt").read_text(encoding="utf-8")
    assert "comfy-cli==1.20.0" in text


def test_cli_start_prints_ok(isolated, capsys):
    rc = cmd_tower(
        Namespace(
            comfy_command="start",
            workspace=None,
            port=None,
            base_url=None,
            no_wait=False,
            no_watch=True,
            extra_model_paths=None,
            as_json=False,
        )
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "comfy start: started" in out
    assert "127.0.0.1:8188" in out
