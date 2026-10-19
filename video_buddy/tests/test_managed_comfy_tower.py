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
    exit_code: int | None = None
    output = ""

    def __init__(self, cmd, **kwargs):
        self.cmd = cmd
        self.kwargs = kwargs
        self.pid = 424242
        stdout = kwargs.get("stdout")
        if self.output and hasattr(stdout, "write"):
            stdout.write(self.output)
            stdout.flush()

    def poll(self):
        return self.exit_code


def _comfy_popens(isolated) -> list[_Popen]:
    return [p for p in isolated["popens"] if "main.py" in p.cmd]


def _watchdog_popens(isolated) -> list[_Popen]:
    return [p for p in isolated["popens"] if p.cmd == watchdog_command()]


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    workspace = tmp_path / "ComfyUI"
    workspace.mkdir()
    (workspace / "main.py").write_text("", encoding="utf-8")
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

    monkeypatch.setattr(_Popen, "exit_code", None)
    monkeypatch.setattr(_Popen, "output", "")
    monkeypatch.setattr(tower.subprocess, "run", fake_run)
    monkeypatch.setattr(tower.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(tower.os, "kill", fake_kill)
    monkeypatch.setattr(tower.os, "killpg", fake_kill, raising=False)
    monkeypatch.setattr(tower.time, "sleep", lambda _s: None)
    probes = {"n": 0, "script": [(False, "down"), (True, "/system_stats ok")]}

    def fake_ready(url, timeout=5.0):
        script = probes["script"]
        item = script[min(probes["n"], len(script) - 1)]
        probes["n"] += 1
        return item

    monkeypatch.setattr(tower, "http_ready", fake_ready)
    monkeypatch.setattr(tower, "sageattention_available", lambda: True)
    owner: dict = {"info": None}
    monkeypatch.setattr(tower, "find_port_owner", lambda port: owner["info"])
    return {
        "owner": owner,
        "root": tmp_path,
        "state_dir": state_dir,
        "workspace": workspace,
        "calls": calls,
        "popens": popens,
        "kills": kills,
        "probes": probes,
    }


def _launch_tail(cmd: list[str]) -> list[str]:
    assert "main.py" in cmd
    return cmd[cmd.index("main.py") + 1 :]


def test_launch_argv_is_local_loopback_and_hides_browser(isolated):
    tower_obj = ManagedComfyTower(ready_timeout_s=1)
    result = tower_obj.start(watch=True)
    assert result["status"] == "started"
    assert result["ready"] is True
    assert isolated["calls"] == [], "start launches ComfyUI itself, not comfy-cli"
    launched = _comfy_popens(isolated)
    assert len(launched) == 1
    proc = launched[0]
    cmd = proc.cmd
    assert cmd[0] == tower.sys.executable  # no workspace .venv in this fixture
    assert proc.kwargs["cwd"] == str(isolated["workspace"])
    assert proc.kwargs.get("start_new_session") is True
    assert proc.kwargs["stderr"] is tower.subprocess.STDOUT
    assert proc.kwargs["env"]["COMFY_WHERE"] == "local"
    assert proc.kwargs["env"]["PYTHONUNBUFFERED"] == "1"
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
    watchdogs = _watchdog_popens(isolated)
    assert watchdogs, "start should spawn a detached watchdog"
    assert watchdogs[0].kwargs.get("start_new_session") is True
    saved = json.loads((isolated["state_dir"] / "managed_comfy.json").read_text(encoding="utf-8"))
    assert saved["want_running"] is True
    assert saved["comfy_mode"] == "managed"
    assert saved["watchdog_pid"] == 424242
    assert saved["extra"]["comfy_pid"] == 424242
    assert saved["workspace"] == "", "an env/default workspace is not pinned into state"


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
        (False, "down"),
        (True, "/system_stats ok"),
    ]
    result = ManagedComfyTower(ready_timeout_s=1).restart()
    assert result["status"] == "restarted"
    assert result["restart_count"] == 1
    assert any(cmd[-1] == "stop" for cmd in isolated["calls"])
    assert len(_comfy_popens(isolated)) == 1


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
    assert len(_comfy_popens(isolated)) == 1


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
        ManagedComfyTower().stop()


def test_start_without_comfyui_installed_names_the_folder(isolated):
    (isolated["workspace"] / "main.py").unlink()
    isolated["probes"]["script"] = [(False, "down")]
    with pytest.raises(TowerError) as exc:
        ManagedComfyTower().start(watch=False)
    text = str(exc.value)
    assert str(isolated["workspace"]) in text
    assert "from MANAGED_COMFY_ROOT" in text
    assert "no main.py" in text
    assert _comfy_popens(isolated) == []
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
    tail = _launch_tail(_comfy_popens(isolated)[0].cmd)
    assert tail[-3:] == [
        "--extra-model-paths-config",
        str(yaml_path),
        "--use-sage-attention",
    ]
    assert _watchdog_popens(isolated) == []


def test_launch_args_always_use_sage_attention(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(tower, "sageattention_available", lambda: True)
    tower_obj = ManagedComfyTower()
    without = tower_obj._launch_args(None)
    assert without[0] == "--disable-auto-launch"
    assert without[-1] == "--use-sage-attention"
    assert "--extra-model-paths-config" not in without

    yaml_path = tmp_path / "extra.yaml"
    yaml_path.write_text("comfyui:\n", encoding="utf-8")
    with_yaml = tower_obj._launch_args(yaml_path)
    assert with_yaml[-1] == "--use-sage-attention"
    config_at = with_yaml.index("--extra-model-paths-config")
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


TRACEBACK = (
    "Traceback (most recent call last):\n"
    '  File "main.py", line 1, in <module>\n'
    "ModuleNotFoundError: No module named 'torch'\n"
)


def test_launch_failure_surfaces_exit_code_command_and_output(isolated, monkeypatch):
    monkeypatch.setattr(_Popen, "exit_code", 1)
    monkeypatch.setattr(_Popen, "output", TRACEBACK)
    isolated["probes"]["script"] = [(False, "down")]
    with pytest.raises(tower.ComfyLaunchError) as exc:
        ManagedComfyTower(ready_timeout_s=30).start(watch=False)
    text = str(exc.value)
    assert "exited with code 1" in text
    assert "ModuleNotFoundError: No module named 'torch'" in text
    assert f"cwd:     {isolated['workspace']}" in text
    assert f"python:  {tower.sys.executable}" in text
    assert "main.py --disable-auto-launch --port 8188" in text
    assert "(from MANAGED_COMFY_ROOT)" in text
    details = exc.value.details
    assert details["exit_code"] == 1
    assert details["log"] == str(tower.comfy_log_path(8188))
    assert any("No module named 'torch'" in line for line in details["log_tail"])
    st = load_state()
    assert st.want_running is False
    assert "No module named 'torch'" in st.last_error
    assert st.extra["last_launch_failure"]["exit_code"] == 1


def test_launch_failure_with_no_output_says_so(isolated, monkeypatch):
    monkeypatch.setattr(_Popen, "exit_code", 3)
    isolated["probes"]["script"] = [(False, "down")]
    with pytest.raises(tower.ComfyLaunchError, match="ComfyUI printed nothing"):
        ManagedComfyTower().start(watch=False)


def test_no_wait_still_reports_an_instant_crash(isolated, monkeypatch):
    monkeypatch.setattr(_Popen, "exit_code", 2)
    monkeypatch.setattr(_Popen, "output", "error: unrecognized arguments: --use-sage-attention\n")
    isolated["probes"]["script"] = [(False, "down")]
    with pytest.raises(tower.ComfyLaunchError, match="unrecognized arguments"):
        ManagedComfyTower().start(wait=False, watch=False)


def test_no_wait_returns_while_comfy_is_still_loading(isolated, monkeypatch):
    monkeypatch.setattr(tower, "EARLY_EXIT_GRACE_S", 0.0)
    isolated["probes"]["script"] = [(False, "down")]
    result = ManagedComfyTower().start(wait=False, watch=False)
    assert result["status"] == "started"
    assert result["ready"] is False
    assert result["detail"] == "launched pid 424242"


def test_ready_timeout_reports_still_running_and_keeps_process(isolated, monkeypatch):
    monkeypatch.setattr(_Popen, "output", "Loading custom nodes...\n")
    isolated["probes"]["script"] = [(False, "down")]
    with pytest.raises(tower.ComfyLaunchError, match="still running") as exc:
        ManagedComfyTower(ready_timeout_s=0).start(watch=False)
    assert "Loading custom nodes" in str(exc.value)
    assert isolated["kills"] == []
    assert load_state().want_running is True


def test_cli_start_failure_prints_the_comfy_output(isolated, monkeypatch, capsys):
    monkeypatch.setattr(_Popen, "exit_code", 1)
    monkeypatch.setattr(_Popen, "output", TRACEBACK)
    isolated["probes"]["script"] = [(False, "down")]
    args = dict(
        comfy_command="start", workspace=None, port=None, base_url=None,
        no_wait=False, no_watch=True, extra_model_paths=None,
    )
    assert cmd_tower(Namespace(**args, as_json=False)) == 1
    out = capsys.readouterr().out
    assert out.startswith("FAIL  comfy start: ComfyUI exited with code 1")
    assert "No module named 'torch'" in out
    assert "Execution error" not in out

    assert cmd_tower(Namespace(**args, as_json=True)) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "error"
    assert payload["launch"]["exit_code"] == 1
    assert payload["launch"]["cwd"] == str(isolated["workspace"])
    assert any("No module named 'torch'" in line for line in payload["launch"]["log_tail"])


def test_agent_health_summary_carries_the_last_launch_failure(isolated, monkeypatch):
    from master_agent import agent_api

    monkeypatch.setattr(_Popen, "exit_code", 1)
    monkeypatch.setattr(_Popen, "output", TRACEBACK)
    isolated["probes"]["script"] = [(False, "down")]
    with pytest.raises(tower.ComfyLaunchError):
        ManagedComfyTower().start(watch=False)
    summary = agent_api._managed_comfy_summary()
    assert summary["workspace"] == str(isolated["workspace"])
    assert summary["workspace_source"] == "MANAGED_COMFY_ROOT"
    assert summary["exit_code"] == 1
    assert "No module named 'torch'" in summary["last_error"]
    assert any("No module named 'torch'" in line for line in summary["log_tail"])


def test_start_uses_the_workspace_venv_python(isolated, monkeypatch):
    monkeypatch.setattr(tower, "comfy_sageattention_available", lambda _ws: False)
    venv_py = isolated["workspace"] / ".venv" / ("Scripts/python.exe" if tower.os.name == "nt" else "bin/python")
    venv_py.parent.mkdir(parents=True)
    venv_py.write_text("", encoding="utf-8")
    ManagedComfyTower(ready_timeout_s=1).start(watch=False)
    proc = _comfy_popens(isolated)[0]
    assert proc.cmd[0] == str(venv_py)
    assert "VIRTUAL_ENV" not in proc.kwargs["env"]


def test_stop_kills_the_launched_pid_not_comfy_cli(isolated, monkeypatch):
    monkeypatch.setattr(tower, "_pid_alive", lambda pid: bool(pid))
    ManagedComfyTower(ready_timeout_s=1).start(watch=False)
    isolated["probes"]["script"] = [(False, "down")]
    isolated["probes"]["n"] = 0
    result = ManagedComfyTower().stop()
    assert result["status"] == "stopped"
    assert result["cli_detail"] == "stopped ComfyUI pid 424242"
    assert isolated["calls"] == []
    assert (424242, int(tower.signal.SIGTERM)) in isolated["kills"]
    assert "comfy_pid" not in (load_state().extra or {})


# --- agent tools: comfy_status / comfy_start / comfy_restart -----------------------


@pytest.fixture
def agent_comfy(isolated, monkeypatch):
    from master_agent import agent_api
    from master_agent.comfy import updates

    monkeypatch.setattr(updates, "startup_report", lambda ws=None: {"updated": False, "lines": []})
    return agent_api


def test_agent_comfy_tools_are_registered_with_docs(agent_comfy):
    for name in ("comfy_status", "comfy_start", "comfy_restart"):
        assert name in agent_comfy.TOOLS
        assert (agent_comfy.TOOLS[name].__doc__ or "").strip()


def test_agent_comfy_start_runs_the_cli_code_detached(agent_comfy, isolated, monkeypatch):
    seen: list[tuple[str, dict]] = []
    real = tower.run_tower_command

    def spy(command, **kw):
        seen.append((command, kw))
        return real(command, **kw)

    monkeypatch.setattr(tower, "run_tower_command", spy)
    result = agent_comfy.call("comfy_start", {"timeout_s": 30})
    assert result["status"] == "started" and result["ready"] is True
    assert result["base_url"] == "http://127.0.0.1:8188"
    assert result["workspace_source"] == "MANAGED_COMFY_ROOT"
    assert seen == [("start", {"wait": True, "ready_timeout_s": 30.0, "adopt": False})]
    (proc,) = _comfy_popens(isolated)
    assert proc.kwargs.get("start_new_session") is True or proc.kwargs.get("creationflags")
    assert load_state().want_running is True


def test_agent_comfy_start_failure_returns_the_log_tail(agent_comfy, isolated, monkeypatch):
    monkeypatch.setattr(_Popen, "exit_code", 1)
    monkeypatch.setattr(_Popen, "output", TRACEBACK)
    isolated["probes"]["script"] = [(False, "down")]
    result = agent_comfy.comfy_start()
    assert result["status"] == "error" and result["command"] == "start"
    assert "exited with code 1" in result["error"]
    launch = result["launch"]
    assert launch["exit_code"] == 1
    assert launch["cwd"] == str(isolated["workspace"])
    assert any("No module named 'torch'" in line for line in launch["log_tail"])
    assert "Do not repeat the same call" in result["next"]


def test_agent_comfy_start_still_loading_is_starting_not_error(agent_comfy, isolated, monkeypatch):
    import itertools

    ticks = itertools.count(0, 50)
    monkeypatch.setattr(tower.time, "monotonic", lambda: float(next(ticks)))
    isolated["probes"]["script"] = [(False, "down")]
    result = agent_comfy.comfy_start(timeout_s=5)
    assert result["status"] == "starting"
    assert result["launch"]["exit_code"] is None
    assert "Poll comfy_status" in result["next"] and "Do not call comfy_start again" in result["next"]
    assert isolated["kills"] == []


def test_agent_comfy_status_down_carries_last_failure(agent_comfy, isolated, monkeypatch):
    monkeypatch.setattr(_Popen, "exit_code", 1)
    monkeypatch.setattr(_Popen, "output", TRACEBACK)
    isolated["probes"]["script"] = [(False, "down")]
    agent_comfy.comfy_start()
    result = agent_comfy.comfy_status()
    assert result["status"] == "down" and result["ok"] is False
    assert result["managed"]["exit_code"] == 1
    assert any("No module named 'torch'" in line for line in result["managed"]["log_tail"])
    assert "comfy_start" in result["next"]


def test_agent_comfy_status_up(agent_comfy, isolated):
    isolated["probes"]["script"] = [(True, "/system_stats ok")]
    result = agent_comfy.comfy_status()
    assert result["status"] == "up" and result["ok"] is True
    assert "next" not in result


def test_agent_comfy_restart_external_mode_is_refused(agent_comfy, monkeypatch):
    monkeypatch.setenv("COMFY_MODE", "external")
    result = agent_comfy.comfy_restart()
    assert result["status"] == "error"
    assert "refusing to restart" in result["error"]


def test_agent_cli_comfy_start_prints_one_json_document(agent_comfy, isolated, monkeypatch, capsys):
    from master_agent.__main__ import main

    monkeypatch.setattr(_Popen, "exit_code", 1)
    monkeypatch.setattr(_Popen, "output", TRACEBACK)
    isolated["probes"]["script"] = [(False, "down")]
    rc = main(["agent", "comfy_start", "--args", '{"timeout_s": 20}'])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert payload["status"] == "error"
    assert payload["launch"]["exit_code"] == 1


# --- Windows portable builds, COMFY_EXTRA_ARGS, foreign servers ---------------------

PORTABLE_CMDLINE = [
    ".\\python_embeded\\python.exe",
    "-s",
    "ComfyUI\\main.py",
    "--windows-standalone-build",
    "--port",
    "8188",
    "--preview-method",
    "auto",
    "--use-sage-attention",
]


@pytest.fixture
def portable(isolated, monkeypatch):
    """``MANAGED_COMFY_ROOT`` = a ComfyUI_windows_portable top folder, like the tower."""
    top = isolated["root"] / "ComfyUI_windows_portable"
    (top / "python_embeded").mkdir(parents=True)
    embedded = top / "python_embeded" / "python.exe"
    embedded.write_text("", encoding="utf-8")
    code = top / "ComfyUI"
    code.mkdir()
    (code / "main.py").write_text("", encoding="utf-8")
    monkeypatch.setenv("MANAGED_COMFY_ROOT", str(top))
    monkeypatch.setattr(tower, "comfy_sageattention_available", lambda ws: True)
    monkeypatch.delenv("COMFY_EXTRA_ARGS", raising=False)
    return {"top": top.resolve(), "code": code.resolve(), "python": embedded.resolve()}


def _portable_popens(isolated) -> list:
    return [p for p in isolated["popens"] if any(str(a).endswith("main.py") for a in p.cmd)]


def test_portable_launches_like_its_bat(isolated, portable):
    result = ManagedComfyTower(ready_timeout_s=1).start(watch=False)
    assert result["status"] == "started"
    assert result["workspace"] == str(portable["code"])
    (proc,) = _portable_popens(isolated)
    cmd = proc.cmd
    assert cmd[:4] == [
        str(portable["python"]),
        "-s",
        f"ComfyUI{tower.os.sep}main.py",
        "--windows-standalone-build",
    ]
    assert cmd[4:9] == ["--disable-auto-launch", "--port", "8188", "--listen", "127.0.0.1"]
    assert cmd[-1] == "--use-sage-attention"
    assert proc.kwargs["cwd"] == str(portable["top"])
    assert "VIRTUAL_ENV" not in proc.kwargs["env"]
    failure_details = load_state().extra
    assert failure_details["comfy_command"] == cmd
    assert isolated["calls"] == [], "no comfy-cli and no install"


def test_comfy_extra_args_apply_to_portable_and_dedupe(isolated, portable, monkeypatch):
    monkeypatch.setenv("COMFY_EXTRA_ARGS", "--preview-method auto --use-sage-attention")
    ManagedComfyTower(ready_timeout_s=1).start(watch=False)
    (proc,) = _portable_popens(isolated)
    assert proc.cmd[-3:] == ["--use-sage-attention", "--preview-method", "auto"]
    assert proc.cmd.count("--use-sage-attention") == 1
    assert proc.cmd.count("--windows-standalone-build") == 1


def test_comfy_extra_args_apply_to_venv_layout(isolated, monkeypatch):
    venv_py = isolated["workspace"] / ".venv" / ("Scripts/python.exe" if tower.os.name == "nt" else "bin/python")
    venv_py.parent.mkdir(parents=True)
    venv_py.write_text("", encoding="utf-8")
    monkeypatch.setattr(tower, "comfy_sageattention_available", lambda ws: False)
    monkeypatch.setenv("COMFY_EXTRA_ARGS", "--preview-method auto --use-sage-attention")
    ManagedComfyTower(ready_timeout_s=1).start(watch=False)
    (proc,) = _comfy_popens(isolated)
    assert proc.cmd[:2] == [str(venv_py), "main.py"]
    assert "-s" not in proc.cmd and "--windows-standalone-build" not in proc.cmd
    assert proc.cmd[-3:] == ["--preview-method", "auto", "--use-sage-attention"]
    assert proc.kwargs["cwd"] == str(isolated["workspace"])


def test_bad_comfy_extra_args_fail_before_launch(isolated, portable, monkeypatch):
    monkeypatch.setenv("COMFY_EXTRA_ARGS", '--output-directory "D:\\unterminated')
    with pytest.raises(TowerError, match="COMFY_EXTRA_ARGS"):
        ManagedComfyTower(ready_timeout_s=1).start(watch=False)
    assert _portable_popens(isolated) == []


def _foreign_owner(portable, cmdline=None, cwd=None) -> dict:
    return {"pid": 7777, "cmdline": list(cmdline or PORTABLE_CMDLINE), "cwd": str(cwd or portable["top"])}


def test_status_reports_a_foreign_server_as_up_with_its_owner(isolated, portable):
    isolated["probes"]["script"] = [(True, "/system_stats ok")]
    isolated["owner"]["info"] = _foreign_owner(portable)
    st = ManagedComfyTower().status()
    assert st["ok"] is True and st["foreign"] is True
    assert st["owner"]["pid"] == 7777
    assert st["owner"]["cmdline"] == PORTABLE_CMDLINE
    assert st["owner"]["matches_workspace"] is True
    assert st["owner"]["main_py"] == str(portable["code"] / "main.py")


def test_agent_status_is_up_for_a_foreign_server(agent_comfy, isolated, portable):
    isolated["probes"]["script"] = [(True, "/system_stats ok")]
    isolated["owner"]["info"] = _foreign_owner(portable)
    result = agent_comfy.call("comfy_status", {})
    assert result["status"] == "up"
    assert result["foreign"] is True and result["owner"]["pid"] == 7777


def test_restart_refuses_foreign_without_adopt_and_says_how(isolated, portable):
    isolated["probes"]["script"] = [(True, "/system_stats ok")]
    isolated["owner"]["info"] = _foreign_owner(portable)
    with pytest.raises(tower.ForeignComfyError) as err:
        ManagedComfyTower().restart()
    msg = str(err.value)
    assert "pid 7777 runs this workspace's ComfyUI" in msg
    assert "python -m master_agent comfy restart --adopt" in msg
    assert 'comfy_restart with {"adopt": true}' in msg
    assert "COMFY_MODE=external" in msg
    assert isolated["kills"] == [] and isolated["calls"] == [] and isolated["popens"] == []
    payload = tower.tower_error_payload("restart", err.value)
    assert payload["foreign"]["owner"]["pid"] == 7777
    assert payload["foreign"]["adopt_command"] == "python -m master_agent comfy restart --adopt"


def test_restart_adopt_stops_the_matching_foreign_server_and_launches_ours(isolated, portable):
    # restart probe: foreign up; after kill: down; stop; start: down; await: up.
    isolated["probes"]["script"] = [
        (True, "/system_stats ok"),
        (False, "down"),
        (False, "down"),
        (False, "down"),
        (True, "/system_stats ok"),
    ]
    isolated["owner"]["info"] = _foreign_owner(portable)
    result = ManagedComfyTower(ready_timeout_s=1).restart(adopt=True, watch=False)
    assert result["status"] == "restarted"
    assert result["adopted"]["pid"] == 7777
    assert (7777, int(tower.signal.SIGTERM)) in isolated["kills"]
    (proc,) = _portable_popens(isolated)
    assert proc.kwargs["cwd"] == str(portable["top"])
    assert load_state().want_running is True


def test_adopt_matches_an_absolute_main_py_too(isolated, portable):
    isolated["probes"]["script"] = [(True, "ok"), (False, "down"), (False, "down"), (True, "ok")]
    isolated["owner"]["info"] = _foreign_owner(
        portable, cmdline=[str(portable["python"]), "-s", str(portable["code"] / "main.py")], cwd="/elsewhere"
    )
    result = ManagedComfyTower(ready_timeout_s=1).start(adopt=True, watch=False)
    assert result["status"] == "started" and result["adopted"]["pid"] == 7777


def test_adopt_refuses_a_different_comfyui(isolated, portable, tmp_path):
    isolated["probes"]["script"] = [(True, "/system_stats ok")]
    other = tmp_path / "OtherComfy"
    isolated["owner"]["info"] = _foreign_owner(portable, cmdline=["python", "main.py", "--port", "8188"], cwd=other)
    with pytest.raises(tower.ForeignComfyError) as err:
        ManagedComfyTower().restart(adopt=True)
    msg = str(err.value)
    assert "runs a different ComfyUI" in msg and "even with --adopt" in msg
    assert str(other / "main.py") in msg
    assert "pid 7777" in msg and ("taskkill /PID 7777" in msg or "kill 7777" in msg)
    assert isolated["kills"] == [] and isolated["popens"] == []


def test_adopt_refuses_when_the_owner_is_unknown(isolated, portable):
    isolated["probes"]["script"] = [(True, "/system_stats ok")]
    with pytest.raises(tower.ForeignComfyError, match="could not tell which process owns port 8188"):
        ManagedComfyTower().start(adopt=True)
    assert isolated["kills"] == [] and isolated["popens"] == []


def test_adopt_reports_a_server_that_will_not_die(isolated, portable, monkeypatch):
    monkeypatch.setattr(tower, "STOP_WAIT_S", 0.0)
    isolated["probes"]["script"] = [(True, "/system_stats ok")]
    isolated["owner"]["info"] = _foreign_owner(portable)
    with pytest.raises(tower.ForeignComfyError, match="still answers"):
        ManagedComfyTower().start(adopt=True)
    assert _portable_popens(isolated) == []


def test_agent_restart_passes_adopt_and_explains_foreign(agent_comfy, isolated, portable, monkeypatch):
    isolated["probes"]["script"] = [(True, "/system_stats ok")]
    isolated["owner"]["info"] = _foreign_owner(portable)
    seen: list[dict] = []
    real = tower.run_tower_command

    def spy(command, **kw):
        seen.append(kw)
        return real(command, **kw)

    monkeypatch.setattr(tower, "run_tower_command", spy)
    refused = agent_comfy.call("comfy_restart", {})
    assert refused["status"] == "foreign"
    assert 'comfy_restart with {"adopt": true}' in refused["next"]
    assert refused["foreign"]["owner"]["matches_workspace"] is True

    isolated["probes"]["n"] = 0
    isolated["probes"]["script"] = [(True, "ok"), (False, "down"), (False, "down"), (False, "down"), (True, "ok")]
    result = agent_comfy.call("comfy_restart", {"adopt": True, "timeout_s": 30})
    assert seen[-1]["adopt"] is True
    assert result["status"] == "restarted" and result["adopted"]["pid"] == 7777


def test_agent_foreign_mismatch_says_do_not_retry(agent_comfy, isolated, portable, tmp_path):
    isolated["probes"]["script"] = [(True, "/system_stats ok")]
    isolated["owner"]["info"] = _foreign_owner(portable, cmdline=["python", "main.py"], cwd=tmp_path / "x")
    result = agent_comfy.call("comfy_start", {"adopt": True})
    assert result["status"] == "foreign"
    assert "Do not retry" in result["next"]


@pytest.mark.parametrize("flag", ["--adopt", "--force"])
def test_cli_adopt_flag_reaches_the_tower(isolated, monkeypatch, flag, capsys):
    from master_agent.cli.parser import main

    seen: list[tuple[str, dict]] = []

    def fake(command, **kw):
        seen.append((command, kw))
        return {"status": "restarted", "base_url": "http://127.0.0.1:8188"}

    monkeypatch.setattr(tower, "run_tower_command", fake)
    assert main(["comfy", "restart", flag, "--as-json"]) == 0
    assert seen[-1][0] == "restart" and seen[-1][1]["adopt"] is True
    assert main(["comfy", "start", "--as-json"]) == 0
    assert seen[-1][1]["adopt"] is False


def test_cli_text_shows_the_foreign_owner(isolated, portable, capsys):
    isolated["probes"]["script"] = [(True, "/system_stats ok")]
    isolated["owner"]["info"] = _foreign_owner(portable)
    code = cmd_tower(Namespace(comfy_command="status", workspace=None, port=None, base_url=None))
    out = capsys.readouterr().out
    assert code == 0
    assert "comfy status: up (not started by Buddy)" in out
    assert "owner pid=7777" in out


def test_find_port_owner_falls_back_to_a_cmdline_scan(monkeypatch):
    import types

    class _P:
        def __init__(self, pid, cmd, cwd):
            self.pid, self._cmd, self._cwd = pid, cmd, cwd

        def cmdline(self):
            return self._cmd

        def cwd(self):
            return self._cwd

    def denied(kind="tcp"):
        raise PermissionError("needs admin")

    fake = types.SimpleNamespace(
        CONN_LISTEN="LISTEN",
        net_connections=denied,
        process_iter=lambda: [
            _P(1, ["python", "server.py"], "/a"),
            _P(2, ["python", "main.py", "--port", "8190"], "/b"),
            _P(3, PORTABLE_CMDLINE, "F:\\portable"),
        ],
        Process=lambda pid: None,
    )
    monkeypatch.setitem(tower.sys.modules, "psutil", fake)
    got = tower.find_port_owner(8188)
    assert got == {"pid": 3, "cmdline": PORTABLE_CMDLINE, "cwd": "F:\\portable"}
    assert tower.find_port_owner(8190)["pid"] == 2
    assert tower.find_port_owner(9999) is None


def test_find_port_owner_prefers_the_listening_socket(monkeypatch):
    import types

    proc = types.SimpleNamespace(pid=55, cmdline=lambda: ["python", "main.py"], cwd=lambda: "/c")
    conn = types.SimpleNamespace(pid=55, status="LISTEN", laddr=types.SimpleNamespace(port=8188))
    fake = types.SimpleNamespace(
        CONN_LISTEN="LISTEN",
        net_connections=lambda kind="tcp": [conn],
        process_iter=lambda: [],
        Process=lambda pid: proc,
    )
    monkeypatch.setitem(tower.sys.modules, "psutil", fake)
    assert tower.find_port_owner(8188) == {"pid": 55, "cmdline": ["python", "main.py"], "cwd": "/c"}
