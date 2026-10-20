"""ComfyUI-Manager channel fix and the Comfy-venv gguf / colour step.

Only tmp paths are written. Subprocesses are mocked.

Run: python -m pytest tests/test_manager_config.py -q
"""

from __future__ import annotations

import configparser
from pathlib import Path

import pytest

from master_agent import automatic_install as ai
from master_agent.comfy import comfy_venv
from master_agent.comfy import manager_config as mc

LTDRDATA = "https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main"
GOOD = "https://raw.githubusercontent.com/Comfy-Org/ComfyUI-Manager/main"


def _comfy(root: Path, *, system_user_api: bool) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "main.py").write_text("")
    (root / "comfy").mkdir(exist_ok=True)
    api = "def get_system_user_directory(name):\n    pass\n" if system_user_api else ""
    (root / "folder_paths.py").write_text("def get_user_directory():\n    pass\n" + api)
    return root


def _manager_accepts(manager_dir: Path) -> bool:
    """Manager's own check: config channel_url must be a URL listed in channels.list."""
    cp = configparser.ConfigParser(strict=False)
    cp.read(manager_dir / "config.ini")
    channel_url = cp["default"].get("channel_url", LTDRDATA)
    valid = {"default", "local"}
    for line in (manager_dir / "channels.list").read_text().split("\n"):
        parts = line.split("::")
        if len(parts) == 2:
            valid.add(parts[1])
    return channel_url in valid


def test_verified_channel_is_comfy_org_with_hyphen():
    assert mc.CHANNEL_URL == GOOD
    assert "ComfyOrg/" not in mc.CHANNEL_URL
    assert LTDRDATA in mc.BROKEN_CHANNEL_URLS


@pytest.mark.parametrize("api,sub", [(True, ("user", "__manager")), (False, ("user", "default", "ComfyUI-Manager"))])
def test_fresh_install_seeds_where_manager_reads(tmp_path, api, sub):
    code = _comfy(tmp_path / "ComfyUI", system_user_api=api)
    report = mc.ensure_manager_config(code)
    d = code.joinpath(*sub)
    other = code.joinpath(*(("user", "default", "ComfyUI-Manager") if api else ("user", "__manager")))
    assert {a.action for a in report.actions} == {mc.CREATED}
    cp = configparser.ConfigParser(strict=False)
    cp.read(d / "config.ini")
    assert cp["default"]["channel_url"] == GOOD
    assert "auto-updates" not in cp["default"] and "channel" not in cp["default"]
    assert (d / "channels.list").read_text().splitlines()[0] == f"default::{GOOD}"
    assert _manager_accepts(d)
    assert not other.exists()


def test_broken_config_is_repaired_and_other_keys_kept(tmp_path):
    code = _comfy(tmp_path / "ComfyUI", system_user_api=True)
    d = code / "user" / "__manager"
    d.mkdir(parents=True)
    original = (
        "[default]\n"
        "preview_method = auto\n"
        f"channel_url = {LTDRDATA}\n"
        "security_level = normal\n"
        "\n"
        "[extra]\n"
        "channel_url = keep-me\n"
    )
    (d / "config.ini").write_text(original)
    (d / "channels.list").write_text(f"default::{LTDRDATA}\nrecent::{LTDRDATA}/node_db/new\n")
    report = mc.ensure_manager_config(code)
    assert {a.action for a in report.actions} == {mc.FIXED}
    text = (d / "config.ini").read_text()
    assert text == original.replace(f"channel_url = {LTDRDATA}", f"channel_url = {GOOD}")
    assert (d / "channels.list").read_text() == f"default::{GOOD}\nrecent::{LTDRDATA}/node_db/new\n"
    assert _manager_accepts(d)


def test_idempotent_second_run_changes_nothing(tmp_path):
    code = _comfy(tmp_path / "ComfyUI", system_user_api=True)
    d = code / "user" / "__manager"
    d.mkdir(parents=True)
    (d / "config.ini").write_text(f"[default]\nchannel_url = {LTDRDATA}/\n")
    mc.ensure_manager_config(code)
    before = {p.name: p.read_bytes() for p in d.iterdir()}
    again = mc.ensure_manager_config(code)
    assert again.changed == []
    assert {p.name: p.read_bytes() for p in d.iterdir()} == before


def test_custom_channel_url_is_left_alone_but_channel_listed(tmp_path):
    code = _comfy(tmp_path / "ComfyUI", system_user_api=True)
    d = code / "user" / "__manager"
    d.mkdir(parents=True)
    (d / "config.ini").write_text("[default]\nchannel_url = https://example.com/mine\n")
    (d / "channels.list").write_text("default::https://example.com/mine\n")
    mc.ensure_manager_config(code)
    assert "https://example.com/mine" in (d / "config.ini").read_text()
    assert (d / "channels.list").read_text() == f"default::https://example.com/mine\ncomfy-org::{GOOD}\n"


def test_missing_channel_url_or_section_is_added():
    text, detail = mc.fix_config_text("[default]\nuse_uv = true\n")
    assert text == f"[default]\nchannel_url = {GOOD}\nuse_uv = true\n" and detail
    text, detail = mc.fix_config_text("[other]\nx = 1\n")
    assert text.startswith(f"[default]\nchannel_url = {GOOD}\n\n[other]") and detail
    text, _ = mc.fix_config_text(f"[default]\nchannel_url={LTDRDATA}\n")
    assert text == f"[default]\nchannel_url={GOOD}\n"
    text, _ = mc.fix_config_text("[default]\nchannel_url = https://raw.githubusercontent.com/ComfyOrg/ComfyUI-Manager/main\n")
    assert text == f"[default]\nchannel_url = {GOOD}\n"


def test_legacy_config_is_repaired_without_creating_a_second_config(tmp_path):
    """New ComfyUI + leftover legacy config: Manager migrates it, so only fix it in place."""
    code = _comfy(tmp_path / "ComfyUI", system_user_api=True)
    legacy = code / "user" / "default" / "ComfyUI-Manager"
    legacy.mkdir(parents=True)
    (legacy / "config.ini").write_text(f"[default]\nchannel_url = {LTDRDATA}\n")
    mc.ensure_manager_config(code)
    assert (legacy / "config.ini").read_text() == f"[default]\nchannel_url = {GOOD}\n"
    new = code / "user" / "__manager"
    assert not (new / "config.ini").exists()
    assert (new / "channels.list").read_text().startswith(f"default::{GOOD}\n")


def test_dry_run_writes_nothing(tmp_path):
    code = _comfy(tmp_path / "ComfyUI", system_user_api=True)
    report = mc.ensure_manager_config(code, apply=False)
    assert [a.action for a in report.actions] == [mc.CREATED, mc.CREATED]
    assert not (code / "user").exists()


# --- automatic-install steps ---------------------------------------------------


@pytest.fixture
def installed(tmp_path, monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(ai, "_run", lambda cmd, **k: calls.append(list(cmd)) or (0, ""))
    monkeypatch.setattr(ai, "comfy_install_state", lambda ws: "installed")
    return calls


def test_step_manager_config_fixes_then_skips(tmp_path, installed):
    code = _comfy(tmp_path / "ComfyUI", system_user_api=False)
    d = code / "user" / "default" / "ComfyUI-Manager"
    d.mkdir(parents=True)
    (d / "config.ini").write_text(f"[default]\nchannel_url = {LTDRDATA}\n")
    plan = ai.step_manager_config(False, workspace=code)
    assert plan.status == ai.PLANNED and LTDRDATA in (d / "config.ini").read_text()
    first = ai.step_manager_config(True, workspace=code)
    assert first.status == ai.OK and not first.required
    assert any("config.ini" in line for line in first.lines)
    assert ai.step_manager_config(True, workspace=code).status == ai.SKIPPED
    assert _manager_accepts(d)
    assert installed == []


def test_step_manager_config_waits_for_comfy(tmp_path):
    res = ai.step_manager_config(True, workspace=tmp_path / "missing")
    assert res.status == ai.NONFATAL and not res.required


def test_comfy_requirements_file_lists_gguf_and_colour():
    reqs = [
        line.strip() for line in ai.COMFY_REQUIREMENTS.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert reqs == ["gguf", "colour"]


def test_step_comfy_packages_upgrades_into_comfy_venv(tmp_path, installed):
    code = _comfy(tmp_path / "ComfyUI", system_user_api=False)
    py = code / ".venv" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("")
    res = ai.step_comfy_packages(True, workspace=code)
    assert res.status == ai.OK
    assert installed == [[str(py), "-m", "pip", "install", "--upgrade", "-r", str(ai.COMFY_REQUIREMENTS)]]


def test_step_comfy_packages_portable_uses_embedded_python(tmp_path, installed):
    top = tmp_path / "ComfyUI_windows_portable"
    code = _comfy(top / "ComfyUI", system_user_api=True)
    emb = top / "python_embeded" / "python.exe"
    emb.parent.mkdir(parents=True)
    emb.write_text("")
    ai.step_comfy_packages(True, workspace=code)
    assert installed[0][:4] == [str(emb), "-s", "-m", "pip"]


def test_step_comfy_packages_pip_failure_is_nonfatal(tmp_path, monkeypatch, installed):
    code = _comfy(tmp_path / "ComfyUI", system_user_api=False)
    (code / ".venv" / "bin").mkdir(parents=True)
    (code / ".venv" / "bin" / "python").write_text("")
    monkeypatch.setattr(ai, "_run", lambda cmd, **k: (1, "no network"))
    res = ai.step_comfy_packages(True, workspace=code)
    assert res.status == ai.NONFATAL and not res.required and "no network" in res.detail


def test_dry_run_plan_lists_both_new_steps(tmp_path, monkeypatch, capsys):
    """The full plan includes the two steps after custom nodes, before Triton."""
    seen: list[str] = []
    for name in ("step_env_file", "step_playwright", "step_ffmpeg", "step_comfy_cli", "step_model_folders",
                 "step_model_paths", "step_weights", "step_llm_models"):
        monkeypatch.setattr(ai, name, lambda *a, _n=name, **k: seen.append(_n) or ai.StepResult(_n, ai.SKIPPED, ""))
    for name in ("step_comfyui", "step_nodes", "step_comfy_packages", "step_manager_config", "step_accel",
                 "step_llamacpp"):
        monkeypatch.setattr(ai, name, lambda *a, _n=name, **k: seen.append(_n) or ai.StepResult(_n, ai.SKIPPED, ""))
    monkeypatch.setattr(ai, "run_preflight", lambda **k: ai.Preflight(checks=[], gpu={"vendor": "cpu"}))
    monkeypatch.setattr(ai, "print_preflight", lambda pf: None)
    monkeypatch.setattr(ai, "managed_workspace", lambda: tmp_path / "ComfyUI")
    monkeypatch.setattr(comfy_venv, "probe_comfy_env", lambda *a, **k: comfy_venv.ComfyEnv())
    from argparse import Namespace

    assert ai.cmd_automatic_install(Namespace(yes=False, gpu="cpu")) == 0
    i = seen.index
    assert i("step_nodes") < i("step_comfy_packages") < i("step_manager_config") < i("step_accel")
