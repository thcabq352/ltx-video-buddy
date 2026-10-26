"""Windows Defender exclusion during automatic install.

No GPU, no real PowerShell. The subprocess call is mocked.

Run: python -m pytest tests/test_windows_defender_exclusion.py -q
"""

from __future__ import annotations

import subprocess
from argparse import Namespace
from pathlib import Path

import pytest

from master_agent import automatic_install as ai
from master_agent.install_report import NONFATAL

ROOT = Path("C:/vbci/ComfyUI")
ELEVATION = ("sudo", "runas", "Start-Process", "RunAs", "-Verb")


def _exact(root: Path = ROOT) -> str:
    """The command a person pastes into an already-elevated PowerShell.

    ``Path`` follows ``os.name``, so a test that has switched to Windows gets
    the backslash form the step builds.
    """
    shown = str(Path(root)).replace("'", "''")
    return f"Add-MpPreference -ExclusionPath '{shown}'"


def _argv(root: Path = ROOT) -> list[str]:
    return ["powershell", "-NoProfile", "-NonInteractive", "-Command", _exact(root)]


def _windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ai.sys, "platform", "win32")
    monkeypatch.setattr(ai.os, "name", "nt")


def _linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ai.sys, "platform", "linux")
    monkeypatch.setattr(ai.os, "name", "posix")


def _assert_no_elevation(*parts: str) -> None:
    blob = "\n".join(parts)
    for token in ELEVATION:
        assert token not in blob, token


def _completed(code: int, out: str = "", err: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=code, stdout=out, stderr=err)


def _patch_run(monkeypatch: pytest.MonkeyPatch, fn):
    calls: list[tuple[list[str], dict]] = []

    def fake(cmd, **kwargs):
        calls.append((list(cmd), kwargs))
        return fn(cmd, **kwargs)

    monkeypatch.setattr(ai.subprocess, "run", fake)
    return calls


def test_command_quotes_the_comfy_root_and_apostrophes():
    assert ai.defender_exclusion_command(ROOT) == _exact()
    assert ai.defender_powershell_argv(ROOT) == _argv()
    odd = Path("C:/O'Brien/ComfyUI")
    command = ai.defender_exclusion_command(odd)
    assert command == _exact(odd)
    inner = command.removeprefix("Add-MpPreference -ExclusionPath '").removesuffix("'")
    assert inner.replace("''", "'") == str(odd)
    _assert_no_elevation(command, " ".join(ai.defender_powershell_argv(odd)))


def test_windows_success_calls_powershell_without_elevation(monkeypatch):
    _windows(monkeypatch)
    calls = _patch_run(monkeypatch, lambda cmd, **kwargs: _completed(0))
    res = ai.step_windows_defender_exclusion(True, workspace=ROOT)
    assert res.status == ai.OK and not res.failed_required
    assert res.detail == f"excluded {Path(ROOT)}"
    assert calls and calls[0][0] == _argv()
    assert calls[0][1]["timeout"] == ai.DEFENDER_EXCLUSION_TIMEOUT_S
    assert calls[0][1].get("shell") in (None, False)
    assert calls[0][1].get("check") is False
    _assert_no_elevation(" ".join(calls[0][0]), res.detail)


def test_access_denied_prints_needs_you_and_does_not_raise(monkeypatch, capsys):
    _windows(monkeypatch)
    err = "Add-MpPreference : Access is denied. \nCategoryInfo : PermissionDenied"
    calls = _patch_run(monkeypatch, lambda cmd, **kwargs: _completed(1, err=err))
    res = ai.step_windows_defender_exclusion(True, workspace=ROOT)
    ai._print_steps([res])
    out = capsys.readouterr().out
    assert res.status == ai.NEEDS_YOU
    assert not res.failed_required
    assert ai.classify_step(res) == NONFATAL
    assert "NEEDS-YOU" in out
    assert _exact() in out and _exact() in res.detail and _exact() in res.lines
    assert calls[0][0] == _argv()
    _assert_no_elevation(out, res.detail, " ".join(calls[0][0]))


def test_access_denied_on_exit_zero_is_still_needs_you(monkeypatch, capsys):
    """A non-terminating Defender error can leave the process exit code at 0."""
    _windows(monkeypatch)
    _patch_run(monkeypatch, lambda cmd, **kwargs: _completed(0, err="Access is denied."))
    res = ai.step_windows_defender_exclusion(True, workspace=ROOT)
    ai._print_steps([res])
    out = capsys.readouterr().out
    assert res.status == ai.NEEDS_YOU and not res.failed_required
    assert "NEEDS-YOU" in out and _exact() in out


def test_timeout_prints_failed_nonfatal_and_does_not_raise(monkeypatch, capsys):
    _windows(monkeypatch)

    def boom(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 30))

    calls = _patch_run(monkeypatch, boom)
    res = ai.step_windows_defender_exclusion(True, workspace=ROOT)
    ai._print_steps([res])
    out = capsys.readouterr().out
    assert res.status == ai.NONFATAL and not res.failed_required
    assert "FAILED-NONFATAL" in out
    assert _exact() in out and _exact() in res.detail
    assert calls[0][0] == _argv()
    _assert_no_elevation(out, " ".join(calls[0][0]))


def test_nonzero_and_oserror_are_failed_nonfatal(monkeypatch, capsys):
    _windows(monkeypatch)
    calls = _patch_run(
        monkeypatch,
        lambda cmd, **kwargs: _completed(1, err="The term 'Add-MpPreference' is not recognized"),
    )
    res = ai.step_windows_defender_exclusion(True, workspace=ROOT)
    ai._print_steps([res])
    out = capsys.readouterr().out
    assert res.status == ai.NONFATAL and not res.failed_required
    assert "FAILED-NONFATAL" in out and _exact() in out
    assert calls[0][0] == _argv()

    def missing(cmd, **kwargs):
        raise OSError("powershell not found")

    _patch_run(monkeypatch, missing)
    res = ai.step_windows_defender_exclusion(True, workspace=ROOT)
    ai._print_steps([res])
    out = capsys.readouterr().out
    assert res.status == ai.NONFATAL and "FAILED-NONFATAL" in out and _exact() in out
    _assert_no_elevation(out)


def test_non_windows_skips_without_subprocess(monkeypatch):
    _linux(monkeypatch)

    def boom(*_a, **_k):
        raise AssertionError("subprocess must not be called")

    monkeypatch.setattr(ai.subprocess, "run", boom)
    monkeypatch.setenv("COMFY_DIR", str(ROOT))
    monkeypatch.setenv("MANAGED_COMFY_ROOT", str(ROOT))
    res = ai.step_windows_defender_exclusion(True, workspace=ROOT)
    assert res.status == ai.SKIPPED and res.detail == "not Windows"
    assert res.lines == []
    # os.name alone, or sys.platform alone, still counts as Windows.
    monkeypatch.setattr(ai.os, "name", "nt")
    calls = _patch_run(monkeypatch, lambda cmd, **kwargs: _completed(0))
    assert ai.step_windows_defender_exclusion(True, workspace=ROOT).status == ai.OK
    assert calls and calls[0][0] == _argv()


def test_dry_run_on_windows_does_not_call_powershell(monkeypatch):
    _windows(monkeypatch)

    def boom(*_a, **_k):
        raise AssertionError("subprocess must not be called")

    monkeypatch.setattr(ai.subprocess, "run", boom)
    res = ai.step_windows_defender_exclusion(False, workspace=ROOT)
    assert res.status == ai.PLANNED and not res.required
    assert res.lines == [_exact()]
    _assert_no_elevation(res.detail, *res.lines)


def test_root_prefers_workspace_then_comfy_dir_then_managed(monkeypatch):
    _windows(monkeypatch)
    monkeypatch.setenv("COMFY_DIR", "D:/ci/ComfyUI")
    calls = _patch_run(monkeypatch, lambda cmd, **kwargs: _completed(0))
    ai.step_windows_defender_exclusion(True, workspace=ROOT)
    assert calls[0][0] == _argv(ROOT)

    monkeypatch.setattr(ai, "managed_workspace", lambda: pytest.fail("COMFY_DIR is set"))
    calls.clear()
    ai.step_windows_defender_exclusion(True)
    assert calls[0][0] == _argv(Path("D:/ci/ComfyUI"))

    monkeypatch.delenv("COMFY_DIR")
    managed = Path("E:/managed/ComfyUI")
    monkeypatch.setattr(ai, "managed_workspace", lambda: managed)
    calls.clear()
    ai.step_windows_defender_exclusion(True)
    assert calls[0][0] == _argv(managed)


def test_install_continues_and_prints_the_row(monkeypatch, capsys):
    """A Defender failure is a note. The install still ends Install OK."""
    _windows(monkeypatch)
    calls = _patch_run(monkeypatch, lambda cmd, **kwargs: _completed(1, err="Access is denied."))
    monkeypatch.setattr(ai, "run_preflight", lambda **k: ai.Preflight(checks=[], gpu={"vendor": "cpu"}))
    monkeypatch.setattr(ai, "print_preflight", lambda pf: None)
    monkeypatch.setattr(ai, "managed_workspace", lambda: ROOT)
    monkeypatch.setattr(ai, "verify_rows", lambda: [{"ok": True, "name": "ffmpeg", "detail": "ok"}])
    for name in (
        "step_env_file",
        "step_vram_profile",
        "step_playwright",
        "step_ffmpeg",
        "step_comfy_cli",
        "step_comfyui",
        "step_model_folders",
        "step_model_paths",
        "step_nodes",
        "step_node_requirements",
        "step_comfy_packages",
        "step_manager_config",
        "step_accel",
        "step_weights",
        "step_klein_4b",
        "step_llamacpp",
        "step_llm_models",
    ):
        monkeypatch.setattr(
            ai, name, lambda *a, _n=name, **k: ai.StepResult(_n, ai.OK, "ready", required=False)
        )
    args = Namespace(
        yes=True, preflight_only=False, dry_run=False, gpu="cpu",
        skip_weights=False, skip_sage=False, skip_llm=False,
    )
    rc = ai.cmd_automatic_install(args)
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "NEEDS-YOU" in out and "FAILED-NONFATAL" not in out.split("NEEDS-YOU", 1)[0]
    assert _exact() in out
    assert "Install OK" in out
    assert "Still needs you" not in out
    assert "safe to ignore because the install continues" in out
    assert calls and calls[0][0] == _argv()
    _assert_no_elevation(out)
