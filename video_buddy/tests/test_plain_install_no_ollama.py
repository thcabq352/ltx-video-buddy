"""Plain install (`install.py`, `setup --fix`) never calls Ollama; `install.py --check` only reports.

No GPU, no network. Run: python -m pytest tests/test_plain_install_no_ollama.py -q
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from master_agent import automatic_install as ai
from master_agent import config
from master_agent import setup as setup_mod

ROOT = Path(__file__).resolve().parent.parent


def _no_ollama(cmd) -> None:
    argv = cmd if isinstance(cmd, (list, tuple)) else [cmd]
    assert not any("ollama" in str(part).lower() for part in argv), cmd


@pytest.fixture
def plain(tmp_path, monkeypatch):
    """`setup --fix` with every installer stubbed and every subprocess recorded."""
    models = tmp_path / "models"
    monkeypatch.setattr(config, "MODELS_DIR", models)
    monkeypatch.setattr(ai, "models_dir", lambda: models)
    calls: list[list[str]] = []

    def fake_run(cmd, *a, **k):
        _no_ollama(cmd)
        calls.append(list(cmd))
        return 0, ""

    def fake_subprocess(cmd, *a, **k):
        _no_ollama(cmd)
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(setup_mod, "_run", fake_run)
    monkeypatch.setattr(subprocess, "run", fake_subprocess)
    monkeypatch.setattr(subprocess, "call", lambda cmd, *a, **k: fake_subprocess(cmd).returncode)
    monkeypatch.setattr(setup_mod, "_which", lambda name: f"/usr/bin/{name}")
    for name in ("ensure_venv", "install_pip", "install_playwright", "install_env", "install_ffmpeg"):
        monkeypatch.setattr(setup_mod, name, lambda: None)
    monkeypatch.setattr(setup_mod, "print_inventory_preamble", lambda: None)
    monkeypatch.setattr(setup_mod, "snapshot", lambda: [setup_mod.check_llm_models()])
    return SimpleNamespace(models=models, calls=calls)


def test_setup_fix_skips_llm_models_without_ollama(plain, capsys):
    rc = setup_mod.cmd_setup(do_fix=True, yes=True)
    out = capsys.readouterr().out
    assert rc == 0
    assert "Skipped qwen3-vl-heretic: no public GGUF source is defined; continuing." in out
    assert "Skipped nomic-embed-text: no public GGUF source is defined; continuing." in out
    assert "ollama" not in out.lower()
    assert all("ollama" not in " ".join(c).lower() for c in plain.calls)


def test_setup_fix_reports_present_gguf(plain, capsys):
    plain.models.mkdir()
    (plain.models / "qwen3-vl-heretic-Q4_K_M.gguf").write_bytes(b"x")
    rc = setup_mod.fix()
    out = capsys.readouterr().out
    assert rc == 0
    assert "qwen3-vl-heretic (text + vision: brief, director, judge): found" in out
    assert "Skipped nomic-embed-text: no public GGUF source is defined; continuing." in out
    assert "ollama" not in out.lower()


def test_llm_model_step_error_does_not_fail_fix(plain, monkeypatch, capsys):
    def boom(*_a, **_k):
        raise RuntimeError("models dir unreadable")

    monkeypatch.setattr(ai, "step_llm_models", boom)
    assert setup_mod.fix() == 0
    assert "LLM model check skipped (models dir unreadable); continuing." in capsys.readouterr().out


def test_doctor_llm_row_is_informational(plain, capsys):
    row = setup_mod.check_llm_models()
    assert row["name"] == "llm-models" and row["ok"] is False
    assert "no public GGUF source is defined" in row["detail"]
    assert "ollama" not in (row["detail"] + row["fix"]).lower()
    assert setup_mod.print_report([row]) == 0
    assert "All checked dependencies are ready." in capsys.readouterr().out


def test_snapshot_has_no_ollama_row():
    import inspect

    src = inspect.getsource(setup_mod.snapshot)
    assert "check_llm_models" in src and "ollama" not in src.lower()


# --- install.py ---------------------------------------------------------------


@pytest.fixture
def installer(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("vb_install_under_test", ROOT / "install.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    venv = tmp_path / ".venv"
    monkeypatch.setattr(mod, "VENV", venv)
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    calls: list[list[str]] = []

    def fake_call(cmd, *a, **k):
        _no_ollama(cmd)
        calls.append([str(c) for c in cmd])
        return 0

    monkeypatch.setattr(mod.subprocess, "call", fake_call)

    def make_venv() -> Path:
        py = mod._venv_python()
        py.parent.mkdir(parents=True, exist_ok=True)
        py.write_text("")
        (venv / "pyvenv.cfg").write_text("")
        return py

    return SimpleNamespace(mod=mod, calls=calls, make_venv=make_venv, venv=venv)


def _main(installer, monkeypatch, *argv: str) -> int:
    monkeypatch.setattr(installer.mod.sys, "argv", ["install.py", *argv])
    return installer.mod.main()


def test_install_check_only_reports(installer, monkeypatch, capsys):
    py = installer.make_venv()
    assert _main(installer, monkeypatch, "--check") == 0
    assert installer.calls == [[str(py), "-m", "master_agent", "setup"]]
    assert "--check" not in installer.calls[0]


def test_install_check_with_fix_still_only_reports(installer, monkeypatch, capsys):
    py = installer.make_venv()
    assert _main(installer, monkeypatch, "--fix", "--check") == 0
    assert installer.calls == [[str(py), "-m", "master_agent", "setup"]]
    assert "--check: --fix ignored. Nothing installed." in capsys.readouterr().out


def test_install_check_without_venv_changes_nothing(installer, monkeypatch, capsys):
    assert _main(installer, monkeypatch, "--check") == 1
    assert installer.calls == []
    assert not installer.venv.exists()
    assert "Nothing changed." in capsys.readouterr().out


def test_plain_install_runs_setup_fix_and_no_ollama(installer, monkeypatch):
    py = installer.make_venv()
    assert _main(installer, monkeypatch) == 0
    assert installer.calls[-1] == [str(py), "-m", "master_agent", "setup", "--fix"]
    assert all("ollama" not in " ".join(c).lower() for c in installer.calls)
