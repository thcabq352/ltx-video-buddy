"""ComfyUI Windows portable builds: layout detection, root resolution, install guard.

Windows trees are built under tmp_path (``python_embeded/python.exe`` is an
empty file). No live Comfy, no GPU, no subprocess.

Run: PYTHONPATH=video_buddy python -m pytest tests/test_comfy_portable.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from master_agent.comfy import layout as layout_mod
from master_agent.comfy import tower
from master_agent.comfy.layout import (
    PLAIN,
    PORTABLE,
    VENV,
    code_dir_for,
    comfy_interpreter,
    detect_layout,
    discover_portable,
)
from master_agent.comfy.root import (
    SOURCE_DEFAULT,
    SOURCE_ENV,
    SOURCE_PORTABLE,
    SOURCE_STATE,
    resolve_managed_comfy_root,
)


def make_portable(top: Path) -> Path:
    """``top`` laid out like ComfyUI_windows_portable. Returns the inner ComfyUI folder."""
    (top / "python_embeded").mkdir(parents=True)
    (top / "python_embeded" / "python.exe").write_text("", encoding="utf-8")
    code = top / "ComfyUI"
    (code / "comfy").mkdir(parents=True)
    (code / "main.py").write_text("# comfy\n", encoding="utf-8")
    (top / "run_api_8188.bat").write_text(
        ".\\python_embeded\\python.exe -s ComfyUI\\main.py --windows-standalone-build --port 8188\n",
        encoding="utf-8",
    )
    return code


def make_venv_comfy(code: Path) -> Path:
    (code / "comfy").mkdir(parents=True)
    (code / "main.py").write_text("# comfy\n", encoding="utf-8")
    py = code / ".venv" / ("Scripts/python.exe" if tower.os.name == "nt" else "bin/python")
    py.parent.mkdir(parents=True)
    py.write_text("", encoding="utf-8")
    return py


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    vb = tmp_path / "video_buddy"
    state = vb / "state"
    state.mkdir(parents=True)
    monkeypatch.setattr("master_agent.config.PROJECT_ROOT", vb)
    monkeypatch.setattr("master_agent.config.STATE_DIR", state)
    monkeypatch.setattr("master_agent.config.PORTABLE_ROOT", vb / "ComfyUI_windows_portable")
    monkeypatch.setattr(tower, "STATE_DIR", state)
    monkeypatch.setattr(tower, "PROJECT_ROOT", vb)
    monkeypatch.delenv("MANAGED_COMFY_ROOT", raising=False)
    monkeypatch.delenv("COMFY_MODE", raising=False)
    return {"vb": vb, "state": state, "portable": (vb / "ComfyUI_windows_portable").resolve()}


def _save_workspace(state: Path, workspace: Path | str) -> None:
    (state / "managed_comfy.json").write_text(json.dumps({"workspace": str(workspace)}) + "\n", encoding="utf-8")


# --- layout -------------------------------------------------------------------


def test_portable_top_and_inner_folder_are_the_same_layout(tmp_path):
    top = tmp_path / "ComfyUI_windows_portable"
    code = make_portable(top)
    for given in (top, code):
        got = detect_layout(given)
        assert got is not None and got.kind == PORTABLE
        assert got.code_dir == code
        assert got.top == top
        assert got.cwd == top
        assert got.python == top / "python_embeded" / "python.exe"
        assert got.python_flags == ("-s",)
        assert got.comfy_flags == ("--windows-standalone-build",)
    assert code_dir_for(top) == code
    assert comfy_interpreter(top) == top / "python_embeded" / "python.exe"


def test_portable_main_script_uses_windows_separator(tmp_path, monkeypatch):
    make_portable(tmp_path / "P")
    monkeypatch.setattr(layout_mod.os, "sep", "\\")
    assert detect_layout(tmp_path / "P").main_script == "ComfyUI\\main.py"


def test_embedded_python_inside_the_code_folder_also_counts(tmp_path):
    code = tmp_path / "ComfyUI"
    code.mkdir()
    (code / "main.py").write_text("", encoding="utf-8")
    (code / "python_embeded").mkdir()
    (code / "python_embeded" / "python.exe").write_text("", encoding="utf-8")
    got = detect_layout(code)
    assert got.kind == PORTABLE and got.top == code and got.main_script == "main.py"


def test_venv_and_plain_layouts_unchanged(tmp_path):
    venv_code = tmp_path / "venv-comfy"
    py = make_venv_comfy(venv_code)
    got = detect_layout(venv_code)
    assert (got.kind, got.python, got.cwd, got.main_script) == (VENV, py, venv_code, "main.py")
    assert got.python_flags == () and got.comfy_flags == ()

    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "main.py").write_text("", encoding="utf-8")
    assert detect_layout(plain).kind == PLAIN
    assert detect_layout(plain).python is None
    assert detect_layout(tmp_path / "missing") is None


def test_discover_portable_skips_non_portable_roots(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "main.py").write_text("", encoding="utf-8")
    code = make_portable(tmp_path / "ComfyUI_windows_portable_nvidia")
    got = discover_portable([tmp_path / "nothing", plain, tmp_path / "ComfyUI_windows_portable_nvidia"])
    assert got is not None and got.code_dir == code


def test_real_search_globs_project_root(project, real_portable_search, monkeypatch):
    monkeypatch.setattr("master_agent.config.PORTABLE_ROOT", project["vb"] / "nope")
    code = make_portable(project["vb"] / "ComfyUI_windows_portable_cu128")
    assert discover_portable().code_dir == code


# --- resolver -----------------------------------------------------------------


def test_env_may_name_the_portable_top_folder(project, monkeypatch, tmp_path):
    top = tmp_path / "F" / "ComfyUI_windows_portable"
    code = make_portable(top)
    monkeypatch.setenv("MANAGED_COMFY_ROOT", str(top))
    got = resolve_managed_comfy_root()
    assert (got.path, got.source) == (code.resolve(), SOURCE_ENV)
    monkeypatch.setenv("MANAGED_COMFY_ROOT", str(code))
    assert resolve_managed_comfy_root().path == code.resolve()


def test_auto_detects_portable_when_default_is_missing(project, real_portable_search):
    code = make_portable(project["portable"])
    got = resolve_managed_comfy_root()
    assert (got.path, got.source) == (code, SOURCE_PORTABLE)
    assert got.describe() == f"{code} (auto-detected portable)"
    assert got.notes == ()


def test_default_with_main_py_beats_a_portable(project, real_portable_search):
    make_portable(project["portable"])
    default = project["vb"] / "ComfyUI"
    default.mkdir()
    (default / "main.py").write_text("", encoding="utf-8")
    assert resolve_managed_comfy_root().source == SOURCE_DEFAULT


def test_stale_saved_state_is_ignored_with_a_note(project, real_portable_search, tmp_path):
    # The tower report: state points at video_buddy\ComfyUI, which does not exist.
    gone = project["vb"] / "ComfyUI"
    _save_workspace(project["state"], gone)
    code = make_portable(project["portable"])
    got = resolve_managed_comfy_root()
    assert (got.path, got.source) == (code, SOURCE_PORTABLE)
    assert got.stale_saved == gone.resolve()
    assert any("ignored stale state/managed_comfy.json" in n and "no main.py" in n for n in got.notes)


def test_saved_state_with_main_py_still_wins(project, tmp_path):
    saved = tmp_path / "saved"
    saved.mkdir()
    (saved / "main.py").write_text("", encoding="utf-8")
    _save_workspace(project["state"], saved)
    got = resolve_managed_comfy_root()
    assert (got.path, got.source, got.notes) == (saved.resolve(), SOURCE_STATE, ())


def test_auto_detected_root_is_never_saved_and_stale_one_is_cleared(project, real_portable_search):
    _save_workspace(project["state"], project["vb"] / "ComfyUI")
    make_portable(project["portable"])
    t = tower.ManagedComfyTower()
    assert t.workspace_source == SOURCE_PORTABLE
    t._sync_state(last_error="")
    saved = json.loads((project["state"] / "managed_comfy.json").read_text(encoding="utf-8"))
    assert saved["workspace"] == ""
    assert tower.ManagedComfyTower().notes == []


def test_status_reports_notes_and_layout(project, real_portable_search, monkeypatch):
    _save_workspace(project["state"], project["vb"] / "ComfyUI")
    make_portable(project["portable"])
    monkeypatch.setattr(tower, "http_ready", lambda url, timeout=5.0: (False, "down"))
    st = tower.ManagedComfyTower().status()
    assert st["layout"] == PORTABLE
    assert st["workspace_source"] == SOURCE_PORTABLE
    assert st["notes"] and "ignored stale" in st["notes"][0]
    assert st["foreign"] is False and st["owner"] is None


def test_missing_workspace_error_points_at_the_portable(project, real_portable_search, monkeypatch, tmp_path):
    make_portable(project["portable"])
    monkeypatch.setenv("MANAGED_COMFY_ROOT", str(tmp_path / "typo"))
    with pytest.raises(tower.TowerError) as err:
        tower.ManagedComfyTower()._comfy_python()
    msg = str(err.value)
    assert "no main.py there" in msg
    assert f"set MANAGED_COMFY_ROOT={project['portable']}" in msg


# --- comfy interpreter helpers ---------------------------------------------------


def test_probe_and_venv_strip_use_the_embedded_python(tmp_path):
    from master_agent.comfy.comfy_venv import probe_comfy_env, strip_outer_venv

    top = tmp_path / "ComfyUI_windows_portable"
    code = make_portable(top)

    def runner(cmd, **_kw):
        raise OSError("not a real exe")

    env = probe_comfy_env(code, runner=runner)
    assert env.python_path == str(top / "python_embeded" / "python.exe")
    assert env.source == "portable-embedded"
    out = strip_outer_venv({"VIRTUAL_ENV": "/buddy/.venv", "PATH": "x"}, code)
    assert "VIRTUAL_ENV" not in out and out["PATH"] == "x"


def test_sageattention_probe_runs_in_the_embedded_python(tmp_path, monkeypatch):
    from master_agent.comfy import comfy_venv

    top = tmp_path / "ComfyUI_windows_portable"
    code = make_portable(top)
    seen: list[str] = []
    monkeypatch.setattr(comfy_venv, "module_importable", lambda py, mod: seen.append(str(py)) or True)
    assert tower.comfy_sageattention_available(code) is True
    assert seen == [str(top / "python_embeded" / "python.exe")]


# --- COMFY_EXTRA_ARGS -----------------------------------------------------------


def test_extra_args_default_is_empty(monkeypatch):
    monkeypatch.delenv("COMFY_EXTRA_ARGS", raising=False)
    assert tower.comfy_extra_args() == []
    assert tower.comfy_extra_args("   ") == []


def test_extra_args_split_like_a_command_line():
    assert tower.comfy_extra_args("--preview-method auto --use-sage-attention") == [
        "--preview-method",
        "auto",
        "--use-sage-attention",
    ]
    assert tower.comfy_extra_args('--output-directory "D:\\Video Out" --lowvram') == [
        "--output-directory",
        "D:\\Video Out",
        "--lowvram",
    ]


def test_merge_args_drops_repeated_switches_only():
    base = ["--disable-auto-launch", "--port", "8188", "--use-sage-attention"]
    merged = tower._merge_args(base, ["--use-sage-attention", "--preview-method", "auto", "--port", "8190"])
    assert merged == base + ["--preview-method", "auto", "--port", "8190"]


# --- automatic install never touches a portable ---------------------------------------


@pytest.fixture
def no_install(monkeypatch):
    from master_agent import automatic_install as ai

    def boom(*_a, **_kw):
        raise AssertionError("automatic install must not run comfy install here")

    monkeypatch.setattr(ai, "_run", boom)
    return ai


def test_portable_with_torch_is_installed_and_skipped(tmp_path, monkeypatch, no_install):
    from master_agent.comfy import comfy_venv

    code = make_portable(tmp_path / "ComfyUI_windows_portable")
    probed: list[str] = []
    monkeypatch.setattr(comfy_venv, "module_importable", lambda py, mod: probed.append(str(py)) or True)
    assert no_install.comfy_install_state(code) == "installed"
    assert probed and probed[0].endswith("python.exe")
    res = no_install.step_comfyui(True, workspace=code, state="installed", gpu_flag="--nvidia")
    assert res.status == no_install.SKIPPED


def test_portable_without_torch_is_never_repaired_by_comfy_install(tmp_path, monkeypatch, no_install):
    from master_agent.comfy import comfy_venv

    code = make_portable(tmp_path / "ComfyUI_windows_portable")
    monkeypatch.setattr(comfy_venv, "module_importable", lambda py, mod: False)
    state = no_install.comfy_install_state(code)
    assert state == "partial"
    for apply in (False, True):
        res = no_install.step_comfyui(apply, workspace=code, state=state, gpu_flag="--nvidia")
        assert res.status == no_install.FAILED
        assert "never installs over a portable build" in res.detail
    check = no_install.check_workspace(code, state=state, mode="managed")
    assert check.status == no_install.FAIL and check.blocking


def test_no_second_comfyui_beside_a_portable(project, real_portable_search, monkeypatch, tmp_path, no_install):
    make_portable(project["portable"])
    empty = tmp_path / "E" / "ComfyUI"
    monkeypatch.setenv("MANAGED_COMFY_ROOT", str(empty))
    assert no_install.managed_workspace() == empty.resolve()
    res = no_install.step_comfyui(True, workspace=empty.resolve(), state="absent", gpu_flag="--nvidia")
    assert res.status == no_install.FAILED
    assert f"set MANAGED_COMFY_ROOT={project['portable']}" in res.detail
    assert not empty.exists()
    check = no_install.check_workspace(empty.resolve(), state="absent", mode="managed")
    assert check.status == no_install.FAIL
    assert any("will not install a second ComfyUI" in line for line in check.lines)
