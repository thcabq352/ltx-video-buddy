"""gguf / colour-science and the LTX 2.3 packs' requirements in ComfyUI's own python.

GPU-free: subprocesses are mocked (one test runs the import probe in this interpreter).
Only tmp paths are written.

Run: python -m pytest tests/test_comfy_node_deps.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from master_agent import automatic_install as ai
from master_agent import setup as setup_mod
from master_agent.install_report import FAIL, INFO


def _comfy(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "main.py").write_text("")
    (root / "comfy").mkdir(exist_ok=True)
    return root


def _venv_comfy(tmp_path: Path) -> tuple[Path, Path]:
    code = _comfy(tmp_path / "ComfyUI")
    py = code / ".venv" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("")
    return code, py


class FakeComfyPython:
    """Stands in for ComfyUI's interpreter: answers the import probe and records pip calls."""

    def __init__(self, missing: list[str], *, pip_code: int = 0, pip_out: str = "", pip_fixes: bool = True):
        self.missing = list(missing)
        self.pip_code = pip_code
        self.pip_out = pip_out
        self.pip_fixes = pip_fixes
        self.probes: list[list[str]] = []
        self.pips: list[list[str]] = []
        self.requirement_texts: list[str] = []

    def __call__(self, cmd, **_k):
        cmd = list(cmd)
        if "-c" in cmd and ai._PACKAGE_PROBE in cmd:
            self.probes.append(cmd)
            return 0, "VB_MISSING=[" + ", ".join(f'"{d}"' for d in self.missing) + "]"
        if "pip" in cmd:
            self.pips.append(cmd)
            if "-r" in cmd:
                self.requirement_texts.append(Path(cmd[cmd.index("-r") + 1]).read_text())
            if self.pip_code == 0 and self.pip_fixes:
                self.missing = []
            return self.pip_code, self.pip_out
        raise AssertionError(f"unexpected command {cmd}")


@pytest.fixture
def comfy_installed(monkeypatch):
    monkeypatch.setattr(ai, "comfy_install_state", lambda ws: "installed")


def _use(monkeypatch, fake: FakeComfyPython) -> FakeComfyPython:
    monkeypatch.setattr(ai, "_run", fake)
    return fake


# --- requirements file ---------------------------------------------------------------


def test_comfy_requirements_pin_gguf_and_colour_science():
    reqs = [
        line.strip() for line in ai.COMFY_REQUIREMENTS.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert reqs == ["gguf==0.19.0", "colour-science==0.4.6"]
    assert [r.split("==")[0] for r in reqs] == [dist for dist, _mod in ai.COMFY_PACKAGES]
    assert ("colour-science", "colour") in ai.COMFY_PACKAGES


# --- interpreter choice --------------------------------------------------------------


def test_uses_comfy_venv_python_not_buddys(tmp_path, monkeypatch, comfy_installed):
    code, py = _venv_comfy(tmp_path)
    assert str(py) != sys.executable
    fake = _use(monkeypatch, FakeComfyPython(["gguf", "colour-science"]))
    res = ai.step_comfy_packages(True, workspace=code)
    assert res.status == ai.OK
    assert fake.probes and all(c[0] == str(py) for c in fake.probes)
    assert fake.pips == [[str(py), "-m", "pip", "install", "--upgrade", "-r", str(ai.COMFY_REQUIREMENTS)]]


def test_portable_uses_embedded_python_with_s_flag(tmp_path, monkeypatch, comfy_installed):
    top = tmp_path / "ComfyUI_windows_portable"
    code = _comfy(top / "ComfyUI")
    emb = top / "python_embeded" / "python.exe"
    emb.parent.mkdir(parents=True)
    emb.write_text("")
    fake = _use(monkeypatch, FakeComfyPython(["gguf"]))
    ai.step_comfy_packages(True, workspace=code)
    assert fake.probes[0][:3] == [str(emb), "-s", "-c"]
    assert fake.pips[0][:4] == [str(emb), "-s", "-m", "pip"]


def test_comfy_python_matches_what_comfy_start_launches(tmp_path):
    code, py = _venv_comfy(tmp_path)
    assert ai.comfy_python(code) == (py, [])


# --- install when missing, skip when present -----------------------------------------


def test_installs_when_missing_then_verifies(tmp_path, monkeypatch, comfy_installed):
    code, py = _venv_comfy(tmp_path)
    fake = _use(monkeypatch, FakeComfyPython(["colour-science"]))
    res = ai.step_comfy_packages(True, workspace=code)
    assert res.status == ai.OK and res.required
    assert "colour-science" in res.detail
    assert len(fake.pips) == 1
    assert len(fake.probes) == 2


def test_skipped_when_already_importable(tmp_path, monkeypatch, comfy_installed):
    code, py = _venv_comfy(tmp_path)
    fake = _use(monkeypatch, FakeComfyPython([]))
    res = ai.step_comfy_packages(True, workspace=code)
    assert res.status == ai.SKIPPED
    assert fake.pips == []


def test_probe_checks_colour_science_distribution_not_just_import(tmp_path, monkeypatch, comfy_installed):
    code, _py = _venv_comfy(tmp_path)
    fake = _use(monkeypatch, FakeComfyPython([]))
    ai.step_comfy_packages(True, workspace=code)
    assert fake.probes[0][-2:] == ["gguf:gguf", "colour-science:colour"]


def test_dry_run_plans_without_pip(tmp_path, monkeypatch, comfy_installed):
    code, _py = _venv_comfy(tmp_path)
    fake = _use(monkeypatch, FakeComfyPython(["gguf"]))
    res = ai.step_comfy_packages(False, workspace=code)
    assert res.status == ai.PLANNED and "gguf" in res.detail
    assert fake.pips == []


# --- failing loudly ------------------------------------------------------------------


def test_pip_failure_is_a_required_failure_with_pip_output(tmp_path, monkeypatch, comfy_installed):
    code, py = _venv_comfy(tmp_path)
    _use(monkeypatch, FakeComfyPython(["gguf", "colour-science"], pip_code=1, pip_out="ERROR: no network"))
    res = ai.step_comfy_packages(True, workspace=code)
    assert res.status == ai.FAILED and res.failed_required
    assert "pip could not install gguf, colour-science" in res.detail
    assert str(py) in res.detail and "no network" in res.detail
    assert any("automatic-install --yes" in line for line in res.lines)


def test_still_missing_after_pip_fails(tmp_path, monkeypatch, comfy_installed):
    code, _py = _venv_comfy(tmp_path)
    _use(monkeypatch, FakeComfyPython(["colour-science"], pip_fixes=False))
    res = ai.step_comfy_packages(True, workspace=code)
    assert res.failed_required and "still cannot be imported" in res.detail


def test_waits_for_comfy_is_required(tmp_path):
    res = ai.step_comfy_packages(True, workspace=tmp_path / "missing")
    assert res.failed_required


def test_unreadable_probe_counts_everything_missing(monkeypatch):
    monkeypatch.setattr(ai, "_run", lambda cmd, **k: (1, "Traceback ..."))
    assert ai.missing_comfy_packages(Path("py")) == ["gguf", "colour-science"]


def test_import_warnings_after_the_result_are_ignored(monkeypatch):
    out = 'VB_MISSING=[]\ncolour/utilities/verbose.py:322: ColourUsageWarning: "Matplotlib" ...\n  warn(*args)'
    monkeypatch.setattr(ai, "_run", lambda cmd, **k: (0, out))
    assert ai.missing_comfy_packages(Path("py")) == []


def test_probe_script_runs_in_a_real_interpreter():
    pairs = [("pytest", "pytest"), ("definitely-not-a-dist-vb", "json"), ("pytest", "no_such_module_vb")]
    assert ai.missing_comfy_packages(sys.executable, packages=pairs) == ["definitely-not-a-dist-vb", "pytest"]


# --- each pack's own requirements.txt ------------------------------------------------


def _packs(code: Path) -> None:
    for name, text in (
        ("ComfyUI-LTXVideo", "colour-science>=0.4.4\ntorch>=2.1\ndiffusers\n"),
        ("ComfyUI-GGUF", "gguf>=0.13.0\nsentencepiece\n"),
    ):
        d = code / "custom_nodes" / name
        d.mkdir(parents=True)
        (d / "requirements.txt").write_text(text)


def test_pack_requirements_go_into_comfy_python_with_pins_and_no_torch(tmp_path, monkeypatch, comfy_installed):
    code, py = _venv_comfy(tmp_path)
    _packs(code)
    fake = _use(monkeypatch, FakeComfyPython([]))
    res = ai.step_node_requirements(True, workspace=code)
    assert res.status == ai.OK and res.required
    assert len(fake.pips) == 2
    for cmd in fake.pips:
        assert cmd[:4] == [str(py), "-m", "pip", "install"]
        assert cmd[-2:] == ["-c", str(ai.COMFY_REQUIREMENTS)]
    ltx = fake.requirement_texts[0]
    assert "colour-science>=0.4.4" in ltx and "diffusers" in ltx and "torch" not in ltx


def test_pack_requirements_failure_names_the_pack(tmp_path, monkeypatch, comfy_installed):
    code, _py = _venv_comfy(tmp_path)
    _packs(code)
    _use(monkeypatch, FakeComfyPython([], pip_code=1, pip_out="No matching distribution"))
    res = ai.step_node_requirements(True, workspace=code)
    assert res.failed_required
    assert "ComfyUI-LTXVideo" in res.detail and "No matching distribution" in res.detail


def test_pack_requirements_dry_run_and_no_packs(tmp_path, monkeypatch, comfy_installed):
    code, _py = _venv_comfy(tmp_path)
    fake = _use(monkeypatch, FakeComfyPython([]))
    assert ai.step_node_requirements(True, workspace=code).status == ai.SKIPPED
    _packs(code)
    plan = ai.step_node_requirements(False, workspace=code)
    assert plan.status == ai.PLANNED and len(plan.lines) == 2
    assert fake.pips == []


def test_without_torch():
    text = "torch>=2.1\ntorchvision\nTorchaudio==2.4 ; sys_platform == 'win32'\neinops\ntorchsde\n"
    assert ai.without_torch(text) == "einops\ntorchsde\n"


# --- doctor --------------------------------------------------------------------------


def test_doctor_reports_missing_node_deps(tmp_path, monkeypatch):
    code, py = _venv_comfy(tmp_path)
    monkeypatch.setattr(setup_mod, "_managed_comfy_state", lambda: ("installed", code))
    monkeypatch.setattr(ai, "missing_comfy_packages", lambda p, f: ["colour-science"])
    row = setup_mod.check_comfy_node_deps()
    assert not row["ok"] and "colour-science" in row["detail"] and str(py) in row["detail"]
    assert "--automatic-install --yes" in row["fix"]
    assert setup_mod.classify_row(row) == FAIL
    assert setup_mod.classify_row(row, light=True) == INFO


def test_doctor_ok_when_present_or_comfy_absent(tmp_path, monkeypatch):
    code, _py = _venv_comfy(tmp_path)
    monkeypatch.setattr(ai, "missing_comfy_packages", lambda p, f: [])
    monkeypatch.setattr(setup_mod, "_managed_comfy_state", lambda: ("installed", code))
    assert setup_mod.check_comfy_node_deps()["ok"]
    monkeypatch.setattr(setup_mod, "_managed_comfy_state", lambda: ("absent", tmp_path / "x"))
    row = setup_mod.check_comfy_node_deps()
    assert row["ok"] and "not checked" in row["detail"]
