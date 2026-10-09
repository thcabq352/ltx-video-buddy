"""Comfy venv probe, Triton + SageAttention into the Comfy venv, launch flag.

Subprocess and network are mocked. Only tmp paths.

Run: python -m pytest tests/test_comfy_venv.py -q
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from master_agent.comfy import comfy_venv as cv
from master_agent.comfy import tower
from master_agent.comfy.tower import ManagedComfyTower


def _workspace(tmp_path: Path) -> tuple[Path, Path]:
    ws = tmp_path / "ComfyUI"
    py = ws / ".venv" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("")
    return ws, py


class FakeComfyPython:
    """Stands in for ``<comfy-venv>/bin/python``: answers the probe, pip and imports."""

    def __init__(self, *, system="Linux", machine="x86_64", torch="2.9.1+cu128", cuda="12.8",
                 triton_req="triton==3.5.1", pip_fail=(), import_fail=(), have=()):
        self.info = {
            "python": "3.12.3", "system": system, "machine": machine, "torch": torch, "cuda": cuda,
            "hip": None, "cuda_available": bool(cuda), "torch_triton_requirement": triton_req,
            "triton": "3.5.1" if "triton" in have else None,
            "sageattention": "2.2.0" if "sageattention" in have else None, "errors": {},
        }
        self.installed = set(have)
        self.pip_fail = set(pip_fail)
        self.import_fail = set(import_fail)
        self.calls: list[list[str]] = []

    def __call__(self, cmd, **kw):
        cmd = [str(c) for c in cmd]
        self.calls.append(cmd)
        if cmd[1] == "-c" and cmd[2].startswith("import "):
            mod = cmd[2].split()[1]
            ok = mod in self.installed and mod not in self.import_fail
            return subprocess.CompletedProcess(cmd, 0 if ok else 1, "", "")
        if cmd[1] == "-c":
            return subprocess.CompletedProcess(cmd, 0, json.dumps(self.info) + "\n", "")
        if cmd[1:4] == ["-m", "pip", "install"]:
            spec = cmd[4]
            if spec in self.pip_fail:
                return subprocess.CompletedProcess(cmd, 1, "", "ERROR: no matching distribution")
            self.installed.add("triton" if "triton" in spec else "sageattention")
            return subprocess.CompletedProcess(cmd, 0, "Successfully installed", "")
        raise AssertionError(cmd)

    def pips(self) -> list[str]:
        return [c[4] for c in self.calls if c[1:4] == ["-m", "pip", "install"]]


def _install(ws, fake, *, vendor="nvidia", apply=True, assets=(), host_system="Linux", host_machine="x86_64"):
    return cv.install_accel(
        ws, gpu_vendor=vendor, apply=apply, runner=fake, sage_assets=lambda: list(assets),
        progress=lambda msg: None, host_system=host_system, host_machine=host_machine,
    )


# --- interpreter discovery ------------------------------------------------------


def test_workspace_venv_python_and_strip_outer_venv(tmp_path):
    assert cv.workspace_venv_python(tmp_path / "none") is None
    env = {"VIRTUAL_ENV": "/buddy/.venv", "CONDA_PREFIX": "/conda", "PATH": "x"}
    assert cv.strip_outer_venv(env, tmp_path / "none") == env
    ws, py = _workspace(tmp_path)
    assert cv.workspace_venv_python(ws) == py
    assert cv.strip_outer_venv(env, ws) == {"PATH": "x"}


def test_probe_runs_inside_comfy_python_not_buddy(tmp_path):
    ws, py = _workspace(tmp_path)
    fake = FakeComfyPython()
    env = cv.probe_comfy_env(ws, runner=fake)
    assert fake.calls[0][0] == str(py)
    assert env.torch == "2.9.1+cu128" and env.cuda == "12.8" and env.python == "3.12.3"
    assert "Comfy Python 3.12.3" in env.summary() and "CUDA 12.8" in env.summary()


def test_probe_without_venv_reports_unknown(tmp_path):
    env = cv.probe_comfy_env(tmp_path / "ComfyUI", runner=lambda *a, **k: pytest.fail("no subprocess"))
    assert env.python_path is None and "non-venv" in env.summary()


# --- plan / version mapping -------------------------------------------------------


@pytest.mark.parametrize(
    "torch_version,spec,verified",
    [
        ("2.5.1+cu124", "triton-windows>=3.1,<3.2", True),
        ("2.7.0+cu128", "triton-windows>=3.3,<3.4", True),
        ("2.9.1+cu128", "triton-windows>=3.5,<3.6", True),
        ("2.10.0+cu130", "triton-windows>=3.6,<3.7", True),
        ("2.11.0+cu130", "triton-windows>=3.7,<3.8", False),
        ("2.3.1", None, False),
        (None, None, False),
    ],
)
def test_triton_windows_spec(torch_version, spec, verified):
    assert cv.triton_windows_spec(torch_version) == (spec, verified)


def test_triton_linux_spec_follows_torch_pin():
    assert cv.triton_linux_spec(cv.ComfyEnv(torch_triton_requirement="triton==3.5.1")) == "triton==3.5.1"
    assert cv.triton_linux_spec(cv.ComfyEnv()) == "triton"


ASSETS = [
    {"name": "sageattention-2.2.0+cu128torch2.9.1.post6-cp310-abi3-win_amd64.whl", "browser_download_url": "https://x/291.whl"},
    {"name": "sageattention-2.2.0+cu130torch2.10.0andhigher.post6-cp310-abi3-win_amd64.whl", "browser_download_url": "https://x/210plus.whl"},
    {"name": "sageattention-2.2.0+rocm.post7-cp310-abi3-win_amd64.whl", "browser_download_url": "https://x/rocm.whl"},
]


def test_pick_windows_sage_wheel():
    assert cv.pick_windows_sage_wheel(ASSETS, torch_version="2.9.1+cu128", cuda="12.8") == "https://x/291.whl"
    assert cv.pick_windows_sage_wheel(ASSETS, torch_version="2.11.0+cu130", cuda="13.0") == "https://x/210plus.whl"
    assert cv.pick_windows_sage_wheel(ASSETS, torch_version="2.8.0+cu128", cuda="12.8") is None
    assert cv.pick_windows_sage_wheel(ASSETS, torch_version="2.9.1", cuda=None) is None


# --- install order / per-platform commands -------------------------------------------


def test_linux_installs_triton_then_sage_into_comfy_venv(tmp_path):
    ws, py = _workspace(tmp_path)
    fake = FakeComfyPython()
    report = _install(ws, fake)
    assert fake.pips() == ["triton==3.5.1", "sageattention"]
    pip_calls = [c for c in fake.calls if c[1:3] == ["-m", "pip"]]
    assert all(c[0] == str(py) for c in pip_calls)
    imports = [c[2] for c in fake.calls if c[1] == "-c" and c[2].startswith("import ")]
    assert imports == ["import triton", "import sageattention"]
    assert report.status == "OK" and report.triton_present and report.sage_present


def test_windows_triton_windows_then_prebuilt_wheel(tmp_path):
    ws, _ = _workspace(tmp_path)
    fake = FakeComfyPython(system="Windows", machine="AMD64", triton_req=None)
    report = _install(ws, fake, assets=ASSETS, host_system="Windows", host_machine="AMD64")
    assert fake.pips() == ["triton-windows>=3.5,<3.6", "https://x/291.whl"]
    assert report.status == "OK"


def test_windows_wheel_failure_falls_back_to_pypi_then_nonfatal(tmp_path):
    ws, _ = _workspace(tmp_path)
    fake = FakeComfyPython(system="Windows", machine="AMD64", pip_fail={"https://x/291.whl"})
    report = _install(ws, fake, assets=ASSETS, host_system="Windows", host_machine="AMD64")
    assert fake.pips() == ["triton-windows>=3.5,<3.6", "https://x/291.whl", "sageattention"]
    assert report.status == "OK"

    fake = FakeComfyPython(system="Windows", machine="AMD64", import_fail={"sageattention"})
    report = _install(ws, fake, assets=ASSETS, host_system="Windows", host_machine="AMD64")
    assert fake.pips() == ["triton-windows>=3.5,<3.6", "https://x/291.whl", "sageattention"]
    assert report.status == "FAILED-NONFATAL"
    assert report.triton_present and not report.sage_present
    text = "\n".join(report.lines())
    assert "just slower" in text and "without --use-sage-attention" in text


def test_windows_no_wheel_listing_uses_pypi(tmp_path):
    ws, _ = _workspace(tmp_path)
    fake = FakeComfyPython(system="Windows", machine="AMD64")

    def offline():
        raise OSError("offline")

    report = cv.install_accel(ws, gpu_vendor="nvidia", apply=True, runner=fake, sage_assets=offline,
                              progress=lambda m: None, host_system="Windows", host_machine="AMD64")
    assert fake.pips() == ["triton-windows>=3.5,<3.6", "sageattention"]
    assert report.status == "OK"


def test_triton_failure_skips_sage_and_is_nonfatal(tmp_path):
    ws, _ = _workspace(tmp_path)
    fake = FakeComfyPython(pip_fail={"triton==3.5.1"})
    report = _install(ws, fake)
    assert fake.pips() == ["triton==3.5.1"]
    assert report.status == "FAILED-NONFATAL"
    assert [s.name for s in report.steps] == ["triton", "sageattention"]
    assert report.steps[1].status == "SKIPPED" and "needs Triton" in report.steps[1].detail
    assert "No Triton build could be installed for Python 3.12.3 / CUDA 12.8" in report.steps[0].detail


def test_triton_installs_but_does_not_import_is_nonfatal(tmp_path):
    ws, _ = _workspace(tmp_path)
    fake = FakeComfyPython(import_fail={"triton"})
    report = _install(ws, fake)
    assert fake.pips() == ["triton==3.5.1"]
    assert report.status == "FAILED-NONFATAL"


def test_already_installed_is_idempotent(tmp_path):
    ws, _ = _workspace(tmp_path)
    fake = FakeComfyPython(have={"triton", "sageattention"})
    report = _install(ws, fake)
    assert fake.pips() == []
    assert report.status == "OK"


def test_dry_run_plans_without_pip(tmp_path):
    ws, _ = _workspace(tmp_path)
    fake = FakeComfyPython()
    report = _install(ws, fake, apply=False)
    assert fake.pips() == []
    assert [s.status for s in report.steps] == ["PLANNED", "PLANNED"]
    assert report.steps[0].command[-1] == "triton==3.5.1"


@pytest.mark.parametrize(
    "kwargs,vendor,needle",
    [
        ({"system": "Darwin", "machine": "arm64", "cuda": None}, "apple", "macOS"),
        ({"cuda": None, "torch": "2.9.1+cpu"}, "unknown", "No NVIDIA GPU"),
        ({"cuda": None, "torch": "2.9.1+cpu"}, "nvidia", "non-CUDA PyTorch"),
        ({"machine": "riscv64"}, "nvidia", "Unsupported Linux CPU"),
        ({"system": "FreeBSD"}, "nvidia", "Unsupported platform"),
    ],
)
def test_skips_with_plain_message(tmp_path, kwargs, vendor, needle):
    ws, _ = _workspace(tmp_path)
    fake = FakeComfyPython(**kwargs)
    report = _install(ws, fake, vendor=vendor)
    assert fake.pips() == []
    assert report.status == "SKIPPED"
    assert needle in report.steps[0].detail and "just slower" in report.steps[0].detail


def test_unknown_comfy_interpreter_never_pip_installs(tmp_path):
    report = cv.install_accel(tmp_path / "ComfyUI", gpu_vendor="nvidia", apply=True,
                              runner=lambda *a, **k: pytest.fail("no subprocess"), progress=lambda m: None)
    assert report.status == "SKIPPED"


# --- launch flag ------------------------------------------------------------------


def test_launch_flag_probes_comfy_venv(tmp_path, monkeypatch):
    ws, py = _workspace(tmp_path)
    monkeypatch.setenv("COMFY_CPU", "0")
    monkeypatch.setattr(tower, "sageattention_available", lambda: pytest.fail("must probe the Comfy venv"))
    seen: list[list[str]] = []

    def run(cmd, **kw):
        seen.append([str(c) for c in cmd])
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(cv.subprocess, "run", run)
    monkeypatch.setattr(cv, "module_importable", lambda python, module, runner=run: run([python, "-c", f"import {module}"]).returncode == 0)
    t = ManagedComfyTower(workspace=ws)
    assert "--use-sage-attention" in t._launch_args(None)
    assert seen[-1] == [str(py), "-c", "import sageattention"]

    monkeypatch.setattr(cv, "module_importable", lambda python, module, runner=None: False)
    assert "--use-sage-attention" not in t._launch_args(None)


def test_launch_flag_falls_back_to_buddy_interpreter_without_comfy_venv(tmp_path, monkeypatch):
    monkeypatch.setattr(cv, "module_importable", lambda *a, **k: pytest.fail("no Comfy venv to probe"))
    monkeypatch.setattr(tower, "sageattention_available", lambda: True)
    assert tower.comfy_sageattention_available(tmp_path / "ComfyUI") is True
    monkeypatch.setattr(tower, "sageattention_available", lambda: False)
    assert tower.comfy_sageattention_available(tmp_path / "ComfyUI") is False
