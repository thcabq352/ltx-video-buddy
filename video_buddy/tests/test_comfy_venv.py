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


def _install(ws, fake, *, vendor="nvidia", apply=True, assets=(), host_system="Linux", host_machine="x86_64",
             triton_versions=dict, **kw):
    return cv.install_accel(
        ws, gpu_vendor=vendor, apply=apply, runner=fake, sage_assets=lambda: list(assets),
        progress=lambda msg: None, host_system=host_system, host_machine=host_machine,
        triton_versions=triton_versions, **kw,
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
        ("2.11.0+cu130", "triton-windows>=3.6,<3.7", True),
        ("2.12.0+cu130", "triton-windows>=3.7,<3.8", True),
        ("2.14.1+cu130", "triton-windows>=3.8,<3.9", True),
        ("2.15.0+cu130", "triton-windows>=3.9,<3.10", False),
        ("2.3.1", None, False),
        (None, None, False),
    ],
)
def test_triton_windows_spec(torch_version, spec, verified):
    assert cv.triton_windows_spec(torch_version) == (spec, verified)


def test_triton_linux_spec_follows_torch_pin():
    assert cv.triton_linux_spec(cv.ComfyEnv(torch_triton_requirement="triton==3.5.1")) == "triton==3.5.1"
    assert cv.triton_linux_spec(cv.ComfyEnv()) == "triton"
    assert cv.triton_linux_spec(cv.ComfyEnv(torch_triton_requirement="triton~=3.8.0")) == "triton~=3.8.0"


# Shape of fetch_triton_windows_versions() as of 2026-10-22 (PyPI JSON).
REAL_TRITON = {
    "3.5.1.post24": {"cp310", "cp311", "cp312", "cp313", "cp314"},
    "3.6.0.post26": {"cp310", "cp311", "cp312", "cp313", "cp314"},
    "3.7.1.post27": {"cp310", "cp311", "cp312", "cp313", "cp314"},
    "3.8.0.post28": {"cp310", "cp311", "cp312", "cp313", "cp314"},
    "3.8.0.post29": {"cp310", "cp311", "cp312", "cp313", "cp314"},
    "3.3.0.post19": {"cp39", "cp310", "cp311", "cp312", "cp313"},
}


def test_triton_windows_candidates_torch_2141_cu130_real_pin_first():
    tries = cv.triton_windows_candidates("2.14.1+cu130", python="3.12.10", torch_triton_requirement="triton~=3.8.0",
                                         versions=lambda: REAL_TRITON)
    specs = [s for s, _ in tries]
    assert specs[0] == "triton-windows==3.8.0.post29"
    assert specs[-1] == "triton-windows>=3.8,<3.9"
    assert not any("3.10" in s for s in specs)


def test_triton_windows_candidates_network_failure_degrades():
    def boom():
        raise OSError("offline")

    assert cv.triton_windows_candidates("2.14.1", python="3.12", versions=boom) == [("triton-windows>=3.8,<3.9", "")]
    assert cv.triton_windows_candidates("2.3.0", versions=boom) == []


def test_triton_windows_candidates_unverified_torch_uses_newest_real_below():
    tries = cv.triton_windows_candidates("2.16.0", python="3.13.1", versions=lambda: REAL_TRITON)
    specs = [s for s, _ in tries]
    # 2.16 extrapolates to 3.10 which does not exist: newest real 3.8 first, range last.
    assert specs == ["triton-windows==3.8.0.post29", "triton-windows>=3.10,<3.11"]
    # cp tag filtering: Python 3.9 only has old wheels
    assert cv.triton_windows_candidates("2.7.0", python="3.9.1", versions=lambda: REAL_TRITON)[0][0] == "triton-windows==3.3.0.post19"


def test_fetch_triton_windows_versions_pypi_then_github(monkeypatch):
    import io
    pypi = {"releases": {
        "3.8.0.post29": [{"filename": "triton_windows-3.8.0.post29-cp312-cp312-win_amd64.whl"}],
        "3.9.0a0": [],
    }}
    gh = [{"tag_name": "v3.6.0-windows.post25"}, {"tag_name": "junk"}]

    def opener(req, timeout=0):
        url = req if isinstance(req, str) else req.full_url
        if "pypi" in url:
            if opener.pypi_down:
                raise cv.urllib.error.URLError("down")
            return io.BytesIO(json.dumps(pypi).encode())
        return io.BytesIO(json.dumps(gh).encode())

    opener.pypi_down = False
    monkeypatch.setattr(cv.urllib.request, "urlopen", opener)
    assert cv.fetch_triton_windows_versions() == {"3.8.0.post29": {"cp312"}}
    opener.pypi_down = True
    assert cv.fetch_triton_windows_versions() == {"3.6.0.post25": set()}


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


# Real woct0rdho/SageAttention v2.2.0-windows.post6/.post7 asset names (2026-10-22), newest release first.
REAL_SAGE = [
    {"name": "sageattention-2.2.0+rocm10.0torch2.11.0andhigher.post7-cp310-abi3-win_amd64.whl", "browser_download_url": "https://x/rocm.whl"},
    {"name": "sageattention-2.2.0+cu128torch2.10.0andhigher.post6-cp310-abi3-win_amd64.whl", "browser_download_url": "https://x/cu128plus.whl"},
    {"name": "sageattention-2.2.0+cu130torch2.10.0andhigher.post6-cp310-abi3-win_amd64.whl", "browser_download_url": "https://x/cu130plus.whl"},
    {"name": "sageattention-2.2.0+cu130torch2.9.1.post6-cp310-abi3-win_amd64.whl", "browser_download_url": "https://x/cu130_291.whl"},
    {"name": "sageattention-2.2.0+cu128torch2.9.1.post6-cp310-abi3-win_amd64.whl", "browser_download_url": "https://x/cu128_291.whl"},
]


def test_pick_windows_sage_wheel_torch_2141_cu130():
    assert cv.pick_windows_sage_wheel(REAL_SAGE, torch_version="2.14.1+cu130", cuda="13.0") == "https://x/cu130plus.whl"
    assert cv.pick_windows_sage_wheel(REAL_SAGE, torch_version="2.9.1+cu130", cuda="13.0") == "https://x/cu130_291.whl"
    assert cv.pick_windows_sage_wheel(REAL_SAGE, torch_version="2.9.0+cu130", cuda="13.0") is None
    # cu128 'andhigher' is NOT used for a cu130 torch (different CUDA major runtime)
    only128 = [a for a in REAL_SAGE if "cu128" in a["name"]]
    assert cv.pick_windows_sage_wheel(only128, torch_version="2.14.1+cu130", cuda="13.0") is None
    # same major, lower minor is fine: cu128 wheel on a cu129 torch
    assert cv.pick_windows_sage_wheel(only128, torch_version="2.14.1+cu129", cuda="12.9") == "https://x/cu128plus.whl"
    # a cu130 wheel never goes onto cu128 torch
    only130 = [a for a in REAL_SAGE if "cu130" in a["name"]]
    assert cv.pick_windows_sage_wheel(only130, torch_version="2.14.1+cu128", cuda="12.8") is None


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
                              progress=lambda m: None, host_system="Windows", host_machine="AMD64",
                              triton_versions=dict)
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


# --- Blackwell (RTX 5060 Ti, sm_120) + torch 2.14.1+cu130 ------------------------


def _blackwell(**kw):
    fake = FakeComfyPython(system="Windows", machine="AMD64", torch="2.14.1+cu130", cuda="13.0",
                           triton_req="triton~=3.8.0", **kw)
    fake.info["compute_capability"] = "12.0"
    fake.info["gpu_name"] = "NVIDIA GeForce RTX 5060 Ti"
    return fake


def test_probe_captures_compute_capability(tmp_path):
    ws, _ = _workspace(tmp_path)
    env = cv.probe_comfy_env(ws, runner=_blackwell())
    assert env.compute_capability == "12.0" and "sm_120" in env.summary()
    assert "get_device_capability" in cv._PROBE_SCRIPT and "triton~=" in cv._PROBE_SCRIPT


def test_blackwell_windows_installs_real_triton_pin_then_cu130_wheel(tmp_path):
    ws, _ = _workspace(tmp_path)
    fake = _blackwell()
    report = _install(ws, fake, assets=REAL_SAGE, host_system="Windows", host_machine="AMD64",
                      triton_versions=lambda: REAL_TRITON)
    assert fake.pips() == ["triton-windows==3.8.0.post29", "https://x/cu130plus.whl"]
    assert report.status == "OK"


def test_blackwell_triton_falls_through_candidates(tmp_path):
    ws, _ = _workspace(tmp_path)
    fake = _blackwell(pip_fail={"triton-windows==3.8.0.post29"})
    report = _install(ws, fake, assets=REAL_SAGE, host_system="Windows", host_machine="AMD64",
                      triton_versions=lambda: REAL_TRITON)
    assert fake.pips()[:2] == ["triton-windows==3.8.0.post29", "triton-windows>=3.8,<3.9"]
    assert report.triton_present


class BuildingFake(FakeComfyPython):
    def __call__(self, cmd, **kw):
        cmd = [str(c) for c in cmd]
        if cmd[0] == "git":
            self.calls.append(cmd)
            self.git_env = kw.get("env")
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if cmd[1:5] == ["-m", "pip", "install", "--no-build-isolation"]:
            self.calls.append(cmd)
            self.build_env = kw.get("env")
            if "build" in self.pip_fail:
                return subprocess.CompletedProcess(cmd, 1, "", "error: Microsoft Visual C++ 14.0 is required")
            self.installed.add("sageattention")
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return super().__call__(cmd, **kw)


def test_blackwell_source_build_after_wheel_fails_before_pypi(tmp_path):
    ws, _ = _workspace(tmp_path)
    fake = BuildingFake(system="Windows", machine="AMD64", torch="2.14.1+cu130", cuda="13.0",
                        triton_req="triton~=3.8.0", pip_fail={"https://x/cu130plus.whl"})
    fake.info["compute_capability"] = "12.0"
    report = _install(ws, fake, assets=REAL_SAGE, host_system="Windows", host_machine="AMD64",
                      triton_versions=lambda: REAL_TRITON, sage_source_build=True, build_dir=tmp_path / "b")
    assert report.status == "OK" and "source build" in report.steps[-1].detail
    assert any(c[0] == "git" and c[1] == "clone" for c in fake.calls)
    assert fake.build_env["TORCH_CUDA_ARCH_LIST"] == "12.0"
    assert "sageattention" not in fake.pips()  # PyPI v1 not reached


def test_blackwell_source_build_failure_is_nonfatal_then_pypi(tmp_path):
    ws, _ = _workspace(tmp_path)
    fake = BuildingFake(system="Windows", machine="AMD64", torch="2.14.1+cu130", cuda="13.0",
                        triton_req="triton~=3.8.0", pip_fail={"build", "sageattention"})
    report = _install(ws, fake, assets=[], host_system="Windows", host_machine="AMD64",
                      triton_versions=dict, sage_source_build=True, build_dir=tmp_path / "b")
    assert report.status == "FAILED-NONFATAL" and report.triton_present
    assert fake.pips()[-1] == "sageattention"


def test_source_build_off_by_default_and_hinted(tmp_path, monkeypatch):
    monkeypatch.delenv(cv.SAGE_SOURCE_BUILD_ENV, raising=False)
    ws, _ = _workspace(tmp_path)
    fake = BuildingFake(system="Windows", machine="AMD64", torch="2.14.1+cu130", cuda="13.0",
                        pip_fail={"sageattention"})
    report = _install(ws, fake, assets=[], host_system="Windows", host_machine="AMD64")
    assert not any(c[0] == "git" for c in fake.calls)
    assert cv.SAGE_SOURCE_BUILD_ENV in report.steps[-1].detail
    assert cv.sage_source_build_enabled({cv.SAGE_SOURCE_BUILD_ENV: "1"})
