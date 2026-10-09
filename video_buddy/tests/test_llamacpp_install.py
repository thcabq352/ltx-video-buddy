"""Buddy-owned llama.cpp: plan, build-from-source order, prebuilt fallback, non-fatal.

Subprocess and downloads are mocked. Only tmp paths.

Run: python -m pytest tests/test_llamacpp_install.py -q
"""

from __future__ import annotations

import io
import json
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest

from master_agent import config as cfg
from master_agent import llamacpp_install as li
from master_agent import llamacpp_server as srv

REF = "b11389"
TOOLS = {"git": "/usr/bin/git", "cmake": "/usr/bin/cmake", "c++": "/usr/bin/c++"}


def _which(table):
    return lambda name: table.get(name)


def _plan(root, *, vendor="unknown", tools=TOOLS, system="Linux", machine="x86_64", existing=None):
    return li.plan_llamacpp(
        gpu_vendor=vendor, root=root, ref=REF, existing=existing, which=_which(tools),
        system=system, machine=machine,
    )


class FakeToolchain:
    """git / cmake / llama-server --version. ``fail`` holds argv[0:2] prefixes that exit 1."""

    def __init__(self, root: Path, *, fail=(), describe=None, produce=True):
        self.root = root
        self.fail = set(fail)
        self.describe = describe
        self.produce = produce
        self.calls: list[list[str]] = []

    def __call__(self, cmd, **kw):
        cmd = [str(c) for c in cmd]
        self.calls.append(cmd)
        key = " ".join(cmd[:2]) if cmd[0] != "git" else "git " + next(
            a for a in cmd[1:] if a in {"clone", "describe", "fetch", "checkout"}
        )
        if key in self.fail or cmd[0] in self.fail:
            return subprocess.CompletedProcess(cmd, 1, "", "error: boom\nline two")
        if key == "git clone":
            (Path(cmd[-1]) / ".git").mkdir(parents=True)
        elif key == "git describe":
            out = self.describe or ""
            return subprocess.CompletedProcess(cmd, 0 if out else 128, out + "\n", "")
        elif cmd[:2] == ["cmake", "--build"] and self.produce:
            binary = self.root / "build" / "bin" / "llama-server"
            binary.parent.mkdir(parents=True, exist_ok=True)
            binary.write_text("")
        elif cmd[-1] == "--version":
            return subprocess.CompletedProcess(cmd, 0, "version: 0.5.0-dev (build 11389, commit 16c163d56)\n", "")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    def kinds(self) -> list[str]:
        out = []
        for c in self.calls:
            if c[0] == "git":
                out.append("git " + next(a for a in c[1:] if a in {"clone", "describe", "fetch", "checkout"}))
            elif c[0] == "cmake":
                out.append("cmake build" if c[1] == "--build" else "cmake configure")
            else:
                out.append("version")
        return out


def _tarball(dest: Path, *, name="llama-server", folder=f"llama-{REF}") -> None:
    with tarfile.open(dest, "w:gz") as tf:
        data = b"#!/bin/sh\n"
        info = tarfile.TarInfo(f"{folder}/{name}")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))


class FakeDownload:
    def __init__(self, *, fail=False):
        self.urls: list[str] = []
        self.fail = fail

    def __call__(self, url, dest):
        self.urls.append(url)
        if self.fail:
            raise OSError("offline")
        _tarball(dest)


# --- plan ---------------------------------------------------------------------------


def test_pinned_ref_in_config():
    assert cfg.LLAMACPP_REF == REF
    assert cfg.LLAMACPP_ROOT.name == "llama.cpp"


def test_plan_found_on_path_wins(tmp_path):
    plan = _plan(tmp_path, existing="/usr/local/bin/llama-server")
    assert plan.method == "found" and plan.disk_bytes == 0


@pytest.mark.parametrize(
    "vendor,system,machine,tools,backend",
    [
        ("unknown", "Linux", "x86_64", TOOLS, "cpu"),
        ("nvidia", "Linux", "x86_64", {**TOOLS, "nvcc": "/usr/local/cuda/bin/nvcc"}, "cuda"),
        ("apple", "Darwin", "arm64", {**TOOLS, "clang++": "clang++"}, "metal"),
        ("nvidia", "Windows", "AMD64", {"git": "git", "cmake": "cmake", "cl": "cl", "nvcc": "nvcc"}, "cuda"),
    ],
)
def test_plan_builds_with_the_right_backend(tmp_path, vendor, system, machine, tools, backend):
    plan = _plan(tmp_path, vendor=vendor, tools=tools, system=system, machine=machine)
    assert plan.method == "build" and plan.backend == backend


def test_configure_flags_per_backend(tmp_path):
    assert "-DGGML_CUDA=ON" in li.configure_argv(tmp_path, "cuda")
    assert "-DGGML_METAL=ON" in li.configure_argv(tmp_path, "metal")
    cpu = li.configure_argv(tmp_path, "cpu")
    assert "-DGGML_METAL=OFF" in cpu and "-DGGML_CUDA=ON" not in cpu
    assert li.build_argv(tmp_path)[-2:] == ["--target", "llama-server"]
    assert li.clone_argv(tmp_path, REF)[-4:] == ["--branch", REF, li.REPO_URL, str(tmp_path / "src")]


@pytest.mark.parametrize(
    "vendor,system,machine,assets",
    [
        ("unknown", "Linux", "x86_64", [f"llama-{REF}-bin-ubuntu-x64.tar.gz"]),
        ("nvidia", "Linux", "x86_64", [
            f"llama-{REF}-bin-ubuntu-cuda-12.8-x64.tar.gz",
            f"cudart-llama-{REF}-bin-ubuntu-cuda-12.8-x64.tar.gz",
        ]),
        ("unknown", "Windows", "AMD64", [f"llama-{REF}-bin-win-cpu-x64.zip"]),
        ("nvidia", "Windows", "AMD64", [
            f"llama-{REF}-bin-win-cuda-12.4-x64.zip",
            "cudart-llama-bin-win-cuda-12.4-x64.zip",
        ]),
        ("apple", "Darwin", "arm64", [f"llama-{REF}-bin-macos-arm64.tar.gz"]),
        ("unknown", "Linux", "aarch64", [f"llama-{REF}-bin-ubuntu-arm64.tar.gz"]),
    ],
)
def test_plan_prebuilt_when_toolchain_missing(tmp_path, vendor, system, machine, assets):
    plan = _plan(tmp_path, vendor=vendor, tools={}, system=system, machine=machine)
    assert plan.method == "prebuilt"
    assert plan.assets == assets
    assert "git" in plan.missing_tools and "cmake" in plan.missing_tools
    assert plan.commands()[0] == f"download https://github.com/ggml-org/llama.cpp/releases/download/{REF}/{assets[0]}"


def test_plan_nvidia_without_nvcc_uses_cuda_prebuilt(tmp_path):
    plan = _plan(tmp_path, vendor="nvidia", tools=TOOLS)
    assert plan.method == "prebuilt" and plan.backend == "cuda"
    assert plan.missing_tools == ["the CUDA toolkit (nvcc)"]


def test_plan_nvidia_arm64_without_nvcc_builds_cpu(tmp_path):
    plan = _plan(tmp_path, vendor="nvidia", tools=TOOLS, machine="aarch64")
    assert plan.method == "build" and plan.backend == "cpu"
    assert "building CPU-only" in plan.notes[0]


def test_plan_unavailable_lists_what_is_missing(tmp_path):
    plan = _plan(tmp_path, tools={"git": "git"}, machine="riscv64")
    assert plan.method == "unavailable"
    assert plan.missing_tools == ["cmake", "a C++ compiler"]
    assert "missing cmake, a C++ compiler" in plan.describe()
    res = li.install_llamacpp(plan, apply=True, runner=lambda *a, **k: pytest.fail("no subprocess"))
    assert res.status == "NEEDS-YOU"
    assert "sudo apt-get install -y git cmake build-essential" in res.detail
    assert any("renders video without it" in line for line in res.lines)


@pytest.mark.parametrize("system,hint", [
    ("Windows", "winget install Git.Git Kitware.CMake Microsoft.VisualStudio.2022.BuildTools"),
    ("Darwin", "xcode-select --install && brew install cmake"),
])
def test_toolchain_hints(system, hint):
    assert li.toolchain_hint(system) == hint


# --- build ------------------------------------------------------------------------------


def test_build_order_clone_configure_build_verify_marker(tmp_path):
    root = tmp_path / "llama.cpp"
    plan = _plan(root)
    tc = FakeToolchain(root)
    res = li.install_llamacpp(plan, apply=True, runner=tc, download=FakeDownload(), progress=lambda m: None)
    assert res.status == "OK" and "built from source b11389 (cpu)" in res.detail
    assert tc.kinds() == ["git clone", "cmake configure", "cmake build", "version"]
    marker = json.loads((root / "installed.json").read_text())
    assert marker == {"ref": REF, "method": "build", "backend": "cpu", "binary": str(root / "build" / "bin" / "llama-server")}
    again = _plan(root)
    assert again.method == "found" and again.found_via == f"Buddy's own, {REF}"
    assert li.install_llamacpp(again, apply=True, runner=lambda *a, **k: pytest.fail("idempotent")).status == "SKIPPED"


def test_resume_reuses_checkout_at_ref(tmp_path):
    root = tmp_path / "llama.cpp"
    (root / "src" / ".git").mkdir(parents=True)
    tc = FakeToolchain(root, describe=REF)
    res = li.install_llamacpp(_plan(root), apply=True, runner=tc, progress=lambda m: None)
    assert res.status == "OK"
    assert tc.kinds() == ["git describe", "cmake configure", "cmake build", "version"]


def test_resume_moves_old_checkout_to_pinned_ref(tmp_path):
    root = tmp_path / "llama.cpp"
    (root / "src" / ".git").mkdir(parents=True)
    tc = FakeToolchain(root, describe="b11000")
    li.install_llamacpp(_plan(root), apply=True, runner=tc, progress=lambda m: None)
    assert tc.kinds()[:3] == ["git describe", "git fetch", "git checkout"]
    assert tc.calls[1][-2:] == ["tag", REF]


def test_old_ref_marker_triggers_update(tmp_path):
    root = tmp_path / "llama.cpp"
    (root / "build" / "bin").mkdir(parents=True)
    (root / "build" / "bin" / "llama-server").write_text("")
    (root / "installed.json").write_text(json.dumps({"ref": "b11000", "binary": str(root / "build" / "bin" / "llama-server")}))
    plan = _plan(root)
    assert plan.method == "build" and "not at b11389" in plan.notes[0]


def test_build_failure_falls_back_to_prebuilt(tmp_path):
    root = tmp_path / "llama.cpp"
    tc = FakeToolchain(root, fail={"cmake -S"})
    dl = FakeDownload()
    msgs: list[str] = []
    res = li.install_llamacpp(_plan(root), apply=True, runner=tc, download=dl, progress=msgs.append)
    assert res.status == "OK" and "official prebuilt b11389 (cpu)" in res.detail
    assert dl.urls == [f"{li.REPO_URL}/releases/download/{REF}/llama-{REF}-bin-ubuntu-x64.tar.gz"]
    assert any("trying the official prebuilt release" in m and "\n" not in m for m in msgs)
    binary = root / "prebuilt" / REF / f"llama-{REF}" / "llama-server"
    assert res.binary == str(binary) and binary.stat().st_mode & 0o111
    assert json.loads((root / "installed.json").read_text())["method"] == "prebuilt"


def test_build_and_prebuilt_both_fail_is_nonfatal(tmp_path):
    root = tmp_path / "llama.cpp"
    tc = FakeToolchain(root, fail={"git clone"})
    res = li.install_llamacpp(_plan(root), apply=True, runner=tc, download=FakeDownload(fail=True), progress=lambda m: None)
    assert res.status == "FAILED-NONFATAL"
    assert "git checkout of b11389 failed" in res.detail and "download of" in res.detail
    assert any("renders video without it" in line for line in res.lines)
    assert not (root / "installed.json").exists()


def test_src_that_is_not_a_checkout_is_left_alone(tmp_path):
    root = tmp_path / "llama.cpp"
    (root / "src").mkdir(parents=True)
    (root / "src" / "notes.txt").write_text("mine")
    res = li.install_llamacpp(_plan(root), apply=True, runner=FakeToolchain(root),
                              download=FakeDownload(fail=True), progress=lambda m: None)
    assert res.status == "FAILED-NONFATAL" and "is not a git checkout" in res.detail
    assert (root / "src" / "notes.txt").read_text() == "mine"


def test_prebuilt_resumes_from_downloaded_archive(tmp_path):
    root = tmp_path / "llama.cpp"
    plan = _plan(root, tools={})
    (root / "downloads").mkdir(parents=True)
    _tarball(root / "downloads" / plan.assets[0])
    (root / "prebuilt" / f"{REF}.partial" / "junk").mkdir(parents=True)
    res = li.install_llamacpp(plan, apply=True, runner=FakeToolchain(root),
                              download=lambda u, d: pytest.fail("already downloaded"), progress=lambda m: None)
    assert res.status == "OK"
    assert not (root / "prebuilt" / f"{REF}.partial").exists()


def test_prebuilt_rejects_unsafe_archive(tmp_path):
    root = tmp_path / "llama.cpp"
    plan = _plan(root, tools={}, system="Windows", machine="AMD64")

    def evil(url, dest):
        with zipfile.ZipFile(dest, "w") as zf:
            zf.writestr("../../outside.txt", "x")

    res = li.install_llamacpp(plan, apply=True, runner=FakeToolchain(root), download=evil, progress=lambda m: None)
    assert res.status == "FAILED-NONFATAL" and "unsafe path" in res.detail
    assert not (tmp_path / "outside.txt").exists()
    assert not (root / "downloads" / plan.assets[0]).exists()


def test_dry_run_plans_without_running(tmp_path):
    root = tmp_path / "llama.cpp"
    res = li.install_llamacpp(_plan(root), apply=False, runner=lambda *a, **k: pytest.fail("dry run"))
    assert res.status == "PLANNED"
    assert res.lines[0].startswith("git -c advice.detachedHead=false clone --depth 1 --branch b11389")
    assert not root.exists()


def test_version_check_failure_falls_through(tmp_path):
    root = tmp_path / "llama.cpp"
    tc = FakeToolchain(root, fail={str(root / "build" / "bin" / "llama-server")})
    res = li.install_llamacpp(_plan(root), apply=True, runner=tc, download=FakeDownload(), progress=lambda m: None)
    assert res.status == "OK" and "official prebuilt" in res.detail


# --- server lookup ------------------------------------------------------------------------


def test_resolve_binary_finds_buddy_build(tmp_path, monkeypatch):
    root = tmp_path / "llama.cpp"
    monkeypatch.setattr(cfg, "LLAMACPP_BIN", "")
    monkeypatch.setattr(cfg, "LLAMACPP_ROOT", root)
    monkeypatch.setattr(srv.shutil, "which", lambda name: None)
    assert srv.resolve_binary() is None
    binary = root / "build" / "bin" / "llama-server"
    binary.parent.mkdir(parents=True)
    binary.write_text("")
    assert srv.resolve_binary() == str(binary)
    monkeypatch.setattr(srv.shutil, "which", lambda name: "/usr/bin/llama-server")
    assert srv.resolve_binary() == "/usr/bin/llama-server"
