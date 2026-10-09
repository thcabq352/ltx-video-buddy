"""Get a Buddy-owned ``llama-server`` ready (automatic install).

Everything lives under ``LLAMACPP_ROOT`` (default ``video_buddy/llama.cpp``):

- ``src/``       git checkout of the official ggml-org/llama.cpp tag ``LLAMACPP_REF``
- ``build/``     cmake build (CUDA with an NVIDIA GPU + nvcc, Metal on Apple Silicon, else CPU)
- ``prebuilt/``  the official release archive for this platform, when building is not possible
- ``installed.json``  which ref, method, backend and binary were installed

:func:`master_agent.llamacpp_server.resolve_binary` finds that binary after
``LLAMACPP_BIN`` and ``PATH``. Nothing here is fatal: Buddy renders video
without a local LLM.
"""

from __future__ import annotations

import json
import os
import platform as _platform
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

REPO_URL = "https://github.com/ggml-org/llama.cpp"
MARKER = "installed.json"

CLONE_TIMEOUT_S = 1800.0
CONFIGURE_TIMEOUT_S = 1800.0
BUILD_TIMEOUT_S = 7200.0
DOWNLOAD_TIMEOUT_S = 60.0

GB = 1_000_000_000
# Budgets with headroom. Measured on b11389: shallow checkout + CPU build of
# llama-server is ~0.32 GB. CUDA builds and the CUDA runtime archives
# (~0.6 GB packed) are several times larger.
DISK_BYTES = {
    ("build", "cpu"): 1 * GB,
    ("build", "metal"): 1 * GB,
    ("build", "cuda"): 5 * GB,
    ("prebuilt", "cpu"): int(0.3 * GB),
    ("prebuilt", "metal"): int(0.3 * GB),
    ("prebuilt", "cuda"): int(2.5 * GB),
}

# Official release assets (checked against the b11389 release). CUDA 12.x
# runs on older drivers than the 13.x builds.
_PREBUILT: dict[tuple[str, str, str], tuple[str, ...]] = {
    ("Linux", "x64", "cpu"): ("llama-{ref}-bin-ubuntu-x64.tar.gz",),
    ("Linux", "x64", "cuda"): (
        "llama-{ref}-bin-ubuntu-cuda-12.8-x64.tar.gz",
        "cudart-llama-{ref}-bin-ubuntu-cuda-12.8-x64.tar.gz",
    ),
    ("Linux", "arm64", "cpu"): ("llama-{ref}-bin-ubuntu-arm64.tar.gz",),
    ("Windows", "x64", "cpu"): ("llama-{ref}-bin-win-cpu-x64.zip",),
    ("Windows", "x64", "cuda"): (
        "llama-{ref}-bin-win-cuda-12.4-x64.zip",
        "cudart-llama-bin-win-cuda-12.4-x64.zip",
    ),
    ("Darwin", "arm64", "metal"): ("llama-{ref}-bin-macos-arm64.tar.gz",),
    ("Darwin", "x64", "cpu"): ("llama-{ref}-bin-macos-x64.tar.gz",),
}

Runner = Callable[..., subprocess.CompletedProcess]
Which = Callable[[str], "str | None"]
Downloader = Callable[[str, Path], None]


def binary_name(system: str | None = None) -> str:
    return "llama-server.exe" if (system or _platform.system()) == "Windows" else "llama-server"


def _arch(machine: str) -> str:
    m = (machine or "").lower()
    if m in {"x86_64", "amd64", "x64"}:
        return "x64"
    if m in {"arm64", "aarch64"}:
        return "arm64"
    return m


def read_marker(root: Path) -> dict:
    try:
        data = json.loads((Path(root) / MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def buddy_binary(root: Path | str | None) -> Path | None:
    """The ``llama-server`` automatic install put under ``root``, if any."""
    if not root:
        return None
    root = Path(root)
    recorded = read_marker(root).get("binary")
    if recorded and Path(recorded).is_file():
        return Path(recorded)
    name = binary_name()
    for cand in (root / "build" / "bin" / name, root / "build" / "bin" / "Release" / name):
        if cand.is_file():
            return cand
    return None


def backend_for(gpu_vendor: str, system: str, machine: str) -> str:
    if system == "Darwin":
        return "metal" if _arch(machine) == "arm64" else "cpu"
    return "cuda" if gpu_vendor == "nvidia" else "cpu"


def prebuilt_assets(system: str, machine: str, backend: str, ref: str) -> list[str]:
    names = _PREBUILT.get((system, _arch(machine), backend))
    return [n.format(ref=ref) for n in names] if names else []


def asset_url(ref: str, name: str) -> str:
    return f"{REPO_URL}/releases/download/{ref}/{name}"


def missing_build_tools(system: str, backend: str, which: Which) -> list[str]:
    missing = [t for t in ("git", "cmake") if not which(t)]
    compilers = ("cl",) if system == "Windows" else ("c++", "g++", "clang++")
    if not any(which(c) for c in compilers):
        missing.append("a C++ compiler (Visual Studio Build Tools)" if system == "Windows" else "a C++ compiler")
    if backend == "cuda" and not which("nvcc"):
        missing.append("the CUDA toolkit (nvcc)")
    return missing


def toolchain_hint(system: str) -> str:
    if system == "Windows":
        return "winget install Git.Git Kitware.CMake Microsoft.VisualStudio.2022.BuildTools"
    if system == "Darwin":
        return "xcode-select --install && brew install cmake"
    return "sudo apt-get install -y git cmake build-essential"


def configure_argv(root: Path, backend: str) -> list[str]:
    argv = [
        "cmake",
        "-S", str(root / "src"),
        "-B", str(root / "build"),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_EXAMPLES=OFF",
    ]
    if backend == "cuda":
        argv.append("-DGGML_CUDA=ON")
    elif backend == "metal":
        argv.append("-DGGML_METAL=ON")
    else:
        argv.append("-DGGML_METAL=OFF")
    return argv


def build_argv(root: Path) -> list[str]:
    jobs = str(max(1, os.cpu_count() or 1))
    return ["cmake", "--build", str(root / "build"), "--config", "Release", "-j", jobs, "--target", "llama-server"]


def clone_argv(root: Path, ref: str) -> list[str]:
    return ["git", "-c", "advice.detachedHead=false", "clone", "--depth", "1", "--branch", ref, REPO_URL, str(root / "src")]


@dataclass
class LlamaPlan:
    method: str  # found | build | prebuilt | unavailable
    backend: str
    ref: str
    root: Path
    binary: str | None = None
    found_via: str = ""
    missing_tools: list[str] = field(default_factory=list)
    assets: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    system: str = ""
    machine: str = ""

    @property
    def disk_bytes(self) -> int:
        return DISK_BYTES.get((self.method, self.backend), 0)

    def commands(self) -> list[str]:
        if self.method == "build":
            return [" ".join(clone_argv(self.root, self.ref)), " ".join(configure_argv(self.root, self.backend)),
                    " ".join(build_argv(self.root))]
        if self.method == "prebuilt":
            return [f"download {asset_url(self.ref, a)}" for a in self.assets]
        return []

    def describe(self) -> str:
        if self.method == "found":
            return f"llama.cpp found: {self.binary} ({self.found_via})."
        if self.method == "build":
            return (
                f"llama.cpp will be built from source: {REPO_URL} tag {self.ref}, "
                f"{self.backend.upper() if self.backend != 'cpu' else 'CPU'} backend, into {self.root}."
            )
        if self.method == "prebuilt":
            why = f" (cannot build here: missing {', '.join(self.missing_tools)})" if self.missing_tools else ""
            return f"llama.cpp will use the official prebuilt release {self.ref}: {', '.join(self.assets)}{why}."
        return (
            "llama.cpp is unavailable on this machine: no build tools "
            f"(missing {', '.join(self.missing_tools) or 'unknown'}) and no official prebuilt for "
            f"{self.system} {self.backend.upper()}."
        )


def plan_llamacpp(
    *,
    gpu_vendor: str,
    root: Path,
    ref: str,
    existing: str | None,
    which: Which = shutil.which,
    system: str | None = None,
    machine: str | None = None,
) -> LlamaPlan:
    """Pick found / build / prebuilt / unavailable. Reads only."""
    system = system or _platform.system()
    machine = machine or _platform.machine()
    root = Path(root)
    backend = backend_for(gpu_vendor, system, machine)
    plan = LlamaPlan(method="unavailable", backend=backend, ref=ref, root=root, system=system, machine=machine)
    if existing:
        plan.method, plan.binary, plan.found_via = "found", existing, "LLAMACPP_BIN or PATH"
        return plan
    mine = buddy_binary(root)
    if mine is not None and read_marker(root).get("ref") == ref:
        plan.method, plan.binary, plan.found_via = "found", str(mine), f"Buddy's own, {ref}"
        plan.backend = str(read_marker(root).get("backend") or backend)
        return plan
    if mine is not None:
        plan.notes.append(f"Buddy's llama.cpp is not at {ref}; it will be updated.")
    missing = missing_build_tools(system, backend, which)
    if not missing:
        plan.method = "build"
        return plan
    if backend == "cuda":
        assets = prebuilt_assets(system, machine, "cuda", ref)
        if assets:
            plan.method, plan.assets, plan.missing_tools = "prebuilt", assets, missing
            return plan
        cpu_missing = missing_build_tools(system, "cpu", which)
        if not cpu_missing:
            plan.backend, plan.method = "cpu", "build"
            plan.notes.append("No CUDA toolkit (nvcc) and no official CUDA prebuilt here; building CPU-only.")
            return plan
        backend = plan.backend = "cpu"
    assets = prebuilt_assets(system, machine, backend, ref)
    plan.missing_tools = missing
    if assets:
        plan.method, plan.assets = "prebuilt", assets
    return plan


@dataclass
class LlamaResult:
    status: str  # OK | SKIPPED | PLANNED | FAILED-NONFATAL | NEEDS-YOU
    detail: str
    lines: list[str] = field(default_factory=list)
    binary: str | None = None


def _download(url: str, dest: Path) -> None:
    part = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_S) as resp, open(part, "wb") as fh:
        shutil.copyfileobj(resp, fh, length=1 << 20)
    os.replace(part, dest)


def _extract(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    base = dest.resolve()
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                target = (dest / member).resolve()
                if base != target and base not in target.parents:
                    raise ValueError(f"unsafe path in {archive.name}: {member}")
            zf.extractall(dest)
        return
    with tarfile.open(archive, "r:gz") as tf:
        if hasattr(tarfile, "data_filter"):
            tf.extractall(dest, filter="data")
            return
        for member in tf.getmembers():
            target = (dest / member.name).resolve()
            if (base != target and base not in target.parents) or member.issym() or member.islnk():
                raise ValueError(f"unsafe entry in {archive.name}: {member.name}")
        tf.extractall(dest)


def _write_marker(root: Path, *, ref: str, method: str, backend: str, binary: Path) -> None:
    data = {"ref": ref, "method": method, "backend": backend, "binary": str(binary)}
    (root / MARKER).write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")


def _runs(binary: Path, runner: Runner) -> tuple[bool, str]:
    try:
        proc = runner([str(binary), "--version"], capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)[:200]
    text = ((proc.stdout or "") + (proc.stderr or "")).strip()
    return proc.returncode == 0, text.splitlines()[0] if text else ""


def _call(cmd: list[str], runner: Runner, progress: Callable[[str], None], timeout: float) -> tuple[bool, str]:
    progress("$ " + " ".join(cmd))
    try:
        proc = runner(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)[:300]
    text = " ".join(((proc.stderr or "") + "\n" + (proc.stdout or "")).split())
    return proc.returncode == 0, text[-300:]


def _checkout(root: Path, ref: str, runner: Runner, progress: Callable[[str], None]) -> tuple[bool, str]:
    src = root / "src"
    if not src.exists():
        return _call(clone_argv(root, ref), runner, progress, CLONE_TIMEOUT_S)
    if not (src / ".git").exists():
        return False, f"{src} exists but is not a git checkout; delete that folder and run again"
    ok, out = _call(["git", "-C", str(src), "describe", "--tags", "--exact-match"], runner, progress, 60)
    if ok and out.strip().splitlines()[-1:] == [ref]:
        return True, "checkout already at " + ref
    ok, out = _call(["git", "-C", str(src), "fetch", "--depth", "1", "origin", "tag", ref], runner, progress, CLONE_TIMEOUT_S)
    if not ok:
        return False, out
    return _call(["git", "-C", str(src), "checkout", "--force", ref], runner, progress, 300)


def _build(plan: LlamaPlan, runner: Runner, progress: Callable[[str], None]) -> tuple[Path | None, str]:
    root = plan.root
    ok, out = _checkout(root, plan.ref, runner, progress)
    if not ok:
        return None, f"git checkout of {plan.ref} failed: {out}"
    ok, out = _call(configure_argv(root, plan.backend), runner, progress, CONFIGURE_TIMEOUT_S)
    if not ok:
        return None, f"cmake configure failed: {out}"
    ok, out = _call(build_argv(root), runner, progress, BUILD_TIMEOUT_S)
    if not ok:
        return None, f"cmake build failed: {out}"
    name = binary_name(plan.system)
    for cand in (root / "build" / "bin" / name, root / "build" / "bin" / "Release" / name):
        if cand.is_file():
            return cand, ""
    return None, "build finished but llama-server was not produced"


def _prebuilt(
    plan: LlamaPlan, download: Downloader, progress: Callable[[str], None]
) -> tuple[Path | None, str]:
    root = plan.root
    downloads = root / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    final = root / "prebuilt" / plan.ref
    partial = root / "prebuilt" / f"{plan.ref}.partial"
    if partial.exists():
        shutil.rmtree(partial)
    for name in plan.assets:
        archive = downloads / name
        if not archive.is_file():
            progress(f"download {asset_url(plan.ref, name)}")
            try:
                download(asset_url(plan.ref, name), archive)
            except Exception as exc:
                return None, f"download of {name} failed ({exc})"
        try:
            _extract(archive, partial)
        except Exception as exc:
            archive.unlink(missing_ok=True)
            return None, f"could not unpack {name} ({exc}); it will be downloaded again next run"
    if final.exists():
        shutil.rmtree(final)
    os.replace(partial, final)
    name = binary_name(plan.system)
    found = next((p for p in final.rglob(name) if p.is_file()), None)
    if found is None:
        return None, f"{name} not found in {', '.join(plan.assets)}"
    if plan.system != "Windows":
        found.chmod(found.stat().st_mode | 0o111)
    return found, ""


_NO_LLM = (
    "Buddy still renders video without it; the local brief / director / judge LLM and the "
    "knowledge base stay off until llama.cpp is ready."
)


def install_llamacpp(
    plan: LlamaPlan,
    *,
    apply: bool,
    runner: Runner | None = None,
    download: Downloader | None = None,
    progress: Callable[[str], None] = print,
) -> LlamaResult:
    """Carry out ``plan``. Build first; on failure, the prebuilt release. Never raises."""
    runner = runner or subprocess.run
    download = download or _download
    lines = [plan.describe(), *plan.notes]
    if plan.method == "found":
        return LlamaResult("SKIPPED", f"already installed: {plan.binary}", lines, plan.binary)
    if plan.method == "unavailable":
        return LlamaResult(
            "NEEDS-YOU",
            "install the build tools, then run automatic install again:  " + toolchain_hint(plan.system),
            [*lines, _NO_LLM],
        )
    if not apply:
        return LlamaResult("PLANNED", plan.describe(), [*plan.notes, *plan.commands()])
    root = plan.root
    errors: list[str] = []
    attempts = [plan.method]
    fallback_assets = plan.assets or prebuilt_assets(plan.system, plan.machine, plan.backend, plan.ref)
    if plan.method == "build" and fallback_assets:
        attempts.append("prebuilt")
    for method in attempts:
        try:
            root.mkdir(parents=True, exist_ok=True)
            if method == "build":
                binary, err = _build(plan, runner, progress)
            else:
                if not plan.assets:
                    plan.assets = fallback_assets
                binary, err = _prebuilt(plan, download, progress)
        except OSError as exc:
            binary, err = None, f"{method} failed ({exc})"
        if binary is None:
            errors.append(err)
            if method == "build" and len(attempts) > 1:
                progress(f"WARN  llama.cpp build failed ({err}); trying the official prebuilt release")
            continue
        ok, version = _runs(binary, runner)
        if not ok:
            errors.append(f"{binary} --version failed: {version}")
            continue
        try:
            _write_marker(root, ref=plan.ref, method=method, backend=plan.backend, binary=binary)
        except OSError as exc:
            progress(f"WARN  could not record {root / MARKER} ({exc})")
        how = "built from source" if method == "build" else "official prebuilt"
        return LlamaResult("OK", f"{how} {plan.ref} ({plan.backend}): {binary}", [*lines, version], str(binary))
    return LlamaResult(
        "FAILED-NONFATAL",
        "; ".join(errors)[-500:],
        [*lines, _NO_LLM, "Build tools: " + toolchain_hint(plan.system)],
    )
