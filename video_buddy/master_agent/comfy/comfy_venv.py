"""The interpreter the managed ComfyUI runs in, and optional Triton / SageAttention.

comfy-cli 1.20.0 (``comfy_cli/resolve_python.py``) picks the workspace
interpreter in this order: ``VIRTUAL_ENV``, ``CONDA_PREFIX``, then
``<workspace>/.venv`` or ``<workspace>/venv``, then its own ``sys.executable``.
``comfy install`` creates ``<workspace>/.venv`` when comfy-cli itself runs from
a venv or a PEP 668 system Python. Buddy strips ``VIRTUAL_ENV`` /
``CONDA_PREFIX`` for comfy-cli calls once that workspace venv exists, so the
venv found here is the one Comfy launches with.

Triton and SageAttention are optional acceleration. Every failure here is
non-fatal: Comfy runs without them, just slower.
"""

from __future__ import annotations

import json
import os
import platform as _platform
import re
import subprocess
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

VENV_NAMES = (".venv", "venv")
PROBE_TIMEOUT_S = 90.0
PIP_TIMEOUT_S = 1800.0
SAGE_WINDOWS_RELEASES_API = "https://api.github.com/repos/woct0rdho/SageAttention/releases?per_page=20"
TRITON_WINDOWS_PYPI_JSON = "https://pypi.org/pypi/triton-windows/json"
TRITON_WINDOWS_RELEASES_API = "https://api.github.com/repos/woct0rdho/triton-windows/releases?per_page=50"
SAGE_SOURCE_REPO = "https://github.com/woct0rdho/SageAttention.git"
# Opt-in for the slow from-source SageAttention build (needs git, MSVC Build Tools
# and a CUDA toolkit matching torch's CUDA on Windows). "1"/"true"/"yes" enables.
SAGE_SOURCE_BUILD_ENV = "VIDEO_BUDDY_SAGE_SOURCE_BUILD"
SAGE_BUILD_TIMEOUT_S = 5400.0

# triton-windows README: each PyTorch minor is only guaranteed with one Triton minor.
# 2.4/2.5 -> 3.1, then 2.6 -> 3.2 ... 2.10 -> 3.6 (README table).
# 2.11 .. 2.14 are verified against torch's own Linux ``triton`` pin on PyPI
# (torch 2.11.0: triton==3.6.0, 2.12.0: ==3.7.0, 2.13.0: ==3.7.1,
# 2.14.x: ~=3.8.0) and triton-windows 3.6/3.7/3.8 wheels existing on PyPI
# (checked 2026-10-22; newest triton-windows 3.8.0.post29, cp310-cp314).
# Newer torch extrapolates +1 Triton minor per torch minor and is unverified.
_TRITON_WINDOWS_TABLE: dict[tuple[int, int], int] = {
    (2, 4): 1,
    (2, 5): 1,
    (2, 6): 2,
    (2, 7): 3,
    (2, 8): 4,
    (2, 9): 5,
    (2, 10): 6,
    (2, 11): 6,
    (2, 12): 7,
    (2, 13): 7,
    (2, 14): 8,
}

_PROBE_SCRIPT = r"""
import json, platform, sys
out = {
    "python": platform.python_version(),
    "executable": sys.executable,
    "in_venv": sys.prefix != getattr(sys, "base_prefix", sys.prefix),
    "system": platform.system(),
    "machine": platform.machine(),
    "torch": None, "cuda": None, "cuda_available": False, "hip": None,
    "torch_triton_requirement": None,
    "gpu_name": None, "compute_capability": None,
    "triton": None, "sageattention": None, "errors": {},
}
try:
    import torch
    out["torch"] = torch.__version__
    out["cuda"] = getattr(torch.version, "cuda", None)
    out["hip"] = getattr(torch.version, "hip", None)
    try:
        out["cuda_available"] = bool(torch.cuda.is_available())
        if out["cuda_available"]:
            major, minor = torch.cuda.get_device_capability(0)
            out["compute_capability"] = "%d.%d" % (major, minor)
            out["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        out["errors"]["cuda_available"] = str(exc)[:200]
except Exception as exc:
    out["errors"]["torch"] = str(exc)[:200]
try:
    from importlib.metadata import requires
    for req in requires("torch") or []:
        name = req.split(";", 1)[0].strip()
        low = name.lower().replace(" ", "")
        if low.startswith(("triton==", "triton~=", "triton>=", "pytorch-triton==")):
            if "platform_system" in req and "Linux" not in req:
                continue
            out["torch_triton_requirement"] = name.replace(" ", "")
            break
except Exception:
    pass
for mod in ("triton", "sageattention"):
    try:
        m = __import__(mod)
        out[mod] = str(getattr(m, "__version__", "") or "present")
    except Exception as exc:
        out["errors"][mod] = str(exc)[:200]
print(json.dumps(out))
"""

Runner = Callable[..., subprocess.CompletedProcess]


def _python_in(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def workspace_venv_python(workspace: Path | str | None) -> Path | None:
    """``<workspace>/.venv`` or ``<workspace>/venv`` interpreter, if it exists."""
    if not workspace:
        return None
    root = Path(workspace)
    for name in VENV_NAMES:
        py = _python_in(root / name)
        if py.is_file():
            return py
    return None


def strip_outer_venv(env: dict[str, str], workspace: Path | str | None) -> dict[str, str]:
    """Drop ``VIRTUAL_ENV`` / ``CONDA_PREFIX`` once the workspace has its own python.

    That is a ``.venv`` / ``venv`` or a portable build's ``python_embeded``.
    Without this, comfy-cli would run Comfy in Buddy's activated venv.
    """
    from master_agent.comfy.layout import comfy_interpreter

    if comfy_interpreter(workspace) is None:
        return env
    out = dict(env)
    out.pop("VIRTUAL_ENV", None)
    out.pop("CONDA_PREFIX", None)
    return out


@dataclass
class ComfyEnv:
    """What the Comfy interpreter reports. ``python_path`` is None when unknown."""

    python_path: str | None = None
    source: str = "unknown"  # workspace-venv | portable-embedded | unknown
    python: str | None = None
    system: str = ""
    machine: str = ""
    torch: str | None = None
    cuda: str | None = None
    hip: str | None = None
    cuda_available: bool = False
    torch_triton_requirement: str | None = None
    gpu_name: str | None = None
    compute_capability: str | None = None  # "12.0" on Blackwell RTX 50xx
    triton: str | None = None
    sageattention: str | None = None
    errors: dict[str, str] = field(default_factory=dict)
    probe_error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        if not self.python_path:
            return (
                "Comfy interpreter not found (no .venv, venv, or portable python_embeded for the ComfyUI folder). "
                "Comfy may be running in a non-venv interpreter."
            )
        if self.probe_error:
            return f"Comfy Python at {self.python_path} did not answer: {self.probe_error}"
        cuda = self.cuda or ("ROCm " + self.hip if self.hip else "none")
        gpu = ""
        if self.compute_capability:
            gpu = f"; GPU {self.gpu_name or '?'} (sm_{self.compute_capability.replace('.', '')})"
        return (
            f"Comfy Python {self.python or '?'} at {self.python_path}; "
            f"torch {self.torch or 'not installed'}; CUDA {cuda}{gpu}"
        )


def probe_comfy_env(
    workspace: Path | str | None,
    *,
    python: Path | str | None = None,
    runner: Runner = subprocess.run,
) -> ComfyEnv:
    """Run a tiny script inside the Comfy interpreter. Never Buddy's interpreter."""
    from master_agent.comfy.layout import PORTABLE, detect_layout

    layout = None if python else detect_layout(workspace)
    if python:
        py: Path | None = Path(python)
    elif layout is not None:
        py = layout.python
    else:
        py = workspace_venv_python(workspace)
    if py is None:
        return ComfyEnv()
    source = "portable-embedded" if layout is not None and layout.kind == PORTABLE else "workspace-venv"
    env = ComfyEnv(python_path=str(py), source=source)
    try:
        proc = runner(
            [str(py), "-c", _PROBE_SCRIPT],
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        env.probe_error = str(exc)[:300]
        return env
    if proc.returncode != 0:
        env.probe_error = ((proc.stderr or proc.stdout or "").strip() or f"exit {proc.returncode}")[:300]
        return env
    try:
        data = json.loads((proc.stdout or "").strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        env.probe_error = f"unreadable probe output ({exc})"
        return env
    for key in (
        "python",
        "system",
        "machine",
        "torch",
        "cuda",
        "hip",
        "cuda_available",
        "torch_triton_requirement",
        "gpu_name",
        "compute_capability",
        "triton",
        "sageattention",
        "errors",
    ):
        if key in data:
            setattr(env, key, data[key])
    return env


def module_importable(python: Path | str, module: str, *, runner: Runner = subprocess.run) -> bool:
    """``import <module>`` inside ``python`` (a subprocess, not this interpreter)."""
    try:
        proc = runner(
            [str(python), "-c", f"import {module}"],
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


# --- Triton / SageAttention plan --------------------------------------------


def _torch_minor(version: str | None) -> tuple[int, int] | None:
    m = re.match(r"(\d+)\.(\d+)", str(version or ""))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _torch_release(version: str | None) -> tuple[int, ...] | None:
    m = re.match(r"(\d+)\.(\d+)(?:\.(\d+))?", str(version or ""))
    if not m:
        return None
    return tuple(int(x) for x in m.groups(default="0"))


def _triton_windows_minor(torch_version: str | None) -> tuple[int | None, bool]:
    minor = _torch_minor(torch_version)
    if minor is None or minor < (2, 4):
        return None, False
    if minor in _TRITON_WINDOWS_TABLE:
        return _TRITON_WINDOWS_TABLE[minor], True
    last = max(_TRITON_WINDOWS_TABLE)
    if minor[0] == 2 and minor > last:
        return _TRITON_WINDOWS_TABLE[last] + (minor[1] - last[1]), False
    return None, False


def triton_windows_spec(torch_version: str | None) -> tuple[str | None, bool]:
    """``triton-windows>=3.N,<3.N+1`` for the installed torch. Returns ``(spec, verified)``."""
    tri, verified = _triton_windows_minor(torch_version)
    if tri is None:
        return None, False
    return f"triton-windows>=3.{tri},<3.{tri + 1}", verified


def _triton_minor_from_requirement(req: str | None) -> int | None:
    """``triton==3.7.1`` / ``triton~=3.8.0`` -> 8 (torch's own Linux pin)."""
    m = re.match(r"triton(?:==|~=|>=)3\.(\d+)", (req or "").replace(" ", "").lower())
    return int(m.group(1)) if m else None


def _version_key(v: str) -> tuple[int, ...]:
    m = re.match(r"(\d+)\.(\d+)\.(\d+)(?:\.post(\d+))?$", v)
    return tuple(int(x) for x in m.groups(default="0")) if m else ()


def fetch_triton_windows_versions(timeout: float = 10.0) -> dict[str, set[str]]:
    """Real triton-windows releases: ``{"3.8.0.post29": {"cp310", ...}}``.

    PyPI JSON first (has per-wheel cp tags); GitHub release tags as a fallback
    (no wheel info, so the cp set is empty = unknown). Raises on total failure.
    """
    try:
        with urllib.request.urlopen(TRITON_WINDOWS_PYPI_JSON, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        out: dict[str, set[str]] = {}
        for ver, files in (data.get("releases") or {}).items():
            tags = set()
            for f in files or []:
                if f.get("yanked"):
                    continue
                parts = str(f.get("filename") or "").split("-")
                if len(parts) >= 5 and parts[-1].endswith("win_amd64.whl"):
                    tags.add(parts[2])
            if tags:
                out[ver] = tags
        if out:
            return out
    except (urllib.error.URLError, OSError, ValueError, AttributeError):
        pass
    req = urllib.request.Request(TRITON_WINDOWS_RELEASES_API, headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        releases = json.loads(resp.read().decode("utf-8"))
    out = {}
    for rel in releases if isinstance(releases, list) else []:
        m = re.match(r"v(\d+\.\d+\.\d+)-windows\.(post\d+)$", str(rel.get("tag_name") or ""))
        if m and not rel.get("prerelease"):
            out[f"{m.group(1)}.{m.group(2)}"] = set()
    return out


def _newest_real(versions: dict[str, set[str]], tri: int, cp: str | None, *, at_most: bool = False) -> str | None:
    best: tuple[tuple[int, ...], str] | None = None
    for ver, tags in versions.items():
        key = _version_key(ver)
        if not key or key[0] != 3:
            continue
        if (key[1] > tri) if at_most else (key[1] != tri):
            continue
        if cp and tags and cp not in tags:
            continue
        if best is None or key > best[0]:
            best = (key, ver)
    return best[1] if best else None


def triton_windows_candidates(
    torch_version: str | None,
    *,
    python: str | None = None,
    torch_triton_requirement: str | None = None,
    versions: Callable[[], dict[str, set[str]]] | None = fetch_triton_windows_versions,
) -> list[tuple[str, str]]:
    """Ordered ``(pip spec, note)`` tries for triton-windows.

    1. a real ``==`` pin on the Triton minor torch itself pins on Linux,
    2. a real pin on the static-table minor,
    3. the newest real 3.x at or below the expected minor for this Python,
    4. the table / extrapolated range (last resort; pip resolves it).
    A network failure only drops the real pins.
    """
    tri, verified = _triton_windows_minor(torch_version)
    if tri is None:
        return []
    pin_minor = _triton_minor_from_requirement(torch_triton_requirement)
    pm = re.match(r"(\d+)\.(\d+)", str(python or ""))
    cp = f"cp{pm.group(1)}{pm.group(2)}" if pm else None
    real: dict[str, set[str]] = {}
    if versions is not None:
        try:
            real = versions() or {}
        except (urllib.error.URLError, OSError, ValueError, AttributeError):
            real = {}
    out: list[tuple[str, str]] = []

    def add(spec: str, note: str) -> None:
        if spec not in (s for s, _ in out):
            out.append((spec, note))

    if real:
        if pin_minor is not None:
            v = _newest_real(real, pin_minor, cp)
            if v:
                add(f"triton-windows=={v}", f"matches torch's Triton 3.{pin_minor} pin")
        v = _newest_real(real, tri, cp)
        if v:
            add(f"triton-windows=={v}", "verified table" if verified else "extrapolated minor, real release")
        v = _newest_real(real, max(tri, pin_minor or 0), cp, at_most=True)
        if v:
            add(f"triton-windows=={v}", "newest real release not newer than expected (unverified with this torch)")
    if pin_minor is not None and pin_minor != tri:
        add(f"triton-windows>=3.{pin_minor},<3.{pin_minor + 1}", f"torch's Triton 3.{pin_minor} pin")
    add(
        f"triton-windows>=3.{tri},<3.{tri + 1}",
        "" if verified else "torch newer than the triton-windows table; pin extrapolated",
    )
    return out


def triton_linux_spec(env: ComfyEnv) -> str:
    """Torch's own ``triton==`` pin when it declares one, else plain ``triton``."""
    req = (env.torch_triton_requirement or "").strip()
    if req.lower().startswith(("triton==", "triton~=")):
        return req
    return "triton"


def _cuda_tag(cuda: str | None) -> str | None:
    m = re.match(r"(\d+)\.(\d+)", str(cuda or ""))
    if not m:
        return None
    return f"cu{m.group(1)}{m.group(2)}"


def pick_windows_sage_wheel(
    assets: list[dict[str, Any]], *, torch_version: str | None, cuda: str | None
) -> str | None:
    """Best prebuilt SageAttention wheel for this CUDA tag and torch version.

    Asset names look like ``sageattention-2.2.0+cu128torch2.9.1.post6-cp310-abi3-win_amd64.whl``
    or ``...+cu130torch2.10.0andhigher.post6-...``. ``cp310-abi3`` / ``cp39-abi3`` cover newer Pythons.

    * exact torch ``X.Y.Z`` match, or ``andhigher`` with installed torch >= the floor;
    * same CUDA *major* only. An exact tag (cu130 for torch cu130) wins; a lower
      minor of the same major (cu128 wheel on cu129 torch) is accepted. A wheel built
      for another major (cu128 on cu130 torch) is rejected: it links CUDA 12 runtime
      libraries that a cu130 torch does not ship, so it fails to load.
    * rocm / cpu wheels never match. Assets are in newest-release-first order and the
      first best match wins.
    """
    m_cuda = re.match(r"(\d+)\.(\d+)", str(cuda or ""))
    rel = _torch_release(torch_version)
    if not m_cuda or rel is None:
        return None
    want_major, want_minor = int(m_cuda.group(1)), int(m_cuda.group(2))
    pattern = re.compile(r"\+cu(\d+)torch(\d+)\.(\d+)\.(\d+)(andhigher)?\.")
    best: tuple[tuple[int, int], str] | None = None
    for asset in assets:
        name = str(asset.get("name") or "")
        url = str(asset.get("browser_download_url") or "")
        if not name.endswith("win_amd64.whl") or not url:
            continue
        m = pattern.search(name)
        if not m:
            continue
        cu = m.group(1)
        major, minor = int(cu[:-1]), int(cu[-1])
        if major != want_major or minor > want_minor:
            continue
        floor = tuple(int(x) for x in m.groups()[1:4])
        if m.group(5):
            if rel < floor:
                continue
        elif floor != rel:
            continue
        # exact cuda beats lower minor; exact torch beats andhigher
        score = (1 if minor == want_minor else 0, 0 if m.group(5) else 1)
        if best is None or score > best[0]:
            best = (score, url)
    return best[1] if best else None


def sage_source_build_enabled(env: dict[str, str] | None = None) -> bool:
    val = (env if env is not None else os.environ).get(SAGE_SOURCE_BUILD_ENV, "")
    return val.strip().lower() in {"1", "true", "yes", "on"}


def sage_arch_list(compute_capability: str | None) -> str:
    """``TORCH_CUDA_ARCH_LIST`` for the build: the GPU's own capability (``12.0`` on RTX 50xx)."""
    m = re.match(r"(\d+)\.(\d+)", str(compute_capability or ""))
    return f"{m.group(1)}.{m.group(2)}" if m else "8.0;8.6;8.9;9.0;12.0"


def sage_source_build_commands(python: str, build_dir: Path) -> list[list[str]]:
    """git clone + ``pip install --no-build-isolation`` into the Comfy interpreter."""
    src = build_dir / "SageAttention"
    cmds: list[list[str]] = []
    if not (src / "setup.py").is_file():
        cmds.append(["git", "clone", "--depth", "1", SAGE_SOURCE_REPO, str(src)])
    cmds.append([python, "-m", "pip", "install", "setuptools", "wheel", "ninja", "packaging"])
    cmds.append([python, "-m", "pip", "install", "--no-build-isolation", "-v", str(src)])
    return cmds


def fetch_windows_sage_assets(timeout: float = 10.0) -> list[dict[str, Any]]:
    req = urllib.request.Request(
        SAGE_WINDOWS_RELEASES_API, headers={"Accept": "application/vnd.github+json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        releases = json.loads(resp.read().decode("utf-8"))
    out: list[dict[str, Any]] = []
    for rel in releases if isinstance(releases, list) else []:
        out.extend(a for a in rel.get("assets") or [] if isinstance(a, dict))
    return out


@dataclass
class AccelStep:
    name: str
    status: str  # OK | SKIPPED | FAILED-NONFATAL | PLANNED
    detail: str
    command: list[str] = field(default_factory=list)


@dataclass
class AccelReport:
    env: ComfyEnv
    steps: list[AccelStep] = field(default_factory=list)
    triton_present: bool = False
    sage_present: bool = False

    @property
    def status(self) -> str:
        if self.triton_present and self.sage_present:
            return "OK"
        if any(s.status == "FAILED-NONFATAL" for s in self.steps):
            return "FAILED-NONFATAL"
        if any(s.status == "PLANNED" for s in self.steps):
            return "PLANNED"
        return "SKIPPED"

    def lines(self) -> list[str]:
        out = [self.env.summary()]
        out.extend(f"{s.status:<16} {s.name}: {s.detail}" for s in self.steps)
        out.append(
            "triton: " + ("present" if self.triton_present else "absent")
            + "; sageattention: " + ("present" if self.sage_present else "absent")
            + f" (Python {self.env.python or '?'}, CUDA {self.env.cuda or 'none'})"
        )
        if not self.sage_present:
            out.append(
                "SageAttention is unavailable: Comfy launches without --use-sage-attention "
                "and runs, just slower."
            )
        return out


_SLOWER = "Comfy will run, just slower."


def _skip_reason(env: ComfyEnv, *, gpu_vendor: str, host_system: str, host_machine: str) -> str | None:
    system = env.system or host_system
    machine = (env.machine or host_machine or "").lower()
    if system == "Darwin":
        return f"macOS has no Triton / SageAttention build; continuing without Sage Attention. {_SLOWER}"
    if gpu_vendor != "nvidia":
        return f"No NVIDIA GPU (CPU-only or non-CUDA); skipping Triton and Sage Attention. {_SLOWER}"
    if system == "Linux" and machine not in {"x86_64", "amd64", "aarch64", "arm64"}:
        return f"Unsupported Linux CPU {machine!r}; continuing without Sage Attention. {_SLOWER}"
    if system == "Windows" and machine not in {"amd64", "x86_64"}:
        return f"Unsupported Windows CPU {machine!r}; continuing without Sage Attention. {_SLOWER}"
    if system not in {"Linux", "Windows"}:
        return f"Unsupported platform {system!r}; continuing without Sage Attention. {_SLOWER}"
    if not env.torch:
        return f"PyTorch is not installed in the Comfy venv; skipping Triton and Sage Attention. {_SLOWER}"
    if not env.cuda:
        return (
            f"The Comfy venv has a non-CUDA PyTorch ({env.torch}); "
            f"skipping Triton and Sage Attention. {_SLOWER}"
        )
    return None


def install_accel(
    workspace: Path | str,
    *,
    gpu_vendor: str,
    apply: bool,
    runner: Runner = subprocess.run,
    sage_assets: Callable[[], list[dict[str, Any]]] = fetch_windows_sage_assets,
    progress: Callable[[str], None] = print,
    host_system: str | None = None,
    host_machine: str | None = None,
    triton_versions: Callable[[], dict[str, set[str]]] | None = fetch_triton_windows_versions,
    sage_source_build: bool | None = None,
    build_dir: Path | str | None = None,
) -> AccelReport:
    """Triton first, then SageAttention, into the Comfy venv. Never raises."""
    env = probe_comfy_env(workspace, runner=runner)
    report = AccelReport(env=env)
    host_system = host_system or _platform.system()
    host_machine = host_machine or _platform.machine()
    if not env.python_path:
        report.steps.append(
            AccelStep(
                "triton+sageattention",
                "SKIPPED",
                "Could not find the Comfy interpreter (no .venv or venv in the ComfyUI folder); "
                f"Buddy will not pip-install into an unknown Python. {_SLOWER}",
            )
        )
        return report
    if env.probe_error:
        report.steps.append(
            AccelStep("triton+sageattention", "SKIPPED", f"Comfy Python did not answer ({env.probe_error}). {_SLOWER}")
        )
        return report
    py = env.python_path
    report.triton_present = bool(env.triton)
    report.sage_present = bool(env.sageattention)
    if report.triton_present and report.sage_present:
        report.steps.append(AccelStep("triton", "SKIPPED", f"already installed ({env.triton})"))
        report.steps.append(AccelStep("sageattention", "SKIPPED", f"already installed ({env.sageattention})"))
        return report
    reason = _skip_reason(env, gpu_vendor=gpu_vendor, host_system=host_system, host_machine=host_machine)
    if reason:
        report.steps.append(AccelStep("triton+sageattention", "SKIPPED", reason))
        return report
    system = env.system or host_system

    def pip(spec: str) -> list[str]:
        return [py, "-m", "pip", "install", spec]

    def run_pip(cmd: list[str], *, env: dict[str, str] | None = None, timeout: float = PIP_TIMEOUT_S) -> tuple[bool, str]:
        progress("$ " + " ".join(cmd))
        kw: dict[str, Any] = {"env": env} if env is not None else {}
        try:
            proc = runner(cmd, capture_output=True, text=True, timeout=timeout, check=False, **kw)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return False, str(exc)[:300]
        tail = ((proc.stderr or "") + (proc.stdout or "")).strip()[-300:]
        return proc.returncode == 0, tail

    # Triton
    if report.triton_present:
        report.steps.append(AccelStep("triton", "SKIPPED", f"already installed ({env.triton})"))
    else:
        if system == "Windows":
            tries = triton_windows_candidates(
                env.torch,
                python=env.python,
                torch_triton_requirement=env.torch_triton_requirement,
                versions=triton_versions,
            )
            if not tries:
                report.steps.append(
                    AccelStep(
                        "triton",
                        "FAILED-NONFATAL",
                        f"No Triton build is available for Python {env.python} / CUDA {env.cuda} / "
                        f"torch {env.torch} on this machine; continuing without Sage Attention. {_SLOWER}",
                    )
                )
                report.steps.append(AccelStep("sageattention", "SKIPPED", "needs Triton"))
                return report
        else:
            spec = triton_linux_spec(env)
            tries = [(spec, "torch's own triton pin" if spec != "triton" else "")]
        if not apply:
            spec, note = tries[0]
            report.steps.append(AccelStep("triton", "PLANNED", spec + (f" ({note})" if note else ""), pip(spec)))
        else:
            tail = ""
            cmd: list[str] = []
            for spec, note in tries:
                cmd = pip(spec)
                ok, tail = run_pip(cmd)
                if ok and module_importable(py, "triton", runner=runner):
                    report.triton_present = True
                    report.steps.append(
                        AccelStep("triton", "OK", f"installed {spec}" + (f" ({note})" if note else ""), cmd)
                    )
                    break
                if len(tries) > 1:
                    progress(f"INFO  {spec} did not install/import; trying next triton-windows option")
            if not report.triton_present:
                report.steps.append(
                    AccelStep(
                        "triton",
                        "FAILED-NONFATAL",
                        f"No Triton build could be installed for Python {env.python} / CUDA {env.cuda} "
                        f"on this machine ({tail or 'import failed'}); continuing without Sage Attention. "
                        f"{_SLOWER}",
                        cmd,
                    )
                )
                report.steps.append(AccelStep("sageattention", "SKIPPED", "needs Triton"))
                return report

    # SageAttention
    if report.sage_present:
        report.steps.append(AccelStep("sageattention", "SKIPPED", f"already installed ({env.sageattention})"))
        return report
    # Each candidate: (label, list of commands, env overrides, timeout).
    candidates: list[tuple[str, list[list[str]], dict[str, str] | None, float]] = []
    if system == "Windows":
        wheel = None
        try:
            wheel = pick_windows_sage_wheel(sage_assets(), torch_version=env.torch, cuda=env.cuda)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            progress(f"INFO  could not list prebuilt SageAttention wheels ({exc}); trying other options")
        if wheel:
            candidates.append(("prebuilt Windows wheel", [pip(wheel)], None, PIP_TIMEOUT_S))
        build = sage_source_build_enabled() if sage_source_build is None else sage_source_build
        if build:
            # Before PyPI on purpose: PyPI "sageattention" is the 1.0.x Triton-only
            # kernel set; it imports, which would stop us ever reaching SageAttention 2.
            bdir = Path(build_dir) if build_dir else Path(py).resolve().parents[2] / ".buddy-build"
            benv = dict(os.environ)
            arch = sage_arch_list(env.compute_capability)
            benv.update({"TORCH_CUDA_ARCH_LIST": arch, "MAX_JOBS": benv.get("MAX_JOBS", "4"), "DISTUTILS_USE_SDK": "1"})
            # Some forks read a bare CUDA_ARCH / SAGEATTENTION_CUDA_ARCH; harmless otherwise.
            benv.setdefault("SAGEATTENTION_CUDA_ARCH", arch)
            candidates.append(
                (f"source build (TORCH_CUDA_ARCH_LIST={arch})", sage_source_build_commands(py, bdir), benv, SAGE_BUILD_TIMEOUT_S)
            )
    candidates.append(("PyPI package", [pip("sageattention")], None, PIP_TIMEOUT_S))
    if not apply:
        label, cmds, _, _ = candidates[0]
        report.steps.append(AccelStep("sageattention", "PLANNED", f"{label}: {cmds[-1][-1]}", cmds[-1]))
        return report
    last_tail = ""
    for i, (label, cmds, cenv, timeout) in enumerate(candidates):
        ok, tail = True, ""
        for cmd in cmds:
            if label.startswith("source build") and cmd[0] != py:
                if cmd[0] == "git" and cmd[-1]:
                    try:
                        Path(cmd[-1]).parent.mkdir(parents=True, exist_ok=True)
                    except OSError as exc:
                        ok, tail = False, str(exc)[:300]
                        break
            ok, tail = run_pip(cmd, env=cenv, timeout=timeout)
            if not ok:
                break
        if ok and module_importable(py, "sageattention", runner=runner):
            report.sage_present = True
            report.steps.append(AccelStep("sageattention", "OK", f"installed {label}", cmds[-1]))
            return report
        last_tail = tail
        nxt = "trying next option" if i < len(candidates) - 1 else "giving up (non-fatal, see summary)"
        progress(f"INFO  sageattention via {label} did not import; {nxt}")
    hint = ""
    if system == "Windows" and not any(c[0].startswith("source build") for c in candidates):
        hint = (
            f" To try a from-source build (needs git, MSVC Build Tools and CUDA toolkit {env.cuda}), "
            f"set {SAGE_SOURCE_BUILD_ENV}=1 and re-run."
        )
    report.steps.append(
        AccelStep(
            "sageattention",
            "FAILED-NONFATAL",
            f"SageAttention could not be installed for Python {env.python} / CUDA {env.cuda} "
            f"({last_tail or 'import failed'}); continuing without Sage Attention. {_SLOWER}{hint}",
        )
    )
    return report
