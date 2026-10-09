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

# triton-windows README: each PyTorch minor is only guaranteed with one Triton minor.
# 2.4/2.5 -> 3.1, then 2.6 -> 3.2 ... 2.10 -> 3.6. Newer torch extrapolates the
# same +1 step and is reported as unverified.
_TRITON_WINDOWS_TABLE: dict[tuple[int, int], int] = {
    (2, 4): 1,
    (2, 5): 1,
    (2, 6): 2,
    (2, 7): 3,
    (2, 8): 4,
    (2, 9): 5,
    (2, 10): 6,
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
    "mps_available": False, "torch_triton_requirement": None,
    "triton": None, "sageattention": None, "errors": {},
}
try:
    import torch
    out["torch"] = torch.__version__
    out["cuda"] = getattr(torch.version, "cuda", None)
    out["hip"] = getattr(torch.version, "hip", None)
    try:
        out["cuda_available"] = bool(torch.cuda.is_available())
    except Exception as exc:
        out["errors"]["cuda_available"] = str(exc)[:200]
    try:
        out["mps_available"] = bool(torch.backends.mps.is_available())
    except Exception:
        pass
except Exception as exc:
    out["errors"]["torch"] = str(exc)[:200]
try:
    from importlib.metadata import requires
    for req in requires("torch") or []:
        name = req.split(";", 1)[0].strip()
        if name.lower().startswith(("triton==", "triton ==", "pytorch-triton==")):
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
    """Drop ``VIRTUAL_ENV`` / ``CONDA_PREFIX`` once the workspace has its own venv.

    Without this, comfy-cli would run Comfy in Buddy's activated venv.
    """
    if workspace_venv_python(workspace) is None:
        return env
    out = dict(env)
    out.pop("VIRTUAL_ENV", None)
    out.pop("CONDA_PREFIX", None)
    return out


@dataclass
class ComfyEnv:
    """What the Comfy interpreter reports. ``python_path`` is None when unknown."""

    python_path: str | None = None
    source: str = "unknown"  # workspace-venv | unknown
    python: str | None = None
    system: str = ""
    machine: str = ""
    torch: str | None = None
    cuda: str | None = None
    hip: str | None = None
    cuda_available: bool = False
    mps_available: bool = False
    torch_triton_requirement: str | None = None
    triton: str | None = None
    sageattention: str | None = None
    errors: dict[str, str] = field(default_factory=dict)
    probe_error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        if not self.python_path:
            return (
                "Comfy interpreter not found (no .venv or venv in the ComfyUI folder). "
                "Comfy may be running in a non-venv interpreter."
            )
        if self.probe_error:
            return f"Comfy Python at {self.python_path} did not answer: {self.probe_error}"
        cuda = self.cuda or ("ROCm " + self.hip if self.hip else "none")
        return (
            f"Comfy Python {self.python or '?'} at {self.python_path}; "
            f"torch {self.torch or 'not installed'}; CUDA {cuda}"
        )


def probe_comfy_env(
    workspace: Path | str | None,
    *,
    python: Path | str | None = None,
    runner: Runner = subprocess.run,
) -> ComfyEnv:
    """Run a tiny script inside the Comfy interpreter. Never Buddy's interpreter."""
    py = Path(python) if python else workspace_venv_python(workspace)
    if py is None:
        return ComfyEnv()
    env = ComfyEnv(python_path=str(py), source="workspace-venv")
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
        "mps_available",
        "torch_triton_requirement",
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


def needs_cpu_flag(
    env: ComfyEnv, *, gpu_vendor: str, host_system: str, host_machine: str
) -> tuple[bool, str]:
    """Whether ComfyUI must be launched with ``--cpu``. Returns ``(cpu, reason)``.

    ComfyUI calls ``torch.cuda.current_device()`` at startup unless it finds
    MPS or is told ``--cpu``, so a CPU-only PyTorch crashes without the flag.
    The Comfy venv's torch decides when the probe answered; otherwise the
    hardware scan does. A CUDA build on a machine that does have a GPU is left
    alone, so a broken driver fails loudly instead of silently running on CPU.
    """
    vendor = (gpu_vendor or "").strip().lower()
    no_gpu = vendor in {"", "none", "unknown"}
    if env.python_path and not env.probe_error and env.torch:
        if env.cuda or env.hip:
            if env.cuda_available or not no_gpu:
                return False, f"PyTorch {env.torch} has GPU support"
            return True, f"PyTorch {env.torch} has GPU support but no GPU was found"
        if env.mps_available:
            return False, f"PyTorch {env.torch} uses Apple MPS"
        return True, f"the Comfy venv has a CPU-only PyTorch ({env.torch})"
    apple_silicon = host_system == "Darwin" and host_machine.lower() in {"arm64", "aarch64"}
    if no_gpu and not apple_silicon:
        return True, "no GPU detected"
    return False, f"GPU detected ({vendor or 'Apple Silicon'})"


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


def triton_windows_spec(torch_version: str | None) -> tuple[str | None, bool]:
    """``triton-windows<3.N+1`` for the installed torch. Returns ``(spec, verified)``."""
    minor = _torch_minor(torch_version)
    if minor is None or minor < (2, 4):
        return None, False
    if minor in _TRITON_WINDOWS_TABLE:
        tri = _TRITON_WINDOWS_TABLE[minor]
        verified = True
    elif minor[0] == 2 and minor[1] > 10:
        tri = minor[1] - 4
        verified = False
    else:
        return None, False
    return f"triton-windows>=3.{tri},<3.{tri + 1}", verified


def triton_linux_spec(env: ComfyEnv) -> str:
    """Torch's own ``triton==`` pin when it declares one, else plain ``triton``."""
    req = (env.torch_triton_requirement or "").strip()
    if req.lower().startswith("triton=="):
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
    """First prebuilt SageAttention wheel for this CUDA tag and torch version.

    Asset names look like ``sageattention-2.2.0+cu128torch2.9.1.post6-cp310-abi3-win_amd64.whl``
    or ``...+cu128torch2.10.0andhigher.post6-...``. ``cp310-abi3`` covers Python 3.10+.
    """
    tag = _cuda_tag(cuda)
    rel = _torch_release(torch_version)
    if not tag or rel is None:
        return None
    exact = f"torch{rel[0]}.{rel[1]}.{rel[2]}"
    pattern = re.compile(re.escape(tag) + r"torch(\d+)\.(\d+)\.(\d+)(andhigher)?\.")
    for asset in assets:
        name = str(asset.get("name") or "")
        url = str(asset.get("browser_download_url") or "")
        if not name.endswith("win_amd64.whl") or not url:
            continue
        m = pattern.search(name)
        if not m:
            continue
        want = tuple(int(x) for x in m.groups()[:3])
        if m.group(4):
            if rel >= want:
                return url
        elif f"torch{m.group(1)}.{m.group(2)}.{m.group(3)}" == exact:
            return url
    return None


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

    def run_pip(cmd: list[str]) -> tuple[bool, str]:
        progress("$ " + " ".join(cmd))
        try:
            proc = runner(cmd, capture_output=True, text=True, timeout=PIP_TIMEOUT_S, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return False, str(exc)[:300]
        tail = ((proc.stderr or "") + (proc.stdout or "")).strip()[-300:]
        return proc.returncode == 0, tail

    # Triton
    if report.triton_present:
        report.steps.append(AccelStep("triton", "SKIPPED", f"already installed ({env.triton})"))
    else:
        if system == "Windows":
            spec, verified = triton_windows_spec(env.torch)
            if spec is None:
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
            note = "" if verified else " (torch newer than the triton-windows table; pin extrapolated)"
        else:
            spec = triton_linux_spec(env)
            note = " (torch's own triton pin)" if spec != "triton" else ""
        cmd = pip(spec)
        if not apply:
            report.steps.append(AccelStep("triton", "PLANNED", f"{spec}{note}", cmd))
        else:
            ok, tail = run_pip(cmd)
            if ok and module_importable(py, "triton", runner=runner):
                report.triton_present = True
                report.steps.append(AccelStep("triton", "OK", f"installed {spec}{note}", cmd))
            else:
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
    candidates: list[tuple[str, str]] = []
    if system == "Windows":
        wheel = None
        try:
            wheel = pick_windows_sage_wheel(sage_assets(), torch_version=env.torch, cuda=env.cuda)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            progress(f"WARN  could not list prebuilt SageAttention wheels ({exc}); trying PyPI")
        if wheel:
            candidates.append((wheel, "prebuilt Windows wheel"))
    candidates.append(("sageattention", "PyPI package"))
    if not apply:
        spec, label = candidates[0]
        report.steps.append(AccelStep("sageattention", "PLANNED", f"{label}: {spec}", pip(spec)))
        return report
    last_tail = ""
    for spec, label in candidates:
        cmd = pip(spec)
        ok, tail = run_pip(cmd)
        if ok and module_importable(py, "sageattention", runner=runner):
            report.sage_present = True
            report.steps.append(AccelStep("sageattention", "OK", f"installed {label}", cmd))
            return report
        last_tail = tail
        progress(f"WARN  sageattention via {label} did not import; {'trying next option' if spec != candidates[-1][0] else 'giving up'}")
    report.steps.append(
        AccelStep(
            "sageattention",
            "FAILED-NONFATAL",
            f"SageAttention could not be installed for Python {env.python} / CUDA {env.cuda} "
            f"({last_tail or 'import failed'}); continuing without Sage Attention. {_SLOWER}",
        )
    )
    return report
