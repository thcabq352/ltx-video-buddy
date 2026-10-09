"""Automatic install: one command from an empty machine to a working LTX 2.3 generate.

``python -m master_agent automatic-install`` runs a plain-English pre-flight
first, then prints the plan. Nothing changes without ``--yes``.

With ``--yes`` it installs, in order: ffmpeg, comfy-cli at the
requirements.txt pin, ComfyUI into its own folder (``MANAGED_COMFY_ROOT`` or
``PROJECT_ROOT/ComfyUI``), the custom nodes the LTX 2.3 graphs need, Triton +
SageAttention inside the Comfy venv (optional), the LTX 2.3 weights into
``MODELS_DIR``, then llama.cpp into ``LLAMACPP_ROOT`` (optional). LTX 2.3 is
the only video model this command installs. Weights are never written into
the Comfy tree; managed launch keeps passing Buddy's ``extra_model_paths.yaml``.

The local LLM models (``LLAMACPP_MODEL`` for brief / director / judge,
``KB_EMBED_MODEL`` for knowledge-base embeddings) are GGUFs llama-server
serves from ``MODELS_DIR``. The repo defines no public download source for
them, so they are skipped with a plain message and the install continues.
Automatic install never calls Ollama.

``doctor`` stays a scanner. This module is the only thing that installs.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

LTX23_BUNDLE = "ltx23_core"
# Packs whose classes the shipped LTX 2.3 graphs (base / eros / directors) use,
# per state/object_info.json python_module: GuiderParameters / MultimodalGuider /
# LTXVTiledVAEDecode (ComfyUI-LTXVideo) and UnetLoaderGGUF, which the patcher
# swaps in when the GGUF is on disk (ComfyUI-GGUF). Comfy registry ids.
REQUIRED_NODES = ("ComfyUI-LTXVideo", "ComfyUI-GGUF")

GB = 1_000_000_000
# ComfyUI checkout + its own venv with the PyTorch build for the detected GPU
# (NVIDIA CUDA, AMD ROCm, Apple Silicon, or CPU) + custom nodes + Triton.
COMFY_OVERHEAD_BYTES = 15 * GB
DISK_HEADROOM_BYTES = 5 * GB

PASS, FAIL, INFO = "PASS", "FAIL", "INFO"
OK, SKIPPED, FAILED, NONFATAL, PLANNED, NEEDS_YOU = (
    "OK",
    "SKIPPED",
    "FAILED",
    "FAILED-NONFATAL",
    "PLANNED",
    "NEEDS-YOU",
)


# --- small seams (tests replace these) --------------------------------------


def _which(name: str) -> str | None:
    return shutil.which(name)


def git_path() -> str | None:
    return shutil.which("git")


def _run(cmd: list[str], *, timeout: float = 600.0, env: dict[str, str] | None = None) -> tuple[int, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False, env=env)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)
    return proc.returncode, ((proc.stdout or "") + (proc.stderr or "")).strip()


def disk_free(path: Path) -> int:
    return int(shutil.disk_usage(path).free)


def _volume_id(path: Path) -> int:
    return os.stat(path).st_dev


def _is_root() -> bool:
    geteuid = getattr(os, "geteuid", None)
    return bool(geteuid and geteuid() == 0)


def hf_whoami(token: str) -> tuple[bool, str]:
    """Lightweight token check (``/api/whoami-v2``). Never downloads, never prints the token."""
    try:
        from huggingface_hub import whoami

        info = whoami(token=token)
        return True, str(info.get("name") or "account")
    except Exception as exc:
        text = str(exc)
        if "401" in text or "Invalid" in text or "Unauthorized" in text:
            return False, "Hugging Face rejected the token (401 Unauthorized)"
        return False, f"could not reach Hugging Face to check the token ({type(exc).__name__})"


def hf_repo_access(repo_id: str, token: str | None) -> tuple[bool, str]:
    """Metadata request on a repo (no file download)."""
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import GatedRepoError

        try:
            HfApi().model_info(repo_id, token=token)
        except GatedRepoError:
            return False, "access denied (license not accepted on the repo page)"
        return True, "access ok"
    except Exception as exc:
        return False, f"metadata check failed ({type(exc).__name__})"


def detect_gpu() -> dict[str, Any]:
    from master_agent.comfy.hardware import scan_hardware

    return scan_hardware()


# --- paths -------------------------------------------------------------------


def managed_workspace() -> Path:
    from master_agent.comfy.tower import _default_workspace, load_state

    st = load_state()
    return Path(st.workspace or _default_workspace()).expanduser().resolve()


def models_dir() -> Path:
    from master_agent import config

    return Path(config.MODELS_DIR)


def llamacpp_root() -> Path:
    from master_agent import config

    return Path(config.LLAMACPP_ROOT)


def llamacpp_ref() -> str:
    from master_agent import config

    return str(config.LLAMACPP_REF)


def existing_llama_server() -> str | None:
    """``LLAMACPP_BIN`` or ``llama-server`` on PATH. Buddy's own copy is checked by the plan."""
    from master_agent import config

    explicit = (config.LLAMACPP_BIN or "").strip().strip('"')
    if explicit:
        return explicit if Path(explicit).is_file() else _which(explicit)
    return _which("llama-server")


def plan_llamacpp(gpu: dict[str, Any]) -> Any:
    from master_agent.llamacpp_install import plan_llamacpp as _plan

    return _plan(
        gpu_vendor=str(gpu.get("vendor") or ""),
        root=llamacpp_root(),
        ref=llamacpp_ref(),
        existing=existing_llama_server(),
        which=_which,
    )


def llm_models() -> list[tuple[str, str]]:
    """The GGUF names the existing llama.cpp config serves, and what each is for."""
    from master_agent import config

    out: list[tuple[str, str]] = []
    for name, role in (
        (config.LLAMACPP_MODEL, "text + vision: brief, director, judge"),
        (config.KB_EMBED_MODEL, "knowledge-base embeddings"),
    ):
        if name and name not in {n for n, _ in out}:
            out.append((name, role))
    return out


def local_llm_gguf(name: str) -> Path | None:
    """A GGUF llama-server's ``--models-dir`` router would serve as ``name``."""
    want = name.strip().lower()
    root = models_dir()
    if not want or not root.is_dir():
        return None
    for path in sorted(root.glob("*.gguf")) + sorted(root.glob("*/*.gguf")):
        label = (path.parent.name if path.parent != root else path.stem).lower()
        if label == want or label.startswith(want + "-") or label.startswith(want + "."):
            return path
    return None


def existing_ancestor(path: Path) -> Path:
    p = Path(path).expanduser().absolute()
    while not p.exists() and p.parent != p:
        p = p.parent
    return p


def fmt_gb(n: int | float) -> str:
    return f"{float(n) / GB:.1f} GB"


# --- comfy workspace state ---------------------------------------------------


def comfy_install_state(workspace: Path) -> str:
    """``absent`` | ``installed`` | ``partial`` | ``foreign``."""
    from master_agent.comfy.comfy_venv import module_importable, workspace_venv_python

    if not workspace.exists():
        return "absent"
    if workspace.is_dir() and not any(workspace.iterdir()):
        return "absent"
    looks_comfy = (workspace / "main.py").is_file() and (workspace / "comfy").is_dir()
    if not looks_comfy:
        return "foreign"
    py = workspace_venv_python(workspace)
    if py is None or not module_importable(py, "torch"):
        return "partial"
    return "installed"


def missing_nodes(workspace: Path, nodes: Iterable[str] = REQUIRED_NODES) -> list[str]:
    folder = workspace / "custom_nodes"
    have = {p.name.lower() for p in folder.iterdir() if p.is_dir()} if folder.is_dir() else set()
    return [n for n in nodes if n.lower() not in have]


def gpu_install_flag(gpu: dict[str, Any], override: str = "auto") -> str:
    """comfy-cli GPU flag. ``--skip-prompt`` alone would still ask on Linux/Windows."""
    if override and override != "auto":
        return f"--{override}"
    vendor = str(gpu.get("vendor") or "").lower()
    if sys.platform == "darwin":
        return "--m-series" if platform.machine().lower() in {"arm64", "aarch64"} else "--cpu"
    if vendor == "nvidia":
        return "--nvidia"
    if vendor == "amd":
        return "--amd"
    return "--cpu"


def comfy_argv(workspace: Path, *args: str) -> list[str]:
    from master_agent.comfy.tower import resolve_comfy_cli

    return [*resolve_comfy_cli(), f"--workspace={workspace}", "--where", "local", "--skip-prompt", *args]


def comfy_install_env() -> dict[str, str]:
    """comfy-cli would install Comfy into an activated outer venv; force the workspace venv."""
    env = os.environ.copy()
    env["COMFY_WHERE"] = "local"
    env.pop("VIRTUAL_ENV", None)
    env.pop("CONDA_PREFIX", None)
    return env


# --- pre-flight --------------------------------------------------------------


@dataclass
class Check:
    name: str
    status: str  # PASS | FAIL | INFO
    lines: list[str]
    blocking: bool = False

    @property
    def failed(self) -> bool:
        return self.blocking and self.status == FAIL


@dataclass
class Preflight:
    checks: list[Check]
    weights: Any = None  # WeightStatus for LTX23_BUNDLE
    gpu: dict[str, Any] = field(default_factory=dict)
    workspace_state: str = "absent"
    llama: Any = None  # llamacpp_install.LlamaPlan, None with --skip-llm

    @property
    def first_failure(self) -> Check | None:
        return next((c for c in self.checks if c.failed), None)

    @property
    def ok(self) -> bool:
        return self.first_failure is None


def check_disk(
    *,
    weights_bytes: int,
    comfy_bytes: int,
    llama_bytes: int,
    models: Path,
    workspace: Path,
    llama_dir: Path,
) -> Check:
    needs: dict[int, dict[str, Any]] = {}
    for label, path, size, env_var in (
        ("LTX 2.3 weights", models, weights_bytes, "MODELS_DIR"),
        ("ComfyUI + its venv", workspace, comfy_bytes, "MANAGED_COMFY_ROOT"),
        ("llama.cpp", llama_dir, llama_bytes, "LLAMACPP_ROOT"),
    ):
        if size <= 0:
            continue
        anchor = existing_ancestor(path)
        vol = _volume_id(anchor)
        slot = needs.setdefault(vol, {"anchor": anchor, "need": 0, "parts": [], "vars": []})
        slot["need"] += size
        slot["parts"].append(f"{label} {fmt_gb(size)} → {path}")
        slot["vars"].append(env_var)
    lines: list[str] = []
    failed = False
    if not needs:
        return Check("Disk space", PASS, ["Nothing left to download or install."], blocking=True)
    for slot in needs.values():
        need = slot["need"] + DISK_HEADROOM_BYTES
        free = disk_free(slot["anchor"])
        ok = free >= need
        failed = failed or not ok
        lines.append(
            f"{'enough' if ok else 'NOT enough'} space on the drive holding {slot['anchor']}: "
            f"{fmt_gb(free)} free, {fmt_gb(need)} needed "
            f"(includes {fmt_gb(DISK_HEADROOM_BYTES)} headroom)."
        )
        lines.extend(f"  - {part}" for part in slot["parts"])
        if not ok:
            short = need - free
            lines.append(f"  Free up at least {fmt_gb(short)} on that drive before running again.")
            lines.append(
                "  Ideas: empty the Recycle Bin / Trash, delete old downloads or videos you no longer need, "
                "or uninstall large games or apps."
            )
            lines.append(
                "  Or point Buddy at a bigger drive: add "
                + " / ".join(f"{v}=<folder on the bigger drive>" for v in dict.fromkeys(slot["vars"]))
                + " to the .env file, then run again."
            )
    if failed:
        lines.append("Nothing was downloaded.")
    return Check("Disk space", FAIL if failed else PASS, lines, blocking=True)


def check_hf(missing_weights: list[Any], *, env_file: Path) -> Check:
    token = (os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or "").strip()
    gated = [w for w in missing_weights if getattr(w, "gated", False)]
    steps = [
        "  1. Go to https://huggingface.co and sign in (or create a free account).",
        "  2. Open https://huggingface.co/settings/tokens and create a token with the 'Read' role.",
        f"  3. Open the file {env_file} in a text editor (create it if it is missing).",
        "  4. Add one line:  HF_TOKEN=<paste your token here>  and save the file.",
        "  5. Run automatic install again.",
    ]
    if not token:
        if not gated:
            return Check(
                "Hugging Face access",
                PASS,
                [
                    "Not needed: the LTX 2.3 files Buddy downloads are public on Hugging Face, "
                    "so no token is required.",
                ],
                blocking=True,
            )
        return Check(
            "Hugging Face access",
            FAIL,
            ["Some files need a Hugging Face token and none is set. To fix it:", *steps],
            blocking=True,
        )
    ok, who = hf_whoami(token)
    if not ok:
        return Check(
            "Hugging Face access",
            FAIL,
            [
                f"An HF_TOKEN is set but it does not work: {who}.",
                "Even public downloads fail while a bad token is set. Replace it (or remove the line):",
                *steps,
            ],
            blocking=True,
        )
    lines = [f"HF_TOKEN is valid (signed in as {who}). The token itself is never printed."]
    for repo in sorted({w.repo_id for w in gated}):
        allowed, detail = hf_repo_access(repo, token)
        if not allowed:
            return Check(
                "Hugging Face access",
                FAIL,
                [
                    *lines,
                    f"{repo}: {detail}.",
                    f"  Open https://huggingface.co/{repo}, read and accept the license, then run again.",
                ],
                blocking=True,
            )
        lines.append(f"{repo}: {detail}.")
    if not gated:
        lines.append("Not strictly needed: the LTX 2.3 files are public.")
    return Check("Hugging Face access", PASS, lines, blocking=True)


def git_install_hint() -> str:
    if sys.platform == "win32":
        return "winget install --id Git.Git -e   (or download it from https://git-scm.com/download/win)"
    if sys.platform == "darwin":
        return "xcode-select --install   (or: brew install git)"
    return "sudo apt-get install -y git   (or your distribution's package manager, e.g. sudo dnf install git)"


def check_git(*, needed: bool) -> Check:
    """comfy-cli runs ``git clone`` for ComfyUI and the custom node packs."""
    found = git_path()
    if found:
        return Check("git", PASS, [f"git found: {found}"], blocking=True)
    if not needed:
        return Check(
            "git",
            INFO,
            ["git is not on PATH. ComfyUI and its node packs are already installed, so it is not needed now."],
        )
    return Check(
        "git",
        FAIL,
        [
            "git is not installed (or not on PATH). ComfyUI is downloaded with git, so the install cannot work without it.",
            f"  Install it with: {git_install_hint()}",
            "  Then open a new terminal and run automatic install again.",
        ],
        blocking=True,
    )


def check_gpu(gpu: dict[str, Any]) -> Check:
    from master_agent.models.vram_policy import NVFP4_MIN_VRAM_GB

    vendor = str(gpu.get("vendor") or "unknown")
    vram = gpu.get("vram_gb")
    lines: list[str] = []
    if vendor in {"unknown", "none", ""}:
        lines.append("No GPU detected.")
        lines.append(
            "CPU mode: ComfyUI will be installed for CPU and started with --cpu. Video generation will "
            "work but is very slow (expect hours per clip)."
        )
    else:
        shown = f"{float(vram):.0f} GB VRAM" if vram is not None else "VRAM unknown"
        lines.append(f"{gpu.get('name') or vendor.upper()} detected, {shown}.")
        if vram is not None and float(vram) < NVFP4_MIN_VRAM_GB:
            lines.append(
                f"Under {NVFP4_MIN_VRAM_GB:.0f} GB VRAM: the GGUF Q4 loader "
                "(LTX-2.3-22B-distilled-1.1-Q4_K_S.gguf) will be used."
            )
        else:
            lines.append("The GGUF Q4 loader is used for LTX 2.3 whenever the GGUF is on disk.")
    if gpu.get("sentence"):
        lines.append(str(gpu["sentence"]))
    lines.append("This is information only; it never stops the install.")
    return Check("GPU / VRAM", INFO, lines)


def check_llamacpp(plan: Any, *, skip: bool) -> Check:
    if skip:
        return Check("llama.cpp", INFO, ["--skip-llm: llama.cpp and the local LLM models are left out."])
    lines = [plan.describe(), *plan.notes]
    if plan.method == "unavailable":
        lines.append(
            "Buddy still renders video without it. To get the local LLM later, install the build tools: "
            + _toolchain_hint(plan.system)
        )
    for name, role in llm_models():
        if local_llm_gguf(name) is None:
            lines.append(f"{name} ({role}): no public GGUF source is defined; it will be skipped.")
    lines.append("This is information only; it never stops the install.")
    return Check("llama.cpp", INFO, lines)


def _toolchain_hint(system: str) -> str:
    from master_agent.llamacpp_install import toolchain_hint

    return toolchain_hint(system)


def check_workspace(workspace: Path, *, state: str, mode: str) -> Check:
    from master_agent.comfy.comfy_venv import probe_comfy_env

    if mode == "external":
        return Check(
            "ComfyUI folder",
            FAIL,
            [
                "COMFY_MODE is external: Buddy is set to use a ComfyUI that you manage yourself, "
                "so automatic install will not install into it.",
                "  To let Buddy own its ComfyUI, remove COMFY_MODE=external from .env "
                "(or set COMFY_MODE=managed), then run again.",
            ],
            blocking=True,
        )
    lines = [f"Buddy's own ComfyUI folder: {workspace}"]
    if state == "foreign":
        return Check(
            "ComfyUI folder",
            FAIL,
            [
                *lines,
                "That folder already has files in it, but they are not ComfyUI.",
                "  Move or rename that folder, or set MANAGED_COMFY_ROOT=<an empty folder> in .env, then run again.",
            ],
            blocking=True,
        )
    anchor = existing_ancestor(workspace)
    writable = os.access(anchor, os.W_OK)
    if writable:
        try:
            with tempfile.NamedTemporaryFile(dir=anchor, prefix=".vb-write-test-"):
                pass
        except OSError:
            writable = False
    if not writable:
        return Check(
            "ComfyUI folder",
            FAIL,
            [
                *lines,
                f"Buddy cannot write to {anchor}.",
                "  Pick a folder you own: set MANAGED_COMFY_ROOT=<folder in your home directory> in .env, "
                "then run again.",
            ],
            blocking=True,
        )
    if state == "installed":
        lines.append("ComfyUI is already installed there; that step will be skipped.")
        env = probe_comfy_env(workspace)
        lines.append(env.summary())
        lines.append(
            "triton: " + ("present" if env.triton else "absent")
            + "; sageattention: " + ("present" if env.sageattention else "absent")
        )
    elif state == "partial":
        lines.append("A half-finished ComfyUI install is there; automatic install will repair it.")
    else:
        lines.append("The folder is writable; ComfyUI will be installed there.")
    return Check("ComfyUI folder", PASS, lines, blocking=True)


def run_preflight(*, skip_weights: bool = False, skip_llm: bool = False) -> Preflight:
    from master_agent.comfy.tower import effective_mode, load_state
    from master_agent.models.weights import scan_bundle
    from master_agent.setup import ENV_FILE

    workspace = managed_workspace()
    try:
        mode = effective_mode(load_state())
    except Exception:
        mode = "managed"
    state = comfy_install_state(workspace) if mode != "external" else "absent"
    weights = scan_bundle(LTX23_BUNDLE)
    missing_weights = [] if skip_weights else list(weights.missing_mandatory)
    gpu = detect_gpu()
    llama = None if skip_llm else plan_llamacpp(gpu)
    git_needed = state != "installed" or bool(missing_nodes(workspace))
    checks = [
        check_workspace(workspace, state=state, mode=mode),
        check_git(needed=git_needed),
        check_disk(
            weights_bytes=sum(int(w.size_bytes) for w in missing_weights),
            comfy_bytes=0 if state == "installed" else COMFY_OVERHEAD_BYTES,
            llama_bytes=0 if llama is None else int(llama.disk_bytes),
            models=models_dir(),
            workspace=workspace,
            llama_dir=llamacpp_root(),
        ),
        check_hf(missing_weights, env_file=ENV_FILE),
        check_gpu(gpu),
        check_llamacpp(llama, skip=skip_llm),
    ]
    order = {
        "Disk space": 0,
        "git": 1,
        "Hugging Face access": 2,
        "GPU / VRAM": 3,
        "llama.cpp": 4,
        "ComfyUI folder": 5,
    }
    checks.sort(key=lambda c: order.get(c.name, 9))
    return Preflight(checks=checks, weights=weights, gpu=gpu, workspace_state=state, llama=llama)


def print_preflight(pf: Preflight) -> None:
    print("AUTOMATIC INSTALL — pre-flight check (nothing has been changed yet)")
    print()
    for c in pf.checks:
        print(f"  [{c.status}] {c.name}")
        for line in c.lines:
            print(f"         {line}")
    print()
    bad = pf.first_failure
    if bad is None:
        print("Pre-flight passed.")
    else:
        print(f"STOPPED at '{bad.name}'. Fix the item above, then run automatic install again.")
        print("Nothing was downloaded or installed.")
    print()


# --- steps ---------------------------------------------------------------------


@dataclass
class StepResult:
    name: str
    status: str
    detail: str
    required: bool = True
    lines: list[str] = field(default_factory=list)

    @property
    def failed_required(self) -> bool:
        return self.required and self.status in {FAILED, NEEDS_YOU}


def _elevated(cmd: list[str]) -> tuple[list[str] | None, str]:
    """Linux only. Root runs as-is; otherwise ``sudo -n`` (never prompts). Printed, never silent."""
    if _is_root():
        return cmd, ""
    if _which("sudo"):
        code, _ = _run(["sudo", "-n", "true"], timeout=10)
        if code == 0:
            return ["sudo", "-n", *cmd], ""
    return None, "sudo " + " ".join(cmd)


def step_env_file(apply: bool) -> StepResult:
    from master_agent.setup import ENV_EXAMPLE, ENV_FILE

    if ENV_FILE.is_file():
        return StepResult(".env", SKIPPED, f"{ENV_FILE} already exists", required=False)
    if not ENV_EXAMPLE.is_file():
        return StepResult(".env", NONFATAL, f"{ENV_EXAMPLE} missing", required=False)
    if not apply:
        return StepResult(".env", PLANNED, f"copy {ENV_EXAMPLE.name} to {ENV_FILE}", required=False)
    ENV_FILE.write_text(ENV_EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    return StepResult(".env", OK, f"wrote {ENV_FILE}", required=False)


def step_playwright(apply: bool) -> StepResult:
    cmd = [sys.executable, "-m", "playwright", "install", "chromium"]
    if not apply:
        return StepResult("Playwright Chromium", PLANNED, " ".join(cmd), required=False)
    print("$ " + " ".join(cmd))
    code, out = _run(cmd, timeout=900)
    if code != 0:
        return StepResult("Playwright Chromium", NONFATAL, out[-300:], required=False)
    return StepResult("Playwright Chromium", OK, "ready", required=False)


def step_ffmpeg(apply: bool) -> StepResult:
    if _which("ffmpeg"):
        return StepResult("ffmpeg", SKIPPED, "already installed")
    if sys.platform == "win32":
        cmds = [["winget", "install", "--id", "Gyan.FFmpeg", "-e", "--accept-package-agreements", "--accept-source-agreements"]]
        tool = "winget"
    elif sys.platform == "darwin":
        cmds = [["brew", "install", "ffmpeg"]]
        tool = "brew"
    else:
        cmds = [["apt-get", "update"], ["apt-get", "install", "-y", "ffmpeg"]]
        tool = "apt-get"
    if not _which(tool):
        return StepResult(
            "ffmpeg", NEEDS_YOU, f"{tool} not found. Install ffmpeg from https://ffmpeg.org/download.html, then run again."
        )
    if not apply:
        return StepResult("ffmpeg", PLANNED, " && ".join(" ".join(c) for c in cmds))
    for cmd in cmds:
        argv = cmd
        if tool == "apt-get":
            argv, manual = _elevated(cmd)
            if argv is None:
                return StepResult(
                    "ffmpeg",
                    NEEDS_YOU,
                    "needs administrator rights. Run this yourself, then run automatic install again: "
                    "sudo apt-get update && sudo apt-get install -y ffmpeg",
                )
        print("$ " + " ".join(argv))
        code, out = _run(argv, timeout=900)
        if code != 0:
            return StepResult("ffmpeg", FAILED, f"{' '.join(argv)} failed: {out[-300:]}")
    return StepResult("ffmpeg", OK, "installed")


def step_llamacpp(apply: bool, *, plan: Any, skip: bool) -> StepResult:
    """Buddy-owned llama-server: build the pinned tag, else the official prebuilt. Non-fatal."""
    from master_agent.llamacpp_install import install_llamacpp

    if skip or plan is None:
        return StepResult("llama.cpp", SKIPPED, "--skip-llm", required=False)
    res = install_llamacpp(plan, apply=apply)
    return StepResult("llama.cpp", res.status, res.detail, required=False, lines=res.lines)


def step_llm_models(apply: bool, *, skip: bool) -> StepResult:
    """GGUFs for ``LLAMACPP_MODEL`` / ``KB_EMBED_MODEL``. No public source is defined: skip, continue."""
    if skip:
        return StepResult("LLM models", SKIPPED, "--skip-llm", required=False)
    present: list[str] = []
    skipped: list[str] = []
    lines: list[str] = []
    for name, role in llm_models():
        found = local_llm_gguf(name)
        if found is not None:
            present.append(name)
            lines.append(f"{name} ({role}): found {found}")
            continue
        skipped.append(name)
        line = f"Skipped {name}: no public GGUF source is defined; continuing."
        lines.append(line)
        if apply:
            print(line)
    if skipped:
        lines.append(
            f"To use them, put the GGUF files in {models_dir()} (llama-server serves that folder); "
            "nothing else needs to change."
        )
    detail = "; ".join(
        part
        for part in (
            ("present: " + ", ".join(present)) if present else "",
            ("skipped (no public GGUF source): " + ", ".join(skipped)) if skipped else "",
        )
        if part
    )
    return StepResult("LLM models", SKIPPED, detail, required=False, lines=lines)


def step_comfy_cli(apply: bool) -> StepResult:
    from master_agent.comfy.updates import installed_comfy_cli_version, read_comfy_cli_pin

    pin = read_comfy_cli_pin()
    have = installed_comfy_cli_version()
    if have == pin:
        return StepResult("comfy-cli", SKIPPED, f"comfy-cli=={pin} already installed")
    cmd = [sys.executable, "-m", "pip", "install", f"comfy-cli=={pin}"]
    if not apply:
        return StepResult("comfy-cli", PLANNED, " ".join(cmd))
    print("$ " + " ".join(cmd))
    code, out = _run(cmd, timeout=900)
    if code != 0:
        return StepResult("comfy-cli", FAILED, out[-300:])
    return StepResult("comfy-cli", OK, f"comfy-cli=={pin}")


def step_comfyui(apply: bool, *, workspace: Path, state: str, gpu_flag: str) -> StepResult:
    from master_agent.comfy.comfy_venv import probe_comfy_env

    if state == "installed":
        return StepResult("ComfyUI", SKIPPED, f"already installed in {workspace}")
    if state == "foreign":
        return StepResult("ComfyUI", FAILED, f"{workspace} has files that are not ComfyUI")
    args = ["install", gpu_flag] + (["--restore"] if state == "partial" else [])
    argv = comfy_argv(workspace, *args)
    if not apply:
        verb = "repair" if state == "partial" else "install"
        return StepResult("ComfyUI", PLANNED, f"{verb}: " + " ".join(argv))
    workspace.parent.mkdir(parents=True, exist_ok=True)
    print("$ " + " ".join(argv))
    code, out = _run(argv, timeout=3600, env=comfy_install_env())
    final = comfy_install_state(workspace)
    if code != 0 or final != "installed":
        return StepResult("ComfyUI", FAILED, f"comfy install did not finish ({final}): {out[-400:]}")
    env = probe_comfy_env(workspace)
    lines = [env.summary()]
    if env.python_path is None:
        lines.append("ComfyUI has no .venv in its folder; it will run in a non-venv interpreter.")
    return StepResult("ComfyUI", OK, f"installed in {workspace} ({gpu_flag})", lines=lines)


def step_model_paths(apply: bool) -> StepResult:
    from master_agent.comfy.model_paths import write_extra_model_paths
    from master_agent.comfy.tower import state_path

    target = state_path().parent / "extra_model_paths.yaml"
    if not apply:
        return StepResult("model paths", PLANNED, f"write {target} → MODELS_DIR {models_dir()}")
    models_dir().mkdir(parents=True, exist_ok=True)
    written = write_extra_model_paths(directory=state_path().parent)
    return StepResult("model paths", OK, f"{written} (passed to Comfy at launch)")


def step_nodes(apply: bool, *, workspace: Path) -> StepResult:
    missing = missing_nodes(workspace)
    if not missing:
        return StepResult("custom nodes", SKIPPED, "already installed: " + ", ".join(REQUIRED_NODES))
    argv = comfy_argv(workspace, "node", "install", *missing)
    if not apply:
        return StepResult("custom nodes", PLANNED, " ".join(argv))
    if comfy_install_state(workspace) != "installed":
        return StepResult("custom nodes", FAILED, "waits for ComfyUI")
    print("$ " + " ".join(argv))
    from master_agent.comfy.tower import _child_env

    code, out = _run(argv, timeout=1800, env=_child_env(workspace))
    still = missing_nodes(workspace)
    if still:
        return StepResult("custom nodes", FAILED, f"not installed: {', '.join(still)} ({out[-300:]})")
    return StepResult("custom nodes", OK, "installed " + ", ".join(missing))


def step_accel(apply: bool, *, workspace: Path, gpu: dict[str, Any], skip: bool) -> StepResult:
    from master_agent.comfy.comfy_venv import install_accel

    if skip:
        return StepResult("Triton + SageAttention", SKIPPED, "--skip-sage", required=False)
    if apply and comfy_install_state(workspace) != "installed":
        return StepResult("Triton + SageAttention", NONFATAL, "waits for ComfyUI. Comfy will run, just slower.", required=False)
    if not apply and comfy_install_state(workspace) != "installed":
        return StepResult(
            "Triton + SageAttention",
            PLANNED,
            "after ComfyUI: Triton, then SageAttention, into the Comfy venv (NVIDIA on Linux/Windows only)",
            required=False,
        )
    report = install_accel(workspace, gpu_vendor=str(gpu.get("vendor") or ""), apply=apply)
    detail = "; ".join(f"{s.name} {s.status}" for s in report.steps) or report.status
    return StepResult("Triton + SageAttention", report.status, detail, required=False, lines=report.lines())


def step_weights(apply: bool, *, skip: bool) -> StepResult:
    from master_agent.models.weights import download_missing_bundle, scan_bundle

    status = scan_bundle(LTX23_BUNDLE)
    if status.ok:
        return StepResult("LTX 2.3 weights", SKIPPED, "all present in the model folders")
    missing = status.missing_mandatory
    total = sum(int(w.size_bytes) for w in missing)
    lines = [f"{w.filename} → {models_dir() / w.dest_folder} ({w.size_label})  [{w.repo_id}]" for w in missing]
    if skip:
        return StepResult("LTX 2.3 weights", NEEDS_YOU, f"--skip-weights: {fmt_gb(total)} still missing", lines=lines)
    if not apply:
        return StepResult("LTX 2.3 weights", PLANNED, f"download {len(missing)} file(s), {fmt_gb(total)}", lines=lines)
    models_dir().mkdir(parents=True, exist_ok=True)
    try:
        _st, paths = download_missing_bundle(LTX23_BUNDLE, yes=True)
    except Exception as exc:
        return StepResult("LTX 2.3 weights", FAILED, str(exc)[:500], lines=lines)
    after = scan_bundle(LTX23_BUNDLE)
    if not after.ok:
        names = ", ".join(w.filename for w in after.missing_mandatory)
        return StepResult("LTX 2.3 weights", FAILED, f"still missing: {names}", lines=lines)
    return StepResult("LTX 2.3 weights", OK, f"{len(paths)} file(s) ready in {models_dir()}", lines=lines)


def verify_rows() -> list[dict[str, Any]]:
    """Doctor's own scan rows (read-only) for the pieces generate needs."""
    from master_agent.llamacpp_server import resolve_binary
    from master_agent.setup import check_ffmpeg, check_ltx23_weights

    binary = resolve_binary()
    llama = {
        "name": "llama.cpp",
        "ok": bool(binary),
        "detail": binary or "not installed (optional: Buddy renders without a local LLM)",
    }
    return [check_ffmpeg(), check_ltx23_weights(), llama]


# --- command -------------------------------------------------------------------


def _print_steps(results: list[StepResult]) -> None:
    for r in results:
        tag = "" if r.required else " (optional)"
        print(f"  {r.status:<16} {r.name}{tag}: {r.detail}")
        for line in r.lines:
            print(f"                   {line}")


def cmd_automatic_install(args: Any) -> int:
    preflight_only = bool(getattr(args, "preflight_only", False))
    apply = bool(getattr(args, "yes", False)) and not bool(getattr(args, "dry_run", False))
    skip_weights = bool(getattr(args, "skip_weights", False))
    skip_sage = bool(getattr(args, "skip_sage", False))
    skip_llm = bool(getattr(args, "skip_llm", False))
    gpu_override = str(getattr(args, "gpu", "auto") or "auto")

    pf = run_preflight(skip_weights=skip_weights, skip_llm=skip_llm)
    print_preflight(pf)
    if preflight_only:
        return 0 if pf.ok else 1
    if not pf.ok:
        return 1

    workspace = managed_workspace()
    flag = gpu_install_flag(pf.gpu, gpu_override)
    print("AUTOMATIC INSTALL — " + ("installing" if apply else "plan (dry run; nothing will change)"))
    print("Video model: LTX 2.3 only.")
    print()
    steps: list[tuple[str, bool, Callable[[], StepResult]]] = [
        (".env", False, lambda: step_env_file(apply)),
        ("Playwright Chromium", False, lambda: step_playwright(apply)),
        ("ffmpeg", True, lambda: step_ffmpeg(apply)),
        ("comfy-cli", True, lambda: step_comfy_cli(apply)),
        ("ComfyUI", True, lambda: step_comfyui(
            apply, workspace=workspace, state=comfy_install_state(workspace), gpu_flag=flag
        )),
        ("model paths", True, lambda: step_model_paths(apply)),
        ("custom nodes", True, lambda: step_nodes(apply, workspace=workspace)),
        ("Triton + SageAttention", False, lambda: step_accel(apply, workspace=workspace, gpu=pf.gpu, skip=skip_sage)),
        ("LTX 2.3 weights", True, lambda: step_weights(apply, skip=skip_weights)),
        ("llama.cpp", False, lambda: step_llamacpp(apply, plan=pf.llama, skip=skip_llm)),
        ("LLM models", False, lambda: step_llm_models(apply, skip=skip_llm)),
    ]
    results: list[StepResult] = []
    for name, required, make in steps:
        try:
            result = make()
        except Exception as exc:
            status = FAILED if required else NONFATAL
            result = StepResult(name, status, f"{type(exc).__name__}: {exc}"[:400], required=required)
        results.append(result)
        if apply:
            _print_steps([result])
    if not apply:
        _print_steps(results)
        print()
        print("Nothing was installed. When the plan looks right, run:")
        print("  python -m master_agent automatic-install --yes")
        return 0

    print()
    print("AUTOMATIC INSTALL — summary")
    _print_steps(results)
    print()
    print("Check (doctor scan, read-only):")
    for row in verify_rows():
        print(f"  {'OK  ' if row['ok'] else 'NEED'}  {row['name']:<14} {row['detail']}")
    failed = [r for r in results if r.failed_required]
    print()
    skipped_models = [
        line.split(":", 1)[0].removeprefix("Skipped ")
        for r in results
        if r.name == "LLM models"
        for line in r.lines
        if line.startswith("Skipped ")
    ]
    if skipped_models:
        print("Skipped (no public GGUF source is defined): " + ", ".join(skipped_models))
    llama = next((r for r in results if r.name == "llama.cpp"), None)
    if llama is not None and llama.status in {NONFATAL, NEEDS_YOU}:
        print("No local LLM runtime yet: video generation works; the local brief / judge LLM stays off.")
    if failed:
        print("Not ready for generate yet. Still needs you:")
        for r in failed:
            print(f"  - {r.name}: {r.detail}")
        return 1
    print("Ready for generate (LTX 2.3). Next:")
    print("  python -m master_agent comfy start")
    print('  python -m master_agent comfy run --mode generate --variant base --prompt "a test shot"')
    return 0
