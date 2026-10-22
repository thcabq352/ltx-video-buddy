"""Automatic install: one command from an empty machine to a working LTX 2.3 generate.

``python -m master_agent automatic-install`` runs a plain-English pre-flight
first, then prints the plan. Nothing changes without ``--yes``.

With ``--yes`` it installs, in order: ffmpeg, comfy-cli at the
requirements.txt pin, ComfyUI into its own folder (``MANAGED_COMFY_ROOT`` or
``PROJECT_ROOT/ComfyUI``), the custom nodes the LTX 2.3 graphs need, each
pack's own ``requirements.txt``, then ``comfy-requirements.txt`` (gguf,
colour-science) into the Comfy venv when they are not importable, ComfyUI-Manager's
channel (``config.ini`` / ``channels.list``, see ``comfy/manager_config.py``), Triton +
SageAttention inside the Comfy venv (optional), the LTX 2.3 weights into
``MODELS_DIR``, then llama.cpp into ``LLAMACPP_ROOT`` (optional). LTX 2.3 is
the only video model this command installs. Weights are never written into
the Comfy tree; managed launch keeps passing Buddy's ``extra_model_paths.yaml``.

The local LLM models (``LLAMACPP_MODEL`` for brief / director / judge,
``KB_EMBED_MODEL`` for knowledge-base embeddings) are GGUFs llama-server
serves from ``MODELS_DIR/<name>/``. Public sources live in
``models/llm_gguf.py`` (same Hub download as the LTX weights). A vision pair
already on disk (``MODELS_DIR/gguf_vl_heretic/`` first, then any folder with a
model ``*.gguf`` and an ``mmproj*.gguf``) is the brain and the 4B file is not
downloaded. An existing Ollama ``nomic-embed-text`` blob may be referenced
read-only instead of downloading the embed file. Nothing already on disk is
moved or deleted. A file that is already the full size is skipped. A short
file is fetched again. If a slot has no source or the download fails, the step
fails out loud: the summary, ``state/LAUNCH_NOTES.txt``, and doctor's
``llm-models`` row all name the direct link and the manual steps, which brain
and embed are in use, and whether vision is on. Automatic install never
installs Ollama and never runs ``ollama pull``.
Unset ``LLM_PROVIDER`` is set to ``llamacpp`` in ``.env``.

``doctor`` stays a scanner. This module is the only thing that installs.
"""

from __future__ import annotations

import json
import os
import platform
import re
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
# ComfyUI checkout + its own venv with CUDA PyTorch + custom nodes + Triton.
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
    from master_agent.comfy.root import managed_comfy_root

    return managed_comfy_root()


def managed_workspace_source() -> str:
    from master_agent.comfy.root import resolve_managed_comfy_root

    return resolve_managed_comfy_root().source


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
    from master_agent.comfy.comfy_venv import module_importable
    from master_agent.comfy.layout import comfy_interpreter

    if not workspace.exists():
        return "absent"
    if workspace.is_dir() and not any(workspace.iterdir()):
        return "absent"
    looks_comfy = (workspace / "main.py").is_file() and (workspace / "comfy").is_dir()
    if not looks_comfy:
        return "foreign"
    py = comfy_interpreter(workspace)
    if py is None or not module_importable(py, "torch"):
        return "partial"
    return "installed"


def portable_guard(workspace: Path, state: str) -> str | None:
    """Why automatic install must not install ComfyUI here, or None.

    Buddy never installs into, repairs, or installs beside a Windows portable build.
    """
    from master_agent.comfy.layout import PORTABLE, detect_layout, discover_portable

    layout = detect_layout(workspace)
    if layout is not None and layout.kind == PORTABLE:
        if state == "installed":
            return None
        return (
            f"{workspace} is a ComfyUI portable build ({layout.top}), but its python_embeded "
            "cannot import torch. Buddy never installs over a portable build: repair it with the "
            "portable's own update\\update_comfyui_and_python_dependencies.bat, then run again."
        )
    if state == "installed":
        return None
    portable = discover_portable()
    if portable is not None and portable.code_dir != workspace:
        return (
            f"A ComfyUI portable build is already at {portable.top}. Buddy will not install a second "
            f"ComfyUI in {workspace}: set MANAGED_COMFY_ROOT={portable.top} in .env, then run again."
        )
    return None


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


COMFY_CUDA_VERSION_ENV = "VIDEO_BUDDY_COMFY_CUDA_VERSION"
_BLACKWELL_RE = re.compile(r"RTX\s*50\d0|RTX\s*PRO\s*\d+\s*Blackwell|\bB[12]00\b|\bGB[12]0\d", re.I)


def is_blackwell(gpu: dict[str, Any]) -> bool:
    return str(gpu.get("vendor") or "").lower() == "nvidia" and bool(_BLACKWELL_RE.search(str(gpu.get("name") or "")))


def comfy_cuda_args(gpu_flag: str) -> list[str]:
    """Extra ``--cuda-version X`` for ``comfy install --nvidia``.

    comfy-cli 1.20 already picks the torch wheel tag from the NVIDIA driver
    (``cuda_detect.resolve_cuda_wheel``), so a Blackwell (sm_120) card with a
    570+/580+ driver gets cu128/cu130 without help. Only when driver detection
    fails does it fall back to cu126, which has no sm_120 kernels. Set
    ``VIDEO_BUDDY_COMFY_CUDA_VERSION=13.0`` (or 12.8) to force a tag.
    """
    want = os.environ.get(COMFY_CUDA_VERSION_ENV, "").strip()
    if gpu_flag != "--nvidia" or not re.fullmatch(r"\d+\.\d+", want):
        return []
    return ["--cuda-version", want]


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
    llm_bytes: int = 0,
) -> Check:
    needs: dict[int, dict[str, Any]] = {}
    for label, path, size, env_var in (
        ("LTX 2.3 weights", models, weights_bytes, "MODELS_DIR"),
        ("Local LLM GGUFs", models, llm_bytes, "MODELS_DIR"),
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
                    "Not needed: the LTX 2.3 files and the local LLM GGUFs Buddy downloads are "
                    "public on Hugging Face, so no token is required.",
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


def check_gpu(gpu: dict[str, Any]) -> Check:
    from master_agent.models.vram_policy import NVFP4_MIN_VRAM_GB

    vendor = str(gpu.get("vendor") or "unknown")
    vram = gpu.get("vram_gb")
    lines: list[str] = []
    if vendor in {"unknown", "none", ""}:
        lines.append("No GPU detected.")
        lines.append(
            "CPU mode: ComfyUI will be installed for CPU. Video generation will work but is very slow "
            "(expect hours per clip)."
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
    from master_agent.models.llm_gguf import (
        BRAIN_FILENAME,
        configured_slots,
        is_ollama_reference,
        mmproj_present,
        model_path_for,
        projector_how_to,
        slot_dir,
        slot_ready,
    )

    for slot in configured_slots():
        if slot_ready(slot):
            lines.append(f"{slot.name} ({slot.role}): already on disk.")
            model = model_path_for(slot)
            if slot.vision:
                proj = mmproj_present(slot)
                if model is not None:
                    lines.append(f"  brain: {model}")
                if proj is not None:
                    lines.append(f"  projector: {proj}")
                    lines.append("  vision: enabled")
                    canonical = slot_dir(slot) / BRAIN_FILENAME
                    if model is None or model.resolve() != canonical.resolve():
                        lines.append("  The 4B download is skipped; this pair is the brain.")
                else:
                    lines.append("  vision: text-only (no mmproj). The vision judge cannot see images.")
                    lines.append(f"  {projector_how_to()}")
            elif model is not None and is_ollama_reference(model):
                lines.append(f"  embed: {model}")
                lines.append("  read-only Ollama file; not copied or modified")
        elif slot.files:
            spec = slot.files[0]
            lines.append(
                f"{slot.name} ({slot.role}): will download {spec.filename} "
                f"({spec.size_label}) from {spec.repo_id}."
            )
            lines.append(f"  {spec.url}")
            if slot.vision and not slot.publishes_mmproj:
                lines.append(
                    "  That repo publishes no mmproj file. Text works; the vision judge cannot see images."
                )
                lines.append(f"  {projector_how_to()}")
        else:
            lines.append(
                f"{slot.name} ({slot.role}): NO public GGUF source. "
                "The install will fail this slot and print the manual steps."
            )
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
    lines = [f"Buddy's own ComfyUI folder: {workspace} (from {managed_workspace_source()})"]
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
    guard = portable_guard(workspace, state)
    if guard is not None:
        return Check("ComfyUI folder", FAIL, [*lines, guard], blocking=True)
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
    from master_agent.models.llm_gguf import bytes_still_needed

    weights = scan_bundle(LTX23_BUNDLE)
    missing_weights = [] if skip_weights else list(weights.missing_mandatory)
    gpu = detect_gpu()
    llama = None if skip_llm else plan_llamacpp(gpu)
    checks = [
        check_workspace(workspace, state=state, mode=mode),
        check_disk(
            weights_bytes=sum(int(w.size_bytes) for w in missing_weights),
            comfy_bytes=0 if state == "installed" else COMFY_OVERHEAD_BYTES,
            llama_bytes=0 if llama is None else int(llama.disk_bytes),
            llm_bytes=0 if skip_llm else bytes_still_needed(),
            models=models_dir(),
            workspace=workspace,
            llama_dir=llamacpp_root(),
        ),
        check_hf(missing_weights, env_file=ENV_FILE),
        check_gpu(gpu),
        check_llamacpp(llama, skip=skip_llm),
    ]
    order = {"Disk space": 0, "Hugging Face access": 1, "GPU / VRAM": 2, "llama.cpp": 3, "ComfyUI folder": 4}
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


def step_llm_models(apply: bool, *, skip: bool, fetch: bool = True) -> StepResult:
    """Download ``LLAMACPP_MODEL`` / ``KB_EMBED_MODEL`` into ``MODELS_DIR/<name>/``.

    Same shape as the Comfy Python packages step: required when this command
    fetches, skip when the files are already the right size, verify after the
    copy, and fail loudly (link + manual steps + ``LAUNCH_NOTES.txt``) otherwise.
    ``fetch=False`` is the light install: it does not download, and it still
    refuses to skip quietly. ``--skip-llm`` is the only quiet skip.
    """
    if skip:
        return StepResult("LLM models", SKIPPED, "--skip-llm", required=False)
    from master_agent.models import llm_gguf

    outcomes = [
        llm_gguf.install_slot(slot, apply=apply, fetch=fetch)
        for slot in llm_gguf.configured_slots()
    ]
    failed = [o for o in outcomes if o.status == "failed"]
    if failed:
        # Do not pin llama.cpp when the files are missing: an explicit pin would
        # stop the auto fallback to Ollama. Report whatever is already set.
        from master_agent import config

        path = llm_gguf.env_file()
        pinned = llm_gguf.read_provider(path.read_text(encoding="utf-8")) if path.is_file() else None
        provider = pinned or config.LLM_PROVIDER or "auto"
        action = "unchanged" if pinned else "not changed (download failed)"
    else:
        provider, action = llm_gguf.ensure_provider(apply)
    lines: list[str] = []
    for outcome in outcomes:
        lines.extend(outcome.lines)
    lines.append(llm_gguf.provider_sentence(provider) + f" ({action})")
    lines.extend(llm_gguf.vision_note().splitlines())
    lines.extend(llm_gguf.usage_lines())
    preset = llm_gguf.write_preset() if apply else None
    if preset is not None:
        lines.append(f"llama.cpp preset: {preset}")
    notes = None
    if apply:
        notes = llm_gguf.write_launch_notes(
            llm_gguf.render_launch_notes(
                outcomes=outcomes, provider=provider, provider_action=action
            )
        )
        lines.append(f"Wrote {notes}")
    failed = [o for o in outcomes if o.status == "failed"]
    if failed:
        detail = "MISSING: " + ", ".join(o.slot.name for o in failed)
        if notes is not None:
            lines.insert(0, llm_gguf.warning_box(notes, failed))
        # Light install warns and continues. Automatic install treats a failed
        # download as a required failure so the summary cannot say ready.
        return StepResult("LLM models", FAILED, detail, required=bool(fetch), lines=lines)
    downloaded = [o.slot.name for o in outcomes if o.status == "downloaded"]
    present = [o.slot.name for o in outcomes if o.status == "present"]
    planned = [o.slot.name for o in outcomes if o.status == "planned"]
    if planned and not apply:
        return StepResult(
            "LLM models",
            PLANNED,
            "download " + ", ".join(planned),
            required=True,
            lines=lines,
        )
    if downloaded:
        detail = "downloaded: " + ", ".join(downloaded)
        if present:
            detail += "; present: " + ", ".join(present)
        return StepResult("LLM models", OK, detail, required=True, lines=lines)
    detail = "present: " + ", ".join(present) if present else "ready"
    return StepResult("LLM models", SKIPPED, detail, required=True, lines=lines)


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
    guard = portable_guard(workspace, state)
    if guard is not None:
        return StepResult("ComfyUI", FAILED, guard)
    args = ["install", gpu_flag, *comfy_cuda_args(gpu_flag)] + (["--restore"] if state == "partial" else [])
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


def step_model_folders(apply: bool) -> StepResult:
    from master_agent.models.layout import CREATED, WOULD_CREATE, ensure_model_folders

    actions = ensure_model_folders(models_dir(), dry_run=not apply)
    todo = [a for a in actions if a.action in {CREATED, WOULD_CREATE}]
    if not todo:
        return StepResult("model folders", SKIPPED, f"all {len(actions)} already exist in {models_dir()}")
    lines = [f"mkdir {a.path}" for a in todo]
    if not apply:
        return StepResult("model folders", PLANNED, f"create {len(todo)} folder(s) in {models_dir()}", lines=lines)
    return StepResult("model folders", OK, f"created {len(todo)} folder(s) in {models_dir()}", lines=lines)


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


COMFY_REQUIREMENTS = Path(__file__).resolve().parent.parent / "comfy-requirements.txt"
# (pip distribution, import name) the LTX 2.3 packs import at load time:
# ComfyUI-GGUF needs gguf, ComfyUI-LTXVideo needs colour-science. The unrelated
# PyPI ``colour`` package also imports as ``colour``, so the distribution is checked too.
COMFY_PACKAGES = (("gguf", "gguf"), ("colour-science", "colour"))
TORCH_PACKAGES = frozenset({"torch", "torchvision", "torchaudio"})

_PACKAGE_PROBE = (
    "import importlib, importlib.metadata as md, json, sys\n"
    "missing = []\n"
    "for pair in sys.argv[1:]:\n"
    "    dist, mod = pair.split(':', 1)\n"
    "    try:\n"
    "        md.version(dist)\n"
    "        importlib.import_module(mod)\n"
    "    except Exception:\n"
    "        missing.append(dist)\n"
    "print('VB_MISSING=' + json.dumps(missing))\n"
)


def comfy_python(workspace: Path) -> tuple[Path | None, list[str]]:
    """The interpreter ``comfy start`` launches ComfyUI with, plus its flags (``-s`` for portable)."""
    from master_agent.comfy.layout import comfy_interpreter, detect_layout

    layout = detect_layout(workspace)
    if layout is not None:
        return layout.python, list(layout.python_flags)
    return comfy_interpreter(workspace), []


def missing_comfy_packages(
    py: Path | str, flags: Iterable[str] = (), packages: Iterable[tuple[str, str]] = COMFY_PACKAGES
) -> list[str]:
    """Distributions in ``packages`` that ``py`` cannot import. Read-only; runs in that interpreter."""
    packages = list(packages)
    cmd = [str(py), *flags, "-c", _PACKAGE_PROBE, *(f"{dist}:{mod}" for dist, mod in packages)]
    code, out = _run(cmd, timeout=120)
    # _run merges stderr in; colour-science warns on import (no matplotlib).
    marked = [line for line in out.splitlines() if line.startswith("VB_MISSING=")]
    if code == 0 and marked:
        try:
            found = json.loads(marked[-1].removeprefix("VB_MISSING="))
            if isinstance(found, list):
                return [str(d) for d in found]
        except ValueError:
            pass
    return [dist for dist, _mod in packages]


def _pip_cmd(py: Path, flags: list[str], *args: str) -> list[str]:
    return [str(py), *flags, "-m", "pip", "install", *args]


def without_torch(requirements_text: str) -> str:
    """A pack's requirements minus torch*: pip would swap ComfyUI's CUDA wheels for PyPI ones."""
    keep = []
    for raw in requirements_text.splitlines():
        pkg = re.split(r"[<>=!~\[;\s]", raw.strip(), maxsplit=1)[0].lower()
        if pkg in TORCH_PACKAGES:
            continue
        keep.append(raw)
    return "\n".join(keep) + "\n"


def step_node_requirements(apply: bool, *, workspace: Path) -> StepResult:
    """Each LTX 2.3 pack's own requirements.txt into ComfyUI's interpreter (pins as ``-c`` constraints)."""
    name = "custom node requirements"
    if comfy_install_state(workspace) != "installed":
        if not apply:
            return StepResult(name, PLANNED, "after ComfyUI: pip install -r each pack's requirements.txt into the Comfy venv")
        return StepResult(name, FAILED, "waits for ComfyUI")
    py, flags = comfy_python(workspace)
    if py is None:
        return StepResult(name, FAILED, f"no ComfyUI interpreter found in {workspace}")
    folders = {p.name.lower(): p for p in (workspace / "custom_nodes").iterdir() if p.is_dir()} \
        if (workspace / "custom_nodes").is_dir() else {}
    packs = [(n, folders[n.lower()] / "requirements.txt") for n in REQUIRED_NODES if n.lower() in folders]
    packs = [(n, req) for n, req in packs if req.is_file()]
    if not packs:
        if apply:
            return StepResult(name, SKIPPED, "no pack requirements.txt found")
        return StepResult(name, PLANNED, "after custom nodes: pip install -r each pack's requirements.txt")
    lines: list[str] = []
    done: list[str] = []
    for pack, req in packs:
        with tempfile.TemporaryDirectory() as tmp:
            filtered = Path(tmp) / f"{pack}-requirements.txt"
            filtered.write_text(without_torch(req.read_text(encoding="utf-8")), encoding="utf-8")
            cmd = _pip_cmd(py, flags, "-r", str(filtered), "-c", str(COMFY_REQUIREMENTS))
            shown = " ".join(_pip_cmd(py, flags, "-r", str(req), "-c", str(COMFY_REQUIREMENTS)))
            if not apply:
                lines.append(shown + " (torch lines dropped)")
                continue
            print("$ " + shown + "  (torch lines dropped)")
            code, out = _run(cmd, timeout=1800)
        if code != 0:
            return StepResult(
                name,
                FAILED,
                f"pip could not install {pack}'s requirements into {py}, so {pack} will not load in ComfyUI. "
                f"pip said: {out[-400:]}",
                lines=[*lines, f"Fix the error above, then run automatic-install --yes again (or run: {shown})"],
            )
        done.append(pack)
    if not apply:
        return StepResult(name, PLANNED, f"{len(packs)} pack(s) into {py}", lines=lines)
    return StepResult(name, OK, f"{', '.join(done)} requirements installed in {py}")


def step_comfy_packages(apply: bool, *, workspace: Path) -> StepResult:
    """Make gguf and colour-science importable in ComfyUI's own interpreter. Skips when they already are."""
    name = "Comfy Python packages"
    label = ", ".join(dist for dist, _mod in COMFY_PACKAGES)
    if comfy_install_state(workspace) != "installed":
        if not apply:
            return StepResult(name, PLANNED, f"after ComfyUI: make {label} importable in the Comfy venv")
        return StepResult(name, FAILED, f"waits for ComfyUI ({label} not installed)")
    py, flags = comfy_python(workspace)
    if py is None:
        return StepResult(name, FAILED, f"no ComfyUI interpreter found in {workspace}; {label} not installed")
    missing = missing_comfy_packages(py, flags)
    if not missing:
        return StepResult(name, SKIPPED, f"{label} already importable in {py}")
    cmd = _pip_cmd(py, flags, "--upgrade", "-r", str(COMFY_REQUIREMENTS))
    if not apply:
        return StepResult(name, PLANNED, f"missing {', '.join(missing)}: " + " ".join(cmd))
    print("$ " + " ".join(cmd))
    code, out = _run(cmd, timeout=900)
    fix = f"Fix the error above, then run automatic-install --yes again (or run: {' '.join(cmd)})"
    if code != 0:
        return StepResult(
            name,
            FAILED,
            f"pip could not install {', '.join(missing)} into ComfyUI's python {py}; "
            f"ComfyUI-LTXVideo / ComfyUI-GGUF will not load and video generation stays blocked. pip said: {out[-400:]}",
            lines=[fix],
        )
    still = missing_comfy_packages(py, flags)
    if still:
        return StepResult(
            name,
            FAILED,
            f"pip finished but {', '.join(still)} still cannot be imported by {py}",
            lines=[fix],
        )
    return StepResult(name, OK, f"installed {', '.join(missing)} in {py}")


def step_manager_config(apply: bool, *, workspace: Path) -> StepResult:
    """Point ComfyUI-Manager at the working channel; repair a config.ini that has the broken one."""
    from master_agent.comfy.layout import code_dir_for
    from master_agent.comfy.manager_config import CHANNEL_URL, ensure_manager_config

    name = "ComfyUI-Manager channel"
    if comfy_install_state(workspace) != "installed":
        if not apply:
            return StepResult(name, PLANNED, f"after ComfyUI: channel_url = {CHANNEL_URL}", required=False)
        return StepResult(name, NONFATAL, "waits for ComfyUI", required=False)
    report = ensure_manager_config(code_dir_for(workspace), apply=apply)
    if not report.changed:
        return StepResult(name, SKIPPED, f"already on {CHANNEL_URL}", required=False)
    status = OK if apply else PLANNED
    return StepResult(name, status, f"channel_url = {CHANNEL_URL}", required=False, lines=report.lines())


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


WEIGHTS_DEFERRED = "DEFERRED"


def step_weights(apply: bool, *, skip: bool) -> StepResult:
    from master_agent.models.deferred import DOWNLOAD_CMD, REGISTER_CMD, clear_deferred, mark_deferred
    from master_agent.models.weights import download_missing_bundle, scan_bundle

    status = scan_bundle(LTX23_BUNDLE)
    if status.ok:
        if apply:
            clear_deferred()
        return StepResult("LTX 2.3 weights", SKIPPED, "all present in the model folders")
    missing = status.missing_mandatory
    total = sum(int(w.size_bytes) for w in missing)
    lines = [f"{w.filename} → {models_dir() / w.dest_folder} ({w.size_label})  [{w.repo_id}]" for w in missing]
    if skip:
        detail = f"--skip-weights: {fmt_gb(total)} not downloaded"
        if apply:
            marker = mark_deferred([w.filename for w in missing])
            detail += f" (saved {marker})"
        lines += [
            f"Already have them? {REGISTER_CMD}",
            f"Download later:    {DOWNLOAD_CMD}",
        ]
        return StepResult("LTX 2.3 weights", WEIGHTS_DEFERRED, detail, lines=lines)
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
    clear_deferred()
    return StepResult("LTX 2.3 weights", OK, f"{len(paths)} file(s) ready in {models_dir()}", lines=lines)


def verify_rows() -> list[dict[str, Any]]:
    """Doctor's own scan rows (read-only) for the pieces generate needs."""
    from master_agent.llamacpp_server import resolve_binary
    from master_agent.setup import check_ffmpeg, check_llm_models, check_ltx23_weights

    binary = resolve_binary()
    llama = {
        "name": "llama.cpp",
        "ok": bool(binary),
        "detail": binary or "not installed (optional: Buddy renders without a local LLM)",
    }
    return [check_ffmpeg(), check_ltx23_weights(), llama, check_llm_models()]


# --- command -------------------------------------------------------------------


def classify_step(r: StepResult) -> str:
    """OK | INFO | FAILED-NONFATAL | FAIL for the end-of-install summary."""
    from master_agent.install_report import FAIL as R_FAIL, INFO as R_INFO, NONFATAL as R_NONFATAL, OK as R_OK

    if r.failed_required:
        return R_FAIL
    if r.status == WEIGHTS_DEFERRED:
        return R_INFO
    if r.status in {NONFATAL, NEEDS_YOU, FAILED}:
        return R_NONFATAL
    return R_OK


def _print_steps(results: list[StepResult]) -> None:
    from master_agent.install_report import FAIL as R_FAIL, OK as R_OK, STEP_REASONS, safe_to_ignore

    for r in results:
        tag = "" if r.required else " (optional)"
        print(f"  {r.status:<16} {r.name}{tag}: {r.detail}")
        for line in r.lines:
            print(f"                   {line}")
        level = classify_step(r)
        if level not in {R_OK, R_FAIL} and r.name in STEP_REASONS:
            print(f"                   {safe_to_ignore(STEP_REASONS[r.name])}")


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
        ("model folders", True, lambda: step_model_folders(apply)),
        ("model paths", True, lambda: step_model_paths(apply)),
        ("custom nodes", True, lambda: step_nodes(apply, workspace=workspace)),
        ("custom node requirements", True, lambda: step_node_requirements(apply, workspace=workspace)),
        ("Comfy Python packages", True, lambda: step_comfy_packages(apply, workspace=workspace)),
        ("ComfyUI-Manager channel", False, lambda: step_manager_config(apply, workspace=workspace)),
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
    from master_agent.install_report import (
        FAIL as R_FAIL,
        OK as R_OK,
        STEP_REASONS,
        safe_to_ignore,
        summary_line,
    )

    from master_agent.models.deferred import REASON as DEFERRED_REASON

    print("Check (doctor scan, read-only):")
    for row in verify_rows():
        if row["ok"]:
            mark = "OK  "
        elif row["name"] == "llm-models":
            mark = "WARN"
        elif row["name"] == "llama.cpp" or row.get("deferred"):
            mark = "INFO"
        else:
            mark = "NEED"
        print(f"  {mark}  {row['name']:<14} {row['detail']}")
        if mark == "WARN" and row.get("fix"):
            print(f"        -> {row['fix']}")
        if mark == "INFO":
            why = DEFERRED_REASON if row.get("deferred") else STEP_REASONS["llama.cpp"]
            print(f"        {safe_to_ignore(why)}")
    failed = [r for r in results if r.failed_required]
    levels = [classify_step(r) for r in results]
    ok_count = sum(1 for lv in levels if lv == R_OK)
    note_count = sum(1 for lv in levels if lv not in {R_OK, R_FAIL})
    print()
    llm = next((r for r in results if r.name == "LLM models"), None)
    if llm is not None and llm.status == FAILED:
        print("LOCAL LLM FAILED. The lines above name the direct link and the manual steps.")
        print("state/LAUNCH_NOTES.txt has the same warning. Video files may already be installed.")
    llama = next((r for r in results if r.name == "llama.cpp"), None)
    if llama is not None and llama.status in {NONFATAL, NEEDS_YOU}:
        print("No local LLM runtime yet: video generation works; the local brief / judge LLM stays off.")
    if failed:
        print("Not ready for generate yet. Still needs you:")
        for r in failed:
            print(f"  - {r.name}: {r.detail}")
        print()
        print(summary_line(ok=ok_count, notes=note_count, failed=[r.name for r in failed]))
        return 1
    print(summary_line(ok=ok_count, notes=note_count, failed=[]))
    weights = next((r for r in results if r.name == "LTX 2.3 weights"), None)
    if weights is not None and weights.status == WEIGHTS_DEFERRED:
        from master_agent.models.deferred import DOWNLOAD_CMD, REGISTER_CMD

        print("Installed without the LTX 2.3 weights (--skip-weights). Before generate, either:")
        print(f"  {REGISTER_CMD}   # files you already downloaded")
        print(f"  {DOWNLOAD_CMD}")
        print("Then: python -m master_agent comfy start")
        return 0
    print("Ready for generate (LTX 2.3). Next:")
    print("  python -m master_agent comfy start")
    print('  python -m master_agent comfy run --mode generate --variant base --prompt "a test shot"')
    return 0
