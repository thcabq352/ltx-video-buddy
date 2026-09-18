"""Download Flux / LTX 2.5 weights into models/ (idempotent, consent-gated).

Flux t2i set (16GB-friendly fp8, non-gated repos):
  - Comfy-Org/flux1-dev / flux1-dev-fp8.safetensors     -> models/diffusion_models/
  - comfyanonymous/flux_text_encoders / clip_l + t5xxl  -> models/text_encoders/
  - models/vae/ae.safetensors is expected to exist already (warn only).

LTX 2.5 official bf16 split pack lives in ``master_agent.models.weights``
(Hub paths under ``Lightricks/LTX-2.5``). Never auto-download those gated
files — callers must pass ``yes=True``.

Local-first: files already at the destination, in Comfy ``models/``,
``EXTRA_MODELS_DIRS``, ``extra_model_paths.yaml``, or the Hugging Face hub
cache are reused. A clear SKIP log is printed. Nothing is re-downloaded
when a usable local copy exists.

If configured model dirs are missing, this module refuses to start a Hub
fetch and names every path it checked.

Usage:
    python -m master_agent download-flux
    python -m master_agent download-models --ltx25 --yes
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable

from master_agent.config import (
    FLUX_CLIP_L,
    FLUX_T5XXL,
    FLUX_UNET,
    FLUX_VAE,
    MODELS_DIR,
)

# (repo_id, filename, destination subfolder under MODELS_DIR)
FLUX_FILES: list[tuple[str, str, str]] = [
    ("Comfy-Org/flux1-dev", FLUX_UNET, "diffusion_models"),
    ("comfyanonymous/flux_text_encoders", FLUX_CLIP_L, "text_encoders"),
    ("comfyanonymous/flux_text_encoders", FLUX_T5XXL, "text_encoders"),
]


class LocalModelNotFound(RuntimeError):
    """Configured model dirs are missing or unusable; do not start a Hub fetch."""

    def __init__(
        self,
        message: str,
        *,
        paths_checked: list[str],
        filename: str = "",
    ):
        super().__init__(message)
        self.paths_checked = paths_checked
        self.filename = filename


def _usable(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _skip_local(path: Path, dest: Path, progress: Callable[[str], None]) -> Path:
    progress(
        f"SKIP download of {dest.name} — local file found at {path} (not re-downloading)"
    )
    return path


def _configured_model_roots() -> list[Path]:
    """Read live config (tests and runtime env rebinds must be visible)."""
    from master_agent.config import COMFYUI_ROOT as comfy_root
    from master_agent.config import MODELS_DIR as models_dir
    from master_agent.config import extra_models_dirs as extra_dirs

    return [Path(models_dir), Path(comfy_root) / "models", *extra_dirs()]


def _find_existing_local(repo_filename: str, dest: Path) -> Path | None:
    """Reuse dest, Comfy/extra trees, accepted aliases, or the HF hub cache."""
    if _usable(dest):
        return dest
    needle = Path(str(repo_filename).replace("\\", "/")).name or dest.name
    try:
        from master_agent.models.weights import find_weight_file

        for name in (needle, dest.name, repo_filename):
            found = find_weight_file(name)
            if found is not None:
                return found
    except Exception:
        return None
    return None


def download_hub_file(
    *,
    repo_id: str,
    repo_filename: str,
    dest: Path,
    progress: Callable[[str], None] = print,
) -> Path:
    """Reuse a local copy when present. Only hits the Hub after a full scan."""
    found = _find_existing_local(repo_filename, dest)
    if found is not None:
        return _skip_local(found, dest, progress)

    # try_to_load_from_cache with the real repo (no network)
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(repo_id=repo_id, filename=repo_filename)
        if cached and _usable(Path(cached)):
            return _skip_local(Path(cached), dest, progress)
    except Exception:
        pass

    roots: list[Path] = []
    try:
        from master_agent.models.weights import model_search_roots

        roots = list(model_search_roots())
    except Exception:
        roots = []
    configured = _configured_model_roots()
    if not any(p.is_dir() for p in configured):
        checked: list[str] = []
        seen: set[str] = set()
        for path in (*configured, *roots, dest.parent):
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            checked.append(key)
        raise LocalModelNotFound(
            f"Refusing to download {dest.name} from {repo_id}/{repo_filename} — "
            "no usable local model directories were found. "
            "Set MODELS_DIR, COMFYUI_ROOT, EXTRA_MODELS_DIRS, or "
            "Comfy extra_model_paths.yaml.\n"
            "Paths checked:\n" + "\n".join(f"  - {p}" for p in checked),
            paths_checked=checked,
            filename=dest.name,
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import GatedRepoError

    progress(f"Downloading {repo_id}/{repo_filename} ...")
    try:
        cached = hf_hub_download(repo_id=repo_id, filename=repo_filename)
    except GatedRepoError as exc:
        raise RuntimeError(
            f"Gated repo {repo_id} — accept the license on Hugging Face and set "
            f"HF_TOKEN (huggingface-cli login). {exc}"
        ) from exc
    shutil.copyfile(cached, dest)
    progress(f"OK   {dest}")
    return dest


def download_flux_weights(progress: Callable[[str], None] = print) -> list[Path]:
    """Ensure all Flux weights are present under models/. Returns their paths."""
    paths: list[Path] = []
    for repo_id, filename, sub in FLUX_FILES:
        dest = MODELS_DIR / sub / filename
        paths.append(
            download_hub_file(
                repo_id=repo_id,
                repo_filename=filename,
                dest=dest,
                progress=progress,
            )
        )

    vae = MODELS_DIR / "vae" / FLUX_VAE
    found_vae = vae if vae.is_file() and vae.stat().st_size > 0 else None
    if found_vae is None:
        try:
            from master_agent.models.weights import find_weight_file

            found_vae = find_weight_file(FLUX_VAE)
        except Exception:
            found_vae = None
    if found_vae is not None:
        progress(
            f"SKIP download of {FLUX_VAE} — local file found at {found_vae} "
            "(not re-downloading)"
        )
        paths.append(found_vae)
    else:
        progress(
            f"WARN {vae} not found; place {FLUX_VAE} in models/vae manually "
            "(not downloaded automatically)"
        )
    return paths
