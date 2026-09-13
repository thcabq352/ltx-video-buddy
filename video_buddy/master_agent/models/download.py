"""Download Flux / LTX 2.5 weights into models/ (idempotent, consent-gated).

Flux t2i set (16GB-friendly fp8, non-gated repos):
  - Comfy-Org/flux1-dev / flux1-dev-fp8.safetensors     -> models/diffusion_models/
  - comfyanonymous/flux_text_encoders / clip_l + t5xxl  -> models/text_encoders/
  - models/vae/ae.safetensors is expected to exist already (warn only).

LTX 2.5 distilled split pack lives in ``master_agent.models.weights``.
Never auto-download those gated files — callers must pass ``yes=True``.

Files already at their destination are skipped (SKIP). Downloads go through
the huggingface_hub cache and are then copied into models/.

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


def download_hub_file(
    *,
    repo_id: str,
    repo_filename: str,
    dest: Path,
    progress: Callable[[str], None] = print,
) -> Path:
    """Download one Hub file into dest. Skips if dest already exists."""
    if dest.is_file() and dest.stat().st_size > 0:
        progress(f"SKIP {dest} (already exists)")
        return dest
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
    if vae.is_file():
        progress(f"SKIP {vae} (already exists)")
        paths.append(vae)
    else:
        progress(
            f"WARN {vae} not found; place {FLUX_VAE} in models/vae manually "
            "(not downloaded automatically)"
        )
    return paths
