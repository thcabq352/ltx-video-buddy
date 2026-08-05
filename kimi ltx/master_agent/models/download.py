"""Download Flux fp8 weights into models/ (idempotent).

Flux t2i set (16GB-friendly fp8, non-gated repos):
  - Comfy-Org/flux1-dev / flux1-dev-fp8.safetensors     -> models/diffusion_models/
  - comfyanonymous/flux_text_encoders / clip_l + t5xxl  -> models/text_encoders/
  - models/vae/ae.safetensors is expected to exist already (warn only).

Files already at their destination are skipped (SKIP). Downloads go through
the huggingface_hub cache and are then copied into models/.

Usage:
    python -c "from master_agent.models.download import download_flux_weights; download_flux_weights()"
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


def download_flux_weights(progress: Callable[[str], None] = print) -> list[Path]:
    """Ensure all Flux weights are present under models/. Returns their paths."""
    from huggingface_hub import hf_hub_download

    paths: list[Path] = []
    for repo_id, filename, sub in FLUX_FILES:
        dest = MODELS_DIR / sub / filename
        if dest.is_file():
            progress(f"SKIP {dest} (already exists)")
            paths.append(dest)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        progress(f"Downloading {repo_id}/{filename} ...")
        cached = hf_hub_download(repo_id=repo_id, filename=filename)
        shutil.copyfile(cached, dest)
        progress(f"OK   {dest}")
        paths.append(dest)

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
