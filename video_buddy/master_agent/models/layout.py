"""Comfy-style model subfolders under ``MODELS_DIR``.

The list is what Buddy itself reads or writes: every ``WeightFile.dest_folder``
in ``weights.py`` plus the folders the loader nodes in ``workflows/`` read
(CheckpointLoaderSimple, UNETLoader / UnetLoaderGGUF, CLIPLoader /
DualCLIPLoader / LTXAVTextEncoderLoader, VAELoader / LTXVAudioVAELoader,
LoraLoader* / LTXICLoRALoaderModelOnly, LatentUpscaleModelLoader,
ControlNetLoaderAdvanced, IPAdapter's CLIP vision). Every name is also a
folder ``extra_model_paths.yaml`` maps, so ComfyUI sees what lands here.

``ensure_model_folders`` only makes empty directories. It never downloads.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

MODEL_SUBFOLDERS: tuple[tuple[str, str], ...] = (
    ("checkpoints", "all-in-one checkpoints (CheckpointLoaderSimple): LTX 2.3 distilled 1.1 / EROS"),
    ("diffusion_models", "transformers / UNETs incl. GGUF (UNETLoader, UnetLoaderGGUF): LTX 2.3 Q4_K_S, LTX 2.5, H3, Wan, Flux, Krea, Qwen"),
    ("unet", "legacy name for diffusion_models; Comfy reads both"),
    ("text_encoders", "text encoders (CLIPLoader, DualCLIPLoader, LTXAVTextEncoderLoader): Gemma, Qwen3-VL, umt5"),
    ("clip", "legacy name for text_encoders; Comfy reads both"),
    ("vae", "video / audio / image VAEs (VAELoader, LTXVAudioVAELoader)"),
    ("loras", "LoRAs and IC-LoRAs (LoraLoaderModelOnly, LTXICLoRALoaderModelOnly)"),
    ("latent_upscale_models", "LTX latent spatial / temporal upscalers (LatentUpscaleModelLoader only reads this folder)"),
    ("upscale_models", "pixel-space upscalers (UpscaleModelLoader)"),
    ("model_patches", "model patches: LTX 2.5 duration head (optional)"),
    ("clip_vision", "CLIP vision encoders (IPAdapter graphs)"),
    ("controlnet", "ControlNets (ControlNetLoaderAdvanced)"),
)

SUBFOLDER_NAMES: tuple[str, ...] = tuple(name for name, _ in MODEL_SUBFOLDERS)

EXISTS, CREATED, WOULD_CREATE = "exists", "created", "would create"


@dataclass(frozen=True)
class FolderAction:
    path: Path
    action: str  # exists | created | would create


def _root(root: Path | str | None) -> Path:
    if root is not None:
        return Path(root)
    from master_agent import config

    return Path(config.MODELS_DIR)


def model_folder(role: str, root: Path | str | None = None) -> Path:
    """Exact folder a weight of ``role`` goes in (``MODELS_DIR/<role>``)."""
    return _root(root) / role


def missing_model_folders(root: Path | str | None = None) -> list[Path]:
    base = _root(root)
    return [base / name for name in SUBFOLDER_NAMES if not (base / name).is_dir()]


def ensure_model_folders(root: Path | str | None = None, *, dry_run: bool = False) -> list[FolderAction]:
    """Create every subfolder under ``root`` (default ``MODELS_DIR``). Idempotent.

    ``dry_run`` reports what would be made and touches nothing on disk.
    """
    base = _root(root)
    out: list[FolderAction] = []
    for name in SUBFOLDER_NAMES:
        path = base / name
        if path.is_dir():
            out.append(FolderAction(path, EXISTS))
        elif dry_run:
            out.append(FolderAction(path, WOULD_CREATE))
        else:
            path.mkdir(parents=True, exist_ok=True)
            out.append(FolderAction(path, CREATED))
    return out
