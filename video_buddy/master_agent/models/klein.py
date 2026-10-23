"""FLUX.2 [klein] 4B stills weights. Apache 2.0, ungated. Nothing else.

Pinned from the public Hugging Face Hub API on 2026-10-22 (no token):

- ``black-forest-labs/FLUX.2-klein-4B`` — ``gated: false``,
  ``cardData.license: apache-2.0``. Siblings include
  ``flux-2-klein-4b.safetensors``.
- ``black-forest-labs/FLUX.2-klein-4b-fp8`` — ``gated: false``,
  ``license: apache-2.0``. File ``flux-2-klein-4b-fp8.safetensors``
  is 4,070,624,520 bytes. This is the fleet diffusion file (12GB cards).
- ``Comfy-Org/Flux-2-Klein-4B`` — ``gated: false``, ``license: apache-2.0``.
  Text encoder ``split_files/text_encoders/qwen_3_4b_fp4_flux2.safetensors``
  (3,848,213,998 bytes) and VAE ``split_files/vae/flux2-vae.safetensors``
  (336,211,292 bytes). The same repo also has the bf16 diffusion file
  ``flux-2-klein-4b.safetensors`` (7,751,105,712 bytes), accepted when it
  is already on disk and not downloaded.

9B, FLUX.3, Flux.1 dev/schnell/pro/fill/depth/canny/kontext, and any gated
Klein repo are not offered. A gated payload bails before any token is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# Hub API 2026-10-22. Do not point this at a 9B or Flux.1 repo.
DIFFUSION_REPO = "black-forest-labs/FLUX.2-klein-4b-fp8"
DIFFUSION_FILE = "flux-2-klein-4b-fp8.safetensors"
DIFFUSION_BYTES = 4_070_624_520

PACK_REPO = "Comfy-Org/Flux-2-Klein-4B"
TEXT_FILE = "qwen_3_4b_fp4_flux2.safetensors"
TEXT_REPO_FILE = "split_files/text_encoders/qwen_3_4b_fp4_flux2.safetensors"
TEXT_BYTES = 3_848_213_998

VAE_FILE = "flux2-vae.safetensors"
VAE_REPO_FILE = "split_files/vae/flux2-vae.safetensors"
VAE_BYTES = 336_211_292

# Same 4B model, higher precision. Local copies count. Not an install target.
BF16_FILE = "flux-2-klein-4b.safetensors"
BF16_BYTES = 7_751_105_712
BF16_REPO_FILE = "split_files/diffusion_models/flux-2-klein-4b.safetensors"

LICENSE = "apache-2.0"
GATED = False
LICENSE_LINE = (
    "FLUX.2 [klein] 4B is Apache 2.0, ungated, and the only Flux stills model."
)

# Substrings that must never be fetched. Checked against repo id and filename.
_BANNED = (
    "9b",
    "schnell",
    "flux.1",
    "flux1",
    "kontext",
    "flux.3",
    "flux-3",
    "flux3",
    "fill",
    "depth",
    "canny",
    "pro",
)


class KleinRefused(RuntimeError):
    """A Flux pack other than ungated Klein 4B. No download is started."""


@dataclass(frozen=True)
class KleinFile:
    filename: str
    repo_id: str
    repo_filename: str
    size_bytes: int
    dest_folder: str

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo_id}/resolve/main/{self.repo_filename}"


FILES: tuple[KleinFile, ...] = (
    KleinFile(DIFFUSION_FILE, DIFFUSION_REPO, DIFFUSION_FILE, DIFFUSION_BYTES, "diffusion_models"),
    KleinFile(TEXT_FILE, PACK_REPO, TEXT_REPO_FILE, TEXT_BYTES, "text_encoders"),
    KleinFile(VAE_FILE, PACK_REPO, VAE_REPO_FILE, VAE_BYTES, "vae"),
)

ACCEPTED_DIFFUSION: tuple[str, ...] = (DIFFUSION_FILE, BF16_FILE)


def license_line() -> str:
    return LICENSE_LINE


def assert_allowed_name(repo_id: str, filename: str = "") -> None:
    """Refuse 9B, Flux.1, FLUX.3, and the other stripped variants."""
    blob = f"{repo_id} {filename}".lower()
    # "pro" is too short inside unrelated words; match it as a flux token.
    if "flux" in blob and any(
        tok in blob
        for tok in ("schnell", "kontext", "flux.1", "flux1", "9b", "fill", "depth", "canny", "-pro", "_pro")
    ):
        raise KleinRefused(
            f"Refusing {repo_id}/{filename or '(repo)'}: only FLUX.2 [klein] 4B is offered. "
            "9B, FLUX.3, and Flux.1 dev/schnell/pro/fill/depth/canny/kontext are not downloaded."
        )
    if "flux-3" in blob or "flux.3" in blob or "flux3" in blob:
        raise KleinRefused(
            f"Refusing {repo_id}: FLUX.3 is not offered. Klein 4B is the only Flux."
        )
    if "base-4b" in blob or "klein-base" in blob:
        raise KleinRefused(
            f"Refusing {repo_id}: Klein base is not an upgrade. The stills model is distilled 4B."
        )
    if "9b" in blob and "klein" in blob:
        raise KleinRefused(
            f"Refusing {repo_id}/{filename}: Klein 9B is not offered. "
            "The stills model is FLUX.2 [klein] 4B (Apache 2.0)."
        )


def accept_hub_model(payload: dict[str, Any]) -> None:
    """Require a public Hub model payload to be ungated Apache 2.0 Klein 4B.

    A gated model prints the license field and raises. No token is read.
    """
    repo = str(payload.get("id") or payload.get("modelId") or "")
    assert_allowed_name(repo)
    gated = payload.get("gated")
    if gated not in (False, "false"):
        card = payload.get("cardData") if isinstance(payload.get("cardData"), dict) else {}
        terms = card.get("license") or payload.get("license") or "(no license field on the payload)"
        raise KleinRefused(
            "Gated Klein is not offered. "
            f"License terms on the payload: {terms}. "
            "Stopped before any download. No Hugging Face token was used."
        )
    card = payload.get("cardData") if isinstance(payload.get("cardData"), dict) else {}
    license_name = str(card.get("license") or "")
    tags = [str(t) for t in (payload.get("tags") or [])]
    apache = license_name == LICENSE or "license:apache-2.0" in tags
    if not apache:
        raise KleinRefused(
            f"{repo} is not Apache 2.0 (license={license_name or 'missing'}). "
            "Klein 4B is the only Flux this install accepts."
        )
    if "4b" not in repo.lower() and "4B" not in repo:
        raise KleinRefused(f"{repo} is not the Klein 4B repo.")


def destination(spec: KleinFile, root: Path | None = None) -> Path:
    from master_agent.config import MODELS_DIR

    return (root or MODELS_DIR) / spec.dest_folder / spec.filename


def _full(path: Path, size_bytes: int) -> bool:
    try:
        return path.is_file() and path.stat().st_size >= size_bytes > 0
    except OSError:
        return False


def download_klein_4b(
    progress: Callable[[str], None] = print,
    *,
    root: Path | None = None,
    fetch: Callable[..., Path] | None = None,
) -> list[Path]:
    """Resumable, idempotent fetch into ``MODELS_DIR``. Skips a full local file.

    ``fetch`` replaces the Hub helper in tests. A short file at the destination
    is removed so the Hub cache can resume and the copy is written again.
    """
    from master_agent.models.download import download_hub_file

    getter = fetch or download_hub_file
    paths: list[Path] = []
    for spec in FILES:
        assert_allowed_name(spec.repo_id, spec.repo_filename)
        dest = destination(spec, root)
        if _full(dest, spec.size_bytes):
            progress(f"SKIP {spec.filename} — already {spec.size_bytes} bytes at {dest}")
            paths.append(dest)
            continue
        if dest.is_file():
            dest.unlink()
        progress(f"{LICENSE_LINE}")
        got = getter(
            repo_id=spec.repo_id,
            repo_filename=spec.repo_filename,
            dest=dest,
            progress=progress,
        )
        paths.append(Path(got))
    return paths


def present(root: Path | None = None) -> bool:
    return all(_full(destination(spec, root), spec.size_bytes) for spec in FILES)
