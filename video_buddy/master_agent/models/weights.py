"""Required-weight manifest, scan-first, and ask-to-download.

Product flow for any user of Video Buddy (not a one-off tower inventory):

1. Scan configured model dirs + common relative Comfy ``models/`` trees.
2. If every mandatory file for the workflow is present → proceed silently.
3. If something is missing → raise ``MissingWeightsError`` with a clear ask
   (filename, dest folder, size, gated-HF note). Never auto-download.
4. ``python -m master_agent download-models --ltx25 --yes`` pulls only the
   missing set after the user agrees.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

# Research-graph stub names → official Lightricks/LTX-2.5 split-pack filenames.
STUB_ALIASES: dict[str, str] = {
    "ltx-2.5-22b-distilled.safetensors": (
        "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
    ),
    "ltx-2.5-ic-lora.safetensors": "ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors",
    # No public Lightricks file named "msr"; IC-LoRA Ingredients is the closest
    # official adapter the MSR loader can consume. Users may drop a real MSR
    # LoRA in models/loras/ and the patcher will prefer a file that exists.
    "ltx-2.5-msr.safetensors": "ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors",
}

# Optional placeholders in the research templates — never mandatory.
OPTIONAL_STUBS = frozenset({"style.safetensors", "camera-orbit.safetensors"})

HF_LTX25 = "Lightricks/LTX-2.5"
HF_ICLORA = "Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients"
HF_LICENSE = "https://huggingface.co/Lightricks/LTX-2.5"
HF_ICLORA_LICENSE = "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients"


@dataclass(frozen=True)
class WeightFile:
    key: str
    filename: str
    dest_folder: str
    repo_id: str
    repo_filename: str
    size_bytes: int
    mandatory: bool
    gated: bool
    note: str = ""
    license_url: str = HF_LICENSE

    @property
    def size_label(self) -> str:
        n = float(self.size_bytes)
        if n >= 1e9:
            return f"{n / 1e9:.1f} GB"
        if n >= 1e6:
            return f"{n / 1e6:.0f} MB"
        return f"{int(n)} B"


# Official Comfy-ready distilled split pack (docs.comfy.org / Lightricks/LTX-2.5).
WEIGHT_FILES: dict[str, WeightFile] = {
    "transformer": WeightFile(
        key="transformer",
        filename="ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
        dest_folder="diffusion_models",
        repo_id=HF_LTX25,
        repo_filename="diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
        size_bytes=21_500_000_000,
        mandatory=True,
        gated=True,
        note="Comfy-ready int8 distilled transformer. Accept the LTX-2.x Community License first.",
    ),
    "text_encoder": WeightFile(
        key="text_encoder",
        filename="gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
        dest_folder="text_encoders",
        repo_id=HF_LTX25,
        repo_filename="text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
        size_bytes=15_400_000_000,
        mandatory=True,
        gated=True,
        note="Gemma 4 12B text encoder with the LTX 2.5 projection, Comfy int8.",
    ),
    "video_vae": WeightFile(
        key="video_vae",
        filename="ltx-2.5-video-vae-bf16.safetensors",
        dest_folder="vae",
        repo_id=HF_LTX25,
        repo_filename="vae/ltx-2.5-video-vae-bf16.safetensors",
        size_bytes=1_500_000_000,
        mandatory=True,
        gated=True,
        note="Video VAE (bf16).",
    ),
    "audio_vae": WeightFile(
        key="audio_vae",
        filename="ltx-2.5-audio-vae-bf16.safetensors",
        dest_folder="vae",
        repo_id=HF_LTX25,
        repo_filename="vae/ltx-2.5-audio-vae-bf16.safetensors",
        size_bytes=365_000_000,
        mandatory=True,
        gated=True,
        note="Audio VAE (bf16). Required for T2A / A2V and AV joint graphs.",
    ),
    "spatial_upscaler": WeightFile(
        key="spatial_upscaler",
        filename="ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
        dest_folder="latent_upscale_models",
        repo_id=HF_LTX25,
        repo_filename="latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
        size_bytes=996_000_000,
        mandatory=True,
        gated=True,
        note="2× latent spatial upscaler — mandatory for two-stage distilled.",
    ),
    "ic_lora": WeightFile(
        key="ic_lora",
        filename="ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors",
        dest_folder="loras",
        repo_id=HF_ICLORA,
        repo_filename="ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors",
        size_bytes=1_300_000_000,
        mandatory=True,
        gated=True,
        license_url=HF_ICLORA_LICENSE,
        note="Official IC-LoRA Ingredients. Reconciles stub ltx-2.5-ic-lora.safetensors / ltx-2.5-msr.safetensors.",
    ),
    "distilled_lora": WeightFile(
        key="distilled_lora",
        filename="ltx-2.5-22b-distilled-lora-450-bf16.safetensors",
        dest_folder="loras",
        repo_id=HF_LTX25,
        repo_filename="loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors",
        size_bytes=8_900_000_000,
        mandatory=False,
        gated=True,
        note="Optional distilled LoRA (450). Not required for the default distilled transformer pack.",
    ),
    "temporal_upscaler": WeightFile(
        key="temporal_upscaler",
        filename="ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors",
        dest_folder="latent_upscale_models",
        repo_id=HF_LTX25,
        repo_filename="latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors",
        size_bytes=262_000_000,
        mandatory=False,
        gated=True,
        note="Optional 2× latent temporal upscaler.",
    ),
    "duration_head": WeightFile(
        key="duration_head",
        filename="ltx-2.5-duration-head-bf16.safetensors",
        dest_folder="model_patches",
        repo_id=HF_LTX25,
        repo_filename="model_patches/ltx-2.5-duration-head-bf16.safetensors",
        size_bytes=3_800_000,
        mandatory=False,
        gated=True,
        note="Optional auto-duration model patch.",
    ),
}

# Bundle → weight keys. Mandatory flags on WeightFile still apply per key.
BUNDLES: dict[str, tuple[str, ...]] = {
    "ltx25_core": ("transformer", "text_encoder", "video_vae", "audio_vae"),
    "ltx25_two_stage": (
        "transformer",
        "text_encoder",
        "video_vae",
        "audio_vae",
        "spatial_upscaler",
    ),
    "ltx25_iclora": (
        "transformer",
        "text_encoder",
        "video_vae",
        "audio_vae",
        "ic_lora",
    ),
    "ltx25_msr": (
        "transformer",
        "text_encoder",
        "video_vae",
        "audio_vae",
        "ic_lora",
    ),
    "ltx25_all": tuple(WEIGHT_FILES.keys()),
}

# Variant id / alias → bundle
VARIANT_BUNDLES: dict[str, str] = {
    "ltx25_t2v_i2v": "ltx25_core",
    "ltx25_t2v_i2v_two_stage": "ltx25_two_stage",
    "ltx25_flf2v": "ltx25_core",
    "ltx25_a2v": "ltx25_core",
    "ltx25_t2a": "ltx25_core",
    "ltx25_v2v_ic_lora": "ltx25_iclora",
    "ltx25_msr": "ltx25_msr",
    "t2v_i2v": "ltx25_core",
    "t2v_i2v_two_stage": "ltx25_two_stage",
    "flf2v": "ltx25_core",
    "a2v": "ltx25_core",
    "t2a": "ltx25_core",
    "v2v_ic_lora": "ltx25_iclora",
    "msr": "ltx25_msr",
}


class MissingWeightsError(RuntimeError):
    """Queue-time hard stop: list missing files and how to download them."""

    def __init__(self, message: str, missing: list[WeightFile], *, bundle: str = ""):
        super().__init__(message)
        self.missing = missing
        self.bundle = bundle


@dataclass
class WeightStatus:
    present: list[WeightFile] = field(default_factory=list)
    missing_mandatory: list[WeightFile] = field(default_factory=list)
    missing_optional: list[WeightFile] = field(default_factory=list)
    roots: list[str] = field(default_factory=list)
    bundle: str = ""

    @property
    def ok(self) -> bool:
        return not self.missing_mandatory

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "bundle": self.bundle,
            "roots": self.roots,
            "present": [w.filename for w in self.present],
            "missing_mandatory": [_file_row(w) for w in self.missing_mandatory],
            "missing_optional": [_file_row(w) for w in self.missing_optional],
            "ask": format_ask(self) if not self.ok else "",
        }


def _file_row(w: WeightFile) -> dict[str, Any]:
    return {
        "key": w.key,
        "filename": w.filename,
        "dest_folder": w.dest_folder,
        "size": w.size_label,
        "repo": f"{w.repo_id}/{w.repo_filename}",
        "gated": w.gated,
        "mandatory": w.mandatory,
        "license": w.license_url,
        "note": w.note,
    }


def model_search_roots() -> list[Path]:
    """Configured dirs plus common relative Comfy ``models/`` locations."""
    from master_agent.config import COMFYUI_ROOT, MODELS_DIR, PROJECT_ROOT

    roots: list[Path] = []
    seen: set[Path] = set()

    def _add(path: Path) -> None:
        try:
            resolved = path.resolve()
        except OSError:
            return
        if resolved in seen:
            return
        seen.add(resolved)
        roots.append(resolved)

    _add(Path(MODELS_DIR))
    _add(Path(COMFYUI_ROOT) / "models")
    _add(Path(PROJECT_ROOT) / "models")
    _add(Path(PROJECT_ROOT) / "ComfyUI" / "models")
    _add(Path(PROJECT_ROOT) / "ComfyUI_windows_portable" / "ComfyUI" / "models")
    cwd = Path.cwd()
    _add(cwd / "models")
    _add(cwd / "video_buddy" / "models")
    _add(cwd / "ComfyUI" / "models")
    _add(cwd / "ComfyUI_windows_portable" / "ComfyUI" / "models")
    return roots


def find_weight_file(filename: str, roots: Iterable[Path] | None = None) -> Path | None:
    """Return the first existing path for a bare filename (or stub alias)."""
    names = [filename]
    aliased = STUB_ALIASES.get(filename)
    if aliased and aliased not in names:
        names.append(aliased)
    # Also accept the stub if the user actually dropped that name.
    for alias, official in STUB_ALIASES.items():
        if official == filename and alias not in names:
            names.append(alias)
    search = list(roots) if roots is not None else model_search_roots()
    subdirs = (
        "checkpoints",
        "diffusion_models",
        "unet",
        "loras",
        "vae",
        "text_encoders",
        "latent_upscale_models",
        "model_patches",
        "",
    )
    for root in search:
        if not root.is_dir():
            continue
        for name in names:
            direct = root / name
            if direct.is_file():
                return direct
            for sub in subdirs:
                candidate = root / sub / name if sub else root / name
                if candidate.is_file():
                    return candidate
            # Nested pack folders (loras/ltx/..., diffusion_models/ltx/...)
            try:
                hits = list(root.rglob(name))
            except OSError:
                hits = []
            for hit in hits:
                if hit.is_file():
                    return hit
    return None


def files_for_bundle(bundle: str) -> list[WeightFile]:
    keys = BUNDLES.get(bundle) or BUNDLES["ltx25_core"]
    return [WEIGHT_FILES[k] for k in keys if k in WEIGHT_FILES]


def bundle_for_variant(variant: str | None) -> str | None:
    key = (variant or "").strip()
    if key in VARIANT_BUNDLES:
        return VARIANT_BUNDLES[key]
    if key.startswith("ltx25") or key.startswith("ltx-2.5") or "/ltx-2.5/" in key:
        if "two_stage" in key or "two-stage" in key:
            return "ltx25_two_stage"
        if "iclora" in key or "ic_lora" in key or "ic-lora" in key:
            return "ltx25_iclora"
        if "msr" in key:
            return "ltx25_msr"
        return "ltx25_core"
    return None


def scan_bundle(bundle: str, *, roots: Iterable[Path] | None = None) -> WeightStatus:
    search = list(roots) if roots is not None else model_search_roots()
    status = WeightStatus(roots=[str(p) for p in search], bundle=bundle)
    for weight in files_for_bundle(bundle):
        found = find_weight_file(weight.filename, search)
        if found is not None:
            status.present.append(weight)
            continue
        if weight.mandatory:
            status.missing_mandatory.append(weight)
        else:
            status.missing_optional.append(weight)
    return status


def scan_variant(variant: str, *, roots: Iterable[Path] | None = None) -> WeightStatus | None:
    bundle = bundle_for_variant(variant)
    if not bundle:
        return None
    return scan_bundle(bundle, roots=roots)


def format_ask(status: WeightStatus) -> str:
    lines = [
        "Missing required model files for this workflow.",
        "Video Buddy scanned configured model dirs and common Comfy models/ paths.",
        "Nothing was downloaded (consent required).",
        "",
        f"Bundle: {status.bundle or 'ltx25'}",
        "",
        "Missing (mandatory):",
    ]
    for w in status.missing_mandatory:
        gated = "  [gated Hugging Face — accept the LTX-2.x Community License first]" if w.gated else ""
        lines.append(
            f"  - {w.filename}  →  models/{w.dest_folder}/  ({w.size_label}){gated}"
        )
        lines.append(f"      {w.repo_id}  {w.repo_filename}")
        if w.note:
            lines.append(f"      {w.note}")
        lines.append(f"      license: {w.license_url}")
    if status.missing_optional:
        lines.append("")
        lines.append("Optional (not blocking):")
        for w in status.missing_optional:
            lines.append(f"  - {w.filename}  →  models/{w.dest_folder}/  ({w.size_label})")
    lines.append("")
    lines.append("After you agree, download only the missing set with:")
    lines.append("  python -m master_agent download-models --ltx25 --yes")
    lines.append("or:")
    lines.append("  python -m master_agent doctor --fix-models")
    lines.append("")
    lines.append("Requires a Hugging Face token with access to the gated repos")
    lines.append(f"({HF_LICENSE}) via HF_TOKEN / huggingface-cli login.")
    return "\n".join(lines)


def require_weights(variant: str, *, roots: Iterable[Path] | None = None) -> WeightStatus | None:
    """Silent if present; raise MissingWeightsError if mandatory files are gone."""
    status = scan_variant(variant, roots=roots)
    if status is None or status.ok:
        return status
    raise MissingWeightsError(format_ask(status), status.missing_mandatory, bundle=status.bundle)


def download_files(
    files: Iterable[WeightFile],
    *,
    dest_root: Path | None = None,
    progress: Callable[[str], None] = print,
    yes: bool = False,
) -> list[Path]:
    """Copy missing weights into ``dest_root``. Refuses unless ``yes=True``."""
    from master_agent.config import MODELS_DIR

    wanted = list(files)
    if not wanted:
        return []
    if not yes:
        raise MissingWeightsError(
            "Refusing to download without consent. Re-run with --yes after reviewing:\n"
            + "\n".join(
                f"  {w.filename} → models/{w.dest_folder}/ ({w.size_label})"
                for w in wanted
            ),
            list(wanted),
        )
    from master_agent.models.download import download_hub_file

    root = Path(dest_root or MODELS_DIR)
    paths: list[Path] = []
    for weight in wanted:
        dest = root / weight.dest_folder / weight.filename
        paths.append(
            download_hub_file(
                repo_id=weight.repo_id,
                repo_filename=weight.repo_filename,
                dest=dest,
                progress=progress,
            )
        )
    return paths


def download_missing_bundle(
    bundle: str = "ltx25_all",
    *,
    dest_root: Path | None = None,
    progress: Callable[[str], None] = print,
    yes: bool = False,
    include_optional: bool = False,
) -> tuple[WeightStatus, list[Path]]:
    status = scan_bundle(bundle)
    missing = list(status.missing_mandatory)
    if include_optional:
        missing.extend(status.missing_optional)
    if not missing:
        progress(f"OK    all {bundle} weights already present")
        return status, []
    paths = download_files(missing, dest_root=dest_root, progress=progress, yes=yes)
    return status, paths
