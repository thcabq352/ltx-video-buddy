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

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

# Research-graph stub names → official Lightricks/LTX-2.5 split-pack filenames.
# Hub listing 2026-09: https://huggingface.co/Lightricks/LTX-2.5
STUB_ALIASES: dict[str, str] = {
    "ltx-2.5-22b-distilled.safetensors": (
        "ltx-2.5-22b-distilled-transformer-bf16.safetensors"
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
    # Local alternatives that satisfy this slot. First is preferred when several exist.
    # Official ``filename`` is the Hub download target (not the only accepted name).
    accepts: tuple[str, ...] = ()

    @property
    def size_label(self) -> str:
        n = float(self.size_bytes)
        if n >= 1e9:
            return f"{n / 1e9:.1f} GB"
        if n >= 1e6:
            return f"{n / 1e6:.0f} MB"
        return f"{int(n)} B"

    @property
    def candidates(self) -> tuple[str, ...]:
        """Preference-ordered names that count as present (plus research stubs)."""
        seen: list[str] = []
        for name in (*self.accepts, self.filename):
            if name and name not in seen:
                seen.append(name)
        for stub, official in STUB_ALIASES.items():
            if official == self.filename and stub not in seen:
                seen.append(stub)
        return tuple(seen)


# Official distilled split pack on gated Lightricks/LTX-2.5 (Hub file listing).
# Download / default-wire names are the bf16 pack. 16GB-class installs may
# already have comfy-int8-convrot / nvfp4 / GGUF Q4 — those still count.
WEIGHT_FILES: dict[str, WeightFile] = {
    "transformer": WeightFile(
        key="transformer",
        filename="ltx-2.5-22b-distilled-transformer-bf16.safetensors",
        dest_folder="diffusion_models",
        repo_id=HF_LTX25,
        repo_filename="diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
        size_bytes=42_000_000_000,
        mandatory=True,
        gated=True,
        note="Official distilled transformer (bf16, 42 GB). 16GB-class: comfy-int8-convrot, nvfp4, or GGUF Q4 also count.",
        accepts=(
            # When several exist locally, prefer the 16GB-class file first.
            "ltx-2.5-22b-distilled-transformer-bf16-Q4_K_M.gguf",
            "ltx-2.5-22b-distilled-transformer-nvfp4.safetensors",
            "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
            "ltx-2.5-22b-distilled-transformer-bf16.safetensors",
        ),
    ),
    "text_encoder": WeightFile(
        key="text_encoder",
        filename="gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
        dest_folder="text_encoders",
        repo_id=HF_LTX25,
        repo_filename="text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
        size_bytes=26_300_000_000,
        mandatory=True,
        gated=True,
        note="Official Gemma 4 12B + LTX 2.5 projection (bf16, 26.3 GB). Comfy int8 or heretic int8 also count.",
        accepts=(
            "gemma4-12b-heretic-ltx25-int8convrot.safetensors",
            "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
            "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
        ),
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
        note="Video VAE (bf16). The conv-bf16 variant also counts.",
        accepts=(
            "ltx-2.5-video-vae-bf16.safetensors",
            "ltx-2.5-video-vae-conv-bf16.safetensors",
        ),
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
        note="2× latent spatial upscaler — official pack (required for two-stage; part of the Hub split).",
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
        mandatory=True,
        gated=True,
        note="Official duration-head model patch (3.8 MB). Zero-byte placeholders count as missing.",
    ),
}

# Official Hub pack keys (mandatory). IC-LoRA is a separate gated repo.
_LTX25_OFFICIAL = (
    "transformer",
    "text_encoder",
    "video_vae",
    "audio_vae",
    "duration_head",
    "spatial_upscaler",
)

# Bundle → weight keys. Mandatory flags on WeightFile still apply per key.
BUNDLES: dict[str, tuple[str, ...]] = {
    "ltx25_core": _LTX25_OFFICIAL,
    "ltx25_two_stage": _LTX25_OFFICIAL,
    "ltx25_iclora": (*_LTX25_OFFICIAL, "ic_lora"),
    "ltx25_msr": (*_LTX25_OFFICIAL, "ic_lora"),
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


def _is_usable_file(path: Path) -> bool:
    """Present means a real file with bytes. Zero-byte placeholders are missing."""
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _hf_hub_snapshot_roots() -> list[Path]:
    """Common Hugging Face hub snapshot trees (not a one-off machine path)."""
    hubs: list[Path] = []
    env_cache = os.environ.get("HUGGINGFACE_HUB_CACHE") or os.environ.get("HF_HUB_CACHE")
    if env_cache:
        hubs.append(Path(env_cache))
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        hubs.append(Path(hf_home) / "hub")
    hubs.append(Path.home() / ".cache" / "huggingface" / "hub")
    repos = (
        "models--Lightricks--LTX-2.5",
        "models--Lightricks--LTX-2.5-22b-IC-LoRA-Ingredients",
    )
    out: list[Path] = []
    for hub in hubs:
        for repo in repos:
            snap = hub / repo / "snapshots"
            if snap.is_dir():
                out.append(snap)
    return out


def model_search_roots() -> list[Path]:
    """Configured dirs plus common relative Comfy ``models/`` and HF hub layouts."""
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
    for snap in _hf_hub_snapshot_roots():
        _add(snap)
    return roots


def _search_name(name: str, roots: Iterable[Path]) -> Path | None:
    subdirs = (
        "checkpoints",
        "diffusion_models",
        "diffusion_models/gguf",
        "unet",
        "loras",
        "vae",
        "text_encoders",
        "latent_upscale_models",
        "model_patches",
        "",
    )
    for root in roots:
        if not root.is_dir():
            continue
        direct = root / name
        if _is_usable_file(direct):
            return direct
        for sub in subdirs:
            candidate = root.joinpath(*sub.split("/"), name) if sub else root / name
            if _is_usable_file(candidate):
                return candidate
        try:
            hits = root.rglob(name)
        except OSError:
            hits = []
        for hit in hits:
            if _is_usable_file(hit):
                return hit
    return None


def find_weight_file(filename: str, roots: Iterable[Path] | None = None) -> Path | None:
    """Return the first usable path for a bare filename (or stub / family alias)."""
    names = [filename]
    aliased = STUB_ALIASES.get(filename)
    if aliased and aliased not in names:
        names.append(aliased)
    for alias, official in STUB_ALIASES.items():
        if official == filename and alias not in names:
            names.append(alias)
    for weight in WEIGHT_FILES.values():
        if filename in weight.candidates:
            for extra in weight.candidates:
                if extra not in names:
                    names.append(extra)
            break
    search = list(roots) if roots is not None else model_search_roots()
    for name in names:
        found = _search_name(name, search)
        if found is not None:
            return found
    return None


def resolve_weight(weight: WeightFile, roots: Iterable[Path] | None = None) -> Path | None:
    """Best local file for a slot (preference order). None if all candidates missing/empty."""
    search = list(roots) if roots is not None else model_search_roots()
    for name in weight.candidates:
        found = _search_name(name, search)
        if found is not None:
            return found
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
        found = resolve_weight(weight, search)
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
        alts = [n for n in w.candidates if n != w.filename]
        if alts:
            lines.append("      also accepted locally: " + ", ".join(alts))
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
