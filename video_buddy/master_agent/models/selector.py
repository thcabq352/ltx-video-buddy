"""Model selector (Stability Matrix pattern, not their code).

The catalog is grouped by capability (video generation, soundtrack, local
studio). An LTX 2.3 / 2.5 radio swaps the video-generation checklist.
Required rows stay on and are labeled ``required for generation``. Optional
rows carry sizes when this repo already attests a byte size.

The studio Models tab calls ``catalog_document``, ``assess``, and
``apply_selection``. It does not keep a second catalog.

Downloads go through the existing consent paths (``download_files`` and
``download-models --heartmula``) and only after a free-space check. Files
already on disk are skipped. Sulphur and EROS are local detect flags: this
module never builds a Hub URL for them.

Nothing here runs ``comfy install`` or ``comfy update``.
"""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from master_agent.config import DEFAULT_ALL_IN_ONE_CKPT, EROS_GGUF_BYTES
from master_agent.heartmula.doctor import HUB_USED_STORAGE_BYTES
from master_agent.models.vram_policy import LTX23_DISTILLED_GGUF
from master_agent.models.weights import (
    BUNDLES,
    LTX23_INOUTPAINT_LORA,
    LTX23_LATENT_UPSCALER,
    SULPHUR_Q3_GGUF,
    WEIGHT_FILES,
    WeightFile,
)

SCHEMA = "buddy.model_selector/v1"
STATE_SCHEMA = "buddy.model_selector.state/v1"
REQUIRED_LABEL = "required for generation"
DEFAULT_MBPS = 50.0
CATALOG_PATH = Path(__file__).with_name("selector_catalog.json")

# Attested in ltx23_inoutpaint_lora_placement ("about 1.31 GB"), same figure
# the 2.5 Ingredients WeightFile already uses. Not a Hub fetch from this row.
_INOUTPAINT_ATTESTED_BYTES = 1_310_000_000

_LTX23_VAE_NAMES = (
    "taeltx2_3.safetensors",
    "LTX23_video_vae_bf16.safetensors",
)
_LTX23_TE_NAMES = (
    "gemma_3_12B_it_fp4_mixed.safetensors",
    "gemma_3_12B_it_fp8_scaled.safetensors",
)
_LTX23_UPSCALER_NAMES = (
    LTX23_LATENT_UPSCALER,
    "ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
)
_LTX23_DISTILLED_LORA_NAMES = (
    "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
    "ltx-2.3-22b-distilled-lora-384.safetensors",
)
_DEV_FP8_CKPT = "ltx-2.3-22b-dev-fp8.safetensors"


@dataclass(frozen=True)
class CatalogModel:
    """One checklist row. No Hub URL field — downloads resolve by weight_key."""

    id: str
    label: str
    filename: str
    accepts: tuple[str, ...]
    size_bytes: int | None
    required: bool
    package: str
    version: str | None
    capability: str
    downloadable: bool
    weight_key: str | None = None
    local_only: bool = False
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["accepts"] = list(self.accepts)
        payload["size_label"] = format_size(self.size_bytes)
        payload["required_label"] = REQUIRED_LABEL if self.required else ""
        payload["locked"] = self.required
        return payload


@dataclass
class RowView:
    model: CatalogModel
    present: bool
    selected: bool
    detail: str = ""

    @property
    def will_fetch(self) -> bool:
        return (
            self.selected
            and self.model.downloadable
            and not self.model.local_only
            and not self.present
            and self.model.size_bytes is not None
        )

    def to_dict(self) -> dict[str, Any]:
        data = self.model.to_dict()
        data.update(
            present=self.present,
            selected=self.selected,
            detail=self.detail,
            will_fetch=self.will_fetch,
            skipped=self.present and self.selected,
        )
        return data


@dataclass
class Plan:
    version: str
    rows: list[RowView]
    soundtrack: bool
    download_bytes: int
    selected_bytes: int
    eta_seconds: float
    mbps: float
    disk_free_bytes: int | None
    disk_ok: bool
    disk_path: str
    unknown_size_ids: list[str] = field(default_factory=list)
    local_flags: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "version": self.version,
            "soundtrack_studio": self.soundtrack,
            "download_bytes": self.download_bytes,
            "download_label": format_size(self.download_bytes),
            "selected_bytes": self.selected_bytes,
            "selected_label": format_size(self.selected_bytes),
            "eta_seconds": round(self.eta_seconds, 1),
            "eta_label": format_eta(self.eta_seconds),
            "mbps": self.mbps,
            "disk_free_bytes": self.disk_free_bytes,
            "disk_free_label": format_size(self.disk_free_bytes),
            "disk_ok": self.disk_ok,
            "disk_path": self.disk_path,
            "unknown_size_ids": list(self.unknown_size_ids),
            "rows": [row.to_dict() for row in self.rows],
            "local_flags": list(self.local_flags),
        }


def format_size(size_bytes: int | None) -> str:
    if size_bytes is None:
        return "size not attested"
    n = float(size_bytes)
    if n >= 1e9:
        return f"{n / 1e9:.1f} GB"
    if n >= 1e6:
        return f"{n / 1e6:.0f} MB"
    if n <= 0:
        return "0 GB"
    return f"{int(n)} B"


def eta_seconds(size_bytes: int, mbps: float = DEFAULT_MBPS) -> float:
    """Rough download time. ``mbps`` is megabits per second."""
    rate = float(mbps)
    if size_bytes <= 0 or rate <= 0:
        return 0.0
    return (size_bytes * 8) / (rate * 1_000_000)


def format_eta(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _from_weight(
    version: str,
    key: str,
    *,
    required: bool,
    package: str,
) -> CatalogModel:
    weight: WeightFile = WEIGHT_FILES[key]
    return CatalogModel(
        id=f"ltx{version.replace('.', '')}.{key}",
        label=weight.filename,
        filename=weight.filename,
        accepts=weight.candidates,
        size_bytes=weight.size_bytes,
        required=required,
        package=package,
        version=version,
        capability="video_generation",
        downloadable=True,
        weight_key=key,
        local_only=False,
        note=weight.note,
    )


def video_models(version: str) -> list[CatalogModel]:
    """Required and optional video rows for one radio position."""
    if version == "2.5":
        core = BUNDLES["ltx25_core"]
        required_keys = [key for key in core if WEIGHT_FILES[key].mandatory]
        optional_keys = [key for key in BUNDLES["ltx25_all"] if key not in required_keys]
        rows = [
            _from_weight("2.5", key, required=True, package="ltx25_core")
            for key in required_keys
        ]
        rows.extend(
            _from_weight("2.5", key, required=False, package=key) for key in optional_keys
        )
        return rows
    if version == "2.3":
        diffusion_names = tuple(dict.fromkeys((*LTX23_DISTILLED_GGUF, _DEV_FP8_CKPT, DEFAULT_ALL_IN_ONE_CKPT)))
        return [
            CatalogModel(
                id="ltx23.diffusion",
                label=LTX23_DISTILLED_GGUF[0],
                filename=LTX23_DISTILLED_GGUF[0],
                accepts=diffusion_names,
                size_bytes=None,
                required=True,
                package="ltx23_core",
                version="2.3",
                capability="video_generation",
                downloadable=False,
                note=(
                    "LTX 2.3 diffusion. A local GGUF alias, the dev fp8 checkpoint, "
                    "or 10Eros_v1.5-Q4_K_M.gguf counts as already on disk. "
                    "Byte size is not attested here, so this row is not fetched."
                ),
            ),
            CatalogModel(
                id="ltx23.text_encoder",
                label=_LTX23_TE_NAMES[0],
                filename=_LTX23_TE_NAMES[0],
                accepts=_LTX23_TE_NAMES,
                size_bytes=None,
                required=True,
                package="ltx23_core",
                version="2.3",
                capability="video_generation",
                downloadable=False,
                note=(
                    "Gemma text encoder used by the LTX 2.3 graphs. "
                    "A heretic Gemma encoder already on disk also counts. "
                    "Byte size is not attested here, so this row is not fetched."
                ),
            ),
            CatalogModel(
                id="ltx23.vae",
                label=_LTX23_VAE_NAMES[0],
                filename=_LTX23_VAE_NAMES[0],
                accepts=_LTX23_VAE_NAMES,
                size_bytes=None,
                required=True,
                package="ltx23_core",
                version="2.3",
                capability="video_generation",
                downloadable=False,
                note=(
                    "LTX 2.3 video VAE. Byte size is not attested here, "
                    "so this row is not fetched."
                ),
            ),
            CatalogModel(
                id="ltx23.upscaler",
                label=LTX23_LATENT_UPSCALER,
                filename=LTX23_LATENT_UPSCALER,
                accepts=_LTX23_UPSCALER_NAMES,
                size_bytes=None,
                required=False,
                package="latent_upscaler",
                version="2.3",
                capability="video_generation",
                downloadable=False,
                note="Optional spatial upscaler. Scan only.",
            ),
            CatalogModel(
                id="ltx23.distilled_lora",
                label=_LTX23_DISTILLED_LORA_NAMES[0],
                filename=_LTX23_DISTILLED_LORA_NAMES[0],
                accepts=_LTX23_DISTILLED_LORA_NAMES,
                size_bytes=None,
                required=False,
                package="distilled_lora",
                version="2.3",
                capability="video_generation",
                downloadable=False,
                note="Optional distilled LoRA. Scan only.",
            ),
            CatalogModel(
                id="ltx23.inoutpaint_lora",
                label=LTX23_INOUTPAINT_LORA,
                filename=LTX23_INOUTPAINT_LORA,
                accepts=(LTX23_INOUTPAINT_LORA,),
                size_bytes=_INOUTPAINT_ATTESTED_BYTES,
                required=False,
                package="inoutpaint",
                version="2.3",
                capability="video_generation",
                downloadable=False,
                note=(
                    "Optional In-Outpainting IC-LoRA, about 1.31 GB. "
                    "Doctor does not download it; the selector does not either."
                ),
            ),
        ]
    raise ValueError(f"LTX version must be 2.3 or 2.5, got {version!r}")


def soundtrack_models() -> list[CatalogModel]:
    """HeartMuLa pack sizes from attested Hub usedStorage. Toggle wraps consent."""
    from master_agent.heartmula.config import (
        DEFAULT_CODEC_REPO,
        DEFAULT_MULA_REPO,
        DEFAULT_TRANSCRIPTOR_REPO,
    )

    specs = (
        ("soundtrack.mula", "HeartMuLa", DEFAULT_MULA_REPO, "soundtrack_studio"),
        ("soundtrack.codec", "HeartCodec", DEFAULT_CODEC_REPO, "soundtrack_studio"),
        (
            "soundtrack.transcriptor",
            "HeartTranscriptor",
            DEFAULT_TRANSCRIPTOR_REPO,
            "soundtrack_studio",
        ),
    )
    rows: list[CatalogModel] = []
    for row_id, label, repo_id, package in specs:
        rows.append(
            CatalogModel(
                id=row_id,
                label=label,
                filename=label,
                accepts=(),
                size_bytes=HUB_USED_STORAGE_BYTES[repo_id],
                required=False,
                package=package,
                version=None,
                capability="soundtrack",
                downloadable=True,
                note=(
                    "Enable Soundtrack Studio. Fetch stays on "
                    "download-models --heartmula (existing heartlib consent path)."
                ),
            )
        )
    return rows


def local_models() -> list[CatalogModel]:
    """Sulphur / EROS detect flags. Filenames only. No repo id and no URL."""
    return [
        CatalogModel(
            id="local.sulphur_gguf",
            label="Sulphur GGUF",
            filename=SULPHUR_Q3_GGUF,
            accepts=(SULPHUR_Q3_GGUF,),
            size_bytes=None,
            required=False,
            package="local_sulphur",
            version=None,
            capability="local_studio",
            downloadable=False,
            local_only=True,
            note="Detect a Sulphur GGUF already on disk. Buddy does not download it.",
        ),
        CatalogModel(
            id="local.sulphur_lora",
            label="Sulphur LoRA",
            filename="",
            accepts=(),
            size_bytes=None,
            required=False,
            package="local_sulphur",
            version=None,
            capability="local_studio",
            downloadable=False,
            local_only=True,
            note="Detect files under models/loras/sulphur/. Buddy does not download them.",
        ),
        CatalogModel(
            id="local.eros",
            label="10Eros v1.5 GGUF",
            filename=DEFAULT_ALL_IN_ONE_CKPT,
            accepts=(DEFAULT_ALL_IN_ONE_CKPT,),
            size_bytes=EROS_GGUF_BYTES,
            required=False,
            package="local_eros",
            version=None,
            capability="local_studio",
            downloadable=False,
            local_only=True,
            note=(
                "Local diffusion GGUF 10Eros_v1.5-Q4_K_M.gguf "
                "(14,296,161,888 bytes). Buddy does not download it."
            ),
        ),
    ]


def catalog_document() -> dict[str, Any]:
    """Static catalog. Presence is filled in later by :func:`assess`."""
    return {
        "schema": SCHEMA,
        "group_by": "capability",
        "eta_mbps_default": DEFAULT_MBPS,
        "versions": ["2.3", "2.5"],
        "capabilities": [
            {
                "id": "video_generation",
                "label": "Video generation",
                "note": "The 2.3 / 2.5 radio swaps this checklist.",
                "models": [row.to_dict() for row in (*video_models("2.3"), *video_models("2.5"))],
            },
            {
                "id": "soundtrack",
                "label": "Soundtrack",
                "toggle": {
                    "id": "soundtrack_studio",
                    "label": "Enable Soundtrack Studio",
                    "wraps": "download-models --heartmula",
                },
                "models": [row.to_dict() for row in soundtrack_models()],
            },
            {
                "id": "local_studio",
                "label": "Local studio",
                "note": "Optional on-disk flags. No Hub pull.",
                "models": [row.to_dict() for row in local_models()],
            },
        ],
    }


def version_for_variant(variant: str | None) -> str | None:
    """Map a catalog variant onto the selector radio. Other families return None."""
    key = (variant or "").strip().lower().replace("\\", "/")
    if not key:
        return None
    if key.startswith("ltx25") or key.startswith("ltx-2.5") or key in {
        "t2v_i2v",
        "t2v_i2v_two_stage",
        "flf2v",
        "a2v",
        "t2a",
        "v2v_ic_lora",
        "msr",
    }:
        return "2.5"
    if key.startswith("ltx23") or key.startswith("ltx-2.3") or key in {
        "base",
        "eros",
        "directors",
        "lipsync",
    }:
        return "2.3"
    return None


def _models_dir() -> Path:
    from master_agent import config

    return Path(config.MODELS_DIR)


def _state_file() -> Path:
    from master_agent import config

    return Path(config.STATE_DIR) / "model_selector.json"


def load_state() -> dict[str, Any] | None:
    path = _state_file()
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def save_state(payload: dict[str, Any]) -> Path:
    path = _state_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def probe_disk(path: Path) -> int:
    return int(shutil.disk_usage(path).free)


def _disk_target() -> Path:
    models = _models_dir()
    if models.exists():
        return models
    parent = models.parent
    if parent.exists():
        return parent
    return Path.cwd()


def _is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except (ValueError, OSError):
        return False
    return True


def _search_roots(roots: Iterable[Path] | None) -> list[Path]:
    if roots is not None:
        return [Path(root) for root in roots]
    from master_agent.models.weights import model_search_roots

    return list(model_search_roots())


def _usable(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0 and not path.is_symlink()
    except OSError:
        return False


def _find_name(name: str, roots: Sequence[Path]) -> Path | None:
    if not name:
        return None
    from master_agent.models.weights import find_weight_file

    return find_weight_file(name, roots)


def sulphur_lora_count(roots: Sequence[Path]) -> int:
    """Count weight files in ``loras/sulphur/`` only. Does not return names."""
    count = 0
    for root in roots:
        folder = Path(root) / "loras" / "sulphur"
        if not folder.is_dir():
            continue
        try:
            children = list(folder.iterdir())
        except OSError:
            continue
        for path in children:
            if path.suffix.lower() in {".safetensors", ".gguf"} and _usable(path):
                count += 1
    return count


def sulphur_gguf_present(roots: Sequence[Path]) -> bool:
    from master_agent.models.weights import is_sulphur_or_ltx23_gguf

    if _find_name(SULPHUR_Q3_GGUF, roots) is not None:
        return True
    for root in roots:
        if not root.is_dir():
            continue
        try:
            paths = root.rglob("*.gguf")
        except OSError:
            continue
        for path in paths:
            if "sulphur" in path.name.lower() and is_sulphur_or_ltx23_gguf(path.name) and _usable(path):
                return True
    return False


def _row_present(model: CatalogModel, roots: Sequence[Path]) -> tuple[bool, str]:
    if model.id == "local.sulphur_lora":
        count = sulphur_lora_count(roots)
        if count:
            return True, f"{count} file(s) under loras/sulphur"
        return False, "no files under loras/sulphur"
    if model.id == "local.sulphur_gguf":
        if sulphur_gguf_present(roots):
            return True, "Sulphur GGUF on disk"
        return False, "Sulphur GGUF not on disk"
    if model.id == "local.eros":
        found = _find_name(model.filename, roots)
        if found is not None:
            return True, "10Eros GGUF on disk"
        return False, "10Eros GGUF not on disk"
    if model.capability == "soundtrack":
        return _soundtrack_present(model)
    if model.weight_key:
        from master_agent.models.weights import resolve_weight

        found = resolve_weight(WEIGHT_FILES[model.weight_key], roots)
        if found is not None:
            return True, f"already have {found.name}"
        return False, "missing"
    if model.id == "ltx23.diffusion":
        from master_agent.models.weights import resolve_ltx23_gguf

        found = resolve_ltx23_gguf("base", roots)
        if found is not None:
            return True, f"already have {found.name}"
        for name in (_DEV_FP8_CKPT, DEFAULT_ALL_IN_ONE_CKPT):
            alt = _find_name(name, roots)
            if alt is not None:
                return True, f"already have {alt.name}"
        return False, "missing"
    if model.id == "ltx23.text_encoder":
        for name in model.accepts:
            found = _find_name(name, roots)
            if found is not None:
                return True, f"already have {found.name}"
        from master_agent.models.weights import find_compatible_for_weight

        heretic = find_compatible_for_weight(WEIGHT_FILES["text_encoder"], roots)
        if heretic is not None:
            return True, f"already have {heretic.name}"
        return False, "missing"
    for name in model.accepts or ((model.filename,) if model.filename else ()):
        found = _find_name(name, roots)
        if found is not None:
            return True, f"already have {found.name}"
    return False, "missing"


def _soundtrack_present(model: CatalogModel) -> tuple[bool, str]:
    from master_agent.heartmula.config import HeartMuLaConfigError, codec_repo, mula_repo, transcriptor_repo
    from master_agent.heartmula.doctor import scan_slots

    repo_for = {
        "soundtrack.mula": mula_repo,
        "soundtrack.codec": codec_repo,
        "soundtrack.transcriptor": transcriptor_repo,
    }
    lookup = repo_for.get(model.id)
    if lookup is None:
        return False, "unknown soundtrack row"
    try:
        repo_id = lookup()
        rows = scan_slots()
    except HeartMuLaConfigError as exc:
        return False, str(exc)
    matched = [row for row in rows if row["repo_id"] == repo_id]
    if not matched:
        return False, "no attested slots"
    missing = [row for row in matched if not row["present"]]
    if missing:
        return False, f"{len(missing)} file(s) missing"
    return True, "already have this pack"


def optional_ids_for(version: str) -> list[str]:
    return [model.id for model in video_models(version) if not model.required]


def normalize_optional(version: str, optional: Sequence[str] | None, *, all_optional: bool) -> list[str]:
    known = set(optional_ids_for(version))
    chosen = set(known if all_optional else ())
    unknown: list[str] = []
    for raw in optional or ():
        for piece in str(raw).split(","):
            item = piece.strip()
            if not item:
                continue
            if item not in known:
                unknown.append(item)
            else:
                chosen.add(item)
    if unknown:
        known_list = ", ".join(sorted(known))
        raise ValueError(f"Unknown optional id(s): {', '.join(unknown)}. Known: {known_list}")
    return sorted(chosen)


def assess(
    version: str,
    optional_ids: Sequence[str],
    *,
    soundtrack: bool,
    roots: Iterable[Path] | None = None,
    mbps: float = DEFAULT_MBPS,
) -> Plan:
    search = _search_roots(roots)
    selected_optional = set(optional_ids)
    rows: list[RowView] = []
    for model in video_models(version):
        present, detail = _row_present(model, search)
        selected = model.required or model.id in selected_optional
        rows.append(RowView(model=model, present=present, selected=selected, detail=detail))
    for model in soundtrack_models():
        present, detail = _row_present(model, search)
        rows.append(
            RowView(model=model, present=present, selected=soundtrack, detail=detail)
        )
    local_flags: list[dict[str, Any]] = []
    for model in local_models():
        present, detail = _row_present(model, search)
        local_flags.append(
            {
                "id": model.id,
                "label": model.label,
                "present": present,
                "detail": detail,
                "local_only": True,
                "downloadable": False,
            }
        )
        rows.append(RowView(model=model, present=present, selected=False, detail=detail))

    download_bytes = 0
    selected_bytes = 0
    running = 0
    unknown: list[str] = []
    for row in rows:
        if not row.selected:
            continue
        size = row.model.size_bytes
        if size is None:
            if row.model.downloadable and not row.present:
                unknown.append(row.model.id)
            continue
        selected_bytes += size
        if row.will_fetch:
            download_bytes += size
            running += size
            row.detail = f"{row.detail}; running download {format_size(running)}"
        elif row.present:
            row.detail = f"{row.detail}; skipped"
    target = _disk_target()
    try:
        free = probe_disk(target)
    except OSError:
        free = None
    disk_ok = free is not None and free >= download_bytes
    return Plan(
        version=version,
        rows=rows,
        soundtrack=soundtrack,
        download_bytes=download_bytes,
        selected_bytes=selected_bytes,
        eta_seconds=eta_seconds(download_bytes, mbps),
        mbps=mbps,
        disk_free_bytes=free,
        disk_ok=disk_ok,
        disk_path=str(target),
        unknown_size_ids=unknown,
        local_flags=local_flags,
    )


def render_plan(plan: Plan) -> str:
    lines = [
        f"LTX {plan.version}  (radio; required rows stay on)",
    ]
    headings = {
        "video_generation": "Video generation",
        "soundtrack": "Soundtrack  —  Enable Soundtrack Studio",
    }
    running = 0
    seen = ""
    for row in plan.rows:
        model = row.model
        if model.capability == "local_studio":
            continue
        if model.capability != seen:
            seen = model.capability
            lines.append("")
            lines.append(headings.get(model.capability, model.capability))
        if model.required:
            mark = "[required]"
            role = REQUIRED_LABEL
        elif row.selected:
            mark = "[x]"
            role = "optional"
        else:
            mark = "[ ]"
            role = "optional"
        if row.will_fetch and model.size_bytes:
            running += model.size_bytes
            run_label = f"running {format_size(running)}"
        elif row.selected and row.present:
            run_label = "already have — skipped"
        elif row.selected:
            run_label = "selected"
        else:
            run_label = "off"
        state = "have" if row.present else "NEED"
        lines.append(
            f"  {mark:<12} {model.label:<52} {format_size(model.size_bytes):>18}  "
            f"{role:<24} {state:<5} {run_label}"
        )
    lines.append("")
    lines.append(
        f"download {format_size(plan.download_bytes)}   "
        f"ETA {format_eta(plan.eta_seconds)} at {plan.mbps:g} Mbit/s   "
        f"selected sizes {format_size(plan.selected_bytes)}"
    )
    free = format_size(plan.disk_free_bytes)
    gate = "OK" if plan.disk_ok else "SHORT"
    lines.append(f"disk free {free} on {plan.disk_path}   {gate}")
    if plan.soundtrack:
        lines.append("Soundtrack Studio: on (download-models --heartmula consent path)")
    else:
        lines.append("Soundtrack Studio: off")
    lines.append("Local studio (detect only):")
    for flag in plan.local_flags:
        state = "on disk" if flag["present"] else "not on disk"
        lines.append(f"  [local] {flag['label']:<20} {state}  {flag['detail']}")
    if plan.unknown_size_ids:
        lines.append(
            "Size is not attested for: " + ", ".join(plan.unknown_size_ids) + ". Those rows are not fetched."
        )
    return "\n".join(lines)


def version_wipe_filenames(version: str) -> set[str]:
    """Canonical filenames safe to delete. Sulphur and EROS names are excluded."""
    blocked = {model.filename.lower() for model in local_models() if model.filename}
    blocked.add(DEFAULT_ALL_IN_ONE_CKPT.lower())
    blocked.add(SULPHUR_Q3_GGUF.lower())
    names: set[str] = set()
    for model in video_models(version):
        if model.local_only or not model.filename:
            continue
        lowered = model.filename.lower()
        if "sulphur" in lowered or "eros" in lowered:
            continue
        if lowered in blocked:
            continue
        names.add(model.filename)
    return names


def wipe_version(version: str) -> list[Path]:
    """Unlink the previous version's canonical files under ``MODELS_DIR`` only."""
    from master_agent.models.download import assert_pack_download_destination

    root = _models_dir().resolve()
    if not root.is_dir():
        return []
    names = version_wipe_filenames(version)
    removed: list[Path] = []
    try:
        paths = list(root.rglob("*"))
    except OSError:
        return []
    for path in paths:
        if path.name not in names or not path.is_file() or path.is_symlink():
            continue
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if not _is_under(resolved, root):
            continue
        assert_pack_download_destination(resolved)
        path.unlink()
        removed.append(resolved)
    return removed


def ask_keep_or_wipe(previous: str) -> str | None:
    prompt = (
        f"Keep the LTX {previous} files already on disk, or wipe them from MODELS_DIR? "
        "[keep/wipe] "
    )
    if sys.stdin is None or not sys.stdin.isatty():
        print(prompt + "(non-interactive; pass --keep or --wipe)")
        return None
    try:
        answer = input(prompt).strip().lower()
    except EOFError:
        return None
    if answer in {"keep", "k"}:
        return "keep"
    if answer in {"wipe", "w"}:
        return "wipe"
    print("Answer keep or wipe. Nothing deleted. Nothing downloaded.")
    return None


def _weight_files_to_fetch(plan: Plan) -> list[WeightFile]:
    files: list[WeightFile] = []
    for row in plan.rows:
        if not row.will_fetch or not row.model.weight_key:
            continue
        files.append(WEIGHT_FILES[row.model.weight_key])
    return files


def _soundtrack_needs_fetch(plan: Plan) -> bool:
    if not plan.soundtrack:
        return False
    return any(
        row.model.capability == "soundtrack" and row.selected and not row.present for row in plan.rows
    )


def missing_version_message(version: str) -> str:
    other = "2.3" if version == "2.5" else "2.5"
    return (
        f"LTX {version} is missing required files. "
        f"Buddy will not silently switch to LTX {other}. "
        f"Nothing was downloaded. "
        f"Offer: python -m master_agent models select --version {version} "
        f"or python -m master_agent models select --version {other} --keep"
    )


def version_ready(version: str, roots: Iterable[Path] | None = None) -> bool:
    plan = assess(version, [], soundtrack=False, roots=roots)
    required = [row for row in plan.rows if row.model.required and row.model.version == version]
    return bool(required) and all(row.present for row in required)


def job_version_block(variant: str | None, roots: Iterable[Path] | None = None) -> str | None:
    """Report a missing LTX version. No state file means the selector is not in use."""
    state = load_state()
    if not state:
        return None
    version = version_for_variant(variant)
    if version is None:
        return None
    active = str(state.get("active_version") or "")
    kept = {str(item) for item in state.get("kept_versions") or []}
    per_job = bool(state.get("per_job_version"))
    if active and version != active and not (per_job and version in kept):
        return (
            f"This job wants LTX {version}. The selector is on LTX {active}. "
            f"Buddy will not silently use LTX {active} files for an LTX {version} job. "
            f"Offer: python -m master_agent models select --version {version}"
        )
    if version_ready(version, roots):
        return None
    return missing_version_message(version)


def _merge_kept(previous: dict[str, Any] | None, version: str, choice: str | None) -> tuple[list[str], bool]:
    prior = str((previous or {}).get("active_version") or "")
    kept_now = [str(item) for item in (previous or {}).get("kept_versions") or []]
    if prior and prior != version:
        if choice == "wipe":
            return [version], False
        return sorted(set([*kept_now, prior, version])), True
    return sorted(set([*kept_now, version])), len(set([*kept_now, version])) > 1


def apply_selection(
    version: str,
    optional_ids: Sequence[str],
    *,
    soundtrack: bool,
    yes: bool,
    wipe: bool,
    keep: bool,
    scan_only: bool,
    mbps: float = DEFAULT_MBPS,
    roots: Iterable[Path] | None = None,
    progress: Any = print,
) -> tuple[int, Plan]:
    """Print the checklist. Download only with ``yes`` after the disk gate."""
    if wipe and keep:
        progress("FAIL  pass only one of --keep and --wipe")
        plan = assess(version, optional_ids, soundtrack=soundtrack, roots=roots, mbps=mbps)
        return 2, plan
    plan = assess(version, optional_ids, soundtrack=soundtrack, roots=roots, mbps=mbps)
    progress(render_plan(plan))
    if scan_only:
        progress("\n--scan-only: nothing downloaded. Selector state was not changed.")
        return 0 if plan.disk_ok or plan.download_bytes == 0 else 2, plan

    state = load_state()
    prior = str((state or {}).get("active_version") or "")
    choice: str | None = None
    if prior and prior != version:
        if wipe:
            choice = "wipe"
        elif keep:
            choice = "keep"
        else:
            choice = ask_keep_or_wipe(prior)
            if choice is None:
                progress("Nothing deleted. Nothing downloaded. Pass --keep or --wipe.")
                return 2, plan
        if choice == "wipe":
            removed = wipe_version(prior)
            progress(f"Wiped {len(removed)} LTX {prior} file(s) under MODELS_DIR.")
            plan = assess(version, optional_ids, soundtrack=soundtrack, roots=roots, mbps=mbps)

    def persist() -> tuple[list[str], bool]:
        kept, per_job = _merge_kept(state, version, choice)
        save_state(
            {
                "schema": STATE_SCHEMA,
                "active_version": version,
                "kept_versions": kept,
                "per_job_version": per_job,
                "soundtrack_studio": soundtrack,
                "optional_ids": list(optional_ids),
            }
        )
        return kept, per_job

    if choice is not None:
        kept, per_job = persist()
        if per_job:
            progress(
                f"Both LTX packs stay on disk. Per-job version is on "
                f"(kept: {', '.join(kept)}). Active radio: {version}."
            )

    if not yes:
        progress("\nNothing downloaded. Re-run with --yes after you agree.")
        return 2, plan

    if not plan.disk_ok:
        progress(
            f"FAIL  disk free {format_size(plan.disk_free_bytes)} on {plan.disk_path}; "
            f"need {format_size(plan.download_bytes)}. Nothing downloaded."
        )
        return 1, plan

    if plan.unknown_size_ids:
        progress(
            "These selected rows have no attested size and were not fetched: "
            + ", ".join(plan.unknown_size_ids)
        )

    files = _weight_files_to_fetch(plan)
    if files:
        from master_agent.models.weights import download_files

        download_files(files, yes=True, progress=progress)
    if _soundtrack_needs_fetch(plan):
        from master_agent.heartmula.doctor import download_missing_slots

        download_missing_slots(yes=True, progress=progress)
    if not files and not _soundtrack_needs_fetch(plan):
        progress("OK    nothing to download")

    if choice is None:
        _kept, per_job = persist()
        if per_job:
            progress(
                f"Both LTX packs stay on disk. Per-job version is on. Active radio: {version}."
            )
    return 0, plan


def prompt_optional_toggles(
    version: str,
    optional_ids: Sequence[str],
    *,
    soundtrack: bool,
    roots: Iterable[Path] | None,
    mbps: float,
) -> list[str]:
    """On a TTY, toggle optional ids and reprint the running total. Otherwise no-op."""
    if sys.stdin is None or not sys.stdin.isatty():
        return list(optional_ids)
    known = set(optional_ids_for(version))
    chosen = set(optional_ids)
    while True:
        plan = assess(version, sorted(chosen), soundtrack=soundtrack, roots=roots, mbps=mbps)
        print(render_plan(plan))
        print(
            f"download {format_size(plan.download_bytes)}   "
            f"ETA {format_eta(plan.eta_seconds)}"
        )
        try:
            answer = input(
                "Toggle optional ids (comma-separated), or Enter to continue: "
            ).strip()
        except EOFError:
            return sorted(chosen)
        if not answer:
            return sorted(chosen)
        for piece in answer.split(","):
            item = piece.strip()
            if not item:
                continue
            if item not in known:
                print(f"Unknown optional id {item}")
                continue
            if item in chosen:
                chosen.remove(item)
            else:
                chosen.add(item)


def cmd_models_manifest(_args: Any) -> int:
    print(json.dumps(catalog_document(), indent=2))
    return 0


def cmd_models_select(args: Any) -> int:
    version = getattr(args, "ltx_version", None) or getattr(args, "version", None)
    if not version:
        print("Pass --version 2.3 or --version 2.5.")
        print("python -m master_agent models select --version 2.5")
        print("python -m master_agent models manifest")
        return 2
    raw_optional = getattr(args, "optional", None)
    all_optional = bool(getattr(args, "all_optional", False))
    # download-models --optional is a boolean. With --selector it means every optional row.
    if isinstance(raw_optional, bool):
        all_optional = all_optional or raw_optional
        raw_optional = None
    try:
        optional = normalize_optional(
            version,
            raw_optional,
            all_optional=all_optional,
        )
    except ValueError as exc:
        print(f"FAIL  {exc}")
        return 2
    if not bool(getattr(args, "scan_only", False)) and not bool(getattr(args, "yes", False)):
        optional = prompt_optional_toggles(
            version,
            optional,
            soundtrack=bool(getattr(args, "soundtrack", False)),
            roots=None,
            mbps=float(getattr(args, "mbps", None) or DEFAULT_MBPS),
        )
    try:
        code, plan = apply_selection(
            version,
            optional,
            soundtrack=bool(getattr(args, "soundtrack", False)),
            yes=bool(getattr(args, "yes", False)),
            wipe=bool(getattr(args, "wipe", False)),
            keep=bool(getattr(args, "keep", False)),
            scan_only=bool(getattr(args, "scan_only", False)),
            mbps=float(getattr(args, "mbps", None) or DEFAULT_MBPS),
        )
    except Exception as exc:
        print(f"FAIL  {exc}")
        return 1
    if getattr(args, "json", False):
        print(json.dumps(plan.to_dict(), indent=2))
    return code
