"""Default workflow catalog — convention over config.

Every first-class API graph is listed and selectable with no env flags.
Sources (merged, first id wins for a given file):

1. Explicit ``WORKFLOW_FILES`` (manifest slugs + legacy seeds whose JSON
   files exist on disk — gitignored example graphs are not advertised)
2. ``workflows/manifests.yaml`` entries that point at a file that exists
3. Auto-scan ``workflows/ltx-2.5/*.json`` and ``workflows/minimax-h3/*.json``
4. Auto-scan any other ``*_api.json`` under ``workflows/``

Missing tower nodes/weights never hide an id from the menu. Missing
template files do: a slug is not known until ``(WORKFLOWS_DIR / file)``
is present. Queue-time checks raise a clear ask-to-download error instead.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# Research-agent ids → Buddy default ids (both resolve).
RESEARCH_ALIASES: dict[str, str] = {
    "t2v_i2v": "ltx25_t2v_i2v",
    "t2v_i2v_two_stage": "ltx25_t2v_i2v_two_stage",
    "t2a": "ltx25_t2a",
    "a2v": "ltx25_a2v",
    "v2v_ic_lora": "ltx25_v2v_ic_lora",
    "msr": "ltx25_msr",
    "flf2v": "ltx25_flf2v",
}

# MiniMax H3 mode names → Buddy default ids (no experimental flag).
H3_ALIASES: dict[str, str] = {
    "fl2va": "h3_t2v",
    "h3_fl2va": "h3_t2v",
    "ref2va": "h3_r2v",
    "h3_ref2va": "h3_r2v",
    "h3": "h3_t2v",
    "minimax_h3": "h3_t2v",
    "minimax": "h3_t2v",
}

# Stable ids for the LTX 2.5 API pack (filename stem is not the product id).
LTX25_FILES: dict[str, str] = {
    "ltx25_t2v_i2v": "ltx-2.5/LTX-2.5_T2V_I2V_Single_Stage_Distilled_api.json",
    "ltx25_t2v_i2v_two_stage": "ltx-2.5/LTX-2.5_T2V_I2V_Two_Stage_Distilled_api.json",
    "ltx25_flf2v": "ltx-2.5/LTX-2.5_FLF2V_api.json",
    "ltx25_msr": "ltx-2.5/LTX-2.5_MSR_Multi_Reference_api.json",
    "ltx25_v2v_ic_lora": "ltx-2.5/LTX-2.5_V2V_ICLoRA_Single_Stage_Distilled_api.json",
    "ltx25_a2v": "ltx-2.5/LTX-2.5_A2V_Two_Stage_Distilled_api.json",
    "ltx25_t2a": "ltx-2.5/LTX-2.5_T2A_Single_Stage_Distilled_api.json",
}

LTX25_META: dict[str, dict[str, Any]] = {
    "ltx25_t2v_i2v": {
        "description": "LTX 2.5 single-stage distilled T2V / I2V (default 2.5 path)",
        "modes": ["t2v", "i2v"],
        "weight_bundle": "ltx25_core",
    },
    "ltx25_t2v_i2v_two_stage": {
        "description": "LTX 2.5 two-stage distilled T2V / I2V (latent spatial upscale)",
        "modes": ["t2v", "i2v"],
        "weight_bundle": "ltx25_two_stage",
    },
    "ltx25_flf2v": {
        "description": "LTX 2.5 first-last-frame to video",
        "modes": ["flf2v", "i2v"],
        "weight_bundle": "ltx25_core",
    },
    "ltx25_msr": {
        "description": "LTX 2.5 multi-reference (pic1–pic4 + background)",
        "modes": ["msr", "i2v"],
        "weight_bundle": "ltx25_msr",
    },
    "ltx25_v2v_ic_lora": {
        "description": "LTX 2.5 video-to-video IC-LoRA (single-stage distilled)",
        "modes": ["v2v"],
        "weight_bundle": "ltx25_iclora",
    },
    "ltx25_a2v": {
        "description": "LTX 2.5 audio-to-video two-stage distilled",
        "modes": ["a2v"],
        "weight_bundle": "ltx25_core",
    },
    "ltx25_t2a": {
        "description": "LTX 2.5 text-to-audio single-stage distilled",
        "modes": ["t2a"],
        "weight_bundle": "ltx25_core",
    },
}

H3_FILES: dict[str, str] = {
    "h3_t2v": "minimax-h3/MiniMax-H3_T2V_FL2VA_api.json",
    "h3_i2v": "minimax-h3/MiniMax-H3_I2V_FL2VA_api.json",
    "h3_flf": "minimax-h3/MiniMax-H3_FLF_FL2VA_api.json",
    "h3_r2v": "minimax-h3/MiniMax-H3_R2V_REF2VA_api.json",
}

H3_META: dict[str, dict[str, Any]] = {
    "h3_t2v": {
        "description": "MiniMax H3 fl2va text-to-AV (native stereo, GGUF-first)",
        "modes": ["t2v", "fl2va"],
        "weight_bundle": "h3_fl2va",
    },
    "h3_i2v": {
        "description": "MiniMax H3 fl2va image-to-AV (first-frame)",
        "modes": ["i2v", "fl2va"],
        "weight_bundle": "h3_fl2va",
    },
    "h3_flf": {
        "description": "MiniMax H3 fl2va first-last-frame to AV",
        "modes": ["flf", "fl2va"],
        "weight_bundle": "h3_fl2va",
    },
    "h3_r2v": {
        "description": (
            "MiniMax H3 ref2va reference-to-AV (images / video / audio). "
            "H3 speaks your line in the voice of your 2-12 s sample and animates the mouth to it (coarse sync). "
            "For tight lip-sync to an exact recording, use ltx25_a2v."
        ),
        "modes": ["r2v", "ref2va"],
        "weight_bundle": "h3_ref2va",
    },
}


@dataclass(frozen=True)
class CatalogEntry:
    id: str
    path: str
    name: str
    description: str = ""
    kind: str = "variant"
    family: str = "other"
    default: bool = True
    modes: tuple[str, ...] = ()
    weight_bundle: str = ""
    aliases: tuple[str, ...] = ()
    source: str = "scan"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _workflows_dir() -> Path:
    from master_agent.config import WORKFLOWS_DIR

    return Path(WORKFLOWS_DIR)


def extra_ltx25_workflow_dirs() -> list[Path]:
    """Sibling ``ltx_director/workflows/ltx-2.5`` trees (same filenames as Buddy)."""
    from master_agent.config import PROJECT_ROOT

    candidates = (
        Path(PROJECT_ROOT).parent / "ltx_director" / "workflows" / "ltx-2.5",
        Path(PROJECT_ROOT).parent.parent / "ltx_director" / "workflows" / "ltx-2.5",
        Path.cwd() / "ltx_director" / "workflows" / "ltx-2.5",
        Path.cwd().parent / "ltx_director" / "workflows" / "ltx-2.5",
    )
    out: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.is_dir():
            continue
        seen.add(resolved)
        out.append(resolved)
    return out


def _seed_workflow_files() -> dict[str, str]:
    from master_agent.config import WORKFLOW_FILES

    return dict(WORKFLOW_FILES)


def _is_api_graph(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(data, dict):
        return False
    prompt = data.get("prompt")
    nodes = prompt if isinstance(prompt, dict) else data
    if not isinstance(nodes, dict):
        return False
    return any(
        isinstance(node, dict) and "class_type" in node for node in nodes.values()
    )


def _family_for(variant_id: str, rel: str) -> str:
    key = f"{variant_id} {rel}".lower()
    if "ltx-2.5" in key or key.startswith("ltx25") or "ltx 2.5" in key:
        return "ltx25"
    if "minimax" in key or "h3_" in key or key.startswith("h3") or "fl2va" in key or "ref2va" in key:
        return "h3"
    if "wan" in key:
        return "wan"
    if "flux" in key or "krea" in key:
        return "image"
    if "lipsync" in key or "lip" in key:
        return "lipsync"
    if "ltx" in key:
        return "ltx23"
    return "other"


def _slug_from_filename(rel: str) -> str:
    stem = Path(rel).name
    if stem.endswith("_api.json"):
        stem = stem[: -len("_api.json")]
    elif stem.endswith(".json"):
        stem = stem[: -len(".json")]
    slug = stem.lower().replace(" ", "-")
    slug = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in slug)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-_") or "workflow"


def _load_manifests() -> dict[str, Any]:
    path = _workflows_dir() / "manifests.yaml"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def _add(
    by_id: dict[str, CatalogEntry],
    seen_files: set[str],
    entry: CatalogEntry,
) -> None:
    rel = entry.path.replace("\\", "/")
    if entry.id in by_id:
        return
    by_id[entry.id] = entry
    if rel:
        seen_files.add(rel)


@lru_cache(maxsize=1)
def load_catalog() -> tuple[CatalogEntry, ...]:
    """Build the default catalog. Cached for the process; tests can clear it."""
    root = _workflows_dir()
    by_id: dict[str, CatalogEntry] = {}
    seen_files: set[str] = set()

    # 1) LTX 2.5 first-class pack (always default, even if a file is later missing)
    for vid, rel in LTX25_FILES.items():
        extra = LTX25_META.get(vid) or {}
        aliases = tuple(src for src, dest in RESEARCH_ALIASES.items() if dest == vid)
        _add(
            by_id,
            seen_files,
            CatalogEntry(
                id=vid,
                path=rel,
                name=f"{vid} — {Path(rel).name}",
                description=str(extra.get("description") or "LTX 2.5 API workflow"),
                family="ltx25",
                default=True,
                modes=tuple(extra.get("modes") or ()),
                weight_bundle=str(extra.get("weight_bundle") or "ltx25_core"),
                aliases=aliases,
                source="ltx25",
            ),
        )

    # 1b) MiniMax H3 first-class pack (always default, no experimental flag)
    for vid, rel in H3_FILES.items():
        extra = H3_META.get(vid) or {}
        aliases = tuple(src for src, dest in H3_ALIASES.items() if dest == vid)
        _add(
            by_id,
            seen_files,
            CatalogEntry(
                id=vid,
                path=rel,
                name=f"{vid} — {Path(rel).name}",
                description=str(extra.get("description") or "MiniMax H3 API workflow"),
                family="h3",
                default=True,
                modes=tuple(extra.get("modes") or ()),
                weight_bundle=str(extra.get("weight_bundle") or "h3_fl2va"),
                aliases=aliases,
                source="h3",
            ),
        )

    # 2) Seed WORKFLOW_FILES (on-disk director allowlist + legacy ids)
    manifests = _load_manifests()
    for vid, filename in _seed_workflow_files().items():
        rel = str(filename).replace("\\", "/")
        if vid in by_id:
            continue
        if not (root / rel).is_file():
            continue
        meta = manifests.get(vid) if isinstance(manifests.get(vid), dict) else {}
        _add(
            by_id,
            seen_files,
            CatalogEntry(
                id=vid,
                path=rel,
                name=f"{vid} — {Path(rel).name}",
                description=str((meta or {}).get("description") or f"Built-in variant {vid}"),
                family=_family_for(vid, rel),
                default=True,
                source="seed",
            ),
        )

    # 3) manifests.yaml (covers any slug not already seeded)
    for vid, meta in manifests.items():
        if not isinstance(meta, dict):
            continue
        filename = meta.get("file")
        if not filename:
            continue
        rel = str(filename).replace("\\", "/")
        if vid in by_id:
            continue
        if not (root / rel).is_file():
            continue
        _add(
            by_id,
            seen_files,
            CatalogEntry(
                id=str(vid),
                path=rel,
                name=f"{vid} — {Path(rel).name}",
                description=str(meta.get("description") or ""),
                family=_family_for(str(vid), rel),
                default=True,
                source="manifest",
            ),
        )

    # 4) Auto-scan ltx-2.5/*.json, minimax-h3/*.json, and *_api.json
    if root.is_dir():
        candidates: list[Path] = []
        ltx_dir = root / "ltx-2.5"
        if ltx_dir.is_dir():
            candidates.extend(sorted(ltx_dir.glob("*.json")))
        h3_dir = root / "minimax-h3"
        if h3_dir.is_dir():
            candidates.extend(sorted(h3_dir.glob("*.json")))
        candidates.extend(sorted(root.rglob("*_api.json")))
        for extra in extra_ltx25_workflow_dirs():
            candidates.extend(sorted(extra.glob("*.json")))
        for path in candidates:
            try:
                rel = path.resolve().relative_to(root.resolve()).as_posix()
            except ValueError:
                continue
            if rel in seen_files:
                continue
            if not _is_api_graph(path):
                continue
            slug = _slug_from_filename(rel)
            if slug in by_id:
                slug = rel.replace("/", "-").replace(".json", "")
            _add(
                by_id,
                seen_files,
                CatalogEntry(
                    id=slug,
                    path=rel,
                    name=f"{slug} — {rel}",
                    description="Auto-registered API workflow",
                    family=_family_for(slug, rel),
                    default=True,
                    source="scan",
                ),
            )

    return tuple(by_id.values())


def clear_catalog_cache() -> None:
    load_catalog.cache_clear()


def default_entries() -> list[CatalogEntry]:
    return [e for e in load_catalog() if e.default]


def default_variant_ids() -> list[str]:
    return [e.id for e in default_entries()]


def catalog_by_id() -> dict[str, CatalogEntry]:
    out: dict[str, CatalogEntry] = {}
    for entry in load_catalog():
        out[entry.id] = entry
        for alias in entry.aliases:
            out.setdefault(alias, entry)
    return out


def is_known_variant(variant: str | None) -> bool:
    key = (variant or "").strip()
    if not key:
        return False
    return key in catalog_by_id() or key in RESEARCH_ALIASES or key in H3_ALIASES


def resolve_variant(variant: str) -> CatalogEntry:
    key = (variant or "").strip()
    mapped = H3_ALIASES.get(key, RESEARCH_ALIASES.get(key, key))
    by_id = catalog_by_id()
    if mapped in by_id:
        return by_id[mapped]
    # Path-style ids from the Comfy template picker
    norm = key.replace("\\", "/")
    for entry in load_catalog():
        if entry.path == norm or entry.path.endswith("/" + norm):
            return entry
    raise KeyError(f"unknown workflow variant: {variant}")


def resolve_workflow_path(variant: str) -> Path:
    entry = resolve_variant(variant)
    path = _workflows_dir() / entry.path
    if path.is_file():
        return path
    name = Path(entry.path).name
    for extra in extra_ltx25_workflow_dirs():
        candidate = extra / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Workflow template not found for variant={variant}: {path}"
    )


def workflow_files_map() -> dict[str, str]:
    """id → relative path for every default catalog entry."""
    return {e.id: e.path for e in default_entries()}


def list_catalog_items() -> list[dict[str, str]]:
    """Shape used by CLI / ``GET /api/comfy/templates`` / the studio menus."""
    items: list[dict[str, str]] = []
    seen: set[str] = set()
    root = _workflows_dir()
    for entry in default_entries():
        try:
            from master_agent.models.vram_policy import workflow_row

            vram = workflow_row(entry.id)
            vram_fields = {
                "vram_class": vram.vram_class,
                "vram_peak_gb": vram.expected_vram_gb,
                "default_pack": vram.default_pack,
                "safer_alternate": vram.safer_alternate,
            }
        except Exception:
            vram_fields = {}
        items.append(
            {
                "id": entry.id,
                "path": entry.path,
                "name": entry.name,
                "kind": "variant",
                "family": entry.family,
                "description": entry.description,
                **vram_fields,
            }
        )
        seen.add(entry.path.replace("\\", "/"))
        for alias in entry.aliases:
            seen.add(alias)
    if root.is_dir():
        for path in sorted(root.rglob("*.json")):
            try:
                rel = path.resolve().relative_to(root.resolve()).as_posix()
            except ValueError:
                continue
            if rel in seen:
                continue
            items.append({"id": rel, "path": rel, "name": rel, "kind": "file"})
    return items
