"""Model inventory scanner — indexes models/ into state/model_inventory.json.

Walks the model tree (checkpoints, diffusion_models, loras, vae,
text_encoders), classifies each weight by folder role, and checks the
variant bundles from config.MODEL_FILES (base / eros / directors / lipsync)
so the agent knows what it can actually run before queueing a workflow.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from master_agent.config import (
    COMFYUI_ROOT,
    MODEL_FILES,
    MODEL_INVENTORY_JSON,
    MODEL_OPTIONAL_KEYS,
    MODELS_DIR,
    resolve_model_path,
)

WEIGHT_EXTENSIONS = {".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf"}
PARTIAL_SUFFIXES = (".part", ".tmp", ".download")

# Folder name → logical role
FOLDER_ROLES = {
    "checkpoints": "checkpoint",
    "diffusion_models": "diffusion",
    "unet": "diffusion",
    "loras": "lora",
    "vae": "vae",
    "text_encoders": "text_encoder",
    "clip": "text_encoder",
    "clip_vision": "clip_vision",
    "controlnet": "controlnet",
    "upscale_models": "upscale",
    "embeddings": "embedding",
}


@dataclass
class ModelEntry:
    name: str
    rel_path: str  # relative to the scanned root, e.g. "loras/foo.safetensors"
    folder: str
    role: str
    size_bytes: int
    mtime: str
    root: str  # "project" (MODELS_DIR) or "comfyui" (COMFYUI_ROOT/models)
    partial: bool = False


@dataclass
class Inventory:
    generated_at: str
    models_dir: str
    comfyui_models_dir: str
    entries: list[ModelEntry] = field(default_factory=list)
    bundles: dict[str, dict[str, Any]] = field(default_factory=dict)

    def by_name(self) -> dict[str, ModelEntry]:
        # First occurrence wins; project root is scanned before ComfyUI's tree
        out: dict[str, ModelEntry] = {}
        for e in self.entries:
            out.setdefault(e.name, e)
        return out

    def resolve(self, name: str) -> Optional[ModelEntry]:
        """Look up a model by bare filename (as workflows reference them)."""
        if not name:
            return None
        base = Path(str(name)).name
        return self.by_name().get(base)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "models_dir": self.models_dir,
            "comfyui_models_dir": self.comfyui_models_dir,
            "entries": [asdict(e) for e in self.entries],
            "bundles": self.bundles,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _scan_root(root: Path, root_label: str) -> list[ModelEntry]:
    entries: list[ModelEntry] = []
    if not root.is_dir():
        return entries
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        name = path.name
        lower = name.lower()
        partial = lower.endswith(PARTIAL_SUFFIXES)
        ext = path.suffix.lower()
        if partial:
            ext = Path(path.stem).suffix.lower()
        if ext not in WEIGHT_EXTENSIONS:
            continue
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        folder = rel.parts[0] if len(rel.parts) > 1 else ""
        role = FOLDER_ROLES.get(folder, "other")
        stat = path.stat()
        entries.append(
            ModelEntry(
                name=name,
                rel_path=str(rel).replace("\\", "/"),
                folder=folder,
                role=role,
                size_bytes=stat.st_size,
                mtime=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
                root=root_label,
                partial=partial,
            )
        )
    return entries


def check_bundles(entries: list[ModelEntry]) -> dict[str, dict[str, Any]]:
    """
    For each variant in MODEL_FILES, report which required files exist.
    Missing non-optional keys mark the bundle not runnable.
    """
    names = {e.name for e in entries if not e.partial}
    bundles: dict[str, dict[str, Any]] = {}
    for variant, files in MODEL_FILES.items():
        present: dict[str, str] = {}
        missing: dict[str, str] = {}
        for key, filename in files.items():
            if filename in names:
                present[key] = filename
            else:
                missing[key] = filename
        hard_missing = {
            k: v for k, v in missing.items() if k not in MODEL_OPTIONAL_KEYS
        }
        bundles[variant] = {
            "runnable": not hard_missing,
            "present": present,
            "missing": missing,
            "hard_missing": hard_missing,
        }
    return bundles


def scan_inventory(
    models_dir: Path = MODELS_DIR,
    comfyui_root: Path = COMFYUI_ROOT,
    *,
    write: bool = True,
    out_path: Path = MODEL_INVENTORY_JSON,
) -> Inventory:
    entries = _scan_root(models_dir, "project")
    entries += _scan_root(comfyui_root / "models", "comfyui")
    inv = Inventory(
        generated_at=_utc_now(),
        models_dir=str(models_dir),
        comfyui_models_dir=str(comfyui_root / "models"),
        entries=entries,
        bundles=check_bundles(entries),
    )
    if write:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(inv.to_dict(), indent=1), encoding="utf-8"
        )
    return inv


def load_inventory(path: Path = MODEL_INVENTORY_JSON) -> Inventory:
    """Load a previously written inventory; scans fresh if none exists."""
    if not path.is_file():
        return scan_inventory()
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = [ModelEntry(**e) for e in data.get("entries") or []]
    return Inventory(
        generated_at=str(data.get("generated_at") or ""),
        models_dir=str(data.get("models_dir") or ""),
        comfyui_models_dir=str(data.get("comfyui_models_dir") or ""),
        entries=entries,
        bundles=data.get("bundles") or {},
    )


def format_summary(inv: Inventory) -> str:
    lines: list[str] = []
    lines.append(f"Inventory generated: {inv.generated_at}")
    lines.append(f"Scanned: {inv.models_dir}")
    lines.append(f"         {inv.comfyui_models_dir}")
    by_folder: dict[str, list[ModelEntry]] = {}
    for e in inv.entries:
        by_folder.setdefault(f"{e.root}:{e.folder or '.'}", []).append(e)
    lines.append("")
    lines.append(f"{'folder':<34} {'files':>5} {'size':>10}")
    for folder in sorted(by_folder):
        items = by_folder[folder]
        size_gb = sum(i.size_bytes for i in items) / 1e9
        partial = sum(1 for i in items if i.partial)
        suffix = f" ({partial} partial!)" if partial else ""
        lines.append(f"{folder:<34} {len(items):>5} {size_gb:>8.1f}G{suffix}")
    lines.append("")
    lines.append("Variant bundles (config.MODEL_FILES):")
    for variant, info in inv.bundles.items():
        mark = "OK " if info.get("runnable") else "BAD"
        lines.append(f"  [{mark}] {variant}")
        for key, filename in (info.get("missing") or {}).items():
            opt = " (optional)" if key in MODEL_OPTIONAL_KEYS else ""
            lines.append(f"        missing {key}: {filename}{opt}")
    return "\n".join(lines)
