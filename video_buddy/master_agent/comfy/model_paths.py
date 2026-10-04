"""Buddy-owned ComfyUI ``extra_model_paths.yaml``.

This is not ``comfy attach`` (WorkflowPatchPlan / ``buddy.comfy.attach/v1``).
External mode points at a user-owned Comfy install (``comfy_mode=external``,
``EXTERNAL_COMFY_ROOT``). The YAML lives under Buddy state and is passed with
``--extra-model-paths-config``. Sections reference model directories as
read-only sources. ``is_default`` is set only on Buddy ``MODELS_DIR``.

Buddy does not delete or write weights inside an attached tree. Copying this
YAML into that tree requires ``--write-yaml-into-external``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from master_agent import config as cfg

YAML_FILENAME = "extra_model_paths.yaml"

# Folder keys ComfyUI reads. Values are relative to ``base_path`` (a models dir).
_FOLDERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("checkpoints", ("checkpoints",)),
    ("configs", ("configs",)),
    ("loras", ("loras",)),
    ("vae", ("vae",)),
    ("text_encoders", ("text_encoders", "clip")),
    ("diffusion_models", ("diffusion_models", "unet")),
    ("clip_vision", ("clip_vision",)),
    ("embeddings", ("embeddings",)),
    ("controlnet", ("controlnet",)),
    ("upscale_models", ("upscale_models",)),
    ("latent_upscale_models", ("latent_upscale_models",)),
    ("model_patches", ("model_patches",)),
    ("audio_encoders", ("audio_encoders",)),
    ("hypernetworks", ("hypernetworks",)),
    ("style_models", ("style_models",)),
    ("gligen", ("gligen",)),
    ("photomaker", ("photomaker",)),
    ("diffusers", ("diffusers",)),
    ("vae_approx", ("vae_approx",)),
)

__all__ = [
    "ModelPathSource",
    "ModelPathsError",
    "YAML_FILENAME",
    "attached_install_root",
    "buddy_yaml_path",
    "default_sources",
    "external_comfy_root",
    "managed_workspace",
    "path_is_comfy_like",
    "read_only_weight_roots",
    "render_extra_model_paths_yaml",
    "user_model_directories",
    "write_external_yaml_copy",
    "write_extra_model_paths",
]


class ModelPathsError(RuntimeError):
    """Buddy refused to write or place ``extra_model_paths.yaml``."""


@dataclass(frozen=True)
class ModelPathSource:
    """One YAML section. ``is_default`` is reserved for Buddy ``MODELS_DIR``."""

    key: str
    path: Path
    kind: str  # buddy | user
    is_default: bool


def _resolve(path: Path | str) -> Path:
    return Path(path).expanduser().resolve()


def _is_under(path: Path, parent: Path) -> bool:
    try:
        _resolve(path).relative_to(_resolve(parent))
    except (ValueError, OSError):
        return False
    return True


def managed_workspace() -> Path:
    """Buddy-owned Comfy workspace (``MANAGED_COMFY_ROOT`` or ``PROJECT_ROOT/ComfyUI``)."""
    raw = (os.getenv("MANAGED_COMFY_ROOT") or "").strip()
    if raw:
        return _resolve(raw)
    return _resolve(Path(cfg.PROJECT_ROOT) / "ComfyUI")


def external_comfy_root() -> Path | None:
    """Explicit attached install (``EXTERNAL_COMFY_ROOT``). Not the managed workspace."""
    raw = (os.getenv("EXTERNAL_COMFY_ROOT") or "").strip()
    if not raw:
        return None
    return _resolve(raw)


def buddy_yaml_path(directory: Path | None = None) -> Path:
    """Default Buddy-owned YAML path under state/ (or ``directory``)."""
    base = _resolve(directory) if directory is not None else _resolve(cfg.STATE_DIR)
    return base / YAML_FILENAME


def _dedupe(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        try:
            resolved = _resolve(path)
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(resolved)
    return out


def _as_models_dir(path: Path) -> Path:
    """Point a Comfy install root at its ``models`` folder. Leave models dirs as-is."""
    resolved = _resolve(path)
    if resolved.name.lower() == "models":
        return resolved
    if resolved.name.lower() in {"comfyui", "comfyui_windows_portable"}:
        return resolved / "models"
    if (resolved / "main.py").is_file() or (resolved / "comfy").is_dir():
        return resolved / "models"
    nested = resolved / "models"
    if nested.is_dir() and not (
        (resolved / "checkpoints").is_dir() or (resolved / "diffusion_models").is_dir()
    ):
        return nested
    return resolved


def user_model_directories() -> list[Path]:
    """Read-only model directories the YAML should reference.

    Skips Buddy ``MODELS_DIR`` (its own section) and the managed workspace
    (Comfy already scans that tree).
    """
    models = _resolve(cfg.MODELS_DIR)
    managed = managed_workspace()
    candidates: list[Path] = []
    for extra in cfg.extra_models_dirs():
        candidates.append(_as_models_dir(extra))
    ext = external_comfy_root()
    if ext is not None:
        candidates.append(_as_models_dir(ext))
    candidates.append(_as_models_dir(Path(cfg.COMFYUI_ROOT)))
    candidates.append(_as_models_dir(Path(cfg.PORTABLE_ROOT) / "ComfyUI"))
    out: list[Path] = []
    for path in _dedupe(candidates):
        if path == models or _is_under(path, models):
            continue
        if path == managed or _is_under(path, managed):
            continue
        out.append(path)
    return out


def attached_install_root() -> Path | None:
    """User-owned Comfy root. Never the managed workspace.

    ``EXTERNAL_COMFY_ROOT`` wins. Otherwise ``COMFYUI_ROOT`` when that tree
    is not the managed install.
    """
    managed = managed_workspace()
    ext = external_comfy_root()
    if ext is not None:
        if ext == managed or _is_under(ext, managed):
            return None
        return ext
    comfy = _resolve(cfg.COMFYUI_ROOT)
    if comfy == managed or _is_under(comfy, managed) or _is_under(managed, comfy):
        return None
    return comfy


def external_yaml_destination() -> Path:
    """The single consented path inside an attached install."""
    root = attached_install_root()
    if root is None:
        raise ModelPathsError(
            "No attached Comfy root. Set EXTERNAL_COMFY_ROOT (or COMFYUI_ROOT) "
            "to a tree that is not the managed workspace. "
            "Buddy will not create an attached install."
        )
    return root / YAML_FILENAME


def default_sources() -> list[ModelPathSource]:
    models = _resolve(cfg.MODELS_DIR)
    sources = [ModelPathSource("buddy_models", models, "buddy", True)]
    for index, path in enumerate(user_model_directories(), start=1):
        sources.append(ModelPathSource(f"user_models_{index}", path, "user", False))
    return sources


def _yaml_quote(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def render_extra_model_paths_yaml(sources: list[ModelPathSource] | None = None) -> str:
    """Render ComfyUI extra model paths. User sections omit ``is_default``."""
    chosen = list(sources) if sources is not None else default_sources()
    if not any(source.kind == "buddy" and source.is_default for source in chosen):
        raise ModelPathsError("YAML must mark Buddy MODELS_DIR as the only is_default section")
    lines = [
        "# Video Buddy extra_model_paths.yaml",
        "# Buddy-owned. Not comfy attach / WorkflowPatchPlan.",
        "# Pass to ComfyUI with --extra-model-paths-config.",
        "# User sections are read-only (no is_default).",
        "# Pack downloads write only under buddy_models (MODELS_DIR).",
        "# Buddy does not delete or write weights inside attached Comfy trees.",
        "",
    ]
    for source in chosen:
        if source.is_default and source.kind != "buddy":
            raise ModelPathsError("is_default is only allowed on Buddy MODELS_DIR")
        lines.append(f"{source.key}:")
        lines.append(f"  base_path: {_yaml_quote(_resolve(source.path).as_posix())}")
        if source.is_default:
            lines.append("  is_default: true")
        for key, folders in _FOLDERS:
            if len(folders) == 1:
                lines.append(f"  {key}: {folders[0]}")
            else:
                lines.append(f"  {key}: |-")
                for folder in folders:
                    lines.append(f"    {folder}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _state_roots(directory: Path | None) -> list[Path]:
    roots = [_resolve(cfg.STATE_DIR)]
    if directory is not None:
        roots.append(_resolve(directory))
    return _dedupe(roots)


def _write_text(dest: Path, text: str, *, allow_external_write: bool, directory: Path | None) -> Path:
    dest = _resolve(dest)
    under_state = any(_is_under(dest, root) for root in _state_roots(directory))
    if not under_state:
        if not allow_external_write:
            raise ModelPathsError(
                f"Refusing to write {dest.name} to {dest}. "
                "The default file lives under Buddy state/ and is passed with "
                "--extra-model-paths-config. Writing into an attached or external "
                "Comfy tree requires --write-yaml-into-external."
            )
        allowed = _resolve(external_yaml_destination())
        if dest != allowed:
            raise ModelPathsError(
                f"Refusing to write {dest}. Consented writes go only to {allowed}."
            )
        if not dest.parent.is_dir():
            raise ModelPathsError(
                f"Attached Comfy root does not exist: {dest.parent}. "
                "Buddy will not create an attached install."
            )
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(dest)
    return dest


def write_extra_model_paths(
    dest: Path | None = None,
    *,
    directory: Path | None = None,
    allow_external_write: bool = False,
    sources: list[ModelPathSource] | None = None,
) -> Path:
    """Write the Buddy YAML. Default destination is state/extra_model_paths.yaml."""
    target = Path(dest) if dest is not None else buddy_yaml_path(directory)
    text = render_extra_model_paths_yaml(sources)
    return _write_text(
        target,
        text,
        allow_external_write=allow_external_write,
        directory=directory,
    )


def write_external_yaml_copy(source: Path | None = None) -> Path:
    """Copy YAML into the attached install. Caller must have passed the consent flag.

    Does not create the install, delete files, or write weights.
    """
    dest = external_yaml_destination()
    if source is None:
        text = render_extra_model_paths_yaml()
    else:
        text = Path(source).read_text(encoding="utf-8")
    return _write_text(dest, text, allow_external_write=True, directory=None)


def _comfy_like(path: Path) -> bool:
    name = path.name.lower()
    if name in {"models", "comfyui", "comfyui_windows_portable"}:
        return True
    try:
        has_models = (path / "models").is_dir()
    except OSError:
        has_models = False
    if not has_models:
        return False
    return (path / "main.py").is_file() or (path / "comfy").is_dir()


def read_only_weight_roots() -> list[Path]:
    """Trees pack downloads must not write into (attached, external, managed Comfy, extras)."""
    roots: list[Path] = list(user_model_directories())
    roots.append(managed_workspace())
    roots.append(managed_workspace() / "models")
    ext = external_comfy_root()
    if ext is not None:
        roots.append(ext)
        if ext.name.lower() != "models":
            roots.append(ext / "models")
    roots.append(Path(cfg.COMFYUI_ROOT))
    roots.append(Path(cfg.COMFYUI_ROOT) / "models")
    roots.append(Path(cfg.PORTABLE_ROOT))
    roots.append(Path(cfg.PORTABLE_ROOT) / "ComfyUI" / "models")
    roots.extend(cfg.extra_models_dirs())
    attached = attached_install_root()
    if attached is not None:
        roots.append(attached)
        roots.append(_as_models_dir(attached))
    return _dedupe(roots)


def path_is_comfy_like(path: Path) -> bool:
    """True when ``path`` looks like a Comfy install or models folder."""
    try:
        return _comfy_like(_resolve(path))
    except OSError:
        return False
