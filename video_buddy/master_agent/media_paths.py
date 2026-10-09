"""Which local media files agent-facing tools may read.

An agent caller can name any path; without a fence that is an
arbitrary local file read (and upload into Comfy). Paths must resolve
inside one of:

- ``OUTPUTS_DIR`` (Buddy renders)
- ``STATE_DIR/uploads`` (studio uploads)
- ``COMFYUI_ROOT/input`` and ``COMFYUI_OUTPUT_DIR``
- any directory in ``MEDIA_EXTRA_ROOTS`` (``os.pathsep``-separated)
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import unquote, urlparse


def allowed_media_roots() -> list[Path]:
    import master_agent.config as cfg

    roots = [
        cfg.OUTPUTS_DIR,
        cfg.STATE_DIR / "uploads",
        cfg.COMFYUI_ROOT / "input",
        cfg.COMFYUI_OUTPUT_DIR,
    ]
    extra = os.getenv("MEDIA_EXTRA_ROOTS", "")
    roots.extend(Path(p).expanduser() for p in extra.split(os.pathsep) if p.strip())
    out: list[Path] = []
    for root in roots:
        try:
            out.append(Path(root).resolve())
        except OSError:
            continue
    return out


def to_local_path(raw: str) -> Path:
    """Accept a plain path or a ``file://`` URI."""
    text = str(raw).strip()
    if text.lower().startswith("file://"):
        parsed = urlparse(text)
        path = unquote(parsed.path)
        if os.name == "nt" and len(path) > 2 and path[0] == "/" and path[2] == ":":
            path = path[1:]
        return Path(path)
    return Path(text)


def is_bare_name(raw: str) -> bool:
    """A plain file name (no directory part), e.g. an existing Comfy input name."""
    text = str(raw).strip()
    return bool(text) and "/" not in text and "\\" not in text and not text.lower().startswith("file:")


def is_allowed_media_path(raw: str | os.PathLike) -> bool:
    try:
        path = to_local_path(str(raw)).expanduser().resolve()
    except (OSError, ValueError):
        return False
    for root in allowed_media_roots():
        try:
            path.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def media_path_error(label: str, raw: str | None) -> str | None:
    """Return an error message when *raw* is set and outside the allowed roots."""
    if not raw:
        return None
    if is_allowed_media_path(raw):
        return None
    return (
        f"{label} must be under outputs/, state/uploads/, or the ComfyUI input/output "
        f"folders (add more with MEDIA_EXTRA_ROOTS): {raw}"
    )
