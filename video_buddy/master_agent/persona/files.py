"""Shared markdown overlay loading for persona and soul files."""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

T = TypeVar("T")


def heading_name(text: str, fallback: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip() or fallback
    return fallback


def overlay_markdown(bundled: Path, user_dir: Path) -> list[Path]:
    """Bundled files overlaid by the same slug in the user directory."""
    by_slug: dict[str, Path] = {}
    for directory in (bundled, user_dir):
        if directory.is_dir():
            for path in sorted(directory.glob("*.md")):
                by_slug[path.stem.lower()] = path
    return list(by_slug.values())


def pick_slug(items: list[T], slug: str, *, kind: str) -> T:
    for item in items:
        if getattr(item, "slug", None) == slug:
            return item
    available = ", ".join(str(getattr(item, "slug", item)) for item in items) or "(none)"
    raise ValueError(f"unknown {kind} {slug!r} — available: {available}")
