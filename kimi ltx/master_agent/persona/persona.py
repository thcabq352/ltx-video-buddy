"""Persona loading — bundled personas + user overrides in state/personas/.

A persona is a markdown file whose first `# Heading` is the display name and
whose body is injected into the intake system prompt. Drop `<name>.md` in
`state/personas/` to add your own or override a bundled one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from master_agent.config import PERSONA, PERSONA_DIR

BUNDLED_DIR = Path(__file__).resolve().parent / "personas"


@dataclass
class Persona:
    slug: str
    name: str
    path: Path
    system_prompt: str

    def to_dict(self) -> dict:
        return {"slug": self.slug, "name": self.name, "path": str(self.path)}


def _persona_name(text: str, fallback: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip() or fallback
    return fallback


def _load_file(path: Path) -> Persona:
    text = path.read_text(encoding="utf-8")
    slug = path.stem.lower()
    return Persona(
        slug=slug,
        name=_persona_name(text, slug.title()),
        path=path,
        system_prompt=text.strip(),
    )


def list_personas() -> list[Persona]:
    """Bundled personas overlaid with user personas (same slug wins)."""
    by_slug: dict[str, Path] = {}
    for d in (BUNDLED_DIR, PERSONA_DIR):
        if d.is_dir():
            for p in sorted(d.glob("*.md")):
                by_slug[p.stem.lower()] = p
    return [_load_file(p) for p in by_slug.values()]


def load_persona(name: Optional[str] = None) -> Persona:
    """Load by slug; default from config.PERSONA (env PERSONA, default ara)."""
    slug = (name or PERSONA or "ara").strip().lower()
    for persona in list_personas():
        if persona.slug == slug:
            return persona
    available = ", ".join(p.slug for p in list_personas()) or "(none)"
    raise ValueError(f"unknown persona {slug!r} — available: {available}")
