"""Persona loading — bundled personas + user overrides in state/personas/.

A persona is a markdown file whose first `# Heading` is the display name and
whose body is injected into the intake system prompt. Drop `<name>.md` in
`state/personas/` to add your own or override a bundled one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from master_agent.config import PERSONA_DIR
from master_agent.persona.files import heading_name, overlay_markdown, pick_slug

BUNDLED_DIR = Path(__file__).resolve().parent / "personas"


@dataclass
class Persona:
    slug: str
    name: str
    path: Path
    system_prompt: str

    def to_dict(self) -> dict:
        return {"slug": self.slug, "name": self.name, "path": str(self.path)}


def _load_file(path: Path) -> Persona:
    text = path.read_text(encoding="utf-8")
    slug = path.stem.lower()
    return Persona(
        slug=slug,
        name=heading_name(text, slug.title()),
        path=path,
        system_prompt=text.strip(),
    )


def list_personas() -> list[Persona]:
    """Bundled personas overlaid with user personas (same slug wins)."""
    return [_load_file(path) for path in overlay_markdown(BUNDLED_DIR, PERSONA_DIR)]


def load_persona(name: Optional[str] = None) -> Persona:
    """Load by slug; default from the live PERSONA setting (env, then runtime)."""
    import master_agent.config as cfg

    slug = (name or cfg.PERSONA or "ara").strip().lower()
    return pick_slug(list_personas(), slug, kind="persona")


def set_active_persona(slug: str, *, session: str = "cli") -> Persona:
    persona = load_persona(slug)
    from master_agent.control.versioned_config import get_versioned_config

    get_versioned_config().set_values({"persona": persona.slug}, session=session)
    return persona
