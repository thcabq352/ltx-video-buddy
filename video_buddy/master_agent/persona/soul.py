"""Soul loading — bundled souls + user overrides in state/souls/.

A soul is the studio's standing values. Persona is the interview voice;
soul stays put when the voice changes. Drop `<name>.md` in `state/souls/`
to add your own or override a bundled one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from master_agent.config import SOUL_DIR
from master_agent.persona.files import heading_name, overlay_markdown, pick_slug

BUNDLED_DIR = Path(__file__).resolve().parent / "souls"


@dataclass
class Soul:
    slug: str
    name: str
    path: Path
    system_prompt: str

    def to_dict(self) -> dict:
        return {"slug": self.slug, "name": self.name, "path": str(self.path)}


def _load_file(path: Path) -> Soul:
    text = path.read_text(encoding="utf-8")
    slug = path.stem.lower()
    return Soul(
        slug=slug,
        name=heading_name(text, slug.title()),
        path=path,
        system_prompt=text.strip(),
    )


def list_souls() -> list[Soul]:
    return [_load_file(path) for path in overlay_markdown(BUNDLED_DIR, SOUL_DIR)]


def load_soul(name: Optional[str] = None) -> Soul:
    import master_agent.config as cfg

    slug = (name or cfg.SOUL or "studio").strip().lower()
    return pick_slug(list_souls(), slug, kind="soul")


def set_active_soul(slug: str, *, session: str = "cli") -> Soul:
    soul = load_soul(slug)
    from master_agent.control.versioned_config import get_versioned_config

    get_versioned_config().set_values({"soul": soul.slug}, session=session)
    return soul
