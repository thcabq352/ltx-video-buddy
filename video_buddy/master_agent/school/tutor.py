"""Fold school tutor notes into storyboard cards during generation."""

from __future__ import annotations

from typing import Any

from master_agent.school.score import feasibility_check, tutor_fields, want_pass
from master_agent.storyboard.storyboard import ShotCard


def tutor_storyboard(
    shots: list[ShotCard],
    *,
    scene: dict[str, Any] | None = None,
    school: dict[str, Any] | None = None,
) -> list[ShotCard]:
    show = {
        "school": school or {"enabled": True, "help": "nudge", "help_types": ["theory"]},
        "phase": "writer",
        "scenes": [{"num": "1", "slug": (scene or {}).get("slug") or "scene"}],
        "shots": [],
    }
    notes = tutor_fields(show)
    apply = ((notes.get("theory") or {}).get("apply") or (notes.get("craft") or {}).get("line") or "")
    missing_want = scene is not None and not want_pass(scene)
    out: list[ShotCard] = []
    for shot in shots:
        card = ShotCard.from_dict(shot.to_dict())
        feas = feasibility_check(card.ltx_prompt)
        bits = [card.continuity] if card.continuity else []
        if apply:
            bits.append(apply)
        if missing_want:
            bits.append("Name the want on this scene.")
            if "want" not in card.ltx_prompt.lower():
                card.ltx_prompt = (card.ltx_prompt + " A character wants something concrete.").strip()
        if not feas["pass"] and feas.get("rewrite"):
            bits.append(feas["rewrite"])
        card.continuity = " ".join(bits).strip()
        out.append(card)
    return out
