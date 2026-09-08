"""Validate a storyboard end to end with zero renders."""

from __future__ import annotations

from typing import Any

from master_agent.control.cost import estimate_cost
from master_agent.school.score import continuity_check, feasibility_check, school_score
from master_agent.storyboard.storyboard import ShotCard


def _shot_dict(shot: ShotCard) -> dict[str, Any]:
    return {
        "desc": f"{shot.camera} {shot.action} {shot.ltx_prompt}",
        "angle": shot.camera,
        "title": shot.title,
        "ltx_prompt": shot.ltx_prompt,
    }


def dry_run_storyboard(
    shots: list[ShotCard],
    *,
    variant: str = "base",
    scene: dict[str, Any] | None = None,
    frames_per_shot: int = 81,
    threshold_vram_gb: float = 14.5,
    budget: Any | None = None,
) -> dict[str, Any]:
    rows = [_shot_dict(s) for s in shots]
    school = school_score(scene=scene, shots=rows)
    cont = continuity_check(rows)
    costs = [
        estimate_cost(variant, frames_per_shot, threshold_vram_gb=threshold_vram_gb) for _ in shots
    ]
    issues: list[str] = []
    for s in school["scores"]:
        if not s["pass"]:
            issues.append(f"school:{s['id']}")
    if not cont["pass"]:
        issues.append("continuity")
    for shot in shots:
        feas = feasibility_check(shot.ltx_prompt)
        if not feas["pass"]:
            issues.append(f"feasibility:{shot.index}")
    for cost in costs:
        if cost["flagged"]:
            issues.append(f"cost:{cost['variant']}")
            break
    from master_agent.control.budget import RenderBudget, get_project_budget

    live = budget or get_project_budget()
    preview = RenderBudget(cap=live.cap, used=live.used, ephemeral=True)
    scenes = [
        {"id": f"dry:{i}", "cost": cost} for i, cost in enumerate(costs)
    ]
    budget = preview.apply_queue(scenes, commit=True)
    if budget["held"]:
        issues.append("budget")
    return {
        "ok": not issues,
        "queued": False,
        "renders": 0,
        "school": school,
        "continuity": cont,
        "costs": costs,
        "budget": budget,
        "issues": issues,
        "shots": len(shots),
    }
