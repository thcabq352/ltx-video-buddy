"""Negative-example fixtures: crossed eyelines, no-want, text-on-screen."""

from __future__ import annotations

from typing import Any

from master_agent.school.score import (
    continuity_check,
    feasibility_check,
    load_fixture,
    school_score,
    want_pass,
)

CATALOG = ("no-want", "axis-fail", "text-on-screen")


def negative_example(fid: str) -> dict[str, Any]:
    fx = load_fixture(fid)
    scene = fx.get("scene") or {}
    shots = list(fx.get("shots") or [])
    report = school_score(scene=scene, shots=shots)
    want = want_pass(scene)
    axis = next((s for s in report["scores"] if s["id"] == "axis"), None)
    prompt = shots[0].get("ltx_prompt") or shots[0].get("desc") if shots else ""
    feas = feasibility_check(str(prompt or fx.get("title") or ""))
    cont = continuity_check(shots) if shots else {"pass": True}
    rewrite = fx.get("rewrite") or ""
    if not rewrite:
        if not want:
            rewrite = "Name a concrete want in the action line."
        elif axis and not axis["pass"]:
            rewrite = "Reverse coverage must flip eyeline, or motivate the axis cross."
        elif not feas["pass"]:
            rewrite = feas.get("rewrite") or "Remove unrenderable on-screen type."
    return {
        "id": fx.get("id"),
        "title": fx.get("title"),
        "counter_example": fx.get("counter_example") or (axis or {}).get("counter_example") or "",
        "rewrite": rewrite,
        "want_pass": want,
        "axis_pass": True if axis is None else axis["pass"],
        "feasibility_pass": feas["pass"],
        "continuity_pass": cont["pass"],
        "export_allowed": True,
    }


def all_negatives() -> list[dict[str, Any]]:
    return [negative_example(fid) for fid in CATALOG]
