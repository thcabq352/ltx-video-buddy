"""Deterministic school scoring. Ported from lot/crates/lot-core/src/school.rs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PACK_DIR = Path(__file__).resolve().parent
RUBRICS_PATH = PACK_DIR / "rubrics.json"
FIXTURES_DIR = PACK_DIR / "fixtures"

UNRENDERABLE = (
    "8pt",
    "readable text",
    "on-screen text",
    "tiny text",
    "legible caption",
    "buy now",
    "logo lockup",
    "infinite resolution",
    "change physics",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rubrics() -> list[dict[str, Any]]:
    data = _load_json(RUBRICS_PATH)
    return list(data.get("items") or [])


def load_fixture(fid: str) -> dict[str, Any]:
    path = FIXTURES_DIR / f"{fid}.json"
    if not path.is_file():
        raise FileNotFoundError(f"no fixture — {fid}")
    return _load_json(path)


def want_pass(scene: dict[str, Any] | None) -> bool:
    if not scene:
        return False
    blob = f"{scene.get('synopsis') or ''} {scene.get('text') or ''} {' '.join(scene.get('characters') or [])}".lower()
    if "nobody wants" in blob or "no one wants" in blob or "no want" in blob:
        return False
    markers = ("want", "need", "must", "gotta", "have to", "will not", "won't", "don't")
    return any(m in blob for m in markers) or bool(scene.get("characters"))


def _screen_dir(text: str) -> str | None:
    t = (text or "").lower()
    if "camera left" in t or "screen left" in t:
        return "left"
    if "camera right" in t or "screen right" in t:
        return "right"
    return None


def _shot_blob(shot: dict[str, Any]) -> str:
    return f"{shot.get('desc') or ''} {shot.get('angle') or ''} {shot.get('title') or ''} {shot.get('ltx_prompt') or ''}"


def axis_pass(shots: list[dict[str, Any]]) -> bool:
    if any(
        "crossed axis" in _shot_blob(s).lower() or "crosses the axis" in _shot_blob(s).lower()
        for s in shots
    ):
        return False
    prev_dir: str | None = None
    prev_was_setup = False
    for s in shots:
        desc = str(s.get("desc") or "")
        angle = str(s.get("angle") or "")
        direction = _screen_dir(desc) or _screen_dir(angle)
        reverse = "reverse" in angle.lower()
        if reverse and prev_was_setup and prev_dir and direction and prev_dir == direction:
            return False
        if direction:
            prev_dir = direction
            prev_was_setup = not reverse
    return True


def continuity_check(shots: list[dict[str, Any]]) -> dict[str, Any]:
    """Adjacent-shot eyeline / screen-direction continuity."""
    ok = axis_pass(shots)
    return {
        "pass": ok,
        "id": "continuity",
        "reason": "" if ok else "adjacent reverse coverage repeats the same eyeline",
    }


def feasibility_check(prompt: str) -> dict[str, Any]:
    blob = (prompt or "").lower()
    hit = next((m for m in UNRENDERABLE if m in blob), None)
    rewrite = ""
    if hit:
        rewrite = "Drop on-screen type and logos; show the want through action and light."
    return {"pass": hit is None, "id": "feasibility", "reason": hit or "", "rewrite": rewrite}


def score_world(scene: dict[str, Any] | None, shots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in rubrics():
        if r["id"] == "want-vs-need":
            passed = want_pass(scene)
        elif r["id"] == "axis":
            passed = axis_pass(shots)
        else:
            passed = True
        out.append(
            {
                "id": r["id"],
                "pass": passed,
                "rule": r["rule"],
                "counter_example": r["counter_example"],
                "apply": r["apply"],
                "cite": r["cite"],
            }
        )
    return out


def school_score(
    *,
    fixture: str | None = None,
    scene: dict[str, Any] | None = None,
    shots: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if fixture:
        fx = load_fixture(fixture)
        scores = score_world(fx.get("scene"), list(fx.get("shots") or []))
        return {
            "ok": True,
            "fixture": fx.get("id"),
            "scene": (fx.get("scene") or {}).get("id"),
            "scores": scores,
        }
    return {
        "ok": True,
        "fixture": None,
        "scene": (scene or {}).get("id"),
        "scores": score_world(scene, shots or []),
    }


def school_exam(*, fixture: str | None = None, **kwargs) -> dict[str, Any]:
    score = school_score(fixture=fixture, **kwargs)
    scores = score["scores"]
    passed = bool(scores) and all(s["pass"] for s in scores)
    return {
        "ok": True,
        "passed": passed,
        "fixture": score.get("fixture"),
        "scores": scores,
    }


def export_allowed(_report: dict[str, Any] | None = None) -> bool:
    """School never blocks export."""
    return True


def tutor_fields(show: dict[str, Any]) -> dict[str, Any]:
    school = show.get("school") or {}
    if not school.get("enabled"):
        return {}
    amount = school.get("help") or "nudge"
    if amount == "mute":
        return {}
    types = school.get("help_types") or ["theory"]
    phase = show.get("phase") or ""
    rid = "axis" if phase in ("picture", "board", "slate", "dailies", "stage", "motion") else "want-vs-need"
    rubric = next((r for r in rubrics() if r["id"] == rid), None)
    if not rubric:
        return {}
    scenes = show.get("scenes") or []
    sc = scenes[0] if scenes else {}
    slug = sc.get("slug") or sc.get("num") or ""
    apply = f"On scene {sc.get('num') or ''} {slug}: {rubric['apply']}".strip()
    out: dict[str, Any] = {}
    if "theory" in types:
        beat = {"id": rubric["id"], "rule": rubric["rule"], "apply": apply}
        if amount in ("coach", "walkthrough"):
            beat["counter_example"] = rubric["counter_example"]
            beat["next"] = rubric["apply"]
        if amount == "walkthrough":
            beat["steps"] = [
                f"Rule: {rubric['rule']}",
                apply,
                "Skip anytime — School never blocks a production tool.",
            ]
        out["theory"] = beat
    elif "craft" in types:
        out["craft"] = {"line": apply, "id": rubric["id"]}
    if "quiz" in types and amount != "nudge":
        out["quiz"] = {"id": rubric["id"], "prompt": f"Does this scene hold {rubric['id']}?"}
    return out
