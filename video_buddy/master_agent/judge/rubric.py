"""Selectable judge rubrics.

Empty / unset keeps ``prompts/judge.md`` and the existing look-vs-health
ladder. ``rainey1`` loads ``prompts/rainey1.md`` and applies caliber gates
on top of that schema. The judge still does not album-lock.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any

RAINEY1 = "rainey1"
KNOWN_RUBRICS = frozenset({RAINEY1})
# Explicit env values that pin the default judge and ignore brief keywords.
RUBRIC_OFF = frozenset({"", "default", "off", "none", "unset", "judge"})

LOOK_DIMS = (
    "identity_lock",
    "density_escalation",
    "emissive_lighting",
    "anti_slop",
    "pacing_hold",
)

# Auto-reject / retry below this look, even with no hard-fail.
LOOK_RETRY_BELOW = 0.55
# Pass candidate when look clears this and nothing hard-failed.
LOOK_PASS_AT = 0.75
# Soft flag: surface for a human. Do not retry forever.
BRIEF_SOFT_BELOW = 0.5
LOOK_SOFT_AT = 0.7

HARD_IDENTITY = "identity_morph"
HARD_JUNK = "filesize_junk"
HARD_FRAMES = "too_few_frames"
HARD_FAIL_CODES = (HARD_IDENTITY, HARD_JUNK, HARD_FRAMES)

_PROMPTS = Path(__file__).resolve().parent / "prompts"
_KEYWORD = re.compile(r"\brainey1\b", re.IGNORECASE)
# Defect language only. A leading "no/not/without" is a negative prompt, not a fail.
_MORPH_RE = re.compile(
    r"\b(identity[- ]morph(?:ing|ed)?|face[- ]morph(?:ing|ed|s)?|"
    r"morphs|morphed|morphing|twin faces?|melted face|"
    r"identity drift|face[- ]split)\b",
    re.IGNORECASE,
)
_NEG_PREFIX = re.compile(
    r"\b(no|not|without|zero|avoid|never|free of)\b[\w\s-]{0,16}$",
    re.IGNORECASE,
)
_RUBRIC_KEYS = ("judge_rubric", "rubric", "panel_judge_rubric")
_TEXT_KEYS = ("title", "action", "visuals", "ltx_prompt", "prompt", "notes", "recipe")

_HARD_HINTS = {
    HARD_IDENTITY: (
        "same locked character, stable face wardrobe and silhouette, "
        "no morph, no twin faces, no melted face"
    ),
    HARD_JUNK: "full clip, not a junk file under 100KB",
    HARD_FRAMES: (
        "readable music-video hold, LTX length 8n+1 minimum 9 frames, "
        "not a 1-frame collapse"
    ),
}


def _env_rubric() -> tuple[str, bool]:
    """Return ``(token, explicitly_set)`` for ``PANEL_JUDGE_RUBRIC``."""
    if "PANEL_JUDGE_RUBRIC" in os.environ:
        return (os.environ.get("PANEL_JUDGE_RUBRIC") or "").strip().lower(), True
    try:
        from master_agent.config import PANEL_JUDGE_RUBRIC
    except Exception:
        return "", False
    return (PANEL_JUDGE_RUBRIC or "").strip().lower(), False


def _token(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    token = value.strip().lower()
    return token if token in KNOWN_RUBRICS else None


def _from_mapping(obj: object) -> str | None:
    if not isinstance(obj, dict):
        return None
    for key in _RUBRIC_KEYS:
        found = _token(obj.get(key))
        if found:
            return found
    recipe = obj.get("recipe")
    found = _token(recipe)
    if found:
        return found
    if isinstance(recipe, dict):
        found = _from_mapping(recipe)
        if found:
            return found
        if recipe.get(RAINEY1) is True:
            return RAINEY1
    if obj.get(RAINEY1) is True:
        return RAINEY1
    attach = obj.get("attach_recipe")
    if isinstance(attach, dict):
        found = _from_mapping(attach)
        if found:
            return found
    flags = obj.get("flags")
    if isinstance(flags, dict) and flags.get(RAINEY1) is True:
        return RAINEY1
    return None


def _keyword_blob(*parts: object) -> str:
    chunks: list[str] = []
    for part in parts:
        if isinstance(part, str):
            chunks.append(part)
        elif isinstance(part, dict):
            for key in _TEXT_KEYS:
                if part.get(key):
                    chunks.append(str(part.get(key)))
            request = part.get("request")
            if isinstance(request, str):
                chunks.append(request)
    return "\n".join(chunks)


def resolve_judge_rubric(
    *,
    user_request: str = "",
    ltx_prompt: str = "",
    shot: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
    rubric: str | None = None,
) -> str | None:
    """Which rubric prompt to load. ``None`` means the default judge.

    Selection, first hit wins:
    1. Explicit ``rubric=`` argument (``""`` / ``default`` forces the default).
    2. Shot or context recipe flag (``judge_rubric``, ``rubric``, ``recipe``,
       ``rainey1: true``, or the same keys on ``attach_recipe``).
    3. Env ``PANEL_JUDGE_RUBRIC=rainey1`` (or the config default of that env).
    4. Brief keyword ``rainey1``, unless the env explicitly pins the default.
    """
    if rubric is not None:
        return _token(rubric)
    flagged = _from_mapping(context) or _from_mapping(shot)
    if flagged:
        return flagged
    token, explicit = _env_rubric()
    if token in KNOWN_RUBRICS:
        return token
    if explicit and token in RUBRIC_OFF:
        return None
    blob = _keyword_blob(user_request, ltx_prompt, shot, context)
    if _KEYWORD.search(blob):
        return RAINEY1
    return None


def prompt_path(rubric: str | None, *, kind: str = "judge") -> Path:
    """File for ``kind`` of ``judge`` or ``vision``. Unknown rubrics use the default."""
    if rubric == RAINEY1:
        name = "rainey1.md" if kind == "judge" else "rainey1_vision.md"
    else:
        name = "vision.md" if kind == "vision" else "judge.md"
    return _PROMPTS / name


def load_rubric_text(rubric: str | None, *, kind: str = "judge") -> str:
    """Read the prompt file. Missing rainey1 falls back to the default file."""
    path = prompt_path(rubric, kind=kind)
    if path.is_file():
        return path.read_text(encoding="utf-8")
    if rubric:
        fallback = prompt_path(None, kind=kind)
        if fallback.is_file():
            return fallback.read_text(encoding="utf-8")
    if kind == "vision":
        return "Review video frames. Return JSON score/pass/issues/reason."
    return (
        "Judge video quality. Return JSON pass/score/issues/"
        "prompt_rewrite/param_hints/reason."
    )


def judge_rubric_debug(
    *,
    user_request: str = "",
    ltx_prompt: str = "",
    shot: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Small status dump: which prompt the next judge call will load."""
    selected = resolve_judge_rubric(
        user_request=user_request,
        ltx_prompt=ltx_prompt,
        shot=shot,
        context=context,
    )
    path = prompt_path(selected, kind="judge")
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16] if text else ""
    return {
        "rubric": selected or "",
        "prompt_file": path.name,
        "loaded": bool(text.strip()),
        "instruction_sha256": digest,
    }


def clamp01(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if number > 1.0 and number <= 100.0:
        number = number / 100.0
    if number < 0.0 or number > 1.0:
        return None
    return number


def extract_rubric_scores(*payloads: dict[str, Any] | None) -> dict[str, float]:
    """Look-dimension scores. Earlier payloads win; missing keys stay absent."""
    scores: dict[str, float] = {}
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        for key in LOOK_DIMS:
            if key in scores or key not in payload or payload.get(key) is None:
                continue
            value = clamp01(payload.get(key))
            if value is not None:
                scores[key] = value
    return scores


def aggregate_look(scores: dict[str, float]) -> float | None:
    """Mean of whichever look dimensions are present. Brief is not included."""
    vals = [scores[key] for key in LOOK_DIMS if key in scores]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 3)


def _as_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "on"}:
            return True
        if token in {"0", "false", "no", "off"}:
            return False
    return None


def _normalize_hard_code(value: object) -> str | None:
    token = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if token in {HARD_IDENTITY, "morph", "face_morph", "identity_drift"}:
        return HARD_IDENTITY
    if token in {HARD_JUNK, "tiny_file", "junk_filesize", "junk", "filesize"}:
        return HARD_JUNK
    if token in {HARD_FRAMES, "lt_3_frames", "few_frames", "under_3_frames"}:
        return HARD_FRAMES
    return None


def _defect(text: str, pattern: re.Pattern[str]) -> bool:
    for match in pattern.finditer(text or ""):
        prefix = text[max(0, match.start() - 24) : match.start()]
        if _NEG_PREFIX.search(prefix):
            continue
        return True
    return False


def collect_hard_fails(
    heuristic_issues: list[dict[str, Any]] | None = None,
    *,
    texts: list[object] | tuple[object, ...] = (),
    structured: dict[str, Any] | None = None,
) -> list[str]:
    """Identity morph, junk filesize, and <3 frames.

    Metadata noise (short duration, container tags) is not a hard-fail.
    Negated phrases such as "no morph" are not a morph.
    """
    found: list[str] = []
    codes = {
        str(item.get("code"))
        for item in (heuristic_issues or [])
        if isinstance(item, dict)
    }
    if "tiny_file" in codes:
        found.append(HARD_JUNK)
    if "too_few_frames" in codes:
        found.append(HARD_FRAMES)
    struct = structured or {}
    if _as_bool(struct.get("identity_morph")) is True:
        found.append(HARD_IDENTITY)
    declared = struct.get("hard_fails")
    if isinstance(declared, str):
        declared = [declared]
    if isinstance(declared, list):
        for item in declared:
            code = _normalize_hard_code(item)
            if code:
                found.append(code)
    blob = "\n".join(str(item or "") for item in texts)
    if blob and _defect(blob, _MORPH_RE):
        found.append(HARD_IDENTITY)
    ordered: list[str] = []
    for code in HARD_FAIL_CODES:
        if code in found and code not in ordered:
            ordered.append(code)
    return ordered


def rewrite_for_gates(
    *,
    hard_fails: list[str],
    look: float,
    ltx_prompt: str = "",
    prompt_rewrite: str = "",
) -> str:
    """Keep a model rewrite. Otherwise attach an actionable retry line."""
    existing = (prompt_rewrite or "").strip()
    if existing:
        return existing
    if not hard_fails and look >= LOOK_RETRY_BELOW:
        return ""
    bits = [_HARD_HINTS[code] for code in hard_fails if code in _HARD_HINTS]
    if look < LOOK_RETRY_BELOW:
        bits.append(
            "raise identity lock, additive density, emissive multi-light, "
            "anti-slop, and a stable music-video hold"
        )
    extra = ", ".join(bits)
    base = (ltx_prompt or "").strip()
    if base and extra:
        return f"{base}, {extra}"
    return extra or base


def apply_rainey1_gates(
    *,
    decision: str,
    look: float,
    brief_adherence: float | None,
    hard_fails: list[str],
    judge_retries: int,
    max_rounds: int,
    prompt_rewrite: str = "",
    ltx_prompt: str = "",
) -> tuple[str, str]:
    """Rainey1 decision overlay. Caller skips this when the rubric is unset.

    Hard-fails and look < 0.55 retry until the round cap. A low brief with
    look >= 0.7 is a human veto (no further automatic retry). Look >= 0.75
    with no hard-fail is a pass candidate. Album lock is never returned.
    """
    rewrite = rewrite_for_gates(
        hard_fails=hard_fails,
        look=look,
        ltx_prompt=ltx_prompt,
        prompt_rewrite=prompt_rewrite,
    )
    retry = bool(hard_fails) or float(look) < LOOK_RETRY_BELOW
    if retry:
        if judge_retries >= max_rounds or decision == "exhausted":
            return "exhausted", rewrite
        if decision in {"rewrite", "retune"}:
            return decision, rewrite
        return "rewrite", rewrite
    if (
        brief_adherence is not None
        and float(brief_adherence) < BRIEF_SOFT_BELOW
        and float(look) >= LOOK_SOFT_AT
    ):
        return "human_veto", prompt_rewrite
    if float(look) >= LOOK_PASS_AT and decision != "human_veto":
        return "accept", prompt_rewrite
    return decision, prompt_rewrite


def _has_phrase(text: str, *phrases: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in phrases)


def _probe_issues_from_description(text: str) -> list[dict[str, Any]]:
    """Fixture-only reading of a mock frame description. Not used on live probes."""
    issues: list[dict[str, Any]] = []
    for match in re.finditer(r"\b(\d+(?:\.\d+)?)\s*(kb|kib|bytes|byte)\b", text, re.I):
        number = float(match.group(1))
        unit = match.group(2).lower()
        nbytes = number * 1000.0 if unit.startswith("k") else number
        if nbytes < 100_000:
            issues.append(
                {
                    "code": "tiny_file",
                    "severity": 1.0,
                    "detail": f"mock description {match.group(0)}",
                }
            )
        break
    if _defect(text, re.compile(r"\b(junk file|tiny file|filesize junk)\b", re.I)):
        if not any(item["code"] == "tiny_file" for item in issues):
            issues.append(
                {"code": "tiny_file", "severity": 1.0, "detail": "mock junk file"}
            )
    counts = [
        int(match.group(1))
        for match in re.finditer(r"\b(\d+)\s*[- ]?frames?\b", text, re.I)
    ]
    if re.search(r"\b(one|a single)[- ]frame\b", text, re.I):
        counts.append(1)
    if re.search(r"<\s*3\s*frames?\b", text, re.I):
        counts.append(1)
    if counts and min(counts) < 3:
        issues.append(
            {
                "code": "too_few_frames",
                "severity": 1.0,
                "detail": f"mock description min frames {min(counts)}",
            }
        )
    return issues


def score_mock_frame_description(description: str) -> dict[str, Any]:
    """Turn a mock frame write-up into rubric scores. Fixture path only.

    Live clips are scored by the LLM/vision prompts. This mapper lets tests
    feed frame descriptions without a model or a GPU.
    """
    text = description or ""
    morph = _defect(text, _MORPH_RE)
    identity = 0.12 if morph else (
        0.92
        if _has_phrase(text, "locked", "same face", "same silhouette", "identity stable")
        else 0.55
    )
    if _defect(text, re.compile(r"\b(sparse melt|random cuts?|empty world)\b", re.I)):
        density = 0.2
    elif _has_phrase(text, "flora", "cables", "densif", "cast", "craft"):
        density = 0.9
    else:
        density = 0.55
    light_hits = sum(
        1
        for word in ("ufo", "rgb", "cyan", "magenta", "mushroom", "beam", "aurora", "monitor")
        if word in text.lower()
    )
    if _defect(text, re.compile(r"\b(flat lighting|single glow|glow soup|glow-soup)\b", re.I)):
        emissive = 0.22
    elif light_hits >= 2 and _has_phrase(text, "deep black"):
        emissive = 0.91
    elif light_hits >= 2:
        emissive = 0.8
    else:
        emissive = 0.55
    slop_hit = _defect(
        text,
        re.compile(
            r"\b(glow soup|glow-soup|rainbow noise|plastic skin|plastic beauty|"
            r"watermark|subtitle|ui overlay|text burn)\b",
            re.I,
        ),
    )
    anti_slop = 0.15 if slop_hit else 0.93
    if _defect(text, re.compile(r"\b(jitter|1-frame trash|slideshow junk)\b", re.I)):
        pacing = 0.18
    elif _has_phrase(text, "music-video hold", "readable motion", "stable hold"):
        pacing = 0.9
    else:
        pacing = 0.6
    if _defect(
        text,
        re.compile(r"\b(off brief|wrong brief|does not match the brief)\b", re.I),
    ):
        brief = 0.25
    else:
        brief = 0.88
    issues = _probe_issues_from_description(text)
    reason_bits = []
    if morph:
        reason_bits.append("identity morph")
    if any(item["code"] == "tiny_file" for item in issues):
        reason_bits.append("filesize junk")
    if any(item["code"] == "too_few_frames" for item in issues):
        reason_bits.append("under 3 frames")
    return {
        "identity_lock": identity,
        "density_escalation": density,
        "emissive_lighting": emissive,
        "anti_slop": anti_slop,
        "pacing_hold": pacing,
        "brief_adherence": brief,
        "identity_morph": morph,
        "heuristic_issues": issues,
        "reason": ", ".join(reason_bits) or "mock frame description",
    }


def verdict_from_frame_description(
    description: str,
    *,
    brief_adherence: float | None = None,
) -> dict[str, Any]:
    """Machine-parseable Rainey1 verdict for one mock frame description."""
    parsed = score_mock_frame_description(description)
    scores = {key: float(parsed[key]) for key in LOOK_DIMS}
    look = aggregate_look(scores) or 0.0
    brief = parsed["brief_adherence"] if brief_adherence is None else float(brief_adherence)
    hard = collect_hard_fails(
        parsed["heuristic_issues"],
        texts=[description],
        structured={"identity_morph": parsed["identity_morph"]},
    )
    decision, rewrite = apply_rainey1_gates(
        decision="accept",
        look=look,
        brief_adherence=brief,
        hard_fails=hard,
        judge_retries=0,
        max_rounds=3,
        prompt_rewrite="",
        ltx_prompt="locked character plate",
    )
    passed = decision == "accept" and not hard and look >= LOOK_PASS_AT
    return {
        "rubric": RAINEY1,
        "pass": passed,
        "score": look,
        "look_score": look,
        "brief_adherence": brief,
        "rubric_scores": scores,
        "hard_fails": hard,
        "decision": decision,
        "human_veto": decision == "human_veto",
        "HUMAN_VETO": decision == "human_veto",
        "album_lock": False,
        "issues": [f"hard_fail.{code}" for code in hard],
        "prompt_rewrite": rewrite,
        "param_hints": {},
        "reason": parsed["reason"],
    }
