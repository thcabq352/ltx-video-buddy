"""Cheap Quality Bar fail rules (buddy-core ids as plain strings).

Mirrors sibling your-video-buddy ``buddy-core`` quality_bar rule ids. This is
**not** an iteration-controller API — the orchestrator already owns
judge → revise → re-run. These helpers only name fail reasons and build a
structured revise plan (prompt deltas + param deltas).

Rule ids
--------
a  missing_music_bed     — MV / music-video intent with no bed planned or muxed
b  face_similarity       — skipped here (no invented face scores)
c  thin_still_i2v        — still→I2V with no shot plan; attach previs_source
d  unused_control_pack   — control pack present but no channel marked used

Vision rule b is intentionally unported.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

# buddy-core quality_bar fail rule ids (simple strings)
RULE_A = "a"
RULE_B = "b"
RULE_C = "c"
RULE_D = "d"

RULE_CODES = {
    RULE_A: "missing_music_bed",
    RULE_B: "face_similarity",
    RULE_C: "thin_still_i2v",
    RULE_D: "unused_control_pack",
}

# Rule b is the vision/face bar — never invent a score.
SKIPPED_RULES = (RULE_B,)
EVALUATED_RULES = (RULE_A, RULE_C, RULE_D)

MUSIC_KEYWORDS = (
    "music video",
    "music-video",
    "music bed",
    "soundtrack",
    "song",
    "mv for",
)
I2V_HINTS = ("i2v", "image-to-video", "image to video", "still to video", "still→i2v")
SHOT_PLAN_FIELDS = ("camera", "action", "visuals", "continuity")
CONTROL_CHANNELS = ("openpose", "depth", "edges", "camera")
CONTROL_PROMPT_DELTAS = {
    "openpose": "OpenPose skeleton lock from the control pack",
    "depth": "depth-map control from the previs pack",
    "edges": "canny / edge control from the previs pack",
    "camera": "motivated camera move from the previs camera channel",
}


@dataclass
class QualityBarFail:
    id: str
    code: str
    detail: str
    skipped: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RevisePlan:
    """Structured revise plan — prompt deltas + param deltas. No new controller."""

    fail_ids: list[str] = field(default_factory=list)
    prompt_deltas: list[str] = field(default_factory=list)
    param_deltas: dict[str, Any] = field(default_factory=dict)
    shot_patch: dict[str, Any] = field(default_factory=dict)
    control_pack_used: dict[str, bool] = field(default_factory=dict)
    music_bed_attached: bool = False
    reason: str = ""

    @property
    def actionable(self) -> bool:
        return bool(
            self.prompt_deltas
            or self.param_deltas
            or self.shot_patch
            or self.control_pack_used
            or self.music_bed_attached
        )

    def as_prompt_rewrite(self, base_prompt: str) -> str:
        text = (base_prompt or "").rstrip()
        for delta in self.prompt_deltas:
            piece = (delta or "").strip()
            if piece and piece not in text:
                sep = "" if not text or text.endswith((",", ";", ".", "\n")) else ","
                text = f"{text}{sep} {piece}".strip()
        return text

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def music_intent(
    request: str = "",
    *,
    kind: str = "",
    audio_name: Optional[str] = None,
    audio_path: Optional[str] = None,
) -> bool:
    if (kind or "").strip().lower() in {"music_video", "mv", "music"}:
        return True
    # Presence of a source track alone is not MV intent (lipsync / a2v).
    text = f" {(request or '').lower()} "
    if " mv" in text or text.strip() == "mv":
        return True
    return any(k in text for k in MUSIC_KEYWORDS)


def has_music_bed(
    *,
    audio_name: Optional[str] = None,
    audio_path: Optional[str] = None,
    music_bed_attached: bool = False,
    has_audio: bool = False,
) -> bool:
    return bool(audio_name or audio_path or music_bed_attached or has_audio)


def is_still_i2v(
    *,
    image_name: Optional[str] = None,
    variant: Optional[str] = None,
    request: str = "",
) -> bool:
    if image_name:
        return True
    text = (request or "").lower()
    if any(h in text for h in I2V_HINTS):
        return True
    v = (variant or "").lower()
    return "i2v" in v and bool(image_name)


def shot_plan_thin(shot: Optional[dict[str, Any]]) -> bool:
    if not shot or not isinstance(shot, dict):
        return True
    filled = [f for f in SHOT_PLAN_FIELDS if str(shot.get(f) or "").strip()]
    return len(filled) < 2


def unused_control_pack(
    *,
    control_pack_present: bool = False,
    control_pack_used: Optional[dict[str, Any]] = None,
) -> bool:
    if not control_pack_present:
        return False
    used = control_pack_used or {}
    return not (isinstance(used, dict) and any(bool(v) for v in used.values()))


def attach_rule_passes(record: dict[str, Any]) -> dict[str, bool]:
    """Bookkeeping view of rules (c)/(d) used by ``comfy attach`` run JSON.

    (c) previs_source is recorded when a recipe was applied.
    (d) if a control pack was present, at least one channel is marked used.
    """
    previs = str(record.get("previs_source") or "").strip()
    present = bool(record.get("control_pack_present"))
    return {
        RULE_C: bool(previs),
        RULE_D: not unused_control_pack(
            control_pack_present=present,
            control_pack_used=record.get("control_pack_used") or {},
        ),
    }


def evaluate_quality_bar(context: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Return a JSON-friendly quality_bar verdict. No GPU, no live Comfy."""
    ctx = context or {}
    fails: list[QualityBarFail] = []

    if music_intent(
        str(ctx.get("request") or ""),
        kind=str(ctx.get("kind") or ""),
        audio_name=ctx.get("audio_name"),
        audio_path=ctx.get("audio_path"),
    ) and not has_music_bed(
        audio_name=ctx.get("audio_name"),
        audio_path=ctx.get("audio_path"),
        music_bed_attached=bool(ctx.get("music_bed_attached")),
        has_audio=bool(ctx.get("has_audio")),
    ):
        fails.append(
            QualityBarFail(
                id=RULE_A,
                code=RULE_CODES[RULE_A],
                detail="music-video intent with no music bed planned or muxed",
            )
        )

    still = is_still_i2v(
        image_name=ctx.get("image_name"),
        variant=ctx.get("variant"),
        request=str(ctx.get("request") or ""),
    )
    missing_previs = bool(ctx.get("attach_recipe")) and not str(
        ctx.get("previs_source") or ""
    ).strip()
    if still and shot_plan_thin(ctx.get("shot")):
        fails.append(
            QualityBarFail(
                id=RULE_C,
                code=RULE_CODES[RULE_C],
                detail="still→I2V with no shot plan (camera/action/visuals/continuity)",
            )
        )
    elif missing_previs:
        fails.append(
            QualityBarFail(
                id=RULE_C,
                code="missing_previs_source",
                detail="attach recipe applied but previs_source is empty",
            )
        )

    if unused_control_pack(
        control_pack_present=bool(ctx.get("control_pack_present")),
        control_pack_used=ctx.get("control_pack_used"),
    ):
        fails.append(
            QualityBarFail(
                id=RULE_D,
                code=RULE_CODES[RULE_D],
                detail="control pack present but unused (no openpose/depth/edges/camera channel)",
            )
        )

    return {
        "evaluated": list(EVALUATED_RULES),
        "skipped": [
            {"id": RULE_B, "code": RULE_CODES[RULE_B], "reason": "vision face scores not ported"}
        ],
        "fails": [f.to_dict() for f in fails],
        "pass": not fails,
    }


def build_revise_plan(
    fails: list[dict[str, Any]] | list[QualityBarFail],
    context: Optional[dict[str, Any]] = None,
    *,
    base_prompt: str = "",
    steps: Optional[int] = None,
) -> RevisePlan:
    """Turn quality_bar fail reasons into prompt/param deltas the orchestrator applies."""
    ctx = context or {}
    ids: list[str] = []
    for item in fails or []:
        if isinstance(item, QualityBarFail):
            ids.append(item.id)
        elif isinstance(item, dict) and item.get("id"):
            ids.append(str(item["id"]))
        elif isinstance(item, str):
            ids.append(item)
    ids = [i for i in ids if i in EVALUATED_RULES]
    plan = RevisePlan(fail_ids=ids)

    if RULE_A in ids:
        plan.prompt_deltas.append(
            "music bed / soundtrack mixed under the picture, beat-aware bed"
        )
        plan.music_bed_attached = True
        plan.reason = "quality_bar.a: attach a music bed"

    if RULE_C in ids:
        request = str(ctx.get("request") or base_prompt or "the subject")
        plan.shot_patch = {
            "camera": "slow push-in, locked horizon",
            "action": "subject holds, then a clear motivated move",
            "visuals": f"start from the still; animate into: {request[:160]}",
            "continuity": "I2V from still — preserve identity, add planned motion",
        }
        if base_prompt:
            plan.shot_patch["ltx_prompt"] = (
                f"{base_prompt.rstrip()}, slow camera push-in, subject motion, "
                "shot planned from the still"
            )
        plan.prompt_deltas.append(
            "slow camera push-in, planned I2V motion from the still, clear action"
        )
        if steps is not None:
            plan.param_deltas["steps"] = max(int(steps), 16)
        else:
            plan.param_deltas["steps"] = 16
        plan.reason = (plan.reason + "; " if plan.reason else "") + "quality_bar.c: plan the still→I2V shot"

    if RULE_D in ids:
        used = dict(ctx.get("control_pack_used") or {})
        for name in CONTROL_CHANNELS:
            if not used.get(name):
                plan.prompt_deltas.append(CONTROL_PROMPT_DELTAS[name])
                plan.control_pack_used[name] = True
        plan.reason = (plan.reason + "; " if plan.reason else "") + (
            "quality_bar.d: apply unused control pack channels"
        )

    if not plan.reason and ids:
        plan.reason = "quality_bar: " + ",".join(ids)
    return plan


def apply_revise_plan(target: Any, plan: RevisePlan) -> list[str]:
    """Mutate a RunState-like object. Returns applied field names."""
    applied: list[str] = []
    if plan.prompt_deltas:
        new_prompt = plan.as_prompt_rewrite(getattr(target, "prompt", "") or "")
        if new_prompt and new_prompt != getattr(target, "prompt", ""):
            target.prompt = new_prompt
            applied.append("prompt")
    if plan.shot_patch:
        current = dict(getattr(target, "shot", None) or {})
        current.update(plan.shot_patch)
        target.shot = current
        applied.append("shot")
    if plan.control_pack_used:
        current = dict(getattr(target, "control_pack_used", None) or {})
        current.update(plan.control_pack_used)
        target.control_pack_used = current
        applied.append("control_pack_used")
    if plan.music_bed_attached:
        target.music_bed_attached = True
        applied.append("music_bed_attached")
    return applied


def context_from_state(st: Any) -> dict[str, Any]:
    """Build the cheap-rule context from a RunState / namespace."""
    return {
        "request": getattr(st, "request", "") or "",
        "kind": getattr(st, "kind", "") or "",
        "variant": getattr(st, "variant", None),
        "image_name": getattr(st, "image_name", None),
        "audio_name": getattr(st, "audio_name", None),
        "audio_path": getattr(st, "audio_path", None),
        "music_bed_attached": bool(getattr(st, "music_bed_attached", False)),
        "has_audio": bool(getattr(st, "has_audio", False)),
        "shot": getattr(st, "shot", None),
        "attach_recipe": getattr(st, "attach_recipe", None),
        "previs_source": getattr(st, "previs_source", "") or "",
        "control_pack_present": bool(getattr(st, "control_pack_present", False)),
        "control_pack_used": getattr(st, "control_pack_used", None) or {},
        "prompt": getattr(st, "prompt", "") or "",
        "steps": getattr(st, "steps", None),
    }
