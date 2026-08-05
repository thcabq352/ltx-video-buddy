"""Storyboard planner for LTX Agent v2 — structured shot cards."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class ShotCard:
    index: int
    title: str
    duration_s: float
    camera: str = ""
    action: str = ""
    visuals: str = ""
    continuity: str = ""
    audio_mood: str = ""
    ltx_prompt: str = ""
    negative_extras: str = ""
    seed_offset: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any], *, index: int | None = None) -> "ShotCard":
        i = int(d.get("index") if d.get("index") is not None else (index or 0))
        return cls(
            index=i,
            title=str(d.get("title") or f"Shot {i + 1}"),
            duration_s=float(d.get("duration_s") or 5.0),
            camera=str(d.get("camera") or ""),
            action=str(d.get("action") or ""),
            visuals=str(d.get("visuals") or ""),
            continuity=str(d.get("continuity") or ""),
            audio_mood=str(d.get("audio_mood") or ""),
            ltx_prompt=str(d.get("ltx_prompt") or ""),
            negative_extras=str(d.get("negative_extras") or ""),
            seed_offset=int(d.get("seed_offset") if d.get("seed_offset") is not None else i * 17),
        )


def should_storyboard(
    mode: str,
    *,
    segment_count: int,
    user_request: str,
    variant: str | None = None,
) -> bool:
    m = (mode or "smart").strip().lower()
    if m in ("off", "0", "false", "no"):
        return False
    if m == "always":
        return True
    if m == "multi_only":
        return segment_count >= 2
    # smart
    if segment_count >= 2:
        return True
    if (variant or "").lower() in ("directors",):
        return True
    req = (user_request or "").lower()
    if re.search(
        r"\b(brand|ad|commercial|promo|product|storyboard|story|narrative|multi[- ]?shot|scotty)\b",
        req,
    ):
        return True
    return True  # single-shot board for pipeline uniformity


def build_storyboard(
    user_request: str,
    segment_durations: list[float],
    *,
    variant: str | None = None,
    quality: str | None = None,
    memory_context: str = "",
    rag_context: str = "",
    global_style: str = "",
) -> tuple[list[ShotCard], str]:
    """Return (shots, global_style). Uses LLM when available, else heuristic."""
    segs = [float(s) for s in (segment_durations or [5.0])]
    if not segs:
        segs = [5.0]

    llm_shots = _llm_storyboard(
        user_request,
        segs,
        variant=variant,
        quality=quality,
        memory_context=memory_context,
        rag_context=rag_context,
    )
    if llm_shots:
        shots, style = llm_shots
        return _align_to_durations(shots, segs), style or global_style

    shots = _heuristic_storyboard(user_request, segs, variant=variant)
    return shots, global_style or _default_style(user_request)


def storyboard_to_markdown(shots: list[ShotCard], *, global_style: str = "") -> str:
    lines = ["### Storyboard"]
    if global_style:
        lines.append(f"**Style:** {global_style}")
    for s in shots:
        lines.append(
            f"**{s.index + 1}. {s.title}** ({s.duration_s:.1f}s) — "
            f"_{s.camera or 'camera n/a'}_"
        )
        if s.action:
            lines.append(f"- action: {s.action}")
        if s.continuity:
            lines.append(f"- continuity: {s.continuity}")
        if s.ltx_prompt:
            lines.append(f"- prompt: {s.ltx_prompt[:220]}{'…' if len(s.ltx_prompt) > 220 else ''}")
    return "\n".join(lines)


def _default_style(request: str) -> str:
    if re.search(r"\b(brand|ad|commercial)\b", request or "", re.I):
        return "premium commercial, cool teal and warm gold, clean reflections"
    return "cinematic, smooth temporal motion, shallow depth of field"


def _heuristic_storyboard(
    user_request: str,
    segs: list[float],
    *,
    variant: str | None = None,
) -> list[ShotCard]:
    cleaned = (user_request or "").strip()
    n = len(segs)
    titles = _default_titles(n, cleaned)
    shots: list[ShotCard] = []
    for i, dur in enumerate(segs):
        cont = "opening shot" if i == 0 else f"continues from previous beat ({titles[i - 1]})"
        part = (
            f"{titles[i]}. {cleaned}. {cont}. "
            f"Camera: cinematic, smooth temporal motion. No text, no watermark."
        )
        if n > 1:
            part = f"[Shot {i + 1}/{n}] {part}"
        shots.append(
            ShotCard(
                index=i,
                title=titles[i],
                duration_s=dur,
                camera="slow cinematic move, shallow depth of field",
                action=cleaned[:200],
                visuals=_default_style(cleaned),
                continuity=cont,
                ltx_prompt=part,
                seed_offset=i * 17,
            )
        )
    return shots


def _default_titles(n: int, request: str) -> list[str]:
    if n == 1:
        return ["Hero beat"]
    if n == 2:
        return ["Hook", "Payoff"]
    if n == 3:
        return ["Hook", "Craft", "Payoff"]
    if n <= 5:
        base = ["Hook", "Establish", "Detail", "Hero", "Payoff"]
        return base[:n]
    base = ["Hook", "Establish", "Craft", "Detail", "Hero", "Resolve"]
    while len(base) < n:
        base.append(f"Beat {len(base) + 1}")
    return base[:n]


def _align_to_durations(shots: list[ShotCard], segs: list[float]) -> list[ShotCard]:
    out: list[ShotCard] = []
    for i, dur in enumerate(segs):
        if i < len(shots):
            s = shots[i]
            s.index = i
            s.duration_s = dur
            s.seed_offset = i * 17
            if not s.ltx_prompt:
                s.ltx_prompt = f"{s.title}. {s.action}. {s.visuals}. {s.camera}. No text."
            out.append(s)
        else:
            out.append(
                ShotCard(
                    index=i,
                    title=f"Shot {i + 1}",
                    duration_s=dur,
                    ltx_prompt=f"Continuous sequence shot {i + 1}. Smooth motion. No text.",
                    seed_offset=i * 17,
                )
            )
    return out


def _storyboard_prompt(
    user_request: str,
    segs: list[float],
    *,
    variant: str | None,
    quality: str | None,
    memory_context: str,
    rag_context: str,
) -> tuple[str, str]:
    """Shared (system, user) prompt for single-LLM and panel storyboard calls."""
    path = Path(__file__).resolve().parent / "prompts" / "storyboard.md"
    system = path.read_text(encoding="utf-8") if path.is_file() else (
        "Plan LTX shot cards. Return JSON with shots array."
    )
    if memory_context:
        system += "\n\n" + memory_context[:2500]
    if rag_context:
        system += "\n\n" + rag_context[:3000]

    payload = {
        "user_request": user_request,
        "segment_durations": segs,
        "segment_count": len(segs),
        "variant": variant,
        "quality": quality,
    }
    user = "Plan the storyboard JSON for this request:\n" + json.dumps(payload, indent=2)
    return system, user


def _parse_storyboard(text: str) -> Optional[tuple[list[ShotCard], str, dict]]:
    """Parse an LLM reply into (shots, style, raw_data); None if invalid."""
    data = _extract_json(text)
    if not data or not isinstance(data.get("shots"), list) or not data["shots"]:
        return None
    shots = [
        ShotCard.from_dict(s if isinstance(s, dict) else {}, index=i)
        for i, s in enumerate(data["shots"])
    ]
    style = str(data.get("global_style") or "")
    return shots, style, data


def build_storyboard_panel(
    user_request: str,
    segment_durations: list[float],
    *,
    members: list[str],
    judge_provider: str | None = None,
    variant: str | None = None,
    quality: str | None = None,
    memory_context: str = "",
    rag_context: str = "",
    global_style: str = "",
) -> tuple[list[ShotCard], str, dict[str, Any]]:
    """Panel storyboard: fan out to members, judge picks the winner.

    Returns (shots, style, panel_meta). 0 valid candidates -> heuristic
    fallback; 1 valid -> used directly; >=2 valid -> judge LLM decides.
    """
    from master_agent.config import PANEL_JUDGE
    from master_agent.llm_panel import format_panel_summary, panel_complete

    segs = [float(s) for s in (segment_durations or [5.0])] or [5.0]
    system, user = _storyboard_prompt(
        user_request, segs,
        variant=variant, quality=quality,
        memory_context=memory_context, rag_context=rag_context,
    )

    meta: dict[str, Any] = {
        "members": list(members),
        "judge": None,
        "winner": None,
        "judge_reason": "",
        "candidates": [],
        "fallback": None,
    }

    candidates = panel_complete(members, system, user, temperature=0.4)
    for c in candidates:
        meta["candidates"].append(
            {
                "provider": c.provider,
                "latency_s": round(c.latency_s, 1),
                "ok": c.ok,
                "error": c.error or None,
            }
        )

    valid: list[tuple[int, list[ShotCard], str, dict]] = []
    for i, c in enumerate(candidates):
        if not c.ok:
            continue
        parsed = _parse_storyboard(c.text)
        if parsed:
            shots, style, data = parsed
            valid.append((i, shots, style, data))
        else:
            meta["candidates"][i]["error"] = "unparseable storyboard JSON"

    if not valid:
        meta["fallback"] = "heuristic (no valid candidates)"
        return _heuristic_storyboard(user_request, segs, variant=variant), (
            global_style or _default_style(user_request)
        ), meta

    if len(valid) == 1:
        i, shots, style, _ = valid[0]
        meta["winner"] = candidates[i].provider
        meta["judge_reason"] = "only valid candidate"
        return _align_to_durations(shots, segs), style or global_style, meta

    # >= 2 valid -> judge picks
    judge_spec = (judge_provider or PANEL_JUDGE or "ollama").strip()
    meta["judge"] = judge_spec
    judge_path = Path(__file__).resolve().parent / "prompts" / "panel_judge.md"
    judge_system = judge_path.read_text(encoding="utf-8") if judge_path.is_file() else (
        "Pick the best storyboard. Return JSON winner/reason."
    )
    judge_payload = {
        "brief": {
            "user_request": user_request,
            "segment_durations": segs,
            "variant": variant,
            "quality": quality,
        },
        "candidates": [
            {"index": i, "provider": candidates[i].provider, "storyboard": data}
            for i, _, _, data in valid
        ],
    }
    verdict = _panel_judge_call(judge_spec, judge_system, judge_payload)

    chosen = valid[0]
    if verdict and isinstance(verdict.get("winner"), int):
        w = verdict["winner"]
        for entry in valid:
            if entry[0] == w:
                chosen = entry
                break
        meta["judge_reason"] = str(verdict.get("reason") or "")
        merged = verdict.get("merged_shots")
        if isinstance(merged, list) and merged:
            merged_cards = [
                ShotCard.from_dict(s if isinstance(s, dict) else {}, index=i)
                for i, s in enumerate(merged)
            ]
            meta["merged"] = True
            meta["winner"] = candidates[chosen[0]].provider + " (merged)"
            return _align_to_durations(merged_cards, segs), (
                chosen[2] or global_style
            ), meta
    else:
        meta["judge_reason"] = "judge unavailable; first valid candidate used"

    i, shots, style, _ = chosen
    meta["winner"] = candidates[i].provider
    return _align_to_durations(shots, segs), style or global_style, meta


def _panel_judge_call(
    judge_spec: str, judge_system: str, judge_payload: dict[str, Any]
) -> Optional[dict[str, Any]]:
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        from master_agent.llm import get_llm

        llm = get_llm(temperature=0.2, provider=judge_spec)
        resp = llm.invoke(
            [
                SystemMessage(content=judge_system),
                HumanMessage(
                    content="Judge these storyboard candidates:\n"
                    + json.dumps(judge_payload, indent=2, default=str)
                ),
            ]
        )
        text = getattr(resp, "content", None) or str(resp)
        data = _extract_json(text)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _llm_storyboard(
    user_request: str,
    segs: list[float],
    *,
    variant: str | None,
    quality: str | None,
    memory_context: str,
    rag_context: str,
) -> Optional[tuple[list[ShotCard], str]]:
    try:
        from master_agent.llm import get_llm
        from langchain_core.messages import HumanMessage, SystemMessage
    except Exception:
        return None

    system, user = _storyboard_prompt(
        user_request, segs,
        variant=variant, quality=quality,
        memory_context=memory_context, rag_context=rag_context,
    )
    try:
        llm = get_llm(temperature=0.4)
        resp = llm.invoke(
            [SystemMessage(content=system), HumanMessage(content=user)]
        )
        text = getattr(resp, "content", None) or str(resp)
        parsed = _parse_storyboard(text)
        if not parsed:
            return None
        shots, style, _ = parsed
        return shots, style
    except Exception:
        return None


def _extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None
