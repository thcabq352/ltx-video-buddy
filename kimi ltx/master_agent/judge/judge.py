"""Judge: heuristic + LLM (Grok) quality scoring for generated clips.

Ported from LTX Project agent/judge.py. The judge never executes; it only
evaluates — the orchestrator decides what to do with the verdict.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from master_agent.config import (
    JUDGE_HEURISTIC_WEIGHT,
    JUDGE_LLM_WEIGHT,
    JUDGE_SCORE_THRESHOLD,
    MAX_JUDGE_ROUNDS,
)
from master_agent.judge.probe import frame_notes


def _vision_review_safe(
    video_path: str | None,
    *,
    user_request: str,
    ltx_prompt: str = "",
    full_video: bool = False,
) -> Optional[dict[str, Any]]:
    """Vision evaluator leg; None when disabled/unavailable/failed."""
    try:
        from master_agent.judge.vision import vision_notes, vision_review

        return vision_review(
            video_path,
            user_request=user_request,
            ltx_prompt=ltx_prompt,
            full_video=full_video,
        )
    except Exception:
        return None


def _vnotes(review: dict[str, Any]) -> str:
    from master_agent.judge.vision import vision_notes

    return vision_notes(review)


@dataclass
class JudgeResult:
    pass_: bool
    score: float
    combined_score: float
    issues: list[str] = field(default_factory=list)
    prompt_rewrite: str = ""
    param_hints: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    decision: str = "accept"  # accept | rewrite | retune | skip
    heuristic_score: float = 0.0
    llm_score: float = 0.0
    vision_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["pass"] = d.pop("pass_")
        return d


def merge_scores(heuristic: float, llm: float | None) -> float:
    h = float(heuristic or 0.0)
    if llm is None:
        return h
    w_h = JUDGE_HEURISTIC_WEIGHT
    w_l = JUDGE_LLM_WEIGHT
    total = w_h + w_l
    return (w_h * h + w_l * float(llm)) / total


def merge_legs(
    heuristic: float,
    llm: float | None,
    vision: float | None,
) -> float:
    """Weighted merge of heuristic / text-LLM / vision legs (renormalized)."""
    from master_agent.config import JUDGE_VISION_WEIGHT

    legs: list[tuple[float, float]] = [(JUDGE_HEURISTIC_WEIGHT, float(heuristic or 0.0))]
    if llm is not None:
        legs.append((JUDGE_LLM_WEIGHT, float(llm)))
    if vision is not None:
        legs.append((JUDGE_VISION_WEIGHT, float(vision)))
    total = sum(w for w, _ in legs)
    return sum(w * v for w, v in legs) / total


def is_critical_fail(heuristic_issues: list[dict[str, Any]] | None) -> bool:
    codes = {i.get("code") for i in (heuristic_issues or [])}
    return bool(codes & {"missing_video", "tiny_file", "short_duration"})


def decide_action(
    *,
    combined: float,
    llm_pass: bool | None,
    prompt_rewrite: str,
    param_hints: dict[str, Any],
    judge_retries: int,
    max_rounds: int,
    critical: bool,
    threshold: float | None = None,
) -> str:
    thr = threshold if threshold is not None else JUDGE_SCORE_THRESHOLD
    if judge_retries >= max_rounds:
        return "accept"  # budget exhausted — keep best
    if critical and not prompt_rewrite and not param_hints:
        return "retune"
    acceptable = combined >= thr and (llm_pass is not False or combined >= thr + 0.05)
    if llm_pass is True and combined >= thr * 0.92:
        acceptable = True
    if acceptable and not critical:
        return "accept"
    if prompt_rewrite and prompt_rewrite.strip():
        return "rewrite"
    if param_hints:
        return "retune"
    if critical:
        return "retune"
    return "accept" if combined >= thr * 0.9 else "rewrite"


def judge_segment(
    *,
    user_request: str,
    ltx_prompt: str,
    shot: dict[str, Any] | None = None,
    video_path: str | None,
    heuristic_score: float,
    heuristic_issues: list[dict[str, Any]] | None,
    judge_retries: int = 0,
    max_rounds: int | None = None,
    threshold: float | None = None,
    judge_enabled: bool = True,
) -> JudgeResult:
    """Heuristic + optional LLM judge for one generated clip."""
    thr = threshold if threshold is not None else JUDGE_SCORE_THRESHOLD
    max_r = max_rounds if max_rounds is not None else MAX_JUDGE_ROUNDS
    issues = [str(i.get("detail") or i.get("code") or i) for i in (heuristic_issues or [])]
    critical = is_critical_fail(heuristic_issues)

    if not judge_enabled:
        combined = float(heuristic_score or 0.0)
        return JudgeResult(
            pass_=combined >= thr and not critical,
            score=combined,
            combined_score=combined,
            issues=issues,
            reason="Judge disabled; heuristic only",
            decision="accept" if combined >= thr and not critical else "retune",
            heuristic_score=combined,
            llm_score=combined,
        )

    notes = frame_notes(video_path)
    review = _vision_review_safe(
        video_path, user_request=user_request, ltx_prompt=ltx_prompt
    )
    vision_score = review["score"] if review else None
    if review:
        notes += "; " + _vnotes(review)
        if review.get("issues"):
            issues = list(
                dict.fromkeys(issues + [f"vision: {x}" for x in review["issues"]])
            )
    llm = _llm_judge(
        user_request=user_request,
        ltx_prompt=ltx_prompt,
        shot=shot,
        heuristic_score=heuristic_score,
        heuristic_issues=heuristic_issues or [],
        frame_notes=notes,
    )

    llm_score = float(llm.get("score", heuristic_score)) if llm else float(heuristic_score)
    llm_pass = bool(llm.get("pass")) if llm and "pass" in llm else None
    rewrite = str(llm.get("prompt_rewrite") or "") if llm else ""
    hints = dict(llm.get("param_hints") or {}) if llm else {}
    if llm and llm.get("issues"):
        issues = list(dict.fromkeys(issues + [str(x) for x in llm["issues"]]))
    reason = str(llm.get("reason") or "") if llm else "Heuristic-only judge (LLM unavailable)"
    if review and review.get("reason"):
        reason = f"{reason} | vision: {review['reason'][:200]}"

    combined = merge_legs(
        float(heuristic_score or 0.0), llm_score if llm else None, vision_score
    )

    # Fallback: strong heuristics alone (unless vision explicitly failed it)
    vision_failed = review is not None and not review.get("pass", True)
    if not llm and float(heuristic_score or 0) >= 0.85 and not critical and not vision_failed:
        decision = "accept"
        llm_pass = True
    else:
        decision = decide_action(
            combined=combined,
            llm_pass=llm_pass,
            prompt_rewrite=rewrite,
            param_hints=hints,
            judge_retries=judge_retries,
            max_rounds=max_r,
            critical=critical,
            threshold=thr,
        )

    passed = decision == "accept"
    return JudgeResult(
        pass_=passed,
        score=llm_score,
        combined_score=combined,
        issues=issues,
        prompt_rewrite=rewrite,
        param_hints=hints,
        reason=reason or f"combined={combined:.2f} decision={decision}",
        decision=decision,
        heuristic_score=float(heuristic_score or 0.0),
        llm_score=llm_score,
        vision_score=vision_score,
    )


def judge_full_video(
    *,
    user_request: str,
    storyboard: list[dict[str, Any]] | None,
    video_path: str | None,
    segment_scores: list[float] | None = None,
    threshold: float | None = None,
) -> JudgeResult:
    """Outer judge for a stitched multi-segment video.

    Heuristic leg = segment-score average (floored by file size); LLM verdict
    may name weak shots as "shot:N" for selective re-generation.
    """
    thr = threshold if threshold is not None else JUDGE_SCORE_THRESHOLD
    notes = frame_notes(video_path)
    seg_avg = (
        sum(segment_scores) / len(segment_scores) if segment_scores else 0.75
    )
    h = seg_avg
    if video_path and Path(video_path).is_file():
        size = Path(video_path).stat().st_size
        if size < 100_000:
            h = min(h, 0.3)
        elif size < 400_000:
            h = min(h, 0.6)
    else:
        h = 0.0

    board_summary = ""
    if storyboard:
        board_summary = "; ".join(
            f"{s.get('index', i)}:{s.get('title', '')}" for i, s in enumerate(storyboard)
        )

    review = _vision_review_safe(
        video_path,
        user_request=user_request,
        ltx_prompt=f"FULL VIDEO storyboard=[{board_summary}]",
        full_video=True,
    )
    vision_score = review["score"] if review else None
    if review:
        notes += "; " + _vnotes(review)

    llm = _llm_judge(
        user_request=user_request,
        ltx_prompt=f"FULL VIDEO storyboard=[{board_summary}]",
        shot={"title": "full_video", "continuity": "stitched multi-segment"},
        heuristic_score=h,
        heuristic_issues=[],
        frame_notes=notes + f", segment_avg={seg_avg:.2f}",
        full_video=True,
    )
    llm_score = float(llm.get("score", h)) if llm else h
    combined = merge_legs(h, llm_score if llm else None, vision_score)
    rewrite = str(llm.get("prompt_rewrite") or "") if llm else ""
    vision_ok = review is None or review.get("pass", True) or vision_score >= thr
    passed = (
        combined >= thr
        and (not llm or bool(llm.get("pass", combined >= thr)))
        and vision_ok
    )
    issues = list(llm.get("issues") or []) if llm else []
    if review and review.get("issues"):
        issues = list(dict.fromkeys(issues + [f"vision: {x}" for x in review["issues"]]))
    reason = str(llm.get("reason") or f"full combined={combined:.2f}")
    if review and review.get("reason"):
        reason = f"{reason} | vision: {review['reason'][:200]}"
    return JudgeResult(
        pass_=passed,
        score=llm_score,
        combined_score=combined,
        issues=issues,
        prompt_rewrite=rewrite,
        param_hints=dict(llm.get("param_hints") or {}) if llm else {},
        reason=reason,
        decision="accept" if passed else "rewrite",
        heuristic_score=h,
        llm_score=llm_score,
        vision_score=vision_score,
    )


def parse_weak_shot_indices(issues: list[str], n_segments: int) -> list[int]:
    """Extract 0-based weak shot indices from judge issues ("shot:N")."""
    found: set[int] = set()
    for iss in issues:
        for m in re.finditer(r"shot\s*[#:]?\s*(\d+)", iss or "", re.I):
            idx = int(m.group(1))
            # allow 1-based or 0-based
            if 1 <= idx <= n_segments:
                found.add(idx - 1)
            elif 0 <= idx < n_segments:
                found.add(idx)
    if not found and n_segments > 0:
        # re-gen middle and last as weak default
        found.add(n_segments // 2)
        if n_segments > 1:
            found.add(n_segments - 1)
    return sorted(found)


def _llm_judge(
    *,
    user_request: str,
    ltx_prompt: str,
    shot: dict[str, Any] | None,
    heuristic_score: float,
    heuristic_issues: list[dict[str, Any]],
    frame_notes: str,
    full_video: bool = False,
) -> Optional[dict[str, Any]]:
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        from master_agent.llm import get_llm
    except Exception:
        return None

    path = Path(__file__).resolve().parent / "prompts" / "judge.md"
    system = path.read_text(encoding="utf-8") if path.is_file() else (
        "Judge video quality. Return JSON pass/score/issues/prompt_rewrite/param_hints/reason."
    )
    if full_video:
        system += "\nThis is a FULL stitched video review. Mention weak shot indices as shot:N if needed."

    payload = {
        "user_request": user_request,
        "ltx_prompt": ltx_prompt,
        "shot": shot,
        "heuristic_score": heuristic_score,
        "heuristic_issues": heuristic_issues[:10],
        "frame_notes": frame_notes,
    }
    try:
        llm = get_llm(temperature=0.2)
        resp = llm.invoke(
            [
                SystemMessage(content=system),
                HumanMessage(content="Judge this output:\n" + json.dumps(payload, indent=2, default=str)),
            ]
        )
        text = getattr(resp, "content", None) or str(resp)
        data = _extract_json(text)
        return data if isinstance(data, dict) else None
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
