"""Director brain — LLM request routing.

Decides which workflow variant a request should use. Hard constraints
(forced variant, source video) always win; otherwise the local LLM picks,
with the old keyword rules as fallback. The LLM never sees secrets and its
answer is validated against the known variant list before use.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from master_agent.config import DIRECTOR_LLM, WORKFLOW_FILES

_VARIANT_KEYWORDS = [
    ("lipsync", ("lip-sync", "lipsync", "lip sync", "lipdub", "dub")),
    ("wan22", ("wan 2.2", "wan2.2", "wan22", "photoreal", "photo-real", "stock photo", "film grain")),
    ("directors", ("director", "storyboard", "scene", "shots", "multi-shot")),
    ("eros", ("eros", "10eros")),
]


def rule_based_variant(request: str) -> str:
    text = (request or "").lower()
    for variant, keywords in _VARIANT_KEYWORDS:
        if any(k in text for k in keywords):
            return variant
    return "base"


def choose_variant(
    request: str,
    *,
    has_video: bool = False,
    force: str | None = None,
) -> tuple[str, str]:
    """Returns (variant, source) where source is forced|input|llm|rules."""
    if force:
        return force, "forced"
    if has_video:
        return "lipsync", "input"
    fallback = rule_based_variant(request)
    if not DIRECTOR_LLM:
        return fallback, "rules"
    llm_variant = _llm_variant(request, fallback=fallback)
    if llm_variant:
        return llm_variant, "llm"
    return fallback, "rules"


def _llm_variant(request: str, *, fallback: str) -> Optional[str]:
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        from master_agent.llm import get_llm
    except Exception:
        return None

    path = Path(__file__).resolve().parent / "prompts" / "director.md"
    system = path.read_text(encoding="utf-8") if path.is_file() else (
        "Pick the workflow variant. Return JSON variant/reason."
    )
    payload = {
        "request": request,
        "allowed_variants": sorted(WORKFLOW_FILES.keys()),
        "rule_based_suggestion": fallback,
    }
    try:
        llm = get_llm(temperature=0.1)
        resp = llm.invoke(
            [
                SystemMessage(content=system),
                HumanMessage(
                    content="Route this request:\n" + json.dumps(payload, indent=2)
                ),
            ]
        )
        text = getattr(resp, "content", None) or str(resp)
        data = _extract_json(text)
        if not isinstance(data, dict):
            return None
        variant = str(data.get("variant") or "").strip().lower()
        return variant if variant in WORKFLOW_FILES else None
    except Exception:
        return None


def _extract_json(text: str) -> Optional[dict[str, Any]]:
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
