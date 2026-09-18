"""LLM panel — fan one prompt out to several providers, collect candidates.

Local-first (default): the local ``qwen3-vl-heretic`` alone.
Named presets:
  - ``default`` / ``local`` — local only
  - ``grok`` — Grok solo (cloud)
  - ``both`` / ``panel`` / ``grok+local`` — Grok + local (judge picks)
  - ``grok+claude`` — Grok + Claude (judge picks)
  - ``duo`` — two local models (VL heretic + gemma4), legacy
Custom comma lists still work (``claude``, ``ollama:<model>``,
``llamacpp:<model>``, ``grok``, …).
Unavailable or failing members are skipped with a log line, never fatal.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

from master_agent.config import (
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    LLAMACPP_MODEL,
    LLM_PANEL,
    OLLAMA_MODEL,
    PANEL_MEMBER_TIMEOUT_S,
)
from master_agent.llm import get_llm, normalize_provider_name, provider_available


def _local_member() -> str:
    """``ollama:<model>`` when Ollama is up, else ``llamacpp:<model>``."""
    if provider_available("ollama"):
        return f"ollama:{OLLAMA_MODEL}"
    if provider_available("llamacpp"):
        return f"llamacpp:{LLAMACPP_MODEL}"
    return f"ollama:{OLLAMA_MODEL}"


def _presets() -> dict[str, str]:
    main = _local_member()
    grok_local = f"grok,{main}"
    grok_claude = "grok,claude"
    return {
        # local-first defaults
        "default": main,
        "local": main,
        # Grok solo (cloud)
        "grok": "grok",
        # Grok + local panel (judge picks)
        "both": grok_local,
        "panel": grok_local,
        "grok+local": grok_local,
        "grok-local": grok_local,
        "grok_local": grok_local,
        # Grok + Claude panel
        "grok+claude": grok_claude,
        "grok-claude": grok_claude,
        "grok_claude": grok_claude,
        # legacy: two local models
        "duo": f"{main},ollama:gemma4:latest",
    }


@dataclass
class PanelCandidate:
    provider: str
    text: str = ""
    latency_s: float = 0.0
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.text.strip())


@dataclass
class PanelResolution:
    members: list[str] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (spec, reason)


def _skip_reason(spec: str) -> str:
    name = normalize_provider_name(spec)
    if name == "claude":
        return "ANTHROPIC_API_KEY not set"
    if name == "ollama":
        return "Ollama not reachable"
    if name == "llamacpp":
        return "llama.cpp not reachable"
    if name == "grok":
        return "Grok auth missing (xai-oauth or XAI_API_KEY)"
    return "provider unavailable"


def resolve_panel(spec: str | None) -> PanelResolution:
    """Preset name or comma list -> available members (+ skipped with reasons)."""
    raw = (spec or LLM_PANEL or "default").strip()
    expanded = _presets().get(raw.lower(), raw)
    out = PanelResolution()
    seen: set[str] = set()
    for part in expanded.split(","):
        member = part.strip()
        if not member or member in seen:
            continue
        seen.add(member)
        if provider_available(member):
            out.members.append(member)
        else:
            out.skipped.append((member, _skip_reason(member)))
    return out


def _complete_one(spec: str, system: str, user: str, temperature: float) -> PanelCandidate:
    start = time.time()
    name = spec.split(":", 1)[0].lower()
    try:
        if name == "claude":
            text = _claude_complete(system, user)
        else:
            from langchain_core.messages import HumanMessage, SystemMessage

            llm = get_llm(temperature=temperature, provider=spec)
            resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
            text = getattr(resp, "content", None) or str(resp)
            if isinstance(text, list):  # some providers return content blocks
                text = "".join(
                    b.get("text", "") if isinstance(b, dict) else str(b) for b in text
                )
        cand = PanelCandidate(provider=spec, text=text, latency_s=time.time() - start)
        if not cand.ok:
            cand.error = "empty response"
        return cand
    except Exception as e:
        return PanelCandidate(
            provider=spec,
            latency_s=time.time() - start,
            error=f"{type(e).__name__}: {str(e)[:300]}",
        )


def _claude_complete(system: str, user: str) -> str:
    import httpx

    resp = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": CLAUDE_MODEL,
            "max_tokens": 8192,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        },
        timeout=PANEL_MEMBER_TIMEOUT_S,
    )
    resp.raise_for_status()
    data = resp.json()
    return "".join(b.get("text", "") for b in data.get("content", []) if isinstance(b, dict))


def panel_complete(
    members: list[str],
    system: str,
    user: str,
    *,
    temperature: float = 0.4,
) -> list[PanelCandidate]:
    """Fan the prompt out to all members in parallel; one candidate per member."""
    if not members:
        return []
    results: dict[str, PanelCandidate] = {}
    with ThreadPoolExecutor(max_workers=len(members)) as pool:
        futures = {
            pool.submit(_complete_one, m, system, user, temperature): m for m in members
        }
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
    return [results[m] for m in members]  # preserve member order


def format_panel_summary(candidates: list[PanelCandidate]) -> str:
    lines = []
    for c in candidates:
        status = "ok" if c.ok else f"error: {c.error}"
        lines.append(f"  - {c.provider}: {c.latency_s:.0f}s {status}")
    return "\n".join(lines)
