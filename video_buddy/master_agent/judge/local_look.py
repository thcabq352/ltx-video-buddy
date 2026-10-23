"""Look and taste checks stay on the local llama.cpp Qwen3-VL heretic judge.

SuperGrok is not this path. Creative direction may opt into Grok; a look
check never does. The seat rule owns the GPU during a render, so this ping
runs before the first still and again after a render, when the server is on
CPU or the render has already released the card.
"""

from __future__ import annotations

from typing import Any


def look_provider_spec() -> str:
    from master_agent.config import VISION_MODEL

    model = (VISION_MODEL or "qwen3-vl-heretic").strip() or "qwen3-vl-heretic"
    return f"llamacpp:{model}"


def configured() -> dict[str, Any]:
    """Static check. Does not start a server."""
    from master_agent.config import JUDGE_ENABLED, VISION_ENABLED, VISION_MODEL

    model = (VISION_MODEL or "").strip()
    if not JUDGE_ENABLED:
        return {"fired": False, "reason": "JUDGE_ENABLED is off"}
    if not VISION_ENABLED:
        return {"fired": False, "reason": "VISION_ENABLED is off"}
    if "qwen3-vl-heretic" not in model:
        return {
            "fired": False,
            "reason": f"VISION_MODEL is {model or 'unset'}; look checks require qwen3-vl-heretic on llama.cpp",
        }
    return {"fired": False, "ready": True, "provider": "llamacpp", "model": model}


def fire_look_check(*, kind: str = "look", ping=None) -> dict[str, Any]:
    """Record that the local judge actually ran.

    ``ping`` replaces the llama.cpp call in tests. The default ping asks
    llama.cpp for a one-line look check and refuses Ollama and Grok.
    """
    state = configured()
    if not state.get("ready"):
        return state
    caller = ping or _ping_llamacpp
    try:
        answer = caller(kind)
    except Exception as exc:
        return {
            "fired": False,
            "reason": f"local judge failed: {type(exc).__name__}: {exc}",
            "provider": "llamacpp",
            "model": state.get("model"),
        }
    if not answer or answer.get("fired") is False:
        return {
            "fired": False,
            "reason": (answer or {}).get("reason") or "local judge returned nothing",
            "provider": "llamacpp",
        }
    return {
        "fired": True,
        "provider": "llamacpp",
        "model": state.get("model"),
        "kind": kind,
        "backend": answer.get("backend") or "llamacpp",
    }


def _ping_llamacpp(kind: str) -> dict[str, Any]:
    from master_agent.llm import get_llm

    llm = get_llm(temperature=0.0, provider=look_provider_spec())
    # The chat class name is the backend. Grok and Ollama must not answer a look check.
    name = type(llm).__name__.lower()
    if "grok" in name or "ollama" in name:
        return {"fired": False, "reason": f"look check resolved to {type(llm).__name__}, not llama.cpp"}
    llm.invoke(f"Look check ({kind}). Reply with the single word ready.")
    return {"fired": True, "backend": "llamacpp"}
