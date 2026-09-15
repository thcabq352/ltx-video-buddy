"""LLM client — local Ollama or llama.cpp first, Grok optional.

Provider chain for ``auto`` (default): ollama -> llamacpp -> grok.
Main model: ``qwen3-vl-heretic`` (local Qwen3-VL 9B-class; see
``OLLAMA_MODEL`` / ``LLAMACPP_MODEL``).

Both local backends speak OpenAI-compat chat at ``{base}/v1``.
Ollama also has native ``/api/chat`` and ``/api/embed``; llama.cpp uses
``/v1/chat/completions`` and ``/v1/embeddings``.
"""

from __future__ import annotations

import os
import socket
import time
from typing import Any
from urllib.parse import urlparse

from langchain_openai import ChatOpenAI

from master_agent.config import (
    LLAMACPP_MODEL,
    LLAMACPP_URL,
    LLM_PROVIDER,
    OLLAMA_MODEL,
    OLLAMA_URL,
    SPACEXAI_MODEL,
    XAI_API_KEY,
    XAI_BASE_URL,
)

AUTO_CHAIN = ("ollama", "llamacpp", "grok")

# Canonical local name -> accepted LLM_PROVIDER / panel prefixes
_LLAMACPP_ALIASES = frozenset(
    {
        "llamacpp",
        "llama.cpp",
        "llama-cpp",
        "llama_cpp",
        "llama-server",
        "llamaserver",
    }
)

# --- availability -----------------------------------------------------------

_endpoint_cache: dict[str, tuple[float, bool]] = {}
_CACHE_TTL_S = 60.0


def reset_endpoint_cache() -> None:
    """Drop TCP reachability cache (tests / after a server start)."""
    _endpoint_cache.clear()


def _host_port(url: str, default_port: int) -> tuple[str, int]:
    parsed = urlparse(url if "://" in url else f"http://{url}")
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or default_port
    return host, int(port)


def endpoint_up(url: str, default_port: int) -> bool:
    """True when ``host:port`` from *url* accepts a TCP connection."""
    key = f"{url}|{default_port}"
    ts, up = _endpoint_cache.get(key, (0.0, False))
    if time.time() - ts < _CACHE_TTL_S:
        return up
    try:
        host, port = _host_port(url, default_port)
        with socket.create_connection((host, port), timeout=1.5):
            up = True
    except OSError:
        up = False
    _endpoint_cache[key] = (time.time(), up)
    return up


def _ollama_up() -> bool:
    return endpoint_up(OLLAMA_URL, 11434)


def _llamacpp_up() -> bool:
    return endpoint_up(LLAMACPP_URL, 8080)


def normalize_provider_name(spec: str) -> str:
    """``llama.cpp:foo`` / ``llamacpp`` → ``llamacpp``; otherwise the prefix."""
    name = (spec or "").split(":", 1)[0].strip().lower()
    if name in _LLAMACPP_ALIASES:
        return "llamacpp"
    return name


def parse_provider_spec(spec: str) -> tuple[str, str]:
    """Return ``(canonical_name, model_or_empty)``."""
    raw = (spec or "").strip()
    name, _, rest = raw.partition(":")
    return normalize_provider_name(name), rest.strip()


def openai_compat_base(url: str) -> str:
    """``http://host:port`` → ``http://host:port/v1`` (idempotent if already /v1)."""
    root = (url or "").rstrip("/")
    if root.endswith("/v1"):
        return root
    return f"{root}/v1"


def provider_available(spec: str) -> bool:
    """spec: ollama[:model] | llamacpp[:model] | grok | claude"""
    name = normalize_provider_name(spec)
    if name == "ollama":
        return _ollama_up()
    if name == "llamacpp":
        return _llamacpp_up()
    if name == "grok":
        if has_valid_api_key():
            return True
        try:
            from master_agent.xai_oauth import resolve_access_token

            resolve_access_token()
            return True
        except Exception:
            return False
    if name == "claude":
        return bool((os.getenv("ANTHROPIC_API_KEY") or "").strip())
    return False


def preferred_local_provider() -> str | None:
    """Ollama if reachable, else llama.cpp. Used by panel ``default`` / ``local``."""
    if provider_available("ollama"):
        return "ollama"
    if provider_available("llamacpp"):
        return "llamacpp"
    return None


def active_local_backend() -> str | None:
    """Local backend for vision / embeddings, honoring an explicit ``LLM_PROVIDER``.

    ``auto`` (and cloud-only providers) try ollama then llamacpp.
    An explicit ``llamacpp`` / ``ollama`` does **not** silently hop to the other.
    """
    name = normalize_provider_name(LLM_PROVIDER)
    if name == "llamacpp":
        return "llamacpp" if provider_available("llamacpp") else None
    if name == "ollama":
        return "ollama" if provider_available("ollama") else None
    return preferred_local_provider()


def local_llm_health() -> dict[str, Any]:
    """Reachability for each local backend — never conflates Ollama with llama.cpp."""
    ollama_up = _ollama_up()
    llamacpp_up = _llamacpp_up()
    preferred = preferred_local_provider()
    return {
        "ollama": {
            "up": ollama_up,
            "url": OLLAMA_URL,
            "model": OLLAMA_MODEL,
        },
        "llamacpp": {
            "up": llamacpp_up,
            "url": LLAMACPP_URL,
            "model": LLAMACPP_MODEL,
        },
        "preferred": preferred,
        "provider": LLM_PROVIDER,
        "active": active_local_backend(),
    }


def attach_llm_health(out: dict[str, Any]) -> dict[str, Any]:
    """Fill health dict keys used by MCP / studio. ``ollama`` stays a bool."""
    detail = local_llm_health()
    out["ollama"] = detail["ollama"]["up"]
    out["llamacpp"] = detail["llamacpp"]["up"]
    out["local_llm"] = detail
    return out


def format_local_llm_health(detail: dict[str, Any] | None = None) -> str:
    data = detail or local_llm_health()
    lines = []
    for key in ("ollama", "llamacpp"):
        row = data.get(key) or {}
        mark = "up  " if row.get("up") else "down"
        lines.append(
            f"      {key:<8} {mark}  {row.get('url')}  model={row.get('model')}"
        )
    pref = data.get("preferred") or "none"
    lines.append(f"      local    {pref}   provider {data.get('provider')}")
    return "\n".join(lines) + "\n"


def chat_client_config(spec: str) -> dict[str, str]:
    """OpenAI-compat client fields for a provider spec (no network)."""
    name, model = parse_provider_spec(spec)
    if name == "ollama":
        return {
            "provider": "ollama",
            "base_url": openai_compat_base(OLLAMA_URL),
            "model": model or OLLAMA_MODEL,
            "api_key": "ollama",
        }
    if name == "llamacpp":
        return {
            "provider": "llamacpp",
            "base_url": openai_compat_base(LLAMACPP_URL),
            "model": model or LLAMACPP_MODEL,
            "api_key": "llamacpp",
        }
    if name == "grok":
        return {
            "provider": "grok",
            "base_url": os.getenv("XAI_BASE_URL", XAI_BASE_URL),
            "model": os.getenv("SPACEXAI_MODEL", SPACEXAI_MODEL),
            "api_key": "grok",
        }
    raise RuntimeError(
        f"unknown LLM provider spec: {spec!r} "
        "(use ollama[:model] | llamacpp[:model] | grok)"
    )


def embeddings_endpoint(backend: str) -> str:
    name = normalize_provider_name(backend)
    if name == "llamacpp":
        return f"{openai_compat_base(LLAMACPP_URL)}/embeddings"
    return f"{OLLAMA_URL.rstrip('/')}/api/embed"


def vision_endpoint(backend: str) -> str:
    name = normalize_provider_name(backend)
    if name == "llamacpp":
        return f"{openai_compat_base(LLAMACPP_URL)}/chat/completions"
    return f"{OLLAMA_URL.rstrip('/')}/api/chat"


# --- model construction -----------------------------------------------------


def _openai_compat_llm(
    model: str, temperature: float, *, base_url: str, api_key: str
) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        timeout=900,  # cold model loads are slow
    )


def _ollama_llm(model: str, temperature: float) -> ChatOpenAI:
    return _openai_compat_llm(
        model,
        temperature,
        base_url=openai_compat_base(OLLAMA_URL),
        api_key="ollama",
    )


def _llamacpp_llm(model: str, temperature: float) -> ChatOpenAI:
    return _openai_compat_llm(
        model,
        temperature,
        base_url=openai_compat_base(LLAMACPP_URL),
        api_key="llamacpp",
    )


def _grok_llm(temperature: float) -> ChatOpenAI:
    api_key, source = resolve_credentials()
    return ChatOpenAI(
        model=os.getenv("SPACEXAI_MODEL", SPACEXAI_MODEL),
        api_key=api_key,
        base_url=os.getenv("XAI_BASE_URL", XAI_BASE_URL),
        temperature=temperature,
        default_headers={"X-LTX-Agent-Auth": source},
    )


def get_llm(temperature: float = 0.2, provider: str | None = None) -> ChatOpenAI:
    """Build a chat model for a provider spec; ``auto`` tries ollama -> llamacpp -> grok."""
    spec = (provider or LLM_PROVIDER or "auto").strip().lower()
    if spec != "auto":
        return _llm_for(spec, temperature)
    errors: list[str] = []
    for candidate in AUTO_CHAIN:
        if not provider_available(candidate):
            continue
        try:
            return _llm_for(candidate, temperature)
        except Exception as e:
            errors.append(f"{candidate}: {e}")
    raise RuntimeError(
        "No LLM provider available (tried ollama -> llamacpp -> grok). "
        + "; ".join(errors)
    )


def _llm_for(spec: str, temperature: float) -> ChatOpenAI:
    name, model = parse_provider_spec(spec)
    if name == "ollama":
        return _ollama_llm(model or OLLAMA_MODEL, temperature)
    if name == "llamacpp":
        return _llamacpp_llm(model or LLAMACPP_MODEL, temperature)
    if name == "grok":
        return _grok_llm(temperature)
    raise RuntimeError(
        f"unknown LLM provider spec: {spec!r} "
        "(use ollama[:model] | llamacpp[:model] | grok)"
    )


# --- grok auth (unchanged) --------------------------------------------------


def _auth_mode() -> str:
    """
    SPACEXAI_AUTH:
      - xai-oauth (default): Hermes SuperGrok / Premium+ OAuth
      - api_key: XAI_API_KEY only
      - auto: oauth if present, else api_key
    """
    raw = (os.getenv("SPACEXAI_AUTH") or "xai-oauth").strip().lower()
    aliases = {
        "xaioauth": "xai-oauth",
        "xai_oauth": "xai-oauth",
        "oauth": "xai-oauth",
        "grok-oauth": "xai-oauth",
        "key": "api_key",
        "apikey": "api_key",
        "xai": "api_key",
    }
    return aliases.get(raw, raw)


def has_valid_api_key() -> bool:
    key = (XAI_API_KEY or "").strip()
    if not key:
        return False
    if key in ("your_key_here", "changeme", "xxx"):
        return False
    if key.startswith("your_"):
        return False
    return True


def resolve_credentials() -> tuple[str, str]:
    """
    Returns (api_key_or_bearer, source_label).
    source_label is ``xai-oauth`` or ``api_key``.
    """
    mode = _auth_mode()

    if mode in ("xai-oauth", "auto"):
        try:
            from master_agent.xai_oauth import resolve_access_token

            token = resolve_access_token()
            return token, "xai-oauth"
        except Exception as e:
            if mode == "xai-oauth":
                # Hard preference for oauth — only fall back if api_key explicitly available
                # and SPACEXAI_AUTH was auto; for xai-oauth raise clear message
                if has_valid_api_key() and os.getenv("SPACEXAI_OAUTH_FALLBACK_API_KEY", "").lower() in (
                    "1",
                    "true",
                    "yes",
                ):
                    return XAI_API_KEY, "api_key"
                raise RuntimeError(
                    f"xai-oauth failed: {e}\n"
                    "Fix: hermes auth add xai-oauth\n"
                    "Or set SPACEXAI_AUTH=api_key and XAI_API_KEY=..."
                ) from e
            # auto mode → try api key
            if has_valid_api_key():
                return XAI_API_KEY, "api_key"
            raise RuntimeError(
                f"No xai-oauth token and no XAI_API_KEY. ({e}) "
                "Run: hermes auth add xai-oauth"
            ) from e

    if mode == "api_key":
        if not has_valid_api_key():
            raise RuntimeError(
                "SPACEXAI_AUTH=api_key but XAI_API_KEY is not set. "
                "Create a key at https://console.x.ai or use SPACEXAI_AUTH=xai-oauth"
            )
        return XAI_API_KEY, "api_key"

    raise RuntimeError(
        f"Unknown SPACEXAI_AUTH={mode!r}. Use xai-oauth, api_key, or auto."
    )
