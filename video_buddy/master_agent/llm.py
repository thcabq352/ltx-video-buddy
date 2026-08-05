"""LLM client — local Ollama main (user pivot: local-only first), Kimi/Grok optional.

Provider chain for ``auto`` (default): ollama -> kimi -> grok.
Main model: ``qwen3.6-27b-fable`` (local GGUF import, see ``OLLAMA_MODEL``).
Kimi authenticates with the kimi-code CLI's OAuth identity
(``master_agent.kimi_oauth``); ``KIMI_API_KEY`` is the fallback.
Note: k3 thinking models reject any temperature other than 1.
"""

from __future__ import annotations

import os
import socket
import time

from langchain_openai import ChatOpenAI

from master_agent.config import (
    KIMI_BASE_URL,
    KIMI_MAX_TOKENS,
    KIMI_MODEL,
    KIMI_API_KEY,
    LLM_PROVIDER,
    OLLAMA_MODEL,
    OLLAMA_URL,
    SPACEXAI_MODEL,
    XAI_API_KEY,
    XAI_BASE_URL,
)

# --- availability -----------------------------------------------------------

_ollama_up_cache: tuple[float, bool] = (0.0, False)


def _ollama_up() -> bool:
    global _ollama_up_cache
    ts, up = _ollama_up_cache
    if time.time() - ts < 60:
        return up
    try:
        host, _, port = OLLAMA_URL.split("://", 1)[-1].rstrip("/").partition(":")
        with socket.create_connection((host, int(port or 11434)), timeout=1.5):
            up = True
    except OSError:
        up = False
    _ollama_up_cache = (time.time(), up)
    return up


def _has_kimi_api_key() -> bool:
    key = (KIMI_API_KEY or "").strip()
    return bool(key) and not key.startswith("your_")


def kimi_available() -> bool:
    if _has_kimi_api_key():
        return True
    try:
        from master_agent.kimi_oauth import token_available

        return token_available()
    except Exception:
        return False


def provider_available(spec: str) -> bool:
    """spec: kimi | ollama[:model] | grok | claude"""
    name = (spec or "").split(":", 1)[0].strip().lower()
    if name == "kimi":
        return kimi_available()
    if name == "ollama":
        return _ollama_up()
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


# --- model construction -----------------------------------------------------


def _kimi_llm(temperature: float) -> ChatOpenAI:
    from master_agent.kimi_oauth import resolve_access_token

    try:
        api_key, source = resolve_access_token(), "kimi-oauth"
    except Exception:
        if not _has_kimi_api_key():
            raise
        api_key, source = KIMI_API_KEY, "api_key"
    # k3 thinking models reject any temperature other than 1
    return ChatOpenAI(
        model=os.getenv("KIMI_MODEL", KIMI_MODEL),
        api_key=api_key,
        base_url=KIMI_BASE_URL,
        temperature=1,
        max_tokens=KIMI_MAX_TOKENS,
        timeout=600,
        default_headers={"X-LTX-Agent-Auth": source},
    )


def _ollama_llm(model: str, temperature: float) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        api_key="ollama",
        base_url=f"{OLLAMA_URL}/v1",
        temperature=temperature,
        timeout=900,  # cold model loads are slow
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
    """Build a chat model for a provider spec; ``auto`` tries kimi -> ollama -> grok."""
    spec = (provider or LLM_PROVIDER or "auto").strip().lower()
    if spec != "auto":
        return _llm_for(spec, temperature)
    errors: list[str] = []
    for candidate in ("ollama", "kimi", "grok"):
        if not provider_available(candidate):
            continue
        try:
            return _llm_for(candidate, temperature)
        except Exception as e:
            errors.append(f"{candidate}: {e}")
    raise RuntimeError(
        "No LLM provider available (tried ollama -> kimi -> grok). " + "; ".join(errors)
    )


def _llm_for(spec: str, temperature: float) -> ChatOpenAI:
    name, _, rest = spec.partition(":")
    name = name.strip().lower()
    if name == "kimi":
        return _kimi_llm(temperature)
    if name == "ollama":
        return _ollama_llm(rest.strip() or OLLAMA_MODEL, temperature)
    if name == "grok":
        return _grok_llm(temperature)
    raise RuntimeError(f"unknown LLM provider spec: {spec!r} (use kimi | ollama[:model] | grok)")


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
            from master_agent.xai_oauth import resolve_access_token, XaiOauthError

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
