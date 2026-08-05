"""Load / refresh xAI Grok OAuth tokens from Hermes ``~/.hermes/auth.json``.

Uses the same provider id as Hermes: ``xai-oauth`` (device-code SuperGrok /
X Premium+ login). No ``XAI_API_KEY`` is required when a valid OAuth grant
is present.
"""

from __future__ import annotations

import base64
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx

XAI_OAUTH_PROVIDER = "xai-oauth"
XAI_OAUTH_ISSUER = "https://auth.x.ai"
XAI_OAUTH_DISCOVERY_URL = f"{XAI_OAUTH_ISSUER}/.well-known/openid-configuration"
# Public Hermes / Grok CLI OAuth client id (same as hermes_cli.auth)
XAI_OAUTH_CLIENT_ID = "b1a00492-073a-47ea-816f-4c329264a828"
# Refresh up to 1h before JWT exp (Hermes default for long agent runs)
REFRESH_SKEW_SECONDS = 3600


class XaiOauthError(RuntimeError):
    def __init__(self, message: str, *, relogin_required: bool = False):
        super().__init__(message)
        self.relogin_required = relogin_required


def auth_json_path() -> Path:
    hermes_home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
    return Path(hermes_home) / "auth.json"


def _load_store() -> dict[str, Any]:
    path = auth_json_path()
    if not path.is_file():
        raise XaiOauthError(
            f"No Hermes auth store at {path}. Run: hermes auth add xai-oauth",
            relogin_required=True,
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise XaiOauthError(f"Could not read {path}: {e}") from e
    if not isinstance(data, dict):
        raise XaiOauthError(f"Invalid auth store shape in {path}")
    return data


def _save_store(data: dict[str, Any]) -> None:
    path = auth_json_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _provider_state(store: dict[str, Any]) -> Optional[dict[str, Any]]:
    providers = store.get("providers")
    if not isinstance(providers, dict):
        return None
    state = providers.get(XAI_OAUTH_PROVIDER)
    if isinstance(state, dict):
        tokens = state.get("tokens")
        if isinstance(tokens, dict) and str(tokens.get("access_token") or "").strip():
            return state

    # Credential pool fallback (Hermes multi-entry)
    pool = store.get("credential_pool")
    entries = pool.get(XAI_OAUTH_PROVIDER) if isinstance(pool, dict) else None
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            access = str(entry.get("access_token") or "").strip()
            refresh = str(entry.get("refresh_token") or "").strip()
            if access and refresh:
                return {
                    "tokens": {
                        "access_token": access,
                        "refresh_token": refresh,
                        "token_type": str(entry.get("token_type") or "Bearer"),
                    },
                    "last_refresh": entry.get("last_refresh"),
                    "auth_mode": "oauth_device_code",
                    "discovery": {},
                }
    return state if isinstance(state, dict) else None


def _jwt_exp(access_token: str) -> Optional[float]:
    try:
        parts = access_token.split(".")
        if len(parts) < 2:
            return None
        payload_b64 = parts[1] + "=" * (-len(parts[1]) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64.encode("ascii")))
        exp = payload.get("exp")
        return float(exp) if isinstance(exp, (int, float)) else None
    except Exception:
        return None


def _is_expiring(access_token: str, skew: int = REFRESH_SKEW_SECONDS) -> bool:
    exp = _jwt_exp(access_token)
    if exp is None:
        return False
    return exp <= (time.time() + max(0, skew))


def _token_endpoint(state: dict[str, Any]) -> str:
    discovery = state.get("discovery") or {}
    if isinstance(discovery, dict):
        ep = str(discovery.get("token_endpoint") or "").strip()
        if ep.startswith("https://") and "x.ai" in ep:
            return ep
    with httpx.Client(timeout=15.0) as client:
        r = client.get(
            XAI_OAUTH_DISCOVERY_URL,
            headers={"Accept": "application/json"},
        )
        r.raise_for_status()
        payload = r.json()
    ep = str(payload.get("token_endpoint") or "").strip()
    if not ep:
        raise XaiOauthError("xAI OIDC discovery missing token_endpoint")
    return ep


def _refresh_tokens(tokens: dict[str, Any], token_endpoint: str) -> dict[str, Any]:
    refresh = str(tokens.get("refresh_token") or "").strip()
    if not refresh:
        raise XaiOauthError(
            "xai-oauth missing refresh_token. Run: hermes auth add xai-oauth",
            relogin_required=True,
        )
    with httpx.Client(timeout=25.0, headers={"Accept": "application/json"}) as client:
        r = client.post(
            token_endpoint,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "refresh_token",
                "client_id": XAI_OAUTH_CLIENT_ID,
                "refresh_token": refresh,
            },
        )
    if r.status_code != 200:
        detail = r.text.strip()
        relogin = r.status_code in (400, 401)
        if r.status_code == 403:
            raise XaiOauthError(
                "xAI OAuth refresh HTTP 403 (tier/entitlement). "
                "Use XAI_API_KEY with SPACEXAI_AUTH=api_key, or upgrade SuperGrok. "
                f"Detail: {detail}",
                relogin_required=False,
            )
        raise XaiOauthError(
            "xAI OAuth refresh failed. "
            + ("Re-login: hermes auth add xai-oauth. " if relogin else "")
            + f"HTTP {r.status_code}: {detail}",
            relogin_required=relogin,
        )
    payload = r.json()
    access = str(payload.get("access_token") or "").strip()
    if not access:
        raise XaiOauthError("Refresh response missing access_token", relogin_required=True)
    updated = dict(tokens)
    updated["access_token"] = access
    if payload.get("refresh_token"):
        updated["refresh_token"] = str(payload["refresh_token"]).strip()
    if payload.get("id_token"):
        updated["id_token"] = str(payload["id_token"]).strip()
    if payload.get("expires_in") is not None:
        updated["expires_in"] = payload["expires_in"]
    if payload.get("token_type"):
        updated["token_type"] = str(payload["token_type"])
    return updated


def _persist_tokens(store: dict[str, Any], tokens: dict[str, Any], token_endpoint: str) -> None:
    providers = store.setdefault("providers", {})
    if not isinstance(providers, dict):
        store["providers"] = {}
        providers = store["providers"]
    state = providers.get(XAI_OAUTH_PROVIDER)
    if not isinstance(state, dict):
        state = {}
        providers[XAI_OAUTH_PROVIDER] = state
    state["tokens"] = tokens
    state["last_refresh"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    state.setdefault("auth_mode", "oauth_device_code")
    discovery = state.get("discovery") if isinstance(state.get("discovery"), dict) else {}
    discovery = dict(discovery)
    discovery["token_endpoint"] = token_endpoint
    state["discovery"] = discovery
    # Clear stale refresh errors on success
    if "last_auth_error" in state:
        del state["last_auth_error"]
    _save_store(store)


def resolve_access_token(*, force_refresh: bool = False) -> str:
    """Return a usable xAI OAuth bearer access token."""
    store = _load_store()
    state = _provider_state(store)
    if not state:
        raise XaiOauthError(
            "No xai-oauth credentials in ~/.hermes/auth.json. "
            "Run: hermes auth add xai-oauth",
            relogin_required=True,
        )
    tokens = state.get("tokens")
    if not isinstance(tokens, dict):
        raise XaiOauthError(
            "xai-oauth state missing tokens. Run: hermes auth add xai-oauth",
            relogin_required=True,
        )
    access = str(tokens.get("access_token") or "").strip()
    if not access:
        raise XaiOauthError(
            "xai-oauth missing access_token. Run: hermes auth add xai-oauth",
            relogin_required=True,
        )

    need_refresh = force_refresh or _is_expiring(access)
    if need_refresh:
        endpoint = _token_endpoint(state)
        tokens = _refresh_tokens(tokens, endpoint)
        _persist_tokens(store, tokens, endpoint)
        access = str(tokens.get("access_token") or "").strip()

    return access


def has_xai_oauth() -> bool:
    try:
        store = _load_store()
        state = _provider_state(store)
        if not state:
            return False
        tokens = state.get("tokens") or {}
        return bool(str(tokens.get("access_token") or "").strip())
    except Exception:
        return False
