"""Kimi Code CLI OAuth — reuse the CLI's own credential store.

Reads ``~/.kimi-code/credentials/kimi-code.json`` (written by the kimi-code
CLI login), refreshes the access token when expired via
``https://auth.kimi.com/api/oauth/token``, and persists rotated refresh
tokens back to the same file. Tokens are never logged.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

OAUTH_HOST = os.getenv("KIMI_OAUTH_HOST", "https://auth.kimi.com").rstrip("/")
TOKEN_URL = f"{OAUTH_HOST}/api/oauth/token"
# Public client_id of the kimi-code CLI (from its open-source oauth package)
CLIENT_ID = "17e5f671-d194-4dfb-9706-5516cb48c098"
# Refresh when less than this many seconds remain
EXPIRY_MARGIN_S = 60


class KimiOauthError(RuntimeError):
    pass


def credentials_path() -> Path:
    raw = os.getenv("KIMI_OAUTH_FILE") or "~/.kimi-code/credentials/kimi-code.json"
    return Path(os.path.expanduser(raw))


def _load(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise KimiOauthError(
            f"kimi OAuth credentials not found: {path} (log in with the kimi-code CLI once)"
        )
    except (OSError, json.JSONDecodeError) as e:
        raise KimiOauthError(f"kimi OAuth credentials unreadable: {e}")
    if not isinstance(data, dict) or not data.get("access_token"):
        raise KimiOauthError(f"kimi OAuth credentials malformed: {path}")
    return data


def _refresh(path: Path, data: dict) -> dict:
    refresh_token = data.get("refresh_token")
    if not refresh_token:
        raise KimiOauthError("kimi OAuth credentials have no refresh_token; re-login with the CLI")
    body = urllib.parse.urlencode(
        {
            "client_id": CLIENT_ID,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        }
    ).encode()
    req = urllib.request.Request(
        TOKEN_URL,
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            resp = json.loads(r.read())
    except urllib.error.HTTPError as e:
        detail = e.read()[:200]
        raise KimiOauthError(
            f"kimi OAuth refresh failed (HTTP {e.code}): {detail!r} — re-login with the CLI"
        )
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
        raise KimiOauthError(f"kimi OAuth refresh failed: {e}")
    if not isinstance(resp, dict) or not resp.get("access_token"):
        raise KimiOauthError("kimi OAuth refresh returned no access_token")

    new = dict(data)
    for k in ("access_token", "refresh_token", "expires_in", "token_type", "scope"):
        if k in resp:
            new[k] = resp[k]
    new["expires_at"] = int(time.time()) + int(resp.get("expires_in", 900))
    try:
        path.write_text(json.dumps(new, indent=2), encoding="utf-8")
    except OSError as e:
        raise KimiOauthError(f"could not persist refreshed kimi credentials: {e}")
    return new


def resolve_access_token() -> str:
    """Return a valid access token, refreshing (and persisting) if expired."""
    path = credentials_path()
    data = _load(path)
    expires_at = int(data.get("expires_at") or 0)
    if expires_at <= int(time.time()) + EXPIRY_MARGIN_S:
        data = _refresh(path, data)
    return str(data["access_token"])


def token_available() -> bool:
    """Cheap check: credentials file exists, parses, and has an access token."""
    try:
        _load(credentials_path())
        return True
    except KimiOauthError:
        return False
