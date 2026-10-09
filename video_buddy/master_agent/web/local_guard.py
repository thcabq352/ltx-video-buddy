"""Keep the studio a loopback-only service.

The studio has no auth. ``LocalOnlyGuard`` rejects requests whose ``Host``
is not a loopback name (DNS rebinding) and requests whose ``Origin`` is
not a loopback page (cross-site POSTs from a browser tab).
"""

from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit

LOOPBACK_NAMES = frozenset({"localhost", "127.0.0.1", "::1"})


def is_loopback_host(host: str) -> bool:
    name = (host or "").strip().strip("[]").lower()
    if name in LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


def _hostname(netloc: str) -> str:
    try:
        return urlsplit(f"//{netloc}").hostname or ""
    except ValueError:
        return ""


class LocalOnlyGuard:
    """Pure ASGI middleware: 403 for non-loopback Host or Origin."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers") or []}
        host = _hostname(headers.get("host", ""))
        origin = headers.get("origin")
        bad = None
        if not is_loopback_host(host):
            bad = "host"
        elif origin and origin != "null" and not is_loopback_host(urlsplit(origin).hostname or ""):
            bad = "origin"
        if bad is None:
            await self.app(scope, receive, send)
            return
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        body = f'{{"detail":"studio is loopback-only ({bad} refused)"}}'.encode()
        await send(
            {
                "type": "http.response.start",
                "status": 403,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
            }
        )
        await send({"type": "http.response.body", "body": body})
