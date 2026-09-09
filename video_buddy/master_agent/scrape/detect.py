"""Heuristics for login walls and challenge pages (not a bypass)."""

from __future__ import annotations

CHALLENGE_MARKERS = (
    "you must log in to continue",
    "login_form",
    "log in to facebook",
    "checkpoint",
    "just a moment",
    "cf-challenge",
    "cf-browser-verification",
    "checking your browser",
)


def looks_like_challenge(html: str, status_code: int = 200) -> bool:
    if status_code in (401, 403):
        return True
    blob = (html or "").lower()
    return any(marker in blob for marker in CHALLENGE_MARKERS)
