"""hermes_pitch + PITCH_SYSTEM. Ported from SOS/hermes_studio.py (logic only)."""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Optional

RequestFn = Callable[..., dict[str, Any]]

PITCH_SYSTEM = (
    "You are a film brief director. Reply with JSON only, no markdown: "
    '{"title":"...","logline":"...","shots":[{"title":"...","prompt":"...","motion":"..."}]} '
    "with exactly 3 cinematic shots. Prompts must be visual stills."
)


def _xai_request(
    method: str,
    path: str,
    *,
    token: str,
    json_body: Optional[dict[str, Any]] = None,
    request: Optional[RequestFn] = None,
) -> dict[str, Any]:
    if request:
        return request(method, path, headers={"Authorization": f"Bearer {token}"}, json=json_body)
    import httpx

    from master_agent.config import XAI_BASE_URL

    url = path if path.startswith("http") else f"{XAI_BASE_URL}{path}"
    with httpx.Client(timeout=120.0) as client:
        response = client.request(
            method,
            url,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=json_body,
        )
    try:
        body = response.json()
    except Exception:
        body = {}
    return {"ok": response.status_code < 400, "status": response.status_code, "body": body, "error": ""}


def hermes_chat(
    payload: dict[str, Any],
    *,
    token: str,
    request: Optional[RequestFn] = None,
) -> dict[str, Any]:
    model = str(payload.get("model") or "grok-4")
    prompt = str(payload.get("prompt") or "")
    system = str(payload.get("system") or PITCH_SYSTEM)
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }
    reply = _xai_request("POST", "/chat/completions", token=token, json_body=body, request=request)
    data = reply.get("body") if isinstance(reply.get("body"), dict) else {}
    text = ""
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(message, dict):
            text = str(message.get("content") or "").strip()
    return {"ok": bool(text), "text": text, "model": model}


def hermes_pitch(
    payload: dict[str, Any],
    *,
    token: str,
    request: Optional[RequestFn] = None,
    make_video: bool | None = None,
) -> dict[str, Any]:
    brief = str(payload.get("brief") or payload.get("prompt") or "").strip()
    if not brief:
        return {"ok": False, "error": "brief is required"}
    if make_video is None:
        make_video = bool(payload.get("make_video", True))
    chat = hermes_chat(
        {
            "model": str(payload.get("model") or "grok-4"),
            "prompt": f"Client brief:\n{brief}\nReturn the JSON storyboard now.",
            "system": PITCH_SYSTEM,
        },
        token=token,
        request=request,
    )
    shots: list[dict[str, Any]] = []
    title = "Pitch"
    logline = ""
    raw = str(chat.get("text") or "")
    match = re.search(r"\{.*\}", raw, re.S)
    if match:
        try:
            parsed = json.loads(match.group(0))
            title = str(parsed.get("title") or title)
            logline = str(parsed.get("logline") or "")
            for item in parsed.get("shots") or []:
                if isinstance(item, dict) and item.get("prompt"):
                    shots.append(
                        {
                            "title": str(item.get("title") or f"Shot {len(shots) + 1}"),
                            "prompt": str(item.get("prompt")),
                            "motion": str(item.get("motion") or ""),
                        }
                    )
        except json.JSONDecodeError:
            shots = []
    if not shots:
        shots = [
            {"title": "Open", "prompt": brief, "motion": "slow push in"},
            {"title": "Turn", "prompt": f"{brief}, tighter detail", "motion": "drift left"},
            {"title": "Close", "prompt": f"{brief}, wide hold", "motion": "hold"},
        ]
    return {
        "ok": True,
        "title": title,
        "logline": logline,
        "brief": brief,
        "shots": shots[:3],
        "stills": [],
        "video": None if not make_video else None,
        "story": chat.get("text") or "",
    }
