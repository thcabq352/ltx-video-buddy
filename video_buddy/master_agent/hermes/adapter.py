"""OpenAI-compatible Hermes facade on the studio (port 8189 only)."""

from __future__ import annotations

import json
import uuid
from typing import Any, Callable, Optional

from master_agent.a2a.protocol import DEFAULT_QUALITY, TaskStore
from master_agent.hermes.gateways import LTX_RESEARCH_SYSTEM

SubmitFn = Callable[[str, dict[str, Any]], None]
RequestFn = Callable[..., dict[str, Any]]

PITCH_HINTS = (
    "pitch",
    "3-shot",
    "3 shot",
    "three shot",
    "three-shot",
    "storyboard only",
    "don't render",
    "do not render",
    "no render",
)


def wants_pitch(text: str, metadata: dict[str, Any] | None = None) -> bool:
    meta = metadata or {}
    flag = meta.get("pitch")
    if flag is True or str(flag).strip().lower() in {"1", "true", "yes"}:
        return True
    low = text.lower()
    return any(hint in low for hint in PITCH_HINTS)


def openai_models(*, profile: str = "ltx") -> dict[str, Any]:
    return {
        "object": "list",
        "data": [{"id": profile, "object": "model", "owned_by": "video-buddy"}],
    }


def _user_text(payload: dict[str, Any]) -> str:
    messages = payload.get("messages") or []
    text = ""
    if isinstance(messages, list):
        for item in messages:
            if isinstance(item, dict) and item.get("role") == "user":
                text = str(item.get("content") or "")
    if not text.strip():
        text = str(payload.get("prompt") or payload.get("brief") or payload.get("request") or "")
    return text.strip()


def _chat_completion(content: str, *, model: str = "ltx", extra: dict[str, Any] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }
    if extra:
        out.update(extra)
    return out


def hermes_complete(
    payload: dict[str, Any],
    *,
    store: TaskStore,
    submit: SubmitFn,
    token: str = "",
    request: Optional[RequestFn] = None,
) -> dict[str, Any]:
    """Chat-completions entry. Pitch only on explicit ask or metadata.pitch=true."""
    text = _user_text(payload)
    if not text:
        return {"error": {"message": "empty message", "type": "invalid_request_error"}}
    meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    if wants_pitch(text, meta):
        from master_agent.hermes.pitch import hermes_pitch

        out = hermes_pitch(
            {"brief": text, "make_video": False},
            token=token or "local",
            request=request,
            make_video=False,
        )
        body = {k: out.get(k) for k in ("ok", "title", "logline", "shots", "brief") if k in out}
        content = json.dumps(body, ensure_ascii=False)
        return _chat_completion(
            content,
            extra={"pitch": out, "system": LTX_RESEARCH_SYSTEM},
        )

    body = {
        "request": text,
        "quality": meta.get("quality") or payload.get("quality") or DEFAULT_QUALITY,
        "dry_run": bool(meta.get("dry_run") or payload.get("dry_run")),
        "variant": meta.get("variant") or payload.get("variant"),
        "duration_s": meta.get("duration_s") or payload.get("duration_s") or 5.0,
    }
    task_id = store.create(text)
    submit(task_id, body)
    content = (
        f"accepted task {task_id} state=working. "
        f"Poll A2A tasks/get on /a2a or GET /tasks/{task_id}."
    )
    return _chat_completion(
        content,
        extra={
            "buddy": {
                "task_id": task_id,
                "state": "working",
                "poll_a2a": {"method": "tasks/get", "url": "/a2a"},
            }
        },
    )
