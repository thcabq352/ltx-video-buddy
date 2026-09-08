"""A2A JSON-RPC + task lifecycle. Logic ported from ltx_director/a2a_server.py."""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

from master_agent import __version__

SubmitFn = Callable[[str, dict[str, Any]], None]


class TaskStore:
    def __init__(self) -> None:
        self._tasks: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create(self, request: str = "") -> str:
        task_id = uuid.uuid4().hex[:12]
        self.set(task_id, state="working", request=request)
        return task_id

    def set(self, task_id: str, **fields: Any) -> dict[str, Any]:
        with self._lock:
            t = self._tasks.setdefault(
                task_id,
                {"id": task_id, "state": "working", "created": datetime.now(timezone.utc).isoformat()},
            )
            t.update(fields)
            t["updated"] = datetime.now(timezone.utc).isoformat()
            return dict(t)

    def get(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            t = self._tasks.get(str(task_id))
            return dict(t) if t else None


STUDIO_PORT = 8189

# Pipeline status → A2A task state. Budget hold is not a failure.
_COMPLETED = frozenset({"completed", "done", "workflow_ready"})
_HELD = frozenset({"paused", "held"})
_WORKING = frozenset({"running", "queued", "working"})


def a2a_task_state(status: str | None) -> str:
    key = (status or "").strip().lower()
    if key in _COMPLETED:
        return "completed"
    if key in _HELD:
        return "input-required"
    if key in _WORKING:
        return "working"
    return "failed"


def agent_card(*, host: str = "127.0.0.1", port: int = STUDIO_PORT) -> dict[str, Any]:
    base = f"http://{host}:{port}/"
    return {
        "name": "video-buddy",
        "description": "Local Video Buddy director via ComfyUI (storyboard, judge, multi-seg).",
        "url": base,
        "version": __version__,
        "protocolVersion": "0.2.9",
        "capabilities": {"streaming": False, "pushNotifications": False},
        "defaultInputModes": ["text"],
        "defaultOutputModes": ["text", "video/mp4"],
        "skills": [
            {
                "id": "generate_video",
                "name": "Generate LTX video",
                "description": "A2A message/send with a video brief, or MCP create_video.",
                "tags": ["ltx", "comfyui", "video"],
            }
        ],
        "endpoints": {
            "health": "/health",
            "card": "/.well-known/agent.json",
            "tasks": "/tasks/{id}",
        },
    }


def _message_text(params: dict[str, Any]) -> str:
    message = params.get("message") or params
    parts = message.get("parts") if isinstance(message, dict) else None
    text = ""
    if isinstance(parts, list):
        for p in parts:
            if isinstance(p, dict) and p.get("type") in ("text", None) and p.get("text"):
                text += str(p["text"]) + "\n"
    if not text and isinstance(message, dict):
        text = str(message.get("text") or message.get("request") or "")
    if not text.strip():
        text = str(params.get("request") or params.get("text") or "")
    return text.strip()


def handle_rpc(
    payload: dict[str, Any],
    *,
    store: TaskStore,
    submit: SubmitFn,
    host: str = "127.0.0.1",
    port: int = STUDIO_PORT,
) -> dict[str, Any]:
    method = payload.get("method") or ""
    req_id = payload.get("id")
    params = payload.get("params") or {}

    def ok(result: Any) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    def err(code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}

    if method in ("message/send", "tasks/send"):
        text = _message_text(params if isinstance(params, dict) else {})
        if not text:
            return err(-32602, "empty message text")
        meta = params.get("metadata") or {}
        body = {
            "request": text,
            "quality": meta.get("quality") or params.get("quality") or "balanced",
            "dry_run": bool(meta.get("dry_run") or params.get("dry_run")),
            "variant": meta.get("variant") or params.get("variant"),
        }
        task_id = store.create(text)
        submit(task_id, body)
        return ok({"id": task_id, "contextId": task_id, "status": {"state": "working"}, "kind": "task"})

    if method == "tasks/get":
        tid = params.get("id") or params.get("task_id")
        t = store.get(str(tid) if tid else "")
        if not t:
            return err(-32001, f"task not found: {tid}")
        state = t.get("state") or "working"
        result = t.get("result") or {}
        artifacts = []
        if result.get("video_path"):
            artifacts.append({"name": "video", "parts": [{"type": "text", "text": result["video_path"]}]})
        return ok(
            {
                "id": tid,
                "status": {
                    "state": state,
                    "message": {"parts": [{"type": "text", "text": result.get("error") or state}]},
                },
                "artifacts": artifacts,
                "metadata": {"result": result},
            }
        )

    if method in ("agent/authenticatedExtendedCard", "agent/getAuthenticatedExtendedCard"):
        return ok(agent_card(host=host, port=port))

    return err(-32601, f"method not found: {method}")


def submit_orchestrator(task_id: str, body: dict[str, Any], store: TaskStore) -> None:
    """Wire A2A tasks to the existing orchestrator pipeline (async thread)."""

    def _run() -> None:
        try:
            from master_agent.orchestrator.pipeline import dry_run_pipeline, run_pipeline

            if body.get("dry_run"):
                code = dry_run_pipeline(
                    body["request"],
                    variant=body.get("variant"),
                    quality=body.get("quality"),
                )
                store.set(task_id, state="completed" if code == 0 else "failed", result={"status": "dry-run", "code": code})
                return
            result = run_pipeline(
                body["request"],
                quality=body.get("quality") or "balanced",
                variant=body.get("variant"),
            )
            status = result.status if hasattr(result, "status") else (result or {}).get("status")
            payload = result.to_dict() if hasattr(result, "to_dict") else dict(result or {})
            store.set(task_id, state=a2a_task_state(status), result=payload)
        except Exception as exc:
            store.set(task_id, state="failed", error=str(exc), result={"error": str(exc)})

    threading.Thread(target=_run, daemon=True).start()
