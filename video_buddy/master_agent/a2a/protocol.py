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
DEFAULT_QUALITY = "draft"

# Pipeline status → A2A task state. Budget hold is not a failure.
# done_with_warnings is a soft success (CLI treats it as OK). started is in-flight.
_COMPLETED = frozenset({"completed", "done", "workflow_ready", "done_with_warnings"})
_HELD = frozenset({"paused", "held", "hold", "input-required"})
_WORKING = frozenset({"running", "queued", "working", "started"})


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
    base = f"http://{host}:{port}"
    return {
        "name": "video-buddy",
        "description": "Local Video Buddy director via ComfyUI (storyboard, judge, multi-seg).",
        "url": f"{base}/a2a",
        "version": __version__,
        "protocolVersion": "0.2.9",
        "capabilities": {"streaming": False, "pushNotifications": False},
        "defaultInputModes": [
            "text",
            "image/png",
            "image/jpeg",
            "image/webp",
            "audio/wav",
            "audio/mpeg",
        ],
        "defaultOutputModes": ["text", "video/mp4"],
        "skills": [
            {
                "id": "generate_video",
                "name": "Generate LTX video",
                "description": (
                    "A2A message/send with a video brief, or MCP create_video. "
                    "Photo + voice defaults to ltx25_a2v. "
                    "H3 speaks your line in the voice of your 2-12 s sample and animates the mouth to it (coarse sync). "
                    "For tight lip-sync to an exact recording, use ltx25_a2v."
                ),
                "tags": ["ltx", "comfyui", "video"],
            }
        ],
        "endpoints": {
            "health": "/health",
            "card": "/.well-known/agent.json",
            "card_v1": "/.well-known/agent-card.json",
            "tasks": "/tasks/{id}",
        },
    }


def task_view(task: dict[str, Any], tid: str | None = None) -> dict[str, Any]:
    """Public A2A task shape used by tasks/get and GET /tasks/{id}."""
    state = task.get("state") or "working"
    result = task.get("result") or {}
    artifacts = []
    if result.get("video_path"):
        artifacts.append({"name": "video", "parts": [{"type": "text", "text": result["video_path"]}]})
    return {
        "id": tid or task.get("id"),
        "status": {
            "state": state,
            "message": {
                "parts": [
                    {
                        "type": "text",
                        "text": (
                            result.get("error")
                            or task.get("error")
                            or result.get("warning")
                            or result.get("notes")
                            or state
                        ),
                    }
                ]
            },
        },
        "artifacts": artifacts,
        "metadata": {"result": result},
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


def _message_media(params: dict[str, Any]) -> dict[str, str | None]:
    """Paths from metadata or file parts (uri/path + mimeType)."""
    meta = params.get("metadata") if isinstance(params.get("metadata"), dict) else {}
    message = params.get("message") or params
    out: dict[str, str | None] = {
        "image_path": meta.get("image_path") or None,
        "audio_path": meta.get("audio_path") or None,
        "video_path": meta.get("video_path") or None,
    }
    parts = message.get("parts") if isinstance(message, dict) else None
    if isinstance(parts, list):
        for part in parts:
            if not isinstance(part, dict):
                continue
            file = part.get("file") if isinstance(part.get("file"), dict) else {}
            uri = file.get("uri") or file.get("path") or part.get("uri") or part.get("path")
            mime = str(file.get("mimeType") or part.get("mimeType") or part.get("mime") or "")
            if not uri:
                continue
            if mime.startswith("image/") and not out["image_path"]:
                out["image_path"] = str(uri)
            elif mime.startswith("audio/") and not out["audio_path"]:
                out["audio_path"] = str(uri)
            elif mime.startswith("video/") and not out["video_path"]:
                out["video_path"] = str(uri)
    return out


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
        if not isinstance(meta, dict):
            meta = {}
        media = _message_media(params if isinstance(params, dict) else {})
        explicit_duration = isinstance(meta, dict) and "duration_s" in meta or (
            isinstance(params, dict) and "duration_s" in params
        )
        if explicit_duration:
            duration_s = meta.get("duration_s") if "duration_s" in meta else params.get("duration_s")
        elif media["image_path"] and media["audio_path"] and not media["video_path"]:
            duration_s = None
        else:
            duration_s = 5.0
        body = {
            "request": text,
            "quality": meta.get("quality") or params.get("quality") or DEFAULT_QUALITY,
            "dry_run": bool(meta.get("dry_run") or params.get("dry_run")),
            "variant": meta.get("variant") or params.get("variant"),
            "duration_s": duration_s,
            "image_path": media["image_path"],
            "audio_path": media["audio_path"],
            "video_path": media["video_path"],
            "spoken_line": (
                meta.get("line")
                or meta.get("dialogue")
                or meta.get("spoken_line")
                or params.get("line")
                or params.get("dialogue")
            ),
        }
        task_id = store.create(text)
        submit(task_id, body)
        result: dict[str, Any] = {
            "id": task_id,
            "contextId": task_id,
            "status": {"state": "working"},
            "kind": "task",
        }
        from master_agent.orchestrator.talking import h3_r2v_audio_warning

        voice_warn = h3_r2v_audio_warning(
            text,
            variant=body.get("variant"),
            has_image=bool(body.get("image_path")),
            has_audio=bool(body.get("audio_path")),
            has_video=bool(body.get("video_path")),
        )
        if voice_warn:
            result["warning"] = voice_warn
            result["notes"] = voice_warn
            from master_agent.orchestrator.h3_voice import (
                h3_missing_line_warning,
                resolve_spoken_line,
            )

            spoken = resolve_spoken_line(body.get("spoken_line"), text, None)
            missing = h3_missing_line_warning(spoken, h3_voice=True)
            if spoken:
                result["spoken_line"] = spoken
            if missing:
                result["line_warning"] = missing
        return ok(result)

    if method == "tasks/get":
        tid = params.get("id") or params.get("task_id")
        t = store.get(str(tid) if tid else "")
        if not t:
            return err(-32001, f"task not found: {tid}")
        return ok(task_view(t, str(tid)))

    if method in ("agent/authenticatedExtendedCard", "agent/getAuthenticatedExtendedCard"):
        return ok(agent_card(host=host, port=port))

    return err(-32601, f"method not found: {method}")


def submit_orchestrator(task_id: str, body: dict[str, Any], store: TaskStore) -> None:
    """Wire A2A / Hermes facade tasks through the studio one-GPU-at-a-time gate."""

    def _run() -> None:
        try:
            from master_agent.orchestrator.pipeline import dry_run_pipeline, run_pipeline
            from master_agent.web.jobs import MANAGER

            from pathlib import Path

            quality = body.get("quality") or DEFAULT_QUALITY
            image_path = body.get("image_path")
            audio_path = body.get("audio_path")
            video_path = body.get("video_path")
            duration_s = body.get("duration_s")
            if duration_s is None and image_path and audio_path and not video_path:
                from master_agent.orchestrator.talking import duration_following_audio

                duration_s, _note = duration_following_audio(str(audio_path))
            duration_s = float(duration_s or 5.0)

            def _name(path: str | None) -> str | None:
                return Path(path).name if path else None

            from master_agent.orchestrator.talking import h3_r2v_audio_warning

            voice_warn = h3_r2v_audio_warning(
                body.get("request"),
                variant=body.get("variant"),
                has_image=bool(image_path),
                has_audio=bool(audio_path),
                has_video=bool(video_path),
            )

            if body.get("dry_run"):
                code = dry_run_pipeline(
                    body["request"],
                    variant=body.get("variant"),
                    quality=quality,
                    duration_s=duration_s,
                    image_name=_name(image_path),
                    audio_name=_name(audio_path),
                    video_name=_name(video_path),
                    spoken_line=body.get("spoken_line"),
                )
                dry_result: dict[str, Any] = {"status": "dry-run", "code": code}
                if voice_warn:
                    dry_result["warning"] = voice_warn
                    dry_result["notes"] = voice_warn
                store.set(
                    task_id,
                    state="completed" if code == 0 else "failed",
                    result=dry_result,
                )
                return
            def _uploaded(path: str | None, upload) -> str | None:
                if not path:
                    return None
                file = Path(path)
                if file.is_file():
                    return upload(file)
                return file.name

            voice_sample = None
            spoken_line = body.get("spoken_line")
            if image_path and audio_path and not video_path:
                from master_agent.orchestrator.h3_voice import (
                    VoiceSampleError,
                    h3_voice_preflight,
                )
                from master_agent.orchestrator.talking import is_h3_voice_route

                if is_h3_voice_route(
                    body.get("request"),
                    variant=body.get("variant"),
                    has_image=True,
                    has_audio=True,
                    has_video=bool(video_path),
                ):
                    try:
                        pre = h3_voice_preflight(
                            request=str(body.get("request") or ""),
                            variant=body.get("variant"),
                            audio_path=str(audio_path),
                            has_image=True,
                            has_video=bool(video_path),
                            line=spoken_line,
                        )
                    except VoiceSampleError as exc:
                        store.set(
                            task_id,
                            state="failed",
                            error=str(exc),
                            result={"status": "error", "error": str(exc), "warning": voice_warn},
                        )
                        return
                    audio_path = pre.audio_path
                    voice_sample = pre.voice_sample
                    spoken_line = pre.spoken_line or spoken_line
                    if pre.line_warning:
                        voice_warn = voice_warn or pre.line_warning

            with MANAGER.gpu_lock():
                from master_agent.comfy.client import ComfyClient

                client = ComfyClient()
                result = run_pipeline(
                    body["request"],
                    quality=quality,
                    variant=body.get("variant"),
                    duration_s=duration_s,
                    image_name=_uploaded(image_path, client.upload_image),
                    audio_name=_uploaded(audio_path, client.upload_audio),
                    audio_path=audio_path,
                    video_name=_uploaded(video_path, client.upload_image),
                    spoken_line=spoken_line,
                    voice_sample=voice_sample,
                    client=client,
                )
            status = result.status if hasattr(result, "status") else (result or {}).get("status")
            payload = result.to_dict() if hasattr(result, "to_dict") else dict(result or {})
            if voice_warn:
                payload["warning"] = voice_warn
                payload["notes"] = voice_warn
            store.set(task_id, state=a2a_task_state(status), result=payload)
        except Exception as exc:
            store.set(task_id, state="failed", error=str(exc), result={"error": str(exc)})

    threading.Thread(target=_run, daemon=True).start()
