"""ComfyUI HTTP API client (queue, poll, free VRAM, uploads, object_info cache)."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import httpx

from master_agent.config import (
    COMFYUI_OUTPUT_DIR,
    COMFYUI_URL,
    JOB_TIMEOUT_S,
    OBJECT_INFO_CACHE,
    POLL_INTERVAL_S,
)


class ComfyClientError(RuntimeError):
    pass


class ComfyClient:
    def __init__(self, base_url: str = COMFYUI_URL, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client_id = str(uuid.uuid4())

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def health(self) -> dict[str, Any]:
        try:
            with httpx.Client(timeout=10.0) as client:
                r = client.get(self._url("/system_stats"))
                r.raise_for_status()
                return r.json()
        except Exception as e:
            raise ComfyClientError(
                f"ComfyUI is not reachable at {self.base_url}. "
                f"Start it with: ComfyUI_windows_portable\\run_api_8188.bat  ({e})"
            ) from e

    def is_up(self) -> bool:
        try:
            self.health()
            return True
        except ComfyClientError:
            return False

    def free_memory(self, unload_models: bool = True, free_memory: bool = True) -> None:
        payload = {"unload_models": unload_models, "free_memory": free_memory}
        try:
            with httpx.Client(timeout=self.timeout) as client:
                r = client.post(self._url("/free"), json=payload)
                # Some builds return 200 empty; ignore 404 if endpoint missing
                if r.status_code not in (200, 204, 404):
                    r.raise_for_status()
        except httpx.HTTPError as e:
            # Non-fatal: VRAM free is best-effort
            print(f"[comfy] /free warning: {e}")

    def queue_prompt(self, workflow: dict[str, Any]) -> str:
        """Submit API-format workflow (node_id → node dict). Returns prompt_id."""
        body = {"prompt": workflow, "client_id": self.client_id}
        with httpx.Client(timeout=self.timeout) as client:
            r = client.post(self._url("/prompt"), json=body)
            if r.status_code >= 400:
                detail = r.text
                try:
                    detail = json.dumps(r.json(), indent=2)
                except Exception:
                    pass
                raise ComfyClientError(f"ComfyUI /prompt failed ({r.status_code}): {detail}")
            data = r.json()
        if "error" in data:
            raise ComfyClientError(f"ComfyUI rejected prompt: {data['error']}")
        prompt_id = data.get("prompt_id")
        if not prompt_id:
            raise ComfyClientError(f"No prompt_id in response: {data}")
        return prompt_id

    def get_history(self, prompt_id: str) -> dict[str, Any]:
        with httpx.Client(timeout=self.timeout) as client:
            r = client.get(self._url(f"/history/{prompt_id}"))
            r.raise_for_status()
            return r.json()

    def get_queue(self) -> dict[str, Any]:
        with httpx.Client(timeout=self.timeout) as client:
            r = client.get(self._url("/queue"))
            r.raise_for_status()
            return r.json()

    def wait_for_prompt(
        self,
        prompt_id: str,
        *,
        poll_interval: float = POLL_INTERVAL_S,
        timeout: float = JOB_TIMEOUT_S,
    ) -> dict[str, Any]:
        """Poll history until the prompt finishes. Returns history entry."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            hist = self.get_history(prompt_id)
            if prompt_id in hist:
                entry = hist[prompt_id]
                status = entry.get("status") or {}
                # completed successfully
                if entry.get("outputs"):
                    if status.get("status_str") == "error" or status.get("completed") is False:
                        msgs = status.get("messages") or []
                        raise ComfyClientError(f"ComfyUI job error: {msgs}")
                    return entry
                if status.get("status_str") == "error":
                    msgs = status.get("messages") or []
                    raise ComfyClientError(f"ComfyUI job error: {msgs}")
            time.sleep(poll_interval)
        raise ComfyClientError(
            f"Timed out after {timeout}s waiting for prompt_id={prompt_id}"
        )

    def upload_image(self, path: Path, image_type: str = "input", overwrite: bool = True) -> str:
        path = Path(path)
        if not path.is_file():
            raise ComfyClientError(f"Image not found: {path}")
        with path.open("rb") as f:
            files = {"image": (path.name, f, "application/octet-stream")}
            data = {"type": image_type, "overwrite": str(overwrite).lower()}
            with httpx.Client(timeout=120.0) as client:
                r = client.post(self._url("/upload/image"), files=files, data=data)
                r.raise_for_status()
                out = r.json()
        return out.get("name") or path.name

    def upload_audio(self, path: Path, overwrite: bool = True) -> str:
        """Upload audio via generic upload if available; else copy name only."""
        path = Path(path)
        if not path.is_file():
            raise ComfyClientError(f"Audio not found: {path}")
        # Prefer /upload/image-style endpoint used by many builds for media
        with path.open("rb") as f:
            files = {"image": (path.name, f, "application/octet-stream")}
            data = {"type": "input", "overwrite": str(overwrite).lower()}
            with httpx.Client(timeout=120.0) as client:
                r = client.post(self._url("/upload/image"), files=files, data=data)
                if r.status_code >= 400:
                    # Fallback: return basename; user must place file in ComfyUI/input
                    return path.name
                out = r.json()
        return out.get("name") or path.name

    def fetch_object_info(self) -> dict[str, Any]:
        """GET /object_info — the live node registry used by the validator."""
        with httpx.Client(timeout=self.timeout) as client:
            r = client.get(self._url("/object_info"))
            r.raise_for_status()
            return r.json()

    def refresh_object_info_cache(
        self, cache_path: Path = OBJECT_INFO_CACHE
    ) -> dict[str, Any]:
        """Fetch /object_info from the running server and persist it to disk."""
        info = self.fetch_object_info()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(info, indent=1), encoding="utf-8")
        return info

    def load_object_info(
        self, cache_path: Path = OBJECT_INFO_CACHE, *, prefer_live: bool = True
    ) -> tuple[dict[str, Any], str]:
        """
        Returns (object_info, source) where source is 'live' or 'cache'.
        Falls back to the on-disk cache when ComfyUI is not running.
        """
        if prefer_live:
            try:
                return self.fetch_object_info(), "live"
            except Exception:
                pass
        if cache_path.is_file():
            return json.loads(cache_path.read_text(encoding="utf-8")), "cache"
        raise ComfyClientError(
            "No /object_info available: ComfyUI is down and no cache exists. "
            "Start ComfyUI and run: python -m master_agent fetch-object-info"
        )

    @staticmethod
    def extract_video_files(history_entry: dict[str, Any]) -> list[dict[str, str]]:
        """Pull video/gifs/files from history outputs."""
        results: list[dict[str, str]] = []
        outputs = history_entry.get("outputs") or {}
        for _node_id, node_out in outputs.items():
            for key in ("gifs", "videos", "images", "files"):
                items = node_out.get(key) or []
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    filename = item.get("filename")
                    if not filename:
                        continue
                    # Prefer video extensions
                    lower = filename.lower()
                    if key in ("gifs", "videos") or lower.endswith(
                        (".mp4", ".webm", ".mkv", ".avi", ".mov", ".gif")
                    ):
                        results.append(
                            {
                                "filename": filename,
                                "subfolder": item.get("subfolder") or "",
                                "type": item.get("type") or "output",
                            }
                        )
        # If only images returned (frame sequence), still surface last image
        if not results:
            for _node_id, node_out in outputs.items():
                for item in node_out.get("images") or []:
                    if isinstance(item, dict) and item.get("filename"):
                        results.append(
                            {
                                "filename": item["filename"],
                                "subfolder": item.get("subfolder") or "",
                                "type": item.get("type") or "output",
                            }
                        )
        return results

    @staticmethod
    def resolve_output_path(
        file_info: dict[str, str],
        output_dir: Path = COMFYUI_OUTPUT_DIR,
    ) -> Path:
        sub = file_info.get("subfolder") or ""
        name = file_info["filename"]
        # type can be output/temp/input
        base = output_dir
        if file_info.get("type") == "temp":
            base = output_dir.parent / "temp"
        path = base / sub / name if sub else base / name
        return path.resolve()
