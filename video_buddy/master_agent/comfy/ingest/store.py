"""Read and write ``state/ingested/<slug>/``. Nothing here is committed."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from master_agent.comfy.ingest.learn import learn_workflow
from master_agent.comfy.ingest.normalize import (
    START_COMFY_OR_API,
    IngestError,
    graph_from_history_payload,
    load_workflow_file,
)
from master_agent.comfy.ingest.validate import buddy_model_inventory, union_inventories

_SLUG = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def _state_dir() -> Path:
    from master_agent import config

    return Path(config.STATE_DIR)


def ingested_root() -> Path:
    return _state_dir() / "ingested"


def slugify(raw: str) -> str:
    text = str(raw or "").strip().lower()
    text = text.replace("\\", "-").replace("/", "-")
    text = re.sub(r"[^a-z0-9_-]+", "-", text)
    text = text.strip("-_")
    if not text or not _SLUG.match(text) or ".." in text:
        raise IngestError(f"invalid slug {raw!r}")
    return text


def bundle_dir(slug: str) -> Path:
    safe = slugify(slug)
    root = ingested_root().resolve()
    dest = (root / safe).resolve()
    try:
        dest.relative_to(root)
    except ValueError as exc:
        raise IngestError(f"invalid slug {slug!r}") from exc
    return dest


def _canonical(workflow: dict[str, Any]) -> str:
    return json.dumps(workflow, sort_keys=True, separators=(",", ":"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_learned(path: Path, learned: dict[str, Any]) -> None:
    path.write_text(
        yaml.safe_dump(learned, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def write_bundle(
    slug: str,
    workflow: dict[str, Any],
    learned: dict[str, Any],
    provenance: dict[str, Any],
    *,
    ui_workflow: dict[str, Any] | None = None,
) -> Path:
    dest = bundle_dir(slug)
    dest.mkdir(parents=True, exist_ok=True)
    _write_json(dest / "workflow_api.json", workflow)
    _write_learned(dest / "learned.yaml", learned)
    _write_json(dest / "provenance.json", provenance)
    ui_path = dest / "workflow_ui.json"
    if ui_workflow is not None:
        _write_json(ui_path, ui_workflow)
    elif ui_path.exists():
        # API re-ingest must not leave a stale UI export behind.
        ui_path.unlink()
    return dest


def load_bundle(slug: str) -> dict[str, Any]:
    dest = bundle_dir(slug)
    api_path = dest / "workflow_api.json"
    learned_path = dest / "learned.yaml"
    provenance_path = dest / "provenance.json"
    if not api_path.is_file() or not learned_path.is_file():
        raise IngestError(
            f"no ingested workflow {slug!r} under {ingested_root()}. "
            "Run: python -m master_agent comfy ingest PATH.json --slug "
            + slug
        )
    workflow = json.loads(api_path.read_text(encoding="utf-8"))
    learned = yaml.safe_load(learned_path.read_text(encoding="utf-8")) or {}
    provenance: dict[str, Any] = {}
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    return {
        "slug": slugify(slug),
        "dir": dest,
        "workflow": workflow,
        "learned": learned if isinstance(learned, dict) else {},
        "provenance": provenance,
    }


def _object_info_hash(object_info: dict[str, Any] | None) -> str | None:
    if not object_info:
        return None
    raw = json.dumps(object_info, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def ingest_graph(
    workflow: dict[str, Any],
    *,
    slug: str,
    source: str = "",
    object_info: dict[str, Any] | None = None,
    ui_workflow: dict[str, Any] | None = None,
    model_inventory: set[str] | None = None,
    provenance_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Learn ``workflow`` and store it. Does not queue and does not download."""
    safe = slugify(slug)
    learned = learn_workflow(
        workflow,
        slug=safe,
        source=source,
        object_info=object_info,
        model_inventory=model_inventory,
    )
    provenance = {
        "source_path": source,
        "ingested_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "format": "api",
        "comfy_version": None,
        "object_info_hash": _object_info_hash(object_info),
        "workflow_sha256": hashlib.sha256(_canonical(workflow).encode("utf-8")).hexdigest(),
        "phase": "A",
    }
    if provenance_extra:
        provenance.update(provenance_extra)
    dest = write_bundle(
        safe, workflow, learned, provenance, ui_workflow=ui_workflow
    )
    return {
        "slug": safe,
        "dir": dest,
        "learned": learned,
        "provenance": provenance,
        "queued": False,
        "next": f"python -m master_agent comfy dry-run {safe}",
    }


def _comfy_model_names(client: Any) -> set[str] | None:
    if client is None or not hasattr(client, "list_model_filenames"):
        return None
    try:
        names = client.list_model_filenames()
    except Exception:
        return None
    if not isinstance(names, list):
        return None
    return {str(name) for name in names if str(name).strip()}


def _merge_inventory(
    model_inventory: set[str] | None,
    client: Any,
) -> set[str] | None:
    return union_inventories(model_inventory, buddy_model_inventory(), _comfy_model_names(client))


def ingest_file(
    path: Path,
    *,
    slug: str | None = None,
    object_info: dict[str, Any] | None = None,
    converter: Any = None,
    client: Any = None,
    model_inventory: set[str] | None = None,
) -> dict[str, Any]:
    """Ingest a local JSON file.

    UI-format JSON uses ``converter`` or ``client.convert_workflow`` (Comfy
    ``/workflow/convert``). When neither is available the call refuses with
    ``start Comfy or supply API JSON``. Both ``workflow_ui.json`` and
    ``workflow_api.json`` are kept after a conversion.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise IngestError(f"workflow not found: {file_path}")
    live = client
    used_convert = False

    def _convert(ui: dict[str, Any]) -> dict[str, Any]:
        nonlocal used_convert
        used_convert = True
        if converter is not None:
            return converter(ui)
        if live is not None and hasattr(live, "convert_workflow"):
            return live.convert_workflow(ui)
        raise IngestError(
            f"{START_COMFY_OR_API}. "
            "This file is UI-format JSON and Comfy /workflow/convert was not called."
        )

    # Peek so API files do not construct a converter call.
    try:
        raw = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise IngestError(f"workflow is not JSON: {file_path}") from exc
    from master_agent.comfy.ingest.normalize import is_ui_workflow

    convert = _convert if is_ui_workflow(raw) else None
    workflow, ui_workflow = load_workflow_file(file_path, converter=convert)
    info = object_info
    if ui_workflow is not None and info is None and live is not None and hasattr(live, "fetch_object_info"):
        try:
            info = live.fetch_object_info()
        except Exception:
            info = None
    extra: dict[str, Any] | None = None
    if ui_workflow is not None:
        extra = {"phase": "B", "source_format": "ui", "converted_via": "/workflow/convert"}
    chosen = slug if slug else file_path.stem
    return ingest_graph(
        workflow,
        slug=chosen,
        source=str(file_path),
        object_info=info,
        ui_workflow=ui_workflow,
        model_inventory=_merge_inventory(model_inventory, live if used_convert else None),
        provenance_extra=extra,
    )


def ingest_history(
    prompt_id: str,
    *,
    slug: str | None = None,
    client: Any = None,
    object_info: dict[str, Any] | None = None,
    model_inventory: set[str] | None = None,
) -> dict[str, Any]:
    """Ingest the queued graph from Comfy ``/history/<prompt_id>``.

    ``client`` defaults to ``ComfyClient``. Tests pass a fake. A down server
    refuses with ``start Comfy or supply API JSON``.
    """
    token = str(prompt_id or "").strip()
    if not token:
        raise IngestError("history ingest requires a prompt_id (history:PROMPT_ID).")
    live = client
    if live is None:
        from master_agent.comfy.client import ComfyClient

        live = ComfyClient()
    try:
        payload = live.get_history(token)
    except IngestError:
        raise
    except Exception as exc:
        raise IngestError(
            f"{START_COMFY_OR_API}. Comfy /history/{token} failed: {exc}"
        ) from exc
    workflow = graph_from_history_payload(payload, token)
    info = object_info
    if info is None and hasattr(live, "fetch_object_info"):
        try:
            info = live.fetch_object_info()
        except Exception:
            info = None
    version = None
    if hasattr(live, "health"):
        try:
            stats = live.health() or {}
            system = stats.get("system") if isinstance(stats, dict) else None
            if isinstance(system, dict):
                version = system.get("comfyui_version") or system.get("version")
        except Exception:
            version = None
    chosen = slug if slug else f"history-{token[:8]}"
    return ingest_graph(
        workflow,
        slug=chosen,
        source=f"history:{token}",
        object_info=info,
        model_inventory=_merge_inventory(model_inventory, live),
        provenance_extra={
            "phase": "B",
            "source_format": "history",
            "prompt_id": token,
            "comfy_version": version,
        },
    )


def refresh_learned(
    slug: str,
    object_info: dict[str, Any] | None = None,
    model_inventory: set[str] | None = None,
) -> dict[str, Any]:
    """Re-run heuristics on the stored API graph and rewrite learned.yaml."""
    bundle = load_bundle(slug)
    source = str((bundle.get("provenance") or {}).get("source_path") or "")
    inventory = model_inventory if model_inventory is not None else buddy_model_inventory()
    learned = learn_workflow(
        bundle["workflow"],
        slug=bundle["slug"],
        source=source,
        object_info=object_info,
        model_inventory=inventory,
    )
    _write_learned(bundle["dir"] / "learned.yaml", learned)
    bundle["learned"] = learned
    return bundle
