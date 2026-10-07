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
from master_agent.comfy.ingest.normalize import IngestError, load_workflow_file

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


def ingest_graph(
    workflow: dict[str, Any],
    *,
    slug: str,
    source: str = "",
    object_info: dict[str, Any] | None = None,
    ui_workflow: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Learn ``workflow`` and store it. Does not queue and does not download."""
    safe = slugify(slug)
    learned = learn_workflow(
        workflow, slug=safe, source=source, object_info=object_info
    )
    provenance = {
        "source_path": source,
        "ingested_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "format": "api",
        "comfy_version": None,
        "object_info_hash": None,
        "workflow_sha256": hashlib.sha256(_canonical(workflow).encode("utf-8")).hexdigest(),
        "phase": "A",
    }
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


def ingest_file(
    path: Path,
    *,
    slug: str | None = None,
    object_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    file_path = Path(path)
    workflow, ui_workflow = load_workflow_file(file_path)
    chosen = slug if slug else file_path.stem
    return ingest_graph(
        workflow,
        slug=chosen,
        source=str(file_path),
        object_info=object_info,
        ui_workflow=ui_workflow,
    )


def refresh_learned(slug: str, object_info: dict[str, Any] | None = None) -> dict[str, Any]:
    """Re-run heuristics on the stored API graph and rewrite learned.yaml."""
    bundle = load_bundle(slug)
    source = str((bundle.get("provenance") or {}).get("source_path") or "")
    learned = learn_workflow(
        bundle["workflow"],
        slug=bundle["slug"],
        source=source,
        object_info=object_info,
    )
    _write_learned(bundle["dir"] / "learned.yaml", learned)
    bundle["learned"] = learned
    return bundle
