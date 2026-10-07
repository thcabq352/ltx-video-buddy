"""Patch an ingested graph with the catalog field writer, then vae_guard.

Does not call sulphur, inoutpaint, or lipsync finalize. Does not delete
outputs. Does not download weights or install nodes.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from master_agent.comfy.ingest.classify import missing_class_types, missing_node_packs
from master_agent.comfy.ingest.store import load_bundle
from master_agent.comfy.ingest.validate import (
    DOCTOR_POINTER,
    DOWNLOAD_POINTER,
    MissingCustomNodeError,
    MissingModelError,
    apply_vae_guard,
)
from master_agent.config import snap_ltx_frames

log = logging.getLogger(__name__)


def _widget(workflow: dict[str, Any], spec: dict[str, Any]) -> Any:
    node = workflow.get(str(spec.get("node_id")))
    if not isinstance(node, dict):
        return None
    return (node.get("inputs") or {}).get(spec.get("input"))


def _overrides(
    *,
    prompt: str | None,
    negative_prompt: str | None,
    seed: int | None,
    width: int | None,
    height: int | None,
    frames: int | None,
    vae: str | None,
    filename_prefix: str | None,
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    if isinstance(prompt, str) and prompt.strip():
        values["prompt"] = prompt
    if isinstance(negative_prompt, str) and negative_prompt.strip():
        values["negative_prompt"] = negative_prompt
    if seed is not None:
        values["seed"] = int(seed)
    if width is not None:
        values["width"] = int(width)
    if height is not None:
        values["height"] = int(height)
    if frames is not None:
        values["frames"] = int(frames)
    if isinstance(vae, str) and vae.strip():
        values["vae"] = vae
    if isinstance(filename_prefix, str) and filename_prefix.strip():
        values["filename_prefix"] = filename_prefix
    return values


def _sync_audio_frames(workflow: dict[str, Any], frames: int) -> None:
    """Keep ``LTXVEmptyLatentAudio.frames_number`` on the video length.

    Same pairing as ``workflow_patcher`` (the audio field is ``frames_number``).
    Linked widgets are left alone.
    """
    from master_agent.comfy.workflow_patcher import (
        LTX_AUDIO_CLASSES,
        _find_nodes_by_class,
        _set_input,
    )

    for class_type in LTX_AUDIO_CLASSES:
        for _node_id, node in _find_nodes_by_class(workflow, class_type):
            current = (node.get("inputs") or {}).get("frames_number")
            if isinstance(current, list):
                continue
            _set_input(node, "frames_number", int(frames))


def _snap_frames(fields: dict[str, Any], values: dict[str, Any]) -> None:
    if "frames" not in values:
        return
    spec = fields.get("frames") or {}
    class_type = str(spec.get("class_type") or "")
    if class_type.startswith("EmptyLTX") or spec.get("input") == "length":
        values["frames"] = snap_ltx_frames(int(values["frames"]))


def prepare_ingested(
    slug: str,
    *,
    prompt: str | None = None,
    negative_prompt: str | None = None,
    seed: int | None = None,
    width: int | None = None,
    height: int | None = None,
    frames: int | None = None,
    vae: str | None = None,
    filename_prefix: str | None = None,
    node_overrides: dict[str, dict[str, Any]] | None = None,
    object_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a patched copy plus the dry-run report. Does not queue."""
    from master_agent.comfy.cli_run import apply_overrides
    from master_agent.comfy.workflow_patcher import _apply_named_fields

    bundle = load_bundle(slug)
    learned = bundle["learned"]
    fields = dict(learned.get("fields") or {})
    warnings = list(learned.get("warnings") or [])
    for warning in warnings:
        log.warning(
            "low confidence role %s: %s",
            warning.get("role"),
            warning.get("detail"),
        )
    values = _overrides(
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        width=width,
        height=height,
        frames=frames,
        vae=vae,
        filename_prefix=filename_prefix,
    )
    _snap_frames(fields, values)
    workflow = json.loads(json.dumps(bundle["workflow"]))
    _apply_named_fields(workflow, fields, values)
    if "frames" in values:
        _sync_audio_frames(workflow, int(values["frames"]))
    if node_overrides:
        workflow = apply_overrides(workflow, node_overrides)
    notes = apply_vae_guard(workflow, vae=vae, node_overrides=node_overrides)
    readiness = dict(learned.get("readiness") or {})
    missing_nodes = missing_class_types(workflow, object_info)
    readiness["missing_nodes"] = missing_nodes
    readiness["missing_node_packs"] = missing_node_packs(workflow, object_info)
    if missing_nodes:
        raise MissingCustomNodeError(missing_nodes)
    params: dict[str, Any] = {}
    for role, spec in fields.items():
        if not isinstance(spec, dict):
            continue
        params[role] = {
            "value": _widget(workflow, spec),
            "node_id": spec.get("node_id"),
            "input": spec.get("input"),
            "class_type": spec.get("class_type"),
            "confidence": spec.get("confidence") or "high",
            "overridden": role in values,
        }
    return {
        "slug": bundle["slug"],
        "queued": False,
        "params": params,
        "warnings": warnings,
        "dangers": list(learned.get("dangers") or []),
        "vae_notes": notes,
        "readiness": readiness,
        "workflow": workflow,
        "workflow_sha256": (bundle.get("provenance") or {}).get("workflow_sha256"),
        "next": f"python -m master_agent comfy run --ingested {bundle['slug']}",
    }


def format_dry_run(report: dict[str, Any]) -> str:
    if report.get("queued"):
        header = "ingested run (queued)"
    else:
        header = "ingested dry-run (not queued)"
    lines = [
        header,
        f"slug: {report.get('slug')}",
    ]
    params = report.get("params") or {}
    if not params:
        lines.append("params: none")
    for role, info in params.items():
        confidence = info.get("confidence") or "high"
        lines.append(
            f"{role}: {info.get('value')!r} [{confidence}] "
            f"node {info.get('node_id')} {info.get('input')}"
        )
    for warning in report.get("warnings") or []:
        lines.append(
            f"WARN  low confidence {warning.get('role')}: {warning.get('detail')}"
        )
    dangers = report.get("dangers") or []
    if dangers:
        for danger in dangers:
            lines.append(
                f"dangers: {danger.get('code')} {danger.get('action')} {danger.get('detail')}"
            )
    else:
        lines.append("dangers: none")
    for note in report.get("vae_notes") or []:
        lines.append(f"vae_guard: {note}")
    readiness = report.get("readiness") or {}
    packs = {
        str(row.get("class_type")): row
        for row in (readiness.get("missing_node_packs") or [])
        if isinstance(row, dict)
    }
    missing = readiness.get("missing_nodes") or []
    if missing:
        rendered: list[str] = []
        for item in missing:
            row = packs.get(str(item)) or {}
            pack = row.get("pack")
            if pack:
                rendered.append(f"{item} ({pack})")
            else:
                rendered.append(str(item))
        lines.append("missing_nodes: " + ", ".join(rendered))
    else:
        lines.append("missing_nodes: none")
    models = readiness.get("missing_models") or []
    if models:
        lines.append("missing_models: " + ", ".join(str(item) for item in models))
        lines.append(
            "missing_models are not substituted. "
            f"{DOCTOR_POINTER}. {DOWNLOAD_POINTER}."
        )
    elif readiness.get("models_checked"):
        lines.append("missing_models: none")
    else:
        lines.append("missing_models: not checked")
    for pointer in readiness.get("pointers") or []:
        lines.append(f"pointer: {pointer}")
    lines.append(f"next: {report.get('next')}")
    return "\n".join(lines)


def dry_run_slug(slug: str, **kwargs: Any) -> dict[str, Any]:
    report = prepare_ingested(slug, **kwargs)
    report["text"] = format_dry_run(report)
    return report


def _write_output_provenance(
    report: dict[str, Any],
    history_files: list[dict[str, Any]],
    *,
    output_dir: Any = None,
) -> list[str]:
    """Write ``buddy.clip.provenance/v1`` next to each output that exists.

    Uses ``build_clip_provenance`` / ``write_clip_provenance``. Does not
    delete the output. Skips names that are not on disk.
    """
    from types import SimpleNamespace

    from master_agent.comfy.client import ComfyClient
    from master_agent.config import COMFYUI_OUTPUT_DIR
    from master_agent.provenance import (
        CLIP_PROVENANCE_SCHEMA,
        build_clip_provenance,
        missing_required,
        write_clip_provenance,
    )

    params = report.get("params") or {}

    def _value(role: str) -> Any:
        info = params.get(role) or {}
        return info.get("value") if isinstance(info, dict) else None

    slug = str(report.get("slug") or "ingested")
    base = output_dir or COMFYUI_OUTPUT_DIR
    written: list[str] = []
    for info in history_files:
        if not isinstance(info, dict) or not info.get("filename"):
            continue
        path = ComfyClient.resolve_output_path(info, base)
        if not path.is_file():
            continue
        state = SimpleNamespace(
            request=str(_value("prompt") or ""),
            prompt=str(_value("prompt") or ""),
            negative_prompt=str(_value("negative_prompt") or ""),
            variant=f"ingested:{slug}",
            seed=_value("seed"),
            steps=_value("steps"),
            cfg=_value("cfg"),
            width=int(_value("width") or 0),
            height=int(_value("height") or 0),
            fps=24,
            duration_s=0.0,
            shot_id="shot-1",
            attempt=1,
            judge_score=0.0,
            shot={},
            revise_history=[],
        )
        payload = build_clip_provenance(state, path=path, revise_notes="ingested run")
        payload["engine"]["workflow_id"] = slug
        digest = report.get("workflow_sha256")
        if digest:
            payload["params"]["workflow_sha256"] = digest
        if payload.get("schema") != CLIP_PROVENANCE_SCHEMA:
            raise RuntimeError("ingested provenance used an unknown schema")
        missing = missing_required(payload)
        if missing:
            raise RuntimeError("ingested provenance missing " + ", ".join(missing))
        dest = write_clip_provenance(path, payload)
        written.append(str(dest))
        if not path.is_file():
            raise RuntimeError(f"output was removed while writing provenance: {path}")
    return written


def run_ingested(
    slug: str,
    *,
    client: Any | None = None,
    object_info: dict[str, Any] | None = None,
    output_dir: Any = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Queue a prepared ingested graph through ``client.queue_prompt``.

    ``client`` defaults to ``ComfyClient`` (live tower). Tests pass a fake.
    Output files already on disk are left in place.
    """
    if client is None:
        from master_agent.comfy.client import ComfyClient

        client = ComfyClient()
        if object_info is None and hasattr(client, "load_object_info"):
            object_info, _source = client.load_object_info(prefer_live=True)
    report = prepare_ingested(slug, object_info=object_info, **kwargs)
    readiness = report.get("readiness") or {}
    missing_models = list(readiness.get("missing_models") or [])
    if readiness.get("models_checked") and missing_models:
        raise MissingModelError([str(name) for name in missing_models])
    workflow = report["workflow"]
    if object_info is not None:
        from master_agent.comfy.cli_run import lint_or_raise

        lint_or_raise(workflow, object_info, file_label=f"ingested:{slug}")
    if hasattr(client, "free_memory"):
        client.free_memory()
    prompt_id = client.queue_prompt(workflow)
    report["queued"] = True
    report["prompt_id"] = prompt_id
    report["text"] = format_dry_run(report)
    if hasattr(client, "wait_for_prompt"):
        entry = client.wait_for_prompt(prompt_id)
        from master_agent.comfy.client import ComfyClient

        files = ComfyClient.extract_video_files(entry or {})
        report["history_files"] = files
        if not files:
            raise RuntimeError("job completed but produced no output files")
        report["provenance_sidecars"] = _write_output_provenance(
            report, files, output_dir=output_dir
        )
    return report
