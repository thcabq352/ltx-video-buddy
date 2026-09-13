"""comfy run: raw JSON, template+overrides, or generate-from-scratch via patcher."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

Mode = Literal["raw", "template", "generate"]


class LintError(RuntimeError):
    def __init__(self, message: str, report: Any | None = None):
        super().__init__(message)
        self.report = report


def unwrap_workflow(data: Any) -> dict[str, Any]:
    """Accept API graphs, or Comfy UI exports wrapped as {prompt: {...}}."""
    if not isinstance(data, dict):
        raise ValueError("workflow must be a JSON object")
    prompt = data.get("prompt")
    if isinstance(prompt, dict) and any(
        isinstance(node, dict) and "class_type" in node for node in prompt.values()
    ):
        return prompt
    return data


def is_link(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], (str, int))
        and isinstance(value[1], int)
    )


def editable_fields(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    """Scalar node inputs (not graph links) for the studio field editor."""
    rows: list[dict[str, Any]] = []
    for node_id, node in workflow.items():
        if not isinstance(node, dict) or "class_type" not in node:
            continue
        title = (node.get("_meta") or {}).get("title") or node.get("class_type") or ""
        for field, value in (node.get("inputs") or {}).items():
            if is_link(value):
                continue
            rows.append(
                {
                    "node": str(node_id),
                    "class_type": node.get("class_type") or "",
                    "title": title,
                    "field": str(field),
                    "value": value,
                    "kind": type(value).__name__,
                }
            )
    return rows


def list_templates() -> list[dict[str, str]]:
    from master_agent.comfy.catalog import list_catalog_items

    return list_catalog_items()


def resolve_template(rel: str | Path) -> Path:
    from master_agent.comfy.catalog import is_known_variant, resolve_workflow_path
    from master_agent.config import WORKFLOW_FILES, WORKFLOWS_DIR

    key = str(rel or "").strip().replace("\\", "/")
    if not key:
        raise ValueError("template path required")
    if is_known_variant(key):
        return resolve_workflow_path(key)
    if key in WORKFLOW_FILES:
        key = WORKFLOW_FILES[key]
    raw = Path(key)
    if raw.is_absolute():
        raise ValueError("template must be a path under workflows/")
    root = WORKFLOWS_DIR.resolve()
    target = (root / raw).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError("template must be a path under workflows/") from exc
    if not target.is_file():
        raise FileNotFoundError(f"no template {key}")
    return target


def apply_overrides(workflow: dict[str, Any], overrides: dict[str, dict[str, Any]] | None) -> dict[str, Any]:
    """Expert field-level overrides keyed by node ID then field name."""
    wf = json.loads(json.dumps(unwrap_workflow(workflow)))
    for node_id, fields in (overrides or {}).items():
        node = wf.get(str(node_id))
        if not isinstance(node, dict):
            raise KeyError(f"unknown node id {node_id}")
        inputs = node.setdefault("inputs", {})
        for field, value in (fields or {}).items():
            inputs[str(field)] = value
    return wf


def prepare_run(
    mode: Mode,
    *,
    workflow: dict[str, Any] | None = None,
    template_path: str | Path | None = None,
    variant: str | None = None,
    prompt: str = "",
    overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if mode == "raw":
        if not isinstance(workflow, dict):
            raise ValueError("raw mode requires a workflow dict")
        return apply_overrides(workflow, overrides)
    if mode == "template":
        raw_path = Path(template_path or "")
        path = raw_path if raw_path.is_file() else resolve_template(template_path or "")
        data = json.loads(path.read_text(encoding="utf-8"))
        return apply_overrides(unwrap_workflow(data), overrides)
    if mode == "generate":
        from master_agent.comfy.workflow_patcher import load_and_patch_workflow

        wf, _meta = load_and_patch_workflow(variant or "base", prompt=prompt or "test")
        return apply_overrides(wf, overrides)
    raise ValueError(f"unknown mode {mode!r}")


def lint_report(
    workflow: dict[str, Any],
    object_info: dict[str, Any],
    *,
    file_label: str = "comfy-run",
    strict: bool = False,
) -> dict[str, Any]:
    from master_agent.comfy.linter import lint_workflow
    from master_agent.comfy.validator import format_report

    report = lint_workflow(
        unwrap_workflow(workflow), object_info, file_label=file_label, strict=strict
    )
    return {
        "ok": report.ok,
        "errors": [str(item) for item in report.errors],
        "warnings": [str(item) for item in report.warnings],
        "text": format_report(report),
    }


def lint_or_raise(workflow: dict[str, Any], object_info: dict[str, Any], *, file_label: str = "comfy-run") -> None:
    report = lint_report(workflow, object_info, file_label=file_label)
    if not report["ok"]:
        details = "; ".join(report["errors"][:5])
        raise LintError(
            f"linter hard gate: {len(report['errors'])} error(s): {details}",
            report=report,
        )


def execute_prepared(
    workflow: dict[str, Any],
    *,
    run_id: str | None = None,
    variant: str | None = None,
) -> dict[str, Any]:
    """Lint, queue, poll, and copy the first video/image into outputs/."""
    import shutil

    from master_agent.comfy.client import ComfyClient
    from master_agent.config import OUTPUTS_DIR
    from master_agent.models.weights import require_weights

    wf = unwrap_workflow(workflow)
    if variant:
        require_weights(str(variant))
    client = ComfyClient()
    object_info, source = client.load_object_info(prefer_live=True)
    print(f"object_info: {source}")
    lint_or_raise(wf, object_info)
    client.free_memory()
    prompt_id = client.queue_prompt(wf)
    print(f"queued prompt_id={prompt_id}")
    entry = client.wait_for_prompt(prompt_id)
    files = ComfyClient.extract_video_files(entry)
    if not files:
        raise RuntimeError("job completed but produced no output files")
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    prefix = run_id or prompt_id[:12]
    video_path = None
    copied: list[str] = []
    for info in files:
        src = ComfyClient.resolve_output_path(info)
        if not src.is_file():
            continue
        dest = OUTPUTS_DIR / f"{prefix}_{src.name}"
        shutil.copy2(src, dest)
        copied.append(str(dest))
        if video_path is None:
            video_path = str(dest)
        print(f"output: {dest}")
    if not video_path:
        raise RuntimeError(f"output files missing on disk: {files}")
    return {
        "status": "done",
        "prompt_id": prompt_id,
        "video_path": video_path,
        "outputs": copied,
        "nodes": len(wf),
    }
