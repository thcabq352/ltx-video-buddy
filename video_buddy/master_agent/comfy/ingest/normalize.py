"""Load a Comfy workflow JSON and normalize it to API form.

Phase A accepts API graphs only (``{node_id: {class_type, inputs}}``), including
a ``{prompt: ...}`` wrapper. UI exports are refused. Conversion via
``/workflow/convert`` is Phase B and does not run here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class IngestError(ValueError):
    """The file is not a Phase A API workflow."""


def _is_api_node(value: Any) -> bool:
    return isinstance(value, dict) and isinstance(value.get("class_type"), str)


def _api_nodes(data: dict[str, Any]) -> dict[str, Any] | None:
    nodes = {str(key): value for key, value in data.items() if _is_api_node(value)}
    if not nodes:
        return None
    # Ignore a wrapper that merely contains one embedded API-looking value.
    if len(nodes) < max(1, len(data) // 2) and "nodes" in data:
        return None
    return nodes


def is_ui_workflow(data: Any) -> bool:
    """True for a Comfy UI export (``nodes`` list of ``type`` objects)."""
    if not isinstance(data, dict):
        return False
    nodes = data.get("nodes")
    if not isinstance(nodes, list):
        workflow = data.get("workflow")
        if isinstance(workflow, dict):
            nodes = workflow.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        return False
    return any(isinstance(node, dict) and "type" in node for node in nodes)


def unwrap_api_workflow(data: Any) -> dict[str, Any]:
    """Return the API node map, or raise ``IngestError``."""
    if not isinstance(data, dict):
        raise IngestError("workflow must be a JSON object")
    prompt = data.get("prompt")
    if isinstance(prompt, dict):
        inner = _api_nodes(prompt)
        if inner:
            return inner
    direct = _api_nodes(data)
    if direct:
        return direct
    if is_ui_workflow(data):
        raise IngestError(
            "UI workflow JSON is not converted in Phase A. "
            "Supply API JSON (node id → class_type/inputs). "
            "Comfy /workflow/convert is a later phase and is not called."
        )
    raise IngestError(
        "not a Comfy API workflow. Expected node id → {class_type, inputs}."
    )


def load_workflow_file(path: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Read ``path``. Return ``(api_graph, ui_original_or_none)``.

    UI files raise. ``ui_original`` stays ``None`` so callers do not write
    ``workflow_ui.json``.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise IngestError(f"workflow not found: {file_path}")
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise IngestError(f"workflow is not JSON: {file_path}") from exc
    return unwrap_api_workflow(data), None
