"""Load a Comfy workflow JSON and normalize it to API form.

API graphs (``{node_id: {class_type, inputs}}``), including a ``{prompt: ...}``
wrapper, load as-is. UI exports convert through Comfy ``/workflow/convert``
when a converter is supplied. With no converter, a UI file is refused:
start Comfy or supply API JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class IngestError(ValueError):
    """The workflow could not be ingested."""


# Exact phrase callers and tests look for when conversion cannot run.
START_COMFY_OR_API = "start Comfy or supply API JSON"


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
            f"{START_COMFY_OR_API}. "
            "This file is UI-format JSON and Comfy /workflow/convert was not called."
        )
    raise IngestError(
        "not a Comfy API workflow. Expected node id → {class_type, inputs}."
    )


def graph_from_history_payload(payload: Any, prompt_id: str) -> dict[str, Any]:
    """Pull the queued API graph out of a Comfy ``/history`` body.

    Comfy stores ``prompt`` as ``[number, prompt_id, graph, extra, outputs]``.
    A dict ``prompt`` is accepted when it is already an API graph.
    """
    if not isinstance(payload, dict):
        raise IngestError(
            f"{START_COMFY_OR_API}. Comfy /history/{prompt_id} was not a JSON object."
        )
    entry = payload.get(prompt_id)
    if not isinstance(entry, dict):
        entry = payload
    prompt = entry.get("prompt") if isinstance(entry, dict) else None
    graph: Any = None
    if isinstance(prompt, list) and len(prompt) >= 3 and isinstance(prompt[2], dict):
        graph = prompt[2]
    elif isinstance(prompt, dict):
        graph = prompt
    if not isinstance(graph, dict):
        raise IngestError(
            f"Comfy /history/{prompt_id} has no queued prompt graph. "
            f"{START_COMFY_OR_API}."
        )
    return unwrap_api_workflow(graph)


def load_workflow_file(
    path: Path,
    *,
    converter: Any = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Read ``path``. Return ``(api_graph, ui_original_or_none)``.

    ``converter`` is ``callable(ui_dict) -> api_dict``. UI files without a
    converter raise ``IngestError`` (start Comfy or supply API JSON). The UI
    object is returned so the caller can keep ``workflow_ui.json``.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise IngestError(f"workflow not found: {file_path}")
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise IngestError(f"workflow is not JSON: {file_path}") from exc
    if is_ui_workflow(data):
        if converter is None:
            raise IngestError(
                f"{START_COMFY_OR_API}. "
                "This file is UI-format JSON and Comfy /workflow/convert was not called."
            )
        try:
            converted = converter(data)
        except IngestError:
            raise
        except Exception as exc:
            raise IngestError(
                f"{START_COMFY_OR_API}. Comfy /workflow/convert failed: {exc}"
            ) from exc
        return unwrap_api_workflow(converted), data
    return unwrap_api_workflow(data), None
