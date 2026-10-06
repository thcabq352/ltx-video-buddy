"""Build a learned.yaml document from role guesses. No LLM."""

from __future__ import annotations

import logging
from typing import Any

from master_agent.comfy.ingest.classify import infer_roles, missing_class_types, requires_packs
from master_agent.comfy.ingest.validate import dangers_for

log = logging.getLogger(__name__)

_FIELD_ORDER = (
    "prompt",
    "negative_prompt",
    "seed",
    "width",
    "height",
    "frames",
    "checkpoint",
    "vae",
    "filename_prefix",
)


def learn_workflow(
    workflow: dict[str, Any],
    *,
    slug: str,
    source: str = "",
    object_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Typed field map plus readiness. Low-confidence roles are kept."""
    guesses, output_id = infer_roles(workflow)
    fields: dict[str, Any] = {}
    warnings: list[dict[str, str]] = []
    ordered = sorted(
        guesses,
        key=lambda item: _FIELD_ORDER.index(item.role) if item.role in _FIELD_ORDER else 99,
    )
    for guess in ordered:
        if guess.role in fields:
            continue
        fields[guess.role] = {
            "node_id": guess.node_id,
            "class_type": guess.class_type,
            "index": guess.index,
            "input": guess.input,
            "confidence": guess.confidence,
        }
        if guess.confidence != "high":
            warning = {
                "role": guess.role,
                "confidence": guess.confidence,
                "detail": guess.detail,
            }
            warnings.append(warning)
            log.warning("low confidence role %s: %s", guess.role, guess.detail)
    learned: dict[str, Any] = {
        "slug": slug,
        "source": source,
        "format": "api",
        "fields": fields,
        "inputs": {},
        "outputs": {"final": {"node_id": output_id}} if output_id else {},
        "requires": requires_packs(workflow),
        "dangers": dangers_for(workflow),
        "vram_class": "unknown",
        "warnings": warnings,
        "readiness": {
            "missing_nodes": missing_class_types(workflow, object_info),
            "missing_models": [],
            "convertible": True,
        },
    }
    return learned
