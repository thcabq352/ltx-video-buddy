"""Build a learned.yaml document from role guesses.

``--llm-assist`` is optional and off here. High-confidence roles stay on
the heuristics. See ``llm_assist.py``.
"""

from __future__ import annotations

import logging
from typing import Any

from master_agent.comfy.ingest.classify import (
    infer_roles,
    missing_class_types,
    missing_node_packs,
    requires_packs,
)
from master_agent.comfy.ingest.fingerprint import (
    decide_route,
    family_warning,
    match_family,
)
from master_agent.comfy.ingest.validate import (
    dangers_for,
    missing_model_filenames,
    pointers_for,
)

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
    model_inventory: set[str] | None = None,
    no_family_route: bool = False,
    llm_assist: bool = False,
    proposer: Any = None,
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
        "requires": requires_packs(workflow, object_info),
        "dangers": dangers_for(workflow),
        "vram_class": "unknown",
        "warnings": warnings,
    }
    family = match_family(workflow)
    route = decide_route(family, no_family_route=no_family_route, stored_route=None)
    # An explicit flag stores generic even when a family matches. Unmatched
    # graphs are generic too. ``stored_route=None`` lets decide_route pick
    # specialized when a family matches and the flag is off.
    if no_family_route:
        route = "generic"
    learned["family"] = family
    learned["family_route"] = route
    note = family_warning(family, route=route)
    if note:
        learned["family_warning"] = note
    missing_nodes = missing_class_types(workflow, object_info)
    missing_models = missing_model_filenames(workflow, model_inventory)
    learned["readiness"] = {
        "missing_nodes": missing_nodes,
        "missing_node_packs": missing_node_packs(workflow, object_info),
        "missing_models": missing_models,
        "models_checked": model_inventory is not None,
        "pointers": pointers_for(missing_nodes=missing_nodes, missing_models=missing_models),
        "convertible": True,
    }
    if llm_assist:
        from master_agent.comfy.ingest.llm_assist import apply_llm_assist

        apply_llm_assist(learned, workflow, proposer=proposer)
    return learned
