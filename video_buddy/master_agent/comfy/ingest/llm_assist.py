"""Optional local-LLM names for low-confidence widgets.

Off unless the caller passes ``llm_assist``. Heuristics stay authoritative
for high-confidence roles. The chain is llama.cpp, then Ollama. This module
does not call ``get_llm("auto")``, does not start a server, and does not
call a cloud model.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

ROLES = (
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

LOCAL_PROVIDERS = ("llamacpp", "ollama")

NO_LOCAL = (
    "--llm-assist: no local LLM (llamacpp, then ollama). "
    "Heuristics kept. No cloud call."
)

Proposer = Callable[[dict[str, Any]], Any]


def _context(workflow: dict[str, Any], role: str, spec: dict[str, Any]) -> dict[str, Any]:
    node_id = str(spec.get("node_id") or "")
    node = workflow.get(node_id) if isinstance(workflow.get(node_id), dict) else {}
    meta = node.get("_meta") if isinstance(node.get("_meta"), dict) else {}
    inputs = node.get("inputs") if isinstance(node.get("inputs"), dict) else {}
    value = inputs.get(spec.get("input"))
    if isinstance(value, str) and len(value) > 180:
        value = value[:180]
    return {
        "current_role": role,
        "class_type": spec.get("class_type") or node.get("class_type"),
        "title": (meta or {}).get("title") or "",
        "input": spec.get("input"),
        "value": value,
        "candidates": list(ROLES),
    }


def _normalize(proposal: Any) -> dict[str, str] | None:
    if proposal is None:
        return None
    data = proposal
    if isinstance(proposal, str):
        data = _parse_json(proposal)
    if not isinstance(data, dict):
        return None
    role = str(data.get("role") or "").strip()
    if role not in ROLES:
        return None
    confidence = str(data.get("confidence") or "low").strip().lower()
    if confidence not in {"low", "medium", "high"}:
        confidence = "low"
    return {"role": role, "confidence": confidence}


def _parse_json(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        loaded = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return loaded if isinstance(loaded, dict) else None


def build_local_proposer() -> Proposer | None:
    """llama.cpp if it is already up, else Ollama. Never autostarts or uses Grok."""
    from master_agent.llm import _llm_for, preferred_local_provider

    name = preferred_local_provider()
    if name not in LOCAL_PROVIDERS:
        return None
    chat = _llm_for(name, 0.0)

    def propose(ctx: dict[str, Any]) -> dict[str, str] | None:
        payload = {
            "current_role": ctx.get("current_role"),
            "class_type": ctx.get("class_type"),
            "title": ctx.get("title"),
            "input": ctx.get("input"),
            "value": ctx.get("value"),
        }
        message = (
            "Name this Comfy widget. Reply with JSON only: "
            '{"role": "<one of ' + ", ".join(ROLES) + '>", "confidence": "low"|"medium"|"high"}. '
            "Do not invent a role outside that list.\n"
            + json.dumps(payload)
        )
        raw = chat.invoke(message)
        text = getattr(raw, "content", raw)
        return _normalize(str(text))

    propose.provider = name  # type: ignore[attr-defined]
    return propose


def apply_llm_assist(
    learned: dict[str, Any],
    workflow: dict[str, Any],
    *,
    proposer: Proposer | None = None,
) -> None:
    """Attach proposals to low-confidence fields. High-confidence fields stay put."""
    resolved = proposer if proposer is not None else build_local_proposer()
    learned["llm_assist"] = True
    learned["llm_proposals"] = []
    if resolved is None:
        learned["llm_warning"] = NO_LOCAL
        return
    provider = getattr(resolved, "provider", "injected")
    fields = learned.get("fields") or {}
    warnings = learned.get("warnings") or []
    for warning in list(warnings):
        if not isinstance(warning, dict):
            continue
        role = str(warning.get("role") or "")
        spec = fields.get(role)
        if not isinstance(spec, dict) or spec.get("confidence") == "high":
            continue
        try:
            proposal = resolved(_context(workflow, role, spec))
        except Exception as exc:
            learned["llm_warning"] = (
                f"--llm-assist: local LLM failed ({exc}). Heuristics kept. No cloud call."
            )
            return
        parsed = _normalize(proposal)
        if not parsed:
            continue
        new_role = parsed["role"]
        confidence = parsed["confidence"]
        record = {
            "from": role,
            "role": new_role,
            "confidence": confidence,
            "source": "llm",
            "node_id": spec.get("node_id"),
            "provider": provider,
            "applied": False,
        }
        if new_role != role and new_role in fields:
            record["detail"] = f"{new_role} already mapped; heuristic kept"
            learned["llm_proposals"].append(record)
            continue
        if new_role != role:
            fields.pop(role)
            fields[new_role] = spec
            warning["role"] = new_role
        spec["source"] = "llm"
        spec["confidence"] = confidence
        warning["confidence"] = confidence
        warning["detail"] = f"llm proposed {new_role} ({confidence})"
        if confidence == "high":
            learned["warnings"] = [item for item in warnings if item is not warning]
            warnings = learned["warnings"]
        record["applied"] = True
        learned["llm_proposals"].append(record)
