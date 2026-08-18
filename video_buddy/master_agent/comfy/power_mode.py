"""Agent power mode — LLM proposes graph ops grounded in object_info + RAG.

Flow:
  1. Summarize workflow + schema snippets for classes in use
  2. Recall similar workflows / runs from the KB (optional)
  3. Ask the LLM for a JSON list of graph ops
  4. apply_ops → validate_workflow
  5. Optional one repair pass if validation fails

Safe defaults: limited ops, max_ops, validate-before-use, never queue here.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from master_agent.comfy.graph_ops import (
    apply_ops,
    object_info_snippets,
    summarize_workflow,
)
from master_agent.comfy.validator import validate_workflow

_POWER_SYSTEM = """You are a ComfyUI power-user assistant for LTX / video workflows.

You receive:
- a user creative brief
- a compact digest of the current API-format workflow (node ids + widgets)
- /object_info schema snippets for classes in that graph
- optional RAG hits from past workflows/runs

Your job: propose a SHORT list of graph ops that improve the graph for the brief
WITHOUT breaking it. Prefer small, high-confidence edits (sampler steps/cfg,
stg, seed, scheduler, prompts on CLIP encode). Avoid random topology rewrites.

Allowed ops (JSON objects):
- {"op":"set_widget","node_id":"3","input":"steps","value":20}
- {"op":"set_widget_by_class","class_type":"KSampler","input":"cfg","value":2.5,"index":0}
- {"op":"rewire","node_id":"5","input":"model","from_node":"4","from_slot":0}
- {"op":"add_node","class_type":"CLIPTextEncode","inputs":{"text":"..."},"node_id":"99"}
- {"op":"remove_node","node_id":"99"}
- {"op":"delete_input","node_id":"3","input":"unused_key"}

Rules:
- Only touch inputs that exist in the schema snippets (or already appear on the node).
- Keep link values as [node_id, output_index] pairs when rewiring.
- Prefer set_widget_by_class when multiple similar nodes exist and index is clear.
- Do NOT invent model filenames you have not seen in the digest or context.
- Max 12 ops. If nothing useful to change, return ops: [].

Respond with JSON only:
{"reason":"one sentence","ops":[ ... ]}
"""


@dataclass
class PowerModeResult:
    workflow: dict[str, Any]
    ops: list[dict[str, Any]] = field(default_factory=list)
    applied: list[dict[str, Any]] = field(default_factory=list)
    reason: str = ""
    valid: bool = False
    validation_errors: list[str] = field(default_factory=list)
    validation_warnings: list[str] = field(default_factory=list)
    repair_attempted: bool = False
    error: str = ""
    skipped: list[str] = field(default_factory=list)
    rag_used: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "ops": self.ops,
            "applied": self.applied,
            "valid": self.valid,
            "validation_errors": self.validation_errors,
            "validation_warnings": self.validation_warnings,
            "repair_attempted": self.repair_attempted,
            "error": self.error,
            "skipped": self.skipped,
            "rag_used": self.rag_used,
            "node_count": len(self.workflow) if isinstance(self.workflow, dict) else 0,
        }


def _extract_json(text: str) -> Optional[dict[str, Any]]:
    if not text:
        return None
    text = text.strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass
    # fenced or prose-wrapped
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def _rag_context(request: str) -> tuple[str, bool]:
    try:
        from master_agent.config import KB_ENABLED

        if not KB_ENABLED:
            return "", False
        from master_agent.kb.recall import recall_similar_runs, recall_workflows

        parts = [p for p in (recall_workflows(request, k=2), recall_similar_runs(request)) if p]
        if not parts:
            return "", False
        return "\n\n".join(parts), True
    except Exception:
        return "", False


def _llm_propose_ops(
    *,
    request: str,
    digest: str,
    schema: str,
    rag: str,
    extra: str = "",
    provider: str | None = None,
) -> tuple[list[dict[str, Any]], str, str]:
    """Returns (ops, reason, error)."""
    try:
        from master_agent.llm import get_llm
        from langchain_core.messages import HumanMessage, SystemMessage
    except Exception as e:
        return [], "", f"LLM unavailable: {e}"

    user = (
        f"## User brief\n{request}\n\n"
        f"## Workflow digest\n{digest}\n\n"
        f"## object_info snippets\n{schema}\n"
    )
    if rag:
        user += f"\n## Knowledge base\n{rag}\n"
    if extra:
        user += f"\n## Prior validation errors (repair pass)\n{extra}\n"
    user += "\nReturn JSON only."

    try:
        llm = get_llm(temperature=0.2, provider=provider)
        resp = llm.invoke(
            [SystemMessage(content=_POWER_SYSTEM), HumanMessage(content=user)]
        )
        text = getattr(resp, "content", None) or str(resp)
        if isinstance(text, list):
            text = "".join(
                b.get("text", "") if isinstance(b, dict) else str(b) for b in text
            )
    except Exception as e:
        return [], "", f"LLM call failed: {e}"

    data = _extract_json(str(text))
    if not data:
        return [], "", "LLM did not return valid JSON"
    ops = data.get("ops") or []
    if not isinstance(ops, list):
        return [], str(data.get("reason") or ""), "ops is not a list"
    clean = [o for o in ops if isinstance(o, dict)]
    reason = str(data.get("reason") or "")
    return clean, reason, ""


def power_tune(
    workflow: dict[str, Any],
    *,
    request: str,
    object_info: dict[str, Any] | None = None,
    provider: str | None = None,
    repair: bool = True,
    max_ops: int = 12,
    log=print,
) -> PowerModeResult:
    """Propose and apply LLM graph ops; validate; optional one repair pass."""
    import copy

    base = copy.deepcopy(workflow)
    result = PowerModeResult(workflow=base)

    if object_info is None:
        try:
            from master_agent.comfy.client import ComfyClient

            object_info, _src = ComfyClient().load_object_info(prefer_live=True)
        except Exception as e:
            result.error = f"could not load object_info: {e}"
            log(f"power-mode: {result.error}")
            return result

    digest = summarize_workflow(base)
    schema = object_info_snippets(base, object_info)
    rag, rag_used = _rag_context(request)
    result.rag_used = rag_used

    ops, reason, err = _llm_propose_ops(
        request=request,
        digest=digest,
        schema=schema,
        rag=rag,
        provider=provider,
    )
    result.reason = reason
    if err:
        result.error = err
        log(f"power-mode: {err}")
        return result

    ops = ops[:max_ops]
    result.ops = ops
    if not ops:
        log("power-mode: no ops proposed")
        # still validate original
        rep = validate_workflow(base, object_info, file_label="power-mode")
        result.valid = rep.ok
        result.validation_errors = [str(i) for i in rep.errors]
        result.validation_warnings = [str(i) for i in rep.warnings]
        return result

    log(f"power-mode: applying {len(ops)} op(s) — {reason or 'no reason'}")
    wf, op_res = apply_ops(base, ops, copy_graph=True, max_ops=max_ops)
    result.applied = op_res.applied
    result.skipped = op_res.skipped
    if op_res.errors:
        result.error = "; ".join(op_res.errors[:5])
        log(f"power-mode apply errors: {result.error}")

    rep = validate_workflow(wf, object_info, file_label="power-mode")
    result.validation_errors = [str(i) for i in rep.errors]
    result.validation_warnings = [str(i) for i in rep.warnings[:8]]

    if rep.ok:
        result.workflow = wf
        result.valid = True
        log("power-mode: validated OK")
        return result

    log(f"power-mode: validation failed ({len(rep.errors)} errors)")
    if not repair:
        result.workflow = base  # discard broken graph
        result.valid = False
        return result

    # One repair pass: show errors, ask for fixed ops, re-apply on original base
    result.repair_attempted = True
    err_text = "\n".join(result.validation_errors[:12])
    ops2, reason2, err2 = _llm_propose_ops(
        request=request,
        digest=summarize_workflow(base),
        schema=schema,
        rag=rag,
        extra=err_text,
        provider=provider,
    )
    if err2 or not ops2:
        result.workflow = base
        result.valid = False
        result.error = (result.error + "; " if result.error else "") + (
            err2 or "repair produced no ops"
        )
        log(f"power-mode repair aborted: {result.error}")
        return result

    if reason2:
        result.reason = (result.reason + " | repair: " + reason2).strip(" |")
    ops2 = ops2[:max_ops]
    result.ops = ops2
    wf2, op_res2 = apply_ops(base, ops2, copy_graph=True, max_ops=max_ops)
    result.applied = op_res2.applied
    result.skipped.extend(op_res2.skipped)
    rep2 = validate_workflow(wf2, object_info, file_label="power-mode-repair")
    result.validation_errors = [str(i) for i in rep2.errors]
    result.validation_warnings = [str(i) for i in rep2.warnings[:8]]
    if rep2.ok:
        result.workflow = wf2
        result.valid = True
        log("power-mode: repair validated OK")
    else:
        result.workflow = base
        result.valid = False
        log("power-mode: repair still invalid — keeping pre-power graph")
    return result


def power_tune_from_variant(
    variant: str,
    request: str,
    *,
    prompt: str | None = None,
    duration_s: float = 5.0,
    quality: str | None = None,
    seed: int | None = None,
    provider: str | None = None,
    log=print,
) -> PowerModeResult:
    """Load + heuristic-patch a variant, then run power_tune (dry orchestration)."""
    from master_agent.comfy.workflow_patcher import load_and_patch_workflow
    from master_agent.config import get_quality_profile

    profile = get_quality_profile(quality)
    wf, meta = load_and_patch_workflow(
        variant,
        prompt=prompt or request,
        duration_s=duration_s,
        seed=seed,
        steps=profile.get("steps"),
        width=int(profile.get("max_width") or 768),
        height=int(profile.get("max_height") or 512),
    )
    result = power_tune(wf, request=request, provider=provider, log=log)
    result.to_dict()  # ensure serializable shape ready
    # stash meta for callers
    result.reason = result.reason or ""
    # attach meta on a private-ish key via dynamic attr is ugly; put in reason path
    # Callers that need seed/steps can re-patch; for CLI we print result.to_dict()
    return result


__all__ = ["PowerModeResult", "power_tune", "power_tune_from_variant"]
