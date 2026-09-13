"""General ComfyUI API-graph operations (power-user mutations).

Ops are plain dicts applied to an API-format workflow (node_id -> node).
After applying, callers should ``validate_workflow`` against live/cached
``object_info`` before queueing.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Optional

# Optional nodes: missing from /object_info is WARNING + bypass, never a hard fail.
# Live Comfy often only registers WanVideoTeaCache / WanVideoTeaCacheKJ — a generic
# LTX TeaCache class_type must still bypass cleanly.
#
# Tower dump 2026-09-13 (Comfy Desk, 4114 classes) confirmed exact YES:
#   TeaCache, WanVideoTeaCache, LanPaint_KSampler, GetWarpedNoiseFromVideo (family;
#   there is NO exact VideoNoiseWarp), MMAudioModelLoader / Sampler / VoCoder.
# Structural payload nodes (WanFunInpaintToVideo, Wan22FunControlToVideo,
# IPAdapterFaceID, ControlNetLoader, CreateVoronoiMask) are NOT optional — if a
# graph needs them and the pack is missing, fail closed.
OPTIONAL_ACCELERATOR_CLASS_TYPES = frozenset(
    {
        "TeaCache",
        "TeaCacheForLTXV",
        "LTXVTeaCache",
        "LTXTeaCache",
        "EasyCache",
        "EasyCacheNode",
        "WanVideoTeaCache",
        "WanVideoTeaCacheKJ",
        "CompileModel",
        "TorchCompileModel",
        "FBCache",
        "FirstBlockCache",
        "MagCache",
        "TaylorSeer",
        "CacheDiffusion",
    }
)
# Enhancer / VFX packs: skip the node if this Comfy does not register it.
OPTIONAL_TOWER_CLASS_TYPES = frozenset(
    {
        "LanPaint_KSampler",
        "LanPaint",  # stale alias; live name is LanPaint_KSampler
        "GetWarpedNoiseFromVideo",
        "GetWarpedNoiseFromVideoAdvanced",
        "MMAudioModelLoader",
        "MMAudioSampler",
        "MMAudioVoCoder",
        "MMAudioFeatureUtils",
    }
)
OPTIONAL_NODE_CLASS_TYPES = (
    OPTIONAL_ACCELERATOR_CLASS_TYPES | OPTIONAL_TOWER_CLASS_TYPES
)
_ACCELERATOR_NAME_MARKERS = (
    "teacache",
    "easycache",
    "fbcache",
    "magcache",
    "firstblockcache",
)


def is_optional_node(class_type: str | None) -> bool:
    """True when a missing class should WARNING+bypass instead of hard-fail."""
    if not class_type:
        return False
    if class_type in OPTIONAL_NODE_CLASS_TYPES:
        return True
    lowered = class_type.lower()
    if any(marker in lowered for marker in _ACCELERATOR_NAME_MARKERS):
        return True
    if lowered.startswith("lanpaint"):
        return True
    if "getwarpednoise" in lowered.replace("_", ""):
        return True
    # Exact MMAudio* gen family — not WanVideoEmptyMMAudioLatents
    if lowered.startswith("mmaudio"):
        return True
    return False


def is_optional_accelerator(class_type: str | None) -> bool:
    """Alias kept for callers/tests; includes tower optional VFX nodes."""
    return is_optional_node(class_type)


def _is_link(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], (str, int))
        and isinstance(value[1], int)
    )


def _upstream_link(node: dict[str, Any]) -> list | None:
    """Prefer a MODEL-named input link; otherwise the first typed link."""
    inputs = node.get("inputs") if isinstance(node, dict) else None
    if not isinstance(inputs, dict):
        return None
    preferred = ("model", "MODEL", "unet", "diffusion_model")
    for key in preferred:
        value = inputs.get(key)
        if _is_link(value):
            return [str(value[0]), int(value[1])]
    for value in inputs.values():
        if _is_link(value):
            return [str(value[0]), int(value[1])]
    return None


def bypass_optional_accelerators(
    workflow: dict[str, Any],
    object_info: dict[str, Any] | None = None,
) -> list[tuple[str, str]]:
    """Rewire MODEL (or the typed upstream link) past missing speed nodes and drop them.

    Does not install custom-node packs. Returns [(node_id, class_type), ...] removed.
    """
    registry = object_info if isinstance(object_info, dict) else {}
    dropped: list[tuple[str, str]] = []
    for nid, node in list(workflow.items()):
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type")
        if not is_optional_node(class_type):
            continue
        if class_type in registry:
            continue
        source = _upstream_link(node)
        sid = str(nid)
        for other in workflow.values():
            if not isinstance(other, dict) or other is node:
                continue
            inputs = other.get("inputs")
            if not isinstance(inputs, dict):
                continue
            for key, value in list(inputs.items()):
                if _is_link(value) and str(value[0]) == sid:
                    if source is not None:
                        inputs[key] = [source[0], source[1]]
                    else:
                        del inputs[key]
        workflow.pop(sid, None)
        dropped.append((sid, str(class_type)))
    return dropped


# Ops the power-mode LLM may emit
ALLOWED_OPS = frozenset(
    {
        "set_widget",
        "set_widget_by_class",
        "rewire",
        "add_node",
        "remove_node",
        "delete_input",
    }
)


@dataclass
class OpResult:
    ok: bool
    applied: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "applied": self.applied,
            "errors": self.errors,
            "skipped": self.skipped,
        }


def summarize_workflow(workflow: dict[str, Any], *, max_nodes: int = 80) -> str:
    """Compact human/LLM-readable digest of an API workflow."""
    lines: list[str] = []
    items = list(workflow.items())[:max_nodes]
    for nid, node in items:
        if not isinstance(node, dict):
            continue
        ct = node.get("class_type") or "?"
        title = ((node.get("_meta") or {}).get("title") or "")[:40]
        inputs = node.get("inputs") or {}
        # show only scalar widgets (skip long links dump)
        widgets = []
        for k, v in list(inputs.items())[:12]:
            if isinstance(v, list) and len(v) == 2:
                widgets.append(f"{k}=[{v[0]},{v[1]}]")
            elif isinstance(v, (str, int, float, bool)) or v is None:
                s = repr(v)
                if len(s) > 48:
                    s = s[:45] + "..."
                widgets.append(f"{k}={s}")
        head = f"- id={nid} {ct}"
        if title:
            head += f' "{title}"'
        if widgets:
            head += " | " + ", ".join(widgets)
        lines.append(head)
    if len(workflow) > max_nodes:
        lines.append(f"... ({len(workflow) - max_nodes} more nodes)")
    return "\n".join(lines)


def object_info_snippets(
    workflow: dict[str, Any],
    object_info: dict[str, Any],
    *,
    max_classes: int = 24,
    max_inputs_per_class: int = 20,
) -> str:
    """Schema snippets for class_types present in the workflow."""
    classes: list[str] = []
    seen: set[str] = set()
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        ct = node.get("class_type")
        if not ct or ct in seen:
            continue
        seen.add(ct)
        classes.append(ct)
        if len(classes) >= max_classes:
            break

    blocks: list[str] = []
    for ct in classes:
        info = object_info.get(ct) or {}
        spec_in = info.get("input") or {}
        parts: list[str] = []
        for section in ("required", "optional"):
            section_map = spec_in.get(section) or {}
            for name, spec in list(section_map.items())[:max_inputs_per_class]:
                # spec is usually [type, {options...}] or [type]
                typ = "?"
                extra = ""
                if isinstance(spec, list) and spec:
                    typ = str(spec[0])
                    if len(spec) > 1 and isinstance(spec[1], dict):
                        conf = spec[1]
                        bits = []
                        if "min" in conf:
                            bits.append(f"min={conf['min']}")
                        if "max" in conf:
                            bits.append(f"max={conf['max']}")
                        if "default" in conf:
                            d = conf["default"]
                            ds = repr(d)
                            if len(ds) > 40:
                                ds = ds[:37] + "..."
                            bits.append(f"default={ds}")
                        opts = conf.get("options") or conf.get("choices")
                        if isinstance(opts, list) and opts:
                            sample = ", ".join(str(o) for o in opts[:8])
                            if len(opts) > 8:
                                sample += ", ..."
                            bits.append(f"choices=[{sample}]")
                        if bits:
                            extra = " (" + "; ".join(bits) + ")"
                parts.append(f"  {section}: {name}: {typ}{extra}")
        outs = info.get("output") or []
        out_s = ", ".join(str(o) for o in outs[:8]) if outs else ""
        block = f"{ct}:\n" + ("\n".join(parts) if parts else "  (no inputs)")
        if out_s:
            block += f"\n  outputs: [{out_s}]"
        blocks.append(block)
    return "\n\n".join(blocks)


def _next_node_id(workflow: dict[str, Any]) -> str:
    nums = []
    for k in workflow:
        try:
            nums.append(int(k))
        except (TypeError, ValueError):
            continue
    n = (max(nums) + 1) if nums else 1
    while str(n) in workflow:
        n += 1
    return str(n)


def _find_by_class(
    workflow: dict[str, Any], class_type: str
) -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    for nid, node in workflow.items():
        if isinstance(node, dict) and node.get("class_type") == class_type:
            out.append((str(nid), node))
    return out


def apply_ops(
    workflow: dict[str, Any],
    ops: list[dict[str, Any]],
    *,
    copy_graph: bool = True,
    max_ops: int = 40,
) -> tuple[dict[str, Any], OpResult]:
    """Apply a list of graph ops. Returns (new_or_same_workflow, result)."""
    wf = copy.deepcopy(workflow) if copy_graph else workflow
    result = OpResult(ok=True)
    if not ops:
        return wf, result

    for i, raw in enumerate(ops[:max_ops]):
        if not isinstance(raw, dict):
            result.errors.append(f"op[{i}]: not an object")
            result.ok = False
            continue
        op = str(raw.get("op") or "").strip().lower()
        if op not in ALLOWED_OPS:
            result.skipped.append(f"op[{i}]: unknown op {op!r}")
            continue
        try:
            if op == "set_widget":
                _op_set_widget(wf, raw)
            elif op == "set_widget_by_class":
                _op_set_widget_by_class(wf, raw)
            elif op == "rewire":
                _op_rewire(wf, raw)
            elif op == "add_node":
                _op_add_node(wf, raw)
            elif op == "remove_node":
                _op_remove_node(wf, raw)
            elif op == "delete_input":
                _op_delete_input(wf, raw)
            result.applied.append(dict(raw))
        except Exception as e:
            result.errors.append(f"op[{i}] {op}: {e}")
            result.ok = False

    if len(ops) > max_ops:
        result.skipped.append(f"truncated {len(ops) - max_ops} ops over max_ops={max_ops}")
    return wf, result


def _op_set_widget(wf: dict[str, Any], op: dict[str, Any]) -> None:
    nid = str(op.get("node_id") or "")
    name = op.get("input") or op.get("name")
    if not nid or name is None:
        raise ValueError("set_widget needs node_id and input")
    if nid not in wf:
        raise ValueError(f"node {nid} not found")
    node = wf[nid]
    if not isinstance(node, dict):
        raise ValueError(f"node {nid} is not a dict")
    inputs = node.setdefault("inputs", {})
    if not isinstance(inputs, dict):
        raise ValueError(f"node {nid} inputs not a dict")
    if "value" not in op:
        raise ValueError("set_widget needs value")
    inputs[str(name)] = op["value"]


def _op_set_widget_by_class(wf: dict[str, Any], op: dict[str, Any]) -> None:
    ct = op.get("class_type")
    name = op.get("input") or op.get("name")
    if not ct or name is None:
        raise ValueError("set_widget_by_class needs class_type and input")
    if "value" not in op:
        raise ValueError("set_widget_by_class needs value")
    matches = _find_by_class(wf, str(ct))
    if not matches:
        raise ValueError(f"no nodes of class_type {ct!r}")
    idx = int(op.get("index") or 0)
    if idx < 0 or idx >= len(matches):
        raise ValueError(f"index {idx} out of range ({len(matches)} {ct} nodes)")
    nid, node = matches[idx]
    inputs = node.setdefault("inputs", {})
    if not isinstance(inputs, dict):
        raise ValueError(f"node {nid} inputs not a dict")
    # optional: only set if key already exists (safer for unknown widgets)
    if op.get("only_existing") and str(name) not in inputs:
        raise ValueError(f"input {name!r} not present on {ct}#{idx}")
    inputs[str(name)] = op["value"]


def _op_rewire(wf: dict[str, Any], op: dict[str, Any]) -> None:
    nid = str(op.get("node_id") or "")
    name = op.get("input") or op.get("name")
    src = op.get("from_node") or op.get("source_node")
    slot = op.get("from_slot", op.get("source_slot", 0))
    if not nid or name is None or src is None:
        raise ValueError("rewire needs node_id, input, from_node")
    if nid not in wf:
        raise ValueError(f"node {nid} not found")
    if str(src) not in wf:
        raise ValueError(f"source node {src} not found")
    try:
        slot_i = int(slot)
    except (TypeError, ValueError) as e:
        raise ValueError(f"from_slot must be int: {e}") from e
    node = wf[nid]
    inputs = node.setdefault("inputs", {})
    if not isinstance(inputs, dict):
        raise ValueError(f"node {nid} inputs not a dict")
    inputs[str(name)] = [str(src), slot_i]


def _op_add_node(wf: dict[str, Any], op: dict[str, Any]) -> None:
    ct = op.get("class_type")
    if not ct:
        raise ValueError("add_node needs class_type")
    nid = str(op.get("node_id") or _next_node_id(wf))
    if nid in wf:
        raise ValueError(f"node_id {nid} already exists")
    inputs = op.get("inputs") if isinstance(op.get("inputs"), dict) else {}
    node: dict[str, Any] = {"class_type": str(ct), "inputs": dict(inputs)}
    meta = op.get("_meta") or op.get("meta")
    if isinstance(meta, dict):
        node["_meta"] = meta
    wf[nid] = node


def _op_remove_node(wf: dict[str, Any], op: dict[str, Any]) -> None:
    nid = str(op.get("node_id") or "")
    if not nid:
        raise ValueError("remove_node needs node_id")
    if nid not in wf:
        raise ValueError(f"node {nid} not found")
    del wf[nid]
    # scrub inbound links that pointed at this node
    for other in wf.values():
        if not isinstance(other, dict):
            continue
        inputs = other.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for k, v in list(inputs.items()):
            if isinstance(v, list) and len(v) >= 1 and str(v[0]) == nid:
                del inputs[k]


def _op_delete_input(wf: dict[str, Any], op: dict[str, Any]) -> None:
    nid = str(op.get("node_id") or "")
    name = op.get("input") or op.get("name")
    if not nid or name is None:
        raise ValueError("delete_input needs node_id and input")
    if nid not in wf:
        raise ValueError(f"node {nid} not found")
    inputs = (wf[nid] or {}).get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError(f"node {nid} has no inputs dict")
    inputs.pop(str(name), None)


__all__ = [
    "ALLOWED_OPS",
    "OPTIONAL_ACCELERATOR_CLASS_TYPES",
    "OPTIONAL_NODE_CLASS_TYPES",
    "OPTIONAL_TOWER_CLASS_TYPES",
    "OpResult",
    "apply_ops",
    "bypass_optional_accelerators",
    "is_optional_accelerator",
    "is_optional_node",
    "object_info_snippets",
    "summarize_workflow",
]
