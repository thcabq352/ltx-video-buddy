"""Validate ComfyUI API-format workflow JSON against the live node registry.

Checks a workflow before it ever hits the queue:
  1. Every class_type exists in /object_info (live server or disk cache).
  2. Required inputs are present; unknown input names are flagged.
  3. Widget values fit their spec (combo choices, numeric min/max, types).
  4. Link integrity: [node_id, output_index] references resolve and the
     source output type matches the target input type.
  5. Model filenames in loader widgets resolve against the local model
     inventory (typos / missing downloads caught before queueing).

Usage:
    python -m master_agent validate workflows/base_t2v_i2v.json
    python -m master_agent validate --all
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from master_agent.comfy.client import ComfyClient
from master_agent.comfy.graph_ops import (
    bypass_optional_accelerators,
    is_optional_node,
)
from master_agent.config import is_valid_ltx_frames, snap_ltx_frames
from master_agent.models.inventory import Inventory, load_inventory

# LTX video length + audio frames_number must stay paired 8n+1 counts.
LTX_LENGTH_CLASSES = frozenset({"EmptyLTXVLatentVideo", "LTXVEmptyLatentVideo"})
LTX_AUDIO_FRAME_CLASSES = frozenset({"LTXVEmptyLatentAudio"})
LTX_LENGTH_KEYS = ("length", "frames", "num_frames", "frame_count")

# Loader widget names that reference model weight files
MODEL_INPUT_NAMES = {
    "ckpt_name",
    "unet_name",
    "lora_name",
    "vae_name",
    "clip_name",
    "clip_name1",
    "clip_name2",
    "clip_name3",
    "clip_name4",
    "model_name",
    "diffusion_model",
    "text_encoder",
    "text_encoder1",
    "text_encoder2",
    "audio_vae_name",
    "style_model_name",
    "control_net_name",
    "upscale_model_name",
    "gligen_name",
    "taesd_name",
    "insightface_provider",
    "ipadapter_file",
    "pulid_file",
}

WEIGHT_SUFFIXES = (
    ".safetensors",
    ".ckpt",
    ".pt",
    ".pth",
    ".bin",
    ".gguf",
)


@dataclass
class Issue:
    severity: str  # "error" | "warning"
    node_id: str
    input_name: str
    message: str

    def __str__(self) -> str:
        loc = f"node {self.node_id}"
        if self.input_name:
            loc += f".{self.input_name}"
        return f"[{self.severity.upper():7}] {loc}: {self.message}"


@dataclass
class ValidationReport:
    file: str
    object_info_source: str = "unknown"
    errors: list[Issue] = field(default_factory=list)
    warnings: list[Issue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def error(self, node_id: str, input_name: str, message: str) -> None:
        self.errors.append(Issue("error", str(node_id), input_name, message))

    def warn(self, node_id: str, input_name: str, message: str) -> None:
        self.warnings.append(Issue("warning", str(node_id), input_name, message))

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "object_info_source": self.object_info_source,
            "ok": self.ok,
            "errors": [str(i) for i in self.errors],
            "warnings": [str(i) for i in self.warnings],
        }


def _unwrap_workflow(data: Any, file_label: str) -> dict[str, Any]:
    """Accept raw API dict or {'prompt': {...}} wrapper; reject UI format."""
    if isinstance(data, dict) and isinstance(data.get("prompt"), dict):
        data = data["prompt"]
    if isinstance(data, dict) and isinstance(data.get("nodes"), list):
        raise ValueError(
            f"{file_label}: this looks like UI graph format (nodes/links). "
            "Export it from ComfyUI as API format first."
        )
    if not isinstance(data, dict) or not data:
        raise ValueError(f"{file_label}: not a non-empty workflow dict")
    return data


def _is_link(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], (str, int))
        and isinstance(value[1], int)
    )


def _combo_choices(spec: Any) -> Optional[list[Any]]:
    """If an input spec is a combo, return its choices list; else None."""
    if isinstance(spec, (list, tuple)) and spec and isinstance(spec[0], (list, tuple)):
        return list(spec[0])
    return None


def _spec_type(spec: Any) -> Optional[str]:
    """Expected input type string for link inputs ('MODEL', 'CLIP', ...)."""
    if isinstance(spec, str):
        return spec
    if isinstance(spec, (list, tuple)) and spec and isinstance(spec[0], str):
        return spec[0]
    return None


def _spec_options(spec: Any) -> dict[str, Any]:
    if isinstance(spec, (list, tuple)) and len(spec) > 1 and isinstance(spec[1], dict):
        return spec[1]
    return {}


def _is_wildcard_type(type_name: Optional[str]) -> bool:
    """'*' and match-type passthroughs (COMFY_MATCHTYPE_*) accept/propagate any type."""
    if not type_name:
        return True
    return type_name == "*" or type_name.startswith("COMFY_MATCHTYPE")


def _types_compatible(expected: Optional[str], actual: Optional[str]) -> bool:
    if _is_wildcard_type(expected) or _is_wildcard_type(actual):
        return True
    # Union types from autogrow templates, e.g. "FLOAT,INT,BOOLEAN"
    if "," in expected:
        return actual in {t.strip() for t in expected.split(",")}
    return expected == actual


def _autogrow_child_type(spec: Any) -> Optional[str]:
    """Expected type for a child of a COMFY_AUTOGROW_V3 input ('values.a')."""
    if not (
        isinstance(spec, (list, tuple))
        and spec
        and spec[0] == "COMFY_AUTOGROW_V3"
        and len(spec) > 1
        and isinstance(spec[1], dict)
    ):
        return None
    try:
        req = spec[1]["template"]["input"]["required"]
        # Single-child template: 'value' for names-style, e.g. 'image' for
        # prefix-style (BatchImagesNode's 'images' → 'images.image0').
        child = next(iter(req.values()))
        return _spec_type(child)
    except (KeyError, TypeError, StopIteration):
        return None


def _validate_scalar(
    report: ValidationReport,
    node_id: str,
    input_name: str,
    value: Any,
    spec: Any,
    inventory: Optional[Inventory],
) -> None:
    choices = _combo_choices(spec)
    if choices is not None:
        if value in choices:
            return
        looks_like_model = input_name in MODEL_INPUT_NAMES or (
            isinstance(value, str) and value.lower().endswith(WEIGHT_SUFFIXES)
        )
        if looks_like_model and isinstance(value, str):
            if inventory is not None and inventory.resolve(value) is not None:
                report.warn(
                    node_id,
                    input_name,
                    f"'{value}' not in server combo choices but exists in local "
                    "inventory (server may need a refresh, or cache is stale)",
                )
            else:
                report.error(
                    node_id,
                    input_name,
                    f"model file '{value}' not in server choices and not found "
                    "in local model inventory (typo or missing download?)",
                )
            return
        preview = ", ".join(repr(c) for c in choices[:6])
        if len(choices) > 6:
            preview += f", ... ({len(choices)} total)"
        report.error(
            node_id,
            input_name,
            f"value {value!r} not in combo choices: {preview}",
        )
        return

    spec_type = _spec_type(spec)
    options = _spec_options(spec)
    if spec_type == "INT":
        if isinstance(value, bool) or not isinstance(value, int):
            report.error(node_id, input_name, f"expected INT, got {value!r}")
            return
        _check_range(report, node_id, input_name, value, options)
    elif spec_type == "FLOAT":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            report.error(node_id, input_name, f"expected FLOAT, got {value!r}")
            return
        _check_range(report, node_id, input_name, float(value), options)
    elif spec_type in ("STRING", "COMBO"):
        if not isinstance(value, str):
            report.error(node_id, input_name, f"expected STRING, got {value!r}")
    elif spec_type == "BOOLEAN":
        if not isinstance(value, bool):
            report.error(node_id, input_name, f"expected BOOLEAN, got {value!r}")
    # Other type strings (IMAGE, MODEL, ...) as scalar widgets are unusual but
    # tolerated — converted widgets accept arbitrary upstream link values.


def _check_range(
    report: ValidationReport,
    node_id: str,
    input_name: str,
    value: float,
    options: dict[str, Any],
) -> None:
    lo = options.get("min")
    hi = options.get("max")
    if lo is not None and value < lo:
        report.error(node_id, input_name, f"value {value} below min {lo}")
    if hi is not None and value > hi:
        report.error(node_id, input_name, f"value {value} above max {hi}")


def _scalar_int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def enforce_ltx_frame_law(
    workflow: dict[str, Any],
    report: ValidationReport,
    *,
    strict: bool = False,
) -> None:
    """WARN + snap illegal 8n+1 counts unless --strict (ERROR). Never leave length=8."""
    video_target: Optional[int] = None
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type")
        if class_type not in LTX_LENGTH_CLASSES:
            continue
        inputs = node.setdefault("inputs", {})
        if not isinstance(inputs, dict):
            continue
        key = next((k for k in LTX_LENGTH_KEYS if k in inputs), "length")
        raw = _scalar_int(inputs.get(key))
        if raw is None:
            continue
        snapped = snap_ltx_frames(raw)
        if not is_valid_ltx_frames(raw):
            msg = (
                f"LTX {key}={raw} is not a valid 8n+1 count (min 9); "
                f"{'refusing' if strict else 'auto-correcting to'} {snapped}"
            )
            if strict:
                report.error(str(node_id), key, msg)
            else:
                report.warn(str(node_id), key, msg)
                inputs[key] = snapped
                raw = snapped
        if video_target is None:
            video_target = raw if strict else int(inputs.get(key) or snapped)

    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        if node.get("class_type") not in LTX_AUDIO_FRAME_CLASSES:
            continue
        inputs = node.setdefault("inputs", {})
        if not isinstance(inputs, dict):
            continue
        raw = _scalar_int(inputs.get("frames_number"))
        if raw is None:
            continue
        desired = video_target if video_target is not None else snap_ltx_frames(raw)
        desired = snap_ltx_frames(desired)
        if raw != desired:
            msg = (
                f"LTX frames_number={raw} must stay paired with video length "
                f"({desired}, 8n+1 min 9); "
                f"{'refusing' if strict else 'auto-correcting'}"
            )
            if strict:
                report.error(str(node_id), "frames_number", msg)
            else:
                report.warn(str(node_id), "frames_number", msg)
                inputs["frames_number"] = desired


def validate_workflow(
    workflow: dict[str, Any],
    object_info: dict[str, Any],
    *,
    file_label: str = "<workflow>",
    object_info_source: str = "unknown",
    inventory: Optional[Inventory] = None,
    strict: bool = False,
) -> ValidationReport:
    report = ValidationReport(file=file_label, object_info_source=object_info_source)
    if inventory is None:
        try:
            inventory = load_inventory()
        except Exception:
            inventory = None

    dropped = bypass_optional_accelerators(workflow, object_info)
    for nid, class_type in dropped:
        report.warn(
            nid,
            "",
            f"optional node '{class_type}' missing from object_info; bypassed "
            "(typed upstream rewired past the node; pack not auto-installed)",
        )

    enforce_ltx_frame_law(workflow, report, strict=strict)

    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            report.error(node_id, "", "node entry is not a dict")
            continue
        class_type = node.get("class_type")
        if not class_type:
            report.error(node_id, "", "missing class_type")
            continue
        info = object_info.get(class_type)
        if info is None:
            if is_optional_node(class_type):
                report.warn(
                    node_id,
                    "",
                    f"optional node '{class_type}' missing from object_info; "
                    "bypass (do not auto-install the pack)",
                )
                continue
            report.error(
                node_id,
                "",
                f"unknown class_type '{class_type}' (missing custom node pack?)",
            )
            continue

        inputs = node.get("inputs") or {}
        if not isinstance(inputs, dict):
            report.error(node_id, "", "'inputs' is not a dict")
            continue

        spec_in = info.get("input") or {}
        known: dict[str, Any] = {}
        for section in ("required", "optional", "hidden"):
            for name, spec in (spec_in.get(section) or {}).items():
                known.setdefault(name, spec)
        required = set((spec_in.get("required") or {}).keys())

        # 1. required inputs present
        for name in sorted(required):
            if name in inputs:
                continue
            spec = known.get(name)
            # Autogrow groups ('values') are satisfied by children ('values.a')
            if _autogrow_child_type(spec) is not None and any(
                k.startswith(name + ".") for k in inputs
            ):
                continue
            report.error(node_id, name, f"required input missing (class {class_type})")

        for name, value in inputs.items():
            spec = known.get(name)
            expected: Optional[str] = None
            if spec is None and "." in name:
                # Child of an autogrow group ('values.a' → 'values')
                base_spec = known.get(name.split(".", 1)[0])
                child_type = _autogrow_child_type(base_spec)
                if child_type is not None:
                    spec = base_spec
                    expected = child_type
            if spec is None:
                report.warn(
                    node_id,
                    name,
                    f"input not declared in /object_info for {class_type}",
                )
                continue

            # 2. link integrity
            if _is_link(value):
                src_id, out_idx = str(value[0]), value[1]
                src_node = workflow.get(src_id)
                if src_node is None:
                    report.error(node_id, name, f"link source node '{src_id}' not in workflow")
                    continue
                src_class = src_node.get("class_type")
                src_info = object_info.get(src_class)
                if src_info is None:
                    # already reported on the source node itself
                    continue
                outputs = src_info.get("output") or []
                if not isinstance(out_idx, int) or out_idx < 0 or out_idx >= len(outputs):
                    report.error(
                        node_id,
                        name,
                        f"link output index {out_idx} out of range for "
                        f"{src_class} ({len(outputs)} outputs)",
                    )
                    continue
                if expected is None:
                    expected = _spec_type(spec)
                actual = outputs[out_idx]
                if not _types_compatible(expected, actual):
                    report.error(
                        node_id,
                        name,
                        f"type mismatch: {src_class} output[{out_idx}] is "
                        f"{actual}, input expects {expected}",
                    )
                continue

            # 3. widget/scalar validation
            _validate_scalar(report, str(node_id), name, value, spec, inventory)

    return report


def validate_workflow_file(
    path: Path,
    *,
    client: Optional[ComfyClient] = None,
    prefer_live: bool = True,
    strict: bool = False,
) -> ValidationReport:
    path = Path(path)
    label = str(path)
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    try:
        workflow = _unwrap_workflow(data, label)
    except ValueError as e:
        report = ValidationReport(file=label)
        report.error("-", "", str(e))
        return report

    client = client or ComfyClient()
    try:
        object_info, source = client.load_object_info(prefer_live=prefer_live)
    except Exception as e:
        report = ValidationReport(file=label)
        report.error("-", "", f"could not load /object_info: {e}")
        return report

    return validate_workflow(
        workflow,
        object_info,
        file_label=label,
        object_info_source=source,
        strict=strict,
    )


def format_report(report: ValidationReport) -> str:
    lines = [
        f"{'PASS' if report.ok else 'FAIL'}  {report.file}  "
        f"(object_info: {report.object_info_source})"
    ]
    for issue in report.errors:
        lines.append(f"  {issue}")
    for issue in report.warnings:
        lines.append(f"  {issue}")
    if report.ok and not report.warnings:
        lines.append("  no issues")
    return "\n".join(lines)
