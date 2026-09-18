"""Consume a previs ``buddy.comfy.attach/v1`` / WorkflowPatchPlan recipe.

Contract source (sibling repo; do not fork it here):
https://github.com/thcabq352/your-video-buddy/blob/main/docs/COMFY_ATTACH_CONTRACT.md

Buddy only patches compatible LTX/Wan API graphs (existing loaders + prompt
widgets). It does not invent topology, face scores, or a live Comfy success.
"""

from __future__ import annotations

import copy
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from master_agent.comfy.client import ComfyClient, ComfyClientError
from master_agent.comfy.cli_run import unwrap_workflow
from master_agent.comfy.graph_ops import is_optional_node
from master_agent.config import RUNS_DIR

ATTACH_SCHEMA = "buddy.comfy.attach/v1"
ATTACH_SCHEMA_ALIASES = frozenset(
    {
        ATTACH_SCHEMA,
        "buddy.comfy.attach",
        "https://buddy.video/schema/comfy.attach/v1",
    }
)
CONTROL_CHANNELS = ("openpose", "depth", "edges", "camera")

_TITLE_MARKERS: dict[str, tuple[str, ...]] = {
    "openpose": ("openpose", "dwpose", "pose", "skeleton"),
    "depth": ("depth", "depthmap", "midas"),
    "edges": ("edges", "canny", "lineart", "scribble"),
}
_CLASS_MARKERS: dict[str, tuple[str, ...]] = {
    "openpose": ("dwpreprocessor", "openposepreprocessor", "openpose"),
    "depth": ("depthanything", "midas", "depthcrafter", "ledepth"),
    "edges": ("canny", "lineart", "scribble", "hedpreprocessor"),
}
_IMAGE_KEYS = ("image", "image_path", "filename", "path", "file")
_VIDEO_KEYS = ("video", "video_path", "control_video")
_PROMPT_KEYS = ("prompt_additive", "prompt", "additive", "text")


class AttachError(ValueError):
    """Recipe is invalid or the graph cannot honestly accept the pack."""


@dataclass
class ControlChannel:
    name: str
    image: Optional[str] = None
    video: Optional[str] = None
    prompt_additive: Optional[str] = None

    @property
    def has_media(self) -> bool:
        return bool(self.image or self.video)

    @property
    def present(self) -> bool:
        return self.has_media or bool(self.prompt_additive)


@dataclass
class AttachRecipe:
    schema: str
    previs_source: str = ""
    channels: dict[str, ControlChannel] = field(default_factory=dict)
    preferred_variants: list[str] = field(default_factory=list)
    required_nodes: list[str] = field(default_factory=list)
    patches: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def control_pack_present(self) -> bool:
        return any(ch.present for ch in self.channels.values())


@dataclass
class AttachApplyResult:
    ok: bool
    workflow: dict[str, Any]
    previs_source: str
    control_pack_present: bool
    control_pack_used: dict[str, bool]
    applied: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _as_dict(raw: Any, *, label: str) -> dict[str, Any]:
    if isinstance(raw, Path) or (isinstance(raw, str) and ("{" not in raw)):
        path = Path(raw)
        if not path.is_file():
            raise AttachError(f"{label} not found: {path}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise AttachError(f"{label} is not valid JSON: {e}") from e
    elif isinstance(raw, str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise AttachError(f"{label} is not valid JSON: {e}") from e
    elif isinstance(raw, AttachRecipe):
        return raw.raw or {"schema": raw.schema}
    elif isinstance(raw, dict):
        data = raw
    else:
        raise AttachError(f"{label} must be a JSON object, path, or dict")
    if not isinstance(data, dict):
        raise AttachError(f"{label} must be a JSON object")
    return data


def _schema_token(data: dict[str, Any]) -> str:
    for key in ("schema", "$schema", "kind", "type"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    inner = data.get("workflow_patch_plan") or data.get("recipe")
    if isinstance(inner, dict):
        return _schema_token(inner)
    return ""


def _is_v1(token: str) -> bool:
    lowered = token.lower().rstrip("/")
    if token in ATTACH_SCHEMA_ALIASES or lowered in ATTACH_SCHEMA_ALIASES:
        return True
    if lowered.endswith("buddy.comfy.attach/v1"):
        return True
    if token == "WorkflowPatchPlan":
        return True
    return False


def _channel_from_mapping(name: str, spec: Any) -> ControlChannel:
    if spec is None:
        return ControlChannel(name=name)
    if isinstance(spec, str):
        if name == "camera":
            return ControlChannel(name=name, prompt_additive=spec)
        return ControlChannel(name=name, image=spec)
    if not isinstance(spec, dict):
        raise AttachError(f"control_pack.{name} must be an object or string")
    image = next((str(spec[k]) for k in _IMAGE_KEYS if spec.get(k)), None)
    video = next((str(spec[k]) for k in _VIDEO_KEYS if spec.get(k)), None)
    prompt = next((str(spec[k]) for k in _PROMPT_KEYS if spec.get(k)), None)
    return ControlChannel(name=name, image=image, video=video, prompt_additive=prompt)


def _parse_channels(data: dict[str, Any]) -> dict[str, ControlChannel]:
    pack = (
        data.get("control_pack")
        or data.get("channels")
        or data.get("pack")
        or {}
    )
    if pack is None:
        pack = {}
    if not isinstance(pack, dict):
        raise AttachError("control_pack must be an object")
    additives = data.get("prompt_additives")
    if isinstance(additives, dict):
        pack = dict(pack)
        for name, text in additives.items():
            if name not in CONTROL_CHANNELS:
                continue
            existing = pack.get(name)
            if isinstance(existing, dict):
                existing = dict(existing)
                existing.setdefault("prompt_additive", text)
                pack[name] = existing
            elif existing is None:
                pack[name] = {"prompt_additive": text}
    out: dict[str, ControlChannel] = {}
    for name in CONTROL_CHANNELS:
        out[name] = _channel_from_mapping(name, pack.get(name))
    return out


def _preferred_variants(data: dict[str, Any]) -> list[str]:
    block = data.get("director") or data.get("director_preferences") or {}
    if not isinstance(block, dict):
        return []
    prefs: list[str] = []
    single = block.get("variant") or block.get("preferred_variant")
    if isinstance(single, str) and single.strip():
        prefs.append(single.strip())
    raw = block.get("preferred_variants") or block.get("variants") or []
    if isinstance(raw, str):
        raw = [raw]
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, str) and item.strip() and item.strip() not in prefs:
                prefs.append(item.strip())
    return prefs


def load_attach_recipe(raw: Any) -> AttachRecipe:
    """Load and type-check a v1 attach recipe / WorkflowPatchPlan."""
    data = _as_dict(raw, label="attach recipe")
    inner = data.get("workflow_patch_plan") or data.get("recipe")
    if isinstance(inner, dict) and "control_pack" not in data and "channels" not in data:
        data = {**inner, **{k: data[k] for k in data if k not in ("workflow_patch_plan", "recipe")}}
    token = _schema_token(data)
    if not token:
        raise AttachError("attach recipe missing schema (expected buddy.comfy.attach/v1)")
    if not _is_v1(token):
        raise AttachError(
            f"unsupported attach schema {token!r}; this Buddy consumes {ATTACH_SCHEMA}"
        )
    required = data.get("required_nodes") or data.get("required_class_types") or []
    if required is None:
        required = []
    if not isinstance(required, list):
        raise AttachError("required_nodes must be a list of class_type names")
    patches = data.get("patches") or data.get("ops") or []
    if patches is None:
        patches = []
    if not isinstance(patches, list):
        raise AttachError("patches must be a list of graph ops")
    previs = data.get("previs_source") or data.get("source") or data.get("previs") or ""
    recipe = AttachRecipe(
        schema=ATTACH_SCHEMA,
        previs_source=str(previs or ""),
        channels=_parse_channels(data),
        preferred_variants=_preferred_variants(data),
        required_nodes=[str(x) for x in required if x],
        patches=[p for p in patches if isinstance(p, dict)],
        raw=data,
    )
    return recipe


def attach_director_override(recipe: Any) -> Optional[str]:
    """First allowlisted director preference when a control pack is present."""
    if recipe is None:
        return None
    if not isinstance(recipe, AttachRecipe):
        try:
            recipe = load_attach_recipe(recipe)
        except AttachError:
            return None
    if not recipe.control_pack_present:
        return None
    from master_agent.comfy.catalog import is_known_variant
    from master_agent.config import WORKFLOW_FILES

    allowed = set(WORKFLOW_FILES)
    for variant in recipe.preferred_variants:
        key = variant.strip().lower()
        if key in allowed or is_known_variant(key):
            return key
    return None


def _iter_nodes(workflow: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    for nid, node in workflow.items():
        if isinstance(node, dict) and node.get("class_type"):
            out.append((str(nid), node))
    return out


def _title(node: dict[str, Any]) -> str:
    return str((node.get("_meta") or {}).get("title") or "").lower()


def _matches_channel(node: dict[str, Any], channel: str) -> bool:
    title = _title(node)
    ctype = str(node.get("class_type") or "").lower()
    if any(m in title for m in _TITLE_MARKERS.get(channel, ())):
        return True
    if any(m in ctype.replace(" ", "") for m in _CLASS_MARKERS.get(channel, ())):
        return True
    return False


def _set_media(node: dict[str, Any], channel: ControlChannel) -> bool:
    inputs = node.setdefault("inputs", {})
    if not isinstance(inputs, dict):
        node["inputs"] = {}
        inputs = node["inputs"]
    ctype = str(node.get("class_type") or "")
    if channel.image and ctype in {"LoadImage", "LoadImageMask", "EasyLoadImage"}:
        inputs["image"] = channel.image
        return True
    if channel.video and ctype in {"LoadVideo", "VHS_LoadVideo", "LoadVideoPath"}:
        for key in ("video", "file", "path"):
            if key in inputs or key == "video":
                inputs[key] = channel.video
                return True
    if channel.image:
        for key in ("image", "control_image", channel.name):
            value = inputs.get(key)
            if key in inputs and not isinstance(value, list):
                inputs[key] = channel.image
                return True
    if channel.video:
        for key in ("video", "control_video", "file"):
            value = inputs.get(key)
            if key in inputs and not isinstance(value, list):
                inputs[key] = channel.video
                return True
    return False


def _append_prompt(text: str, additive: str) -> str:
    add = additive.strip()
    if not add:
        return text
    if add in text:
        return text
    base = text.rstrip()
    if not base:
        return add
    sep = "" if base.endswith((",", ";", ".")) else ","
    return f"{base}{sep} {add}"


def _apply_prompt_additives(workflow: dict[str, Any], additives: list[str]) -> list[str]:
    applied: list[str] = []
    if not additives:
        return applied
    blob = " ".join(additives)
    for _nid, node in _iter_nodes(workflow):
        title = _title(node)
        ctype = str(node.get("class_type") or "")
        if "neg" in title:
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        promptish = ctype in {
            "CLIPTextEncode",
            "CLIPTextEncodeFlux",
            "Wan22FunControlToVideo",
            "WanFunControlToVideo",
        } or any(tok in ctype.lower() for tok in ("cliptextencode", "prompt"))
        if not promptish and "prompt" not in title:
            continue
        for key in ("text", "prompt", "string", "positive"):
            value = inputs.get(key)
            if isinstance(value, str):
                inputs[key] = _append_prompt(value, blob)
                applied.append(f"{_nid}.{key}")
                break
    return applied


def _missing_object_info(
    workflow: dict[str, Any],
    object_info: dict[str, Any],
    extra: list[str],
) -> list[str]:
    missing: list[str] = []
    seen: set[str] = set()
    for _nid, node in _iter_nodes(workflow):
        ctype = str(node.get("class_type") or "")
        if not ctype or ctype in seen:
            continue
        seen.add(ctype)
        if ctype not in object_info and not is_optional_node(ctype):
            missing.append(ctype)
    for ctype in extra:
        if ctype and ctype not in object_info and ctype not in missing:
            missing.append(ctype)
    return missing


def apply_attach_recipe(
    workflow: dict[str, Any],
    recipe: Any,
    *,
    object_info: dict[str, Any] | None = None,
) -> AttachApplyResult:
    """Patch control media + camera prompt additives onto a compatible graph."""
    if not isinstance(recipe, AttachRecipe):
        recipe = load_attach_recipe(recipe)
    wf = copy.deepcopy(unwrap_workflow(workflow))
    if object_info is not None:
        missing = _missing_object_info(wf, object_info, recipe.required_nodes)
        if missing:
            raise AttachError(
                "nodes missing from object_info: " + ", ".join(missing)
            )

    used = {name: False for name in CONTROL_CHANNELS}
    applied: list[str] = []
    for name in ("openpose", "depth", "edges"):
        channel = recipe.channels[name]
        if not channel.has_media:
            continue
        placed = False
        for _nid, node in _iter_nodes(wf):
            if not _matches_channel(node, name):
                continue
            if _set_media(node, channel):
                placed = True
                used[name] = True
                applied.append(f"{name}:{_nid}")
                break
        if not placed:
            raise AttachError(
                f"{name} control pack has media but this graph has no compatible "
                f"LoadImage/LoadVideo (title/class must mention {name})"
            )

    additives = [
        recipe.channels[name].prompt_additive
        for name in CONTROL_CHANNELS
        if recipe.channels[name].prompt_additive
    ]
    prompt_hits = _apply_prompt_additives(wf, [a for a in additives if a])
    if prompt_hits:
        applied.extend(prompt_hits)
        for name in CONTROL_CHANNELS:
            if recipe.channels[name].prompt_additive:
                used[name] = True
    elif any(recipe.channels[name].prompt_additive for name in CONTROL_CHANNELS):
        raise AttachError(
            "camera/prompt additives present but no compatible positive prompt widget"
        )

    if recipe.patches:
        from master_agent.comfy.graph_ops import apply_ops

        wf, op_res = apply_ops(wf, recipe.patches)
        if not op_res.ok:
            raise AttachError("recipe patches failed: " + "; ".join(op_res.errors[:5]))
        applied.extend(f"op:{op.get('op')}" for op in op_res.applied)

    if object_info is not None:
        missing = _missing_object_info(wf, object_info, recipe.required_nodes)
        if missing:
            raise AttachError(
                "nodes missing from object_info: " + ", ".join(missing)
            )

    return AttachApplyResult(
        ok=True,
        workflow=wf,
        previs_source=recipe.previs_source,
        control_pack_present=recipe.control_pack_present,
        control_pack_used=used,
        applied=applied,
    )


def evaluate_attach_judge_rules(record: dict[str, Any]) -> dict[str, bool]:
    """Judge bookkeeping (c)/(d) — not aesthetic scores.

    (c) previs_source is recorded when a recipe was applied.
    (d) if a control pack was present, at least one channel is marked used.
    """
    from master_agent.judge.quality_bar import attach_rule_passes

    return attach_rule_passes(record)


def run_attach(
    *,
    recipe: Any,
    workflow: dict[str, Any],
    client: Optional[ComfyClient] = None,
    submit: bool = False,
    runs_dir: Optional[Path] = None,
    variant: Optional[str] = None,
    object_info: dict[str, Any] | None = None,
    object_info_source: str = "unknown",
) -> dict[str, Any]:
    """Patch + validate. Queue ``/prompt`` only when ``submit`` is true."""
    loaded = recipe if isinstance(recipe, AttachRecipe) else load_attach_recipe(recipe)
    client = client or ComfyClient()
    info = object_info
    source = object_info_source
    if info is None:
        try:
            info, source = client.load_object_info(prefer_live=True)
        except ComfyClientError as e:
            raise AttachError(f"cannot load /object_info: {e}") from e
    applied = apply_attach_recipe(workflow, loaded, object_info=info)
    rec: dict[str, Any] = {
        "ok": True,
        "schema": ATTACH_SCHEMA,
        "previs_source": applied.previs_source,
        "control_pack_present": applied.control_pack_present,
        "control_pack_used": applied.control_pack_used,
        "applied": applied.applied,
        "variant": variant,
        "dry_run": not submit,
        "submitted": False,
        "prompt_id": None,
        "object_info": source,
        "nodes": len(applied.workflow),
        "judge_rules": {},
        "error": None,
    }
    rec["judge_rules"] = evaluate_attach_judge_rules(rec)
    if submit:
        try:
            rec["prompt_id"] = client.queue_prompt(applied.workflow)
        except ComfyClientError as e:
            rec["ok"] = False
            rec["error"] = f"Comfy /prompt failed: {e}"
            rec["submitted"] = False
        else:
            rec["submitted"] = True
            rec["dry_run"] = False
    rec["workflow"] = applied.workflow
    dest = Path(runs_dir) if runs_dir else RUNS_DIR
    dest.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex[:12]
    rec["run_id"] = run_id
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = dest / f"{ts}_{run_id}_attach.json"
    serializable = {k: v for k, v in rec.items() if k != "workflow"}
    path.write_text(json.dumps(serializable, indent=1, default=str) + "\n", encoding="utf-8")
    rec["record_path"] = str(path)
    return rec
