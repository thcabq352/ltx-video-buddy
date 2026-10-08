"""Write a draft catalog entry from an ingested workflow.

This edits the local checkout only: ``workflows/<name>.json`` and an appended
block in ``workflows/manifests.yaml``. It does not commit, push, or open a
pull request, and it does not replace the catalog default (``base`` or any
id ``default_variant_ids()`` already publishes).
"""

from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Any

import yaml

from master_agent.comfy.ingest.normalize import IngestError
from master_agent.comfy.ingest.store import load_bundle, slugify


def _workflows_dir(explicit: Path | None) -> Path:
    if explicit is not None:
        return Path(explicit)
    from master_agent import config

    return Path(config.WORKFLOWS_DIR)


def _default_ids() -> set[str]:
    try:
        from master_agent.comfy.catalog import default_variant_ids

        return set(default_variant_ids())
    except Exception:
        return {"base"}


def _comment_for(role: str, learned: dict[str, Any]) -> str:
    for warning in learned.get("warnings") or []:
        if warning.get("role") == role:
            detail = str(warning.get("detail") or "best guess").replace("\n", " ")
            return f"low confidence: {detail}"
    return ""


def _render_field(role: str, spec: dict[str, Any], learned: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    note = _comment_for(role, learned)
    if note or str(spec.get("confidence") or "high") != "high":
        text = note or f"low confidence: {spec.get('confidence')}"
        lines.append(f"    # {text}")
    lines.append(f"    {role}:")
    if spec.get("node_id"):
        lines.append(f"      node_id: {json.dumps(str(spec['node_id']))}")
    elif spec.get("class_type"):
        lines.append(f"      class_type: {spec['class_type']}")
        lines.append(f"      index: {int(spec.get('index') or 0)}")
    else:
        raise IngestError(f"field {role} has neither node_id nor class_type")
    input_name = spec.get("input") or spec.get("key")
    if not input_name:
        raise IngestError(f"field {role} has no input name")
    lines.append(f"      input: {input_name}")
    return lines


def _render_mapping(label: str, mapping: dict[str, Any], indent: str = "  ") -> list[str]:
    if not mapping:
        return [f"{indent}{label}: {{}}"]
    lines = [f"{indent}{label}:"]
    for key, spec in mapping.items():
        if not isinstance(spec, dict):
            continue
        lines.append(f"{indent}  {key}:")
        if spec.get("node_id"):
            node_id = spec["node_id"]
            if isinstance(node_id, list):
                rendered = json.dumps([str(item) for item in node_id])
                lines.append(f"{indent}    node_id: {rendered}")
            else:
                lines.append(f"{indent}    node_id: {json.dumps(str(node_id))}")
        elif spec.get("class_type"):
            lines.append(f"{indent}    class_type: {spec['class_type']}")
            lines.append(f"{indent}    index: {int(spec.get('index') or 0)}")
        if spec.get("input"):
            lines.append(f"{indent}    input: {spec['input']}")
        if "optional" in spec:
            lines.append(f"{indent}    optional: {str(bool(spec['optional'])).lower()}")
    return lines


def render_manifest_entry(learned: dict[str, Any], *, variant: str, filename: str) -> str:
    """YAML block using the manifests.yaml vocabulary. Low-confidence fields are comments."""
    description = (
        f"Draft promoted from ingested {learned.get('slug') or variant}. "
        "Not the catalog default."
    )
    lines = [
        f"{variant}:",
        f"  file: {filename}",
        f"  description: {json.dumps(description)}",
        f"  vram_class: {learned.get('vram_class') or 'unknown'}",
        "  draft: true",
    ]
    requires = [str(item) for item in (learned.get("requires") or []) if str(item).strip()]
    if requires:
        quoted = ", ".join(json.dumps(item) for item in requires)
        lines.append(f"  requires: [{quoted}]")
    else:
        lines.append("  requires: []")
    lines.append("  fields:")
    fields = learned.get("fields") or {}
    if not fields:
        lines.append("    {}")
    for role, spec in fields.items():
        if isinstance(spec, dict):
            lines.extend(_render_field(str(role), spec, learned))
    inputs = learned.get("inputs") if isinstance(learned.get("inputs"), dict) else {}
    outputs = learned.get("outputs") if isinstance(learned.get("outputs"), dict) else {}
    lines.extend(_render_mapping("inputs", inputs))
    lines.extend(_render_mapping("outputs", outputs))
    return "\n".join(lines).rstrip() + "\n"


def _unified(path: str, before: str, after: str) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )


def promote_slug(
    slug: str,
    *,
    variant_name: str | None = None,
    force: bool = False,
    workflows_dir: Path | None = None,
) -> dict[str, Any]:
    """Write the draft files and return the diff. Does not call git."""
    bundle = load_bundle(slug)
    learned = bundle["learned"]
    readiness = learned.get("readiness") or {}
    missing_nodes = list(readiness.get("missing_nodes") or [])
    missing_models = list(readiness.get("missing_models") or [])
    if (missing_nodes or missing_models) and not force:
        parts = []
        if missing_nodes:
            parts.append("nodes " + ", ".join(str(item) for item in missing_nodes))
        if missing_models:
            parts.append("models " + ", ".join(str(item) for item in missing_models))
        raise IngestError(
            "refuse to promote: readiness has missing "
            + "; ".join(parts)
            + ". Pass --force to write the draft anyway. "
            "Buddy does not install nodes or substitute weights."
        )
    variant = slugify(variant_name or bundle["slug"])
    protected = _default_ids()
    protected.add("base")
    if variant in protected:
        raise IngestError(
            f"refuse to promote {variant!r}: that id is a catalog default. "
            "Pick another --variant-name. An ingested graph is not made the default."
        )
    root = _workflows_dir(workflows_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    filename = f"{variant}.json"
    workflow_path = (root / filename).resolve()
    manifest_path = (root / "manifests.yaml").resolve()
    try:
        workflow_path.relative_to(root)
        manifest_path.relative_to(root)
    except ValueError as exc:
        raise IngestError(f"refuse to write outside workflows/: {variant}") from exc
    before_manifest = manifest_path.read_text(encoding="utf-8") if manifest_path.is_file() else ""
    existing = yaml.safe_load(before_manifest) if before_manifest.strip() else {}
    if not isinstance(existing, dict):
        existing = {}
    if variant in existing:
        raise IngestError(
            f"workflows/manifests.yaml already has {variant!r}. "
            "Promote does not overwrite a catalog entry."
        )
    if "base" in existing and variant == "base":
        raise IngestError("refuse to promote over the catalog default 'base'.")
    before_workflow = workflow_path.read_text(encoding="utf-8") if workflow_path.is_file() else ""
    workflow_text = json.dumps(bundle["workflow"], indent=2) + "\n"
    entry = render_manifest_entry(learned, variant=variant, filename=filename)
    after_manifest = before_manifest
    if after_manifest and not after_manifest.endswith("\n"):
        after_manifest += "\n"
    if after_manifest and not after_manifest.endswith("\n\n"):
        after_manifest += "\n"
    after_manifest += entry
    workflow_path.write_text(workflow_text, encoding="utf-8")
    manifest_path.write_text(after_manifest, encoding="utf-8")
    diff = _unified(f"workflows/{filename}", before_workflow, workflow_text)
    diff += _unified("workflows/manifests.yaml", before_manifest, after_manifest)
    return {
        "variant": variant,
        "slug": bundle["slug"],
        "workflow_path": workflow_path,
        "manifest_path": manifest_path,
        "diff": diff,
        "forced": bool(force and (missing_nodes or missing_models)),
        "catalog_default": "base",
    }


def format_promote(result: dict[str, Any]) -> str:
    lines = [
        f"promoted draft variant={result['variant']} from {result['slug']}",
        f"wrote {result['workflow_path']}",
        f"updated {result['manifest_path']}",
        "catalog default unchanged (base)",
        "not committed, not pushed, no pull request",
        result.get("diff") or "",
    ]
    return "\n".join(lines).rstrip() + "\n"
