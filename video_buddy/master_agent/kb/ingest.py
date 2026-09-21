"""KB ingestion — workflow digests and run records."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from master_agent.config import RUNS_DIR, WORKFLOWS_DIR
from master_agent.kb.store import COLLECTION_RUNS, COLLECTION_WORKFLOWS, upsert_docs

log = logging.getLogger(__name__)


def _workflow_digest(path: Path) -> str:
    """Text digest of a workflow: name, node classes, prompt strings.

    Handles both API format (dict of nodes with class_type) and UI graph
    exports ({"nodes": [...]} with type/widgets_values).
    """
    try:
        wf = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        log.debug("kb workflow digest unreadable: %s", path, exc_info=True)
        return f"workflow {path.stem} (unreadable)"
    if isinstance(wf, dict) and isinstance(wf.get("nodes"), list):
        return _ui_workflow_digest(path, wf["nodes"])
    classes: list[str] = []
    prompts: list[str] = []
    nodes = wf.values() if isinstance(wf, dict) else []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        ct = node.get("class_type")
        if ct:
            classes.append(str(ct))
        inputs = node.get("inputs") or {}
        if ct == "CLIPTextEncode":
            text = inputs.get("text")
            if isinstance(text, str) and text.strip():
                prompts.append(text.strip()[:300])
    seen = sorted(set(classes))
    parts = [
        f"workflow {path.stem}: nodes={len(classes)}",
        f"classes: {', '.join(seen[:40])}",
    ]
    for p in prompts[:3]:
        parts.append(f"prompt: {p}")
    return "\n".join(parts)


def _ui_workflow_digest(path: Path, nodes: list) -> str:
    """Digest for UI graph exports (Mickmumpitz library etc.)."""
    classes: list[str] = []
    prompts: list[str] = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        ct = node.get("type")
        if ct:
            classes.append(str(ct))
        if ct and ("TextEncode" in str(ct) or "Prompt" in str(ct)):
            for w in node.get("widgets_values") or []:
                if isinstance(w, str) and len(w.strip()) > 20:
                    prompts.append(w.strip()[:300])
                    break
    seen = sorted(set(classes))
    parts = [
        f"workflow {path.stem} (ui format): nodes={len(classes)}",
        f"classes: {', '.join(seen[:40])}",
    ]
    for p in prompts[:3]:
        parts.append(f"prompt: {p}")
    return "\n".join(parts)


def _guide_digest(path: Path) -> str:
    """Digest for a markdown guide/doc that ships alongside the workflows."""
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        log.debug("kb guide digest unreadable: %s", path, exc_info=True)
        return f"guide {path.stem} (unreadable)"
    return f"guide {path.stem}:\n{text.strip()[:4000]}"


def ingest_workflows() -> int:
    ids, texts, metas = [], [], []
    for path in sorted(WORKFLOWS_DIR.glob("*.json")):
        ids.append(f"wf:{path.stem}")
        texts.append(_workflow_digest(path))
        metas.append({"kind": "workflow", "name": path.stem, "file": path.name})
    for path in sorted(WORKFLOWS_DIR.glob("*.md")):
        ids.append(f"guide:{path.stem}")
        texts.append(_guide_digest(path))
        metas.append({"kind": "guide", "name": path.stem, "file": path.name})
    return upsert_docs(COLLECTION_WORKFLOWS, ids, texts, metas)


def _run_digest(record: dict[str, Any]) -> str:
    """Text digest of an orchestrator or pipeline run record."""
    req = record.get("request") or ""
    parts = [f"request: {req}"]
    if record.get("variant"):
        parts.append(f"variant: {record['variant']}")
    # pipeline records
    if record.get("storyboard"):
        titles = [s.get("title", "") for s in record["storyboard"] if isinstance(s, dict)]
        if titles:
            parts.append("storyboard: " + "; ".join(titles[:8]))
    if record.get("segment_scores"):
        parts.append(f"segment_scores: {record['segment_scores']}")
    if record.get("full_judge_score") is not None:
        parts.append(
            f"full_judge: score={record.get('full_judge_score')} pass={record.get('full_judge_pass')}"
        )
    if record.get("full_judge_notes"):
        parts.append(f"judge_notes: {str(record['full_judge_notes'])[:300]}")
    # orchestrator records
    if record.get("judge_score") is not None:
        parts.append(f"judge_score: {record.get('judge_score')} decision={record.get('judge_decision')}")
    if record.get("judge_reason"):
        parts.append(f"judge_reason: {str(record['judge_reason'])[:300]}")
    if record.get("status") or record.get("state"):
        parts.append(f"status: {record.get('status') or record.get('state')}")
    if record.get("error"):
        parts.append(f"error: {str(record['error'])[:200]}")
    return "\n".join(parts)


def _run_metadata(record: dict[str, Any]) -> dict[str, Any]:
    score = record.get("full_judge_score", record.get("judge_score", 0.0))
    try:
        score = float(score or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    return {
        "kind": "pipeline" if record.get("segment_paths") is not None else "run",
        "run_id": str(record.get("run_id") or ""),
        "variant": str(record.get("variant") or ""),
        "status": str(record.get("status") or record.get("state") or ""),
        "score": score,
        "passed": bool(record.get("full_judge_pass") or record.get("judge_decision") == "accept"),
        "request": str(record.get("request") or "")[:200],
    }


def ingest_run_record(record: dict[str, Any]) -> int:
    run_id = str(record.get("run_id") or "")
    if not run_id:
        return 0
    kind = "pipeline" if record.get("segment_paths") is not None else "run"
    return upsert_docs(
        COLLECTION_RUNS,
        [f"{kind}:{run_id}"],
        [_run_digest(record)],
        [_run_metadata(record)],
    )


def ingest_run_file(path: Path) -> int:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        log.debug("kb ingest skipped unreadable run file %s", path, exc_info=True)
        return 0
    return ingest_run_record(record)


def ingest_all_runs() -> int:
    total = 0
    for path in sorted(RUNS_DIR.glob("*.json")):
        total += ingest_run_file(path)
    return total
