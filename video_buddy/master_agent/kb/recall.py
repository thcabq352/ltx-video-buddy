"""KB recall — format similar past runs/workflows/knowledge as prompt context."""

from __future__ import annotations

from typing import Any

from master_agent.config import KB_RECALL_K
from master_agent.kb.store import (
    COLLECTION_KNOWLEDGE,
    COLLECTION_RUNS,
    COLLECTION_WORKFLOWS,
    search,
)


def recall_similar_runs(request: str, *, k: int | None = None) -> str:
    """Top-k similar past runs as a prompt-ready block ('' when KB empty/down)."""
    hits = search(COLLECTION_RUNS, request, k=k or KB_RECALL_K)
    if not hits:
        return ""
    lines = ["## Similar past runs (knowledge base)"]
    for h in hits:
        m = h.get("metadata") or {}
        status = m.get("status") or "?"
        score = m.get("score")
        head = f"- [{status} score={score:.2f}] {m.get('request', '')[:120]}"
        lines.append(head)
        # include the judge's own words — that is the learning signal
        text = h.get("text") or ""
        for line in text.splitlines():
            if line.startswith(("judge_reason:", "judge_notes:", "error:")):
                lines.append(f"  {line[:200]}")
    return "\n".join(lines)


def recall_knowledge(request: str, *, k: int | None = None) -> str:
    """Shared git knowledge. Failure notes fill the front of the block.

    ``failures/`` is stored with ``priority=high`` and queried first so a
    known-bad approach is in the prompt even when other notes embed closer.
    """
    try:
        from master_agent.kb.ingest import ensure_knowledge_ingested

        ensure_knowledge_ingested()
    except Exception:
        pass
    limit = k or KB_RECALL_K
    failures = search(
        COLLECTION_KNOWLEDGE,
        request,
        k=limit,
        where={"category": "failures"},
    )
    others = search(
        COLLECTION_KNOWLEDGE,
        request,
        k=limit,
        where={"category": {"$ne": "failures"}},
    )
    chosen = _prefer_failures(failures, others, limit)
    if not chosen:
        return ""
    lines = ["## Shared knowledge (git)"]
    for hit in chosen:
        meta = hit.get("metadata") or {}
        category = meta.get("category") or "?"
        priority = " high" if meta.get("priority") == "high" else ""
        path = meta.get("path") or hit.get("id") or ""
        lines.append(f"- [{category}{priority}] {path}")
        text = " ".join((hit.get("text") or "").split())
        if text:
            lines.append(f"  {text[:240]}")
    return "\n".join(lines)


def _prefer_failures(
    failures: list[dict[str, Any]],
    others: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    """Keep failure hits in front, then fill the remaining slots."""
    chosen: list[dict[str, Any]] = []
    seen: set[str] = set()
    failure_slots = min(2, limit) if limit else 0
    for hit in failures:
        doc_id = str(hit.get("id") or "")
        if not doc_id or doc_id in seen:
            continue
        chosen.append(hit)
        seen.add(doc_id)
        if len(chosen) >= failure_slots:
            break
    for hit in others:
        if len(chosen) >= limit:
            break
        doc_id = str(hit.get("id") or "")
        if not doc_id or doc_id in seen:
            continue
        chosen.append(hit)
        seen.add(doc_id)
    return chosen


def recall_workflows(request: str, *, k: int = 2) -> str:
    """Top-k relevant workflow digests as context ('' when unavailable)."""
    hits = search(COLLECTION_WORKFLOWS, request, k=k)
    if not hits:
        return ""
    lines = ["## Relevant workflow templates (knowledge base)"]
    for h in hits:
        first = (h.get("text") or "").splitlines()[0] if h.get("text") else ""
        lines.append(f"- {first}")
    return "\n".join(lines)
