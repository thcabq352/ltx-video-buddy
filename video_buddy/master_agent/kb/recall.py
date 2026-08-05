"""KB recall — format similar past runs/workflows as prompt context."""

from __future__ import annotations

from master_agent.config import KB_RECALL_K
from master_agent.kb.store import COLLECTION_RUNS, COLLECTION_WORKFLOWS, search


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
