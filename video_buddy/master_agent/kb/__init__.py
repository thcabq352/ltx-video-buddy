"""Knowledge base — local RAG over workflows and run records."""

from master_agent.kb.ingest import ingest_all_runs, ingest_run_record, ingest_workflows
from master_agent.kb.recall import recall_similar_runs, recall_workflows
from master_agent.kb.store import kb_available, search

__all__ = [
    "ingest_all_runs",
    "ingest_run_record",
    "ingest_workflows",
    "kb_available",
    "recall_similar_runs",
    "recall_workflows",
    "search",
]
