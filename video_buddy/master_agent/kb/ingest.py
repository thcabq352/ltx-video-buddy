"""KB ingestion — workflow digests, run records, and git-synced knowledge/."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from master_agent.config import CHROMA_DIR, KNOWLEDGE_DIR, RUNS_DIR, WORKFLOWS_DIR
from master_agent.kb.store import (
    COLLECTION_KNOWLEDGE,
    COLLECTION_RUNS,
    COLLECTION_WORKFLOWS,
    delete_docs,
    kb_available,
    list_doc_metas,
    upsert_docs,
)

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


# ── git-synced knowledge/ ────────────────────────────────────────────
#
# Recorded markdown under <repo>/knowledge/ is embedded into the local
# ``knowledge`` collection. Schema docs (README.md, AGENTS.md) and EXAMPLE
# seeds are skipped — they describe the format, they are not learnings.
# Re-runs upsert by ``knowledge:<relative-path>:<content-hash>[:chunk]``
# and delete the previous id when the file changes, so Chroma does not
# accumulate copies.

_CHUNK_MAX = 4000
_SCHEMA_NAMES = frozenset({"readme.md", "agents.md"})
# Bold may wrap the label, the colon, or both: **status:** example
_STATUS_RE = re.compile(
    r"(?im)^\s*(?:[-*]\s*)?(?:\*\*|__)?status(?:\*\*|__)?\s*:\s*(?:\*\*|__)?\s*(recorded|example)\b"
)
_SECRET_RE = re.compile(
    r"(?ix)"
    r"(?:"
    r"\b(?:api[_-]?key|secret|access[_-]?token|auth[_-]?token|password|hf_token|xai_api_key)\b"
    r"\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{8,}"
    r"|"
    r"\b(?:sk-[A-Za-z0-9]{10,}|ghp_[A-Za-z0-9]{10,}|gho_[A-Za-z0-9]{10,}"
    r"|github_pat_[A-Za-z0-9_]{10,}|hf_[A-Za-z0-9]{10,}|xai-[A-Za-z0-9]{10,}"
    r"|ltxv_[A-Za-z0-9]{8,}|AKIA[0-9A-Z]{16})\b"
    r"|"
    r"\bBearer\s+[A-Za-z0-9._\-]{12,}"
    r")"
)
_ABS_PATH_RE = re.compile(
    r"(?x)"
    r"(?:"
    r"[A-Za-z]:[\\/][^\s\"'`<>|*?\]]+"
    r"|"
    r"\\\\[^\s\\]+\\[^\s\"'`<>|*?\]]+"
    r"|"
    r"/(?:home|Users)/[^\s\"'`<>|*?\]]+"
    r")"
)

_ingest_lock = threading.Lock()
_stamp_lock = threading.Lock()
_disk_stamp_cache: str | None = None
_disk_stamp_count = 0


@dataclass(frozen=True)
class KnowledgeDoc:
    id: str
    text: str
    metadata: dict[str, Any]
    path: str
    content_hash: str


@dataclass(frozen=True)
class KnowledgeSync:
    """Result of one knowledge ingest pass.

    ``docs`` is how many current chunks the collection represents after the
    pass (unchanged + newly written). ``skipped`` counts files left out.
    """

    docs: int
    skipped: int
    written: int
    deleted: int


def _chroma_store_exists() -> bool:
    try:
        return CHROMA_DIR.is_dir() and any(CHROMA_DIR.glob("*.sqlite3"))
    except OSError:
        return False


def _skip_name(name: str) -> str | None:
    lowered = name.lower()
    if lowered in _SCHEMA_NAMES:
        return "schema"
    if lowered.startswith("example"):
        return "example"
    return None


def _status_value(text: str) -> str | None:
    match = _STATUS_RE.search(text)
    if not match:
        return None
    return match.group(1).lower()


def _contains_secret(text: str) -> bool:
    return _SECRET_RE.search(text) is not None


def redact_local_paths(text: str) -> str:
    """Replace absolute local paths with a placeholder. Repo-relative paths stay."""
    return _ABS_PATH_RE.sub("[local-path]", text)


def _chunk_text(text: str, limit: int = _CHUNK_MAX) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    buf: list[str] = []
    size = 0
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if len(para) > limit:
            if buf:
                chunks.append("\n\n".join(buf))
                buf, size = [], 0
            for i in range(0, len(para), limit):
                chunks.append(para[i : i + limit])
            continue
        extra = len(para) if not buf else size + 2 + len(para)
        if buf and extra > limit:
            chunks.append("\n\n".join(buf))
            buf = [para]
            size = len(para)
        else:
            buf.append(para)
            size = len(para) if not size else size + 2 + len(para)
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks


def _doc_id(rel: str, digest: str, index: int, count: int) -> str:
    base = f"knowledge:{rel}:{digest[:16]}"
    if count == 1:
        return base
    return f"{base}:{index}"


def _posix_parts(rel: str) -> tuple[str, ...]:
    return tuple(part for part in rel.split("/") if part)


def _category(rel: str) -> str:
    parts = _posix_parts(rel)
    if len(parts) > 1:
        return parts[0]
    return "root"


def collect_knowledge_docs(
    root: Path | None = None,
) -> tuple[list[KnowledgeDoc], list[tuple[str, str]]]:
    """Walk ``knowledge/**/*.md`` and return ``(docs, [(rel, reason), ...])``.

    Does not touch Chroma. Reasons: ``schema``, ``example``, ``secret``,
    ``outside``, ``unreadable``, ``empty``.
    """
    root = (root or KNOWLEDGE_DIR).resolve()
    docs: list[KnowledgeDoc] = []
    skipped: list[tuple[str, str]] = []
    if not root.is_dir():
        return docs, skipped
    for path in sorted(root.rglob("*.md")):
        if not path.is_file():
            continue
        try:
            rel_path = path.resolve().relative_to(root)
        except ValueError:
            skipped.append((path.name, "outside"))
            log.warning("kb knowledge skip (outside knowledge root): %s", path.name)
            continue
        rel = rel_path.as_posix()
        reason = _skip_name(path.name)
        if reason:
            skipped.append((rel, reason))
            continue
        try:
            raw = path.read_text(encoding="utf-8")
        except Exception:
            log.debug("kb knowledge unreadable: %s", rel, exc_info=True)
            skipped.append((rel, "unreadable"))
            continue
        if _contains_secret(raw):
            log.warning("kb knowledge skip (secret-like content): %s", rel)
            skipped.append((rel, "secret"))
            continue
        if _status_value(raw) == "example":
            skipped.append((rel, "example"))
            continue
        body = redact_local_paths(raw).strip()
        if not body:
            skipped.append((rel, "empty"))
            continue
        category = _category(rel)
        priority = "high" if category == "failures" else "normal"
        digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
        header = f"category: {category}\npath: {rel}\npriority: {priority}\n\n"
        chunks = _chunk_text(body)
        for index, chunk in enumerate(chunks):
            docs.append(
                KnowledgeDoc(
                    id=_doc_id(rel, digest, index, len(chunks)),
                    text=f"{header}{chunk}",
                    metadata={
                        "source": "knowledge",
                        "category": category,
                        "path": rel,
                        "content_hash": digest,
                        "priority": priority,
                        "name": path.stem,
                        "chunk": index,
                        "chunks": len(chunks),
                    },
                    path=rel,
                    content_hash=digest,
                )
            )
    return docs, skipped


def _disk_stamp(root: Path) -> str:
    """Cheap change token: relative path, mtime, and size of candidate files."""
    if not root.is_dir():
        return "missing"
    rows: list[str] = []
    for path in sorted(root.rglob("*.md")):
        if not path.is_file() or _skip_name(path.name):
            continue
        try:
            rel = path.resolve().relative_to(root).as_posix()
            st = path.stat()
        except (OSError, ValueError):
            continue
        rows.append(f"{rel}:{st.st_mtime_ns}:{st.st_size}")
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def ingest_knowledge(root: Path | None = None) -> KnowledgeSync:
    """Upsert recorded ``knowledge/`` markdown into the local Chroma collection.

    Idempotent: unchanged files are not re-embedded. A changed file replaces
    its previous id. Removed files are dropped. Schema docs, EXAMPLE seeds,
    and secret-like files are not written. No-op (and no Chroma open) when
    there is nothing to index and no local store exists yet.
    """
    with _ingest_lock:
        return _ingest_knowledge_locked(root)


def _ingest_knowledge_locked(root: Path | None) -> KnowledgeSync:
    root = (root or KNOWLEDGE_DIR).resolve()
    if not root.is_dir():
        log.debug("kb knowledge dir missing: %s", root)
        return KnowledgeSync(0, 0, 0, 0)
    planned, skipped_rows = collect_knowledge_docs(root)
    skipped = len(skipped_rows)
    if not planned and not _chroma_store_exists():
        return KnowledgeSync(0, skipped, 0, 0)
    if not kb_available():
        log.debug("kb knowledge ingest skipped; knowledge base unavailable")
        return KnowledgeSync(0, skipped, 0, 0)

    existing = list_doc_metas(COLLECTION_KNOWLEDGE)
    by_path: dict[str, list[dict[str, Any]]] = {}
    if existing:
        for row in existing:
            meta = row.get("metadata") or {}
            if meta.get("source") != "knowledge":
                continue
            by_path.setdefault(str(meta.get("path") or ""), []).append(row)

    grouped: dict[str, list[KnowledgeDoc]] = {}
    for doc in planned:
        grouped.setdefault(doc.path, []).append(doc)

    to_write: list[KnowledgeDoc] = []
    stale: list[str] = []
    unchanged = 0
    if existing is None:
        to_write = list(planned)
    else:
        for rel, docs in grouped.items():
            new_ids = [doc.id for doc in docs]
            old = by_path.get(rel, [])
            old_ids = {str(row["id"]) for row in old}
            if old_ids == set(new_ids) and len(old) == len(new_ids):
                unchanged += len(new_ids)
                continue
            to_write.extend(docs)
            stale.extend(doc_id for doc_id in old_ids if doc_id not in set(new_ids))
        planned_paths = set(grouped)
        for rel, rows in by_path.items():
            if rel not in planned_paths:
                stale.extend(str(row["id"]) for row in rows)

    written = 0
    if to_write:
        written = upsert_docs(
            COLLECTION_KNOWLEDGE,
            [doc.id for doc in to_write],
            [doc.text for doc in to_write],
            [doc.metadata for doc in to_write],
        )
        if written != len(to_write):
            log.debug(
                "kb knowledge upsert incomplete (%s/%s); leaving previous ids",
                written,
                len(to_write),
            )
            return KnowledgeSync(unchanged, skipped, 0, 0)
    deleted = delete_docs(COLLECTION_KNOWLEDGE, stale) if stale else 0
    return KnowledgeSync(unchanged + written, skipped, written, deleted)


def ensure_knowledge_ingested() -> int:
    """Ingest ``knowledge/`` when files changed since the last pass in this process.

    Safe to call from storyboard / power-mode recall. A directory of schema
    docs and no Chroma store yet returns without opening the database.
    """
    global _disk_stamp_cache, _disk_stamp_count
    root = KNOWLEDGE_DIR
    stamp = _disk_stamp(root)
    with _stamp_lock:
        if stamp == _disk_stamp_cache:
            return _disk_stamp_count
        try:
            result = ingest_knowledge(root)
        except Exception:
            log.debug("kb knowledge ensure failed", exc_info=True)
            result = KnowledgeSync(0, 0, 0, 0)
        _disk_stamp_cache = stamp
        _disk_stamp_count = result.docs
        return result.docs


def schedule_knowledge_ingest(*, stderr: bool = False) -> None:
    """Ingest ``knowledge/`` on a daemon thread so process startup is not blocked.

    No-op under pytest (``PYTEST_CURRENT_TEST``) and when
    ``KB_STARTUP_INGEST`` is ``0``. MCP must pass ``stderr=True`` — stdout
    is the stdio protocol channel.
    """
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return
    flag = os.environ.get("KB_STARTUP_INGEST", "1").strip().lower()
    if flag in ("0", "false", "no", "off"):
        return

    def _run() -> None:
        try:
            result = ingest_knowledge()
        except Exception:
            log.debug("kb knowledge startup ingest failed", exc_info=True)
            return
        stream = sys.stderr if stderr else sys.stdout
        print(
            f"knowledge ingest: {result.docs} doc(s) in Chroma "
            f"({result.written} written, {result.skipped} skipped)",
            file=stream,
            flush=True,
        )

    threading.Thread(target=_run, name="knowledge-ingest", daemon=True).start()
