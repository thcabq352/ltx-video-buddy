"""Knowledge-folder ingest. No live Chroma and no Ollama.

Run: python -m pytest tests/test_knowledge_ingest.py -q
"""

from __future__ import annotations

from pathlib import Path

import pytest

import master_agent.kb.ingest as ingest
from master_agent.kb.ingest import KnowledgeSync, collect_knowledge_docs, ingest_knowledge
from master_agent.kb.recall import recall_knowledge


@pytest.fixture(autouse=True)
def _reset_stamp():
    ingest._disk_stamp_cache = None
    ingest._disk_stamp_count = 0
    yield
    ingest._disk_stamp_cache = None
    ingest._disk_stamp_count = 0


def _tree(root: Path) -> Path:
    """Schema docs, an EXAMPLE seed, one failure, one prompt, one secret."""
    prompts = root / "prompts"
    failures = root / "failures"
    decisions = root / "decisions"
    for folder in (prompts, failures, decisions):
        folder.mkdir(parents=True)
    (root / "README.md").write_text("# schema\n\nstatus: example\n", encoding="utf-8")
    (root / "AGENTS.md").write_text("# agents\n\n- **status:** example\n", encoding="utf-8")
    (prompts / "README.md").write_text("copy the schema\n", encoding="utf-8")
    (prompts / "EXAMPLE-h3-cold-brew-commercial.md").write_text(
        "- **status:** example\nnot a run\n",
        encoding="utf-8",
    )
    (prompts / "2026-09-29-shape-only.md").write_text(
        "# shape\n\n- **status:** example\nnot recorded\n",
        encoding="utf-8",
    )
    (failures / "2026-09-29-length-8.md").write_text(
        "# length 8\n\n"
        "- **status:** recorded\n"
        "- **priority:** check-before-retry\n"
        "- **symptom:** one frame from C:\\Users\\scott\\Outputs\\clip.mp4\n"
        "- **fix:** snap to 9. also saw /home/jason/renders/clip.mp4\n",
        encoding="utf-8",
    )
    (prompts / "2026-09-29-pour.md").write_text(
        "# pour\n\n- **status:** recorded\n- **prompt:** glass pour on a walnut bar\n",
        encoding="utf-8",
    )
    (decisions / "2026-09-29-leak.md").write_text(
        "- **status:** recorded\napi_key: sk-abc1234567890secret\n",
        encoding="utf-8",
    )
    return root


def _bind_store(monkeypatch, chroma: Path):
    store: list[dict] = []
    upsert_calls: list[list[str]] = []
    deleted: list[str] = []

    monkeypatch.setattr(ingest, "CHROMA_DIR", chroma)
    monkeypatch.setattr(ingest, "kb_available", lambda: True)

    def list_metas(_collection):
        return [{"id": row["id"], "metadata": dict(row["metadata"])} for row in store]

    def upsert(_collection, ids, texts, metas):
        upsert_calls.append(list(ids))
        by_id = {row["id"]: row for row in store}
        for doc_id, text, meta in zip(ids, texts, metas):
            by_id[doc_id] = {"id": doc_id, "text": text, "metadata": dict(meta)}
        store[:] = list(by_id.values())
        return len(ids)

    def delete(_collection, ids):
        drop = set(ids)
        deleted.extend(ids)
        store[:] = [row for row in store if row["id"] not in drop]
        return len(ids)

    monkeypatch.setattr(ingest, "list_doc_metas", list_metas)
    monkeypatch.setattr(ingest, "upsert_docs", upsert)
    monkeypatch.setattr(ingest, "delete_docs", delete)
    return store, upsert_calls, deleted


def test_skip_policy_and_redact_paths(tmp_path: Path):
    root = _tree(tmp_path / "knowledge")
    docs, skipped = collect_knowledge_docs(root)
    reasons = {rel: reason for rel, reason in skipped}
    assert reasons["README.md"] == "schema"
    assert reasons["AGENTS.md"] == "schema"
    assert reasons["prompts/README.md"] == "schema"
    assert reasons["prompts/EXAMPLE-h3-cold-brew-commercial.md"] == "example"
    assert reasons["prompts/2026-09-29-shape-only.md"] == "example"
    assert reasons["decisions/2026-09-29-leak.md"] == "secret"
    paths = {doc.path for doc in docs}
    assert paths == {
        "failures/2026-09-29-length-8.md",
        "prompts/2026-09-29-pour.md",
    }
    failure = next(doc for doc in docs if doc.metadata["category"] == "failures")
    assert failure.metadata["source"] == "knowledge"
    assert failure.metadata["priority"] == "high"
    assert failure.metadata["path"] == "failures/2026-09-29-length-8.md"
    assert "C:\\Users" not in failure.text
    assert "/home/jason" not in failure.text
    assert "[local-path]" in failure.text
    assert failure.id.startswith("knowledge:failures/2026-09-29-length-8.md:")
    prompt = next(doc for doc in docs if doc.metadata["category"] == "prompts")
    assert prompt.metadata["priority"] == "normal"
    assert "sk-abc" not in prompt.text


def test_idempotent_upsert_replaces_on_change(tmp_path: Path, monkeypatch):
    root = _tree(tmp_path / "knowledge")
    chroma = tmp_path / "chroma"
    store, upsert_calls, deleted = _bind_store(monkeypatch, chroma)

    first = ingest_knowledge(root)
    assert first.docs == 2
    assert first.written == 2
    assert first.deleted == 0
    assert len(store) == 2
    first_ids = {row["id"] for row in store}
    assert len(upsert_calls) == 1

    upsert_calls.clear()
    deleted.clear()
    second = ingest_knowledge(root)
    assert second.docs == 2
    assert second.written == 0
    assert upsert_calls == []
    assert deleted == []
    assert {row["id"] for row in store} == first_ids

    failure = root / "failures" / "2026-09-29-length-8.md"
    failure.write_text(failure.read_text(encoding="utf-8") + "\nextra note\n", encoding="utf-8")
    third = ingest_knowledge(root)
    assert third.written == 1
    assert third.docs == 2
    assert len(store) == 2
    new_ids = {row["id"] for row in store}
    assert new_ids != first_ids
    assert deleted
    assert set(deleted).isdisjoint(new_ids)
    assert any(row["metadata"]["category"] == "failures" for row in store)
    joined = "\n".join(row["text"] for row in store)
    assert "extra note" in joined
    assert "sk-abc" not in joined


def test_long_entry_chunks_share_a_hash_and_do_not_duplicate(tmp_path: Path, monkeypatch):
    root = tmp_path / "knowledge" / "decisions"
    root.mkdir(parents=True)
    body = "- **status:** recorded\n\n" + ("\n\n".join(f"paragraph {i} " + ("word " * 400) for i in range(6)))
    path = root / "2026-09-29-long.md"
    path.write_text(body, encoding="utf-8")
    docs, skipped = collect_knowledge_docs(root.parent)
    assert skipped == []
    assert len(docs) > 1
    hashes = {doc.content_hash for doc in docs}
    assert len(hashes) == 1
    assert [doc.metadata["chunk"] for doc in docs] == list(range(len(docs)))
    assert all(doc.metadata["chunks"] == len(docs) for doc in docs)

    _store, upsert_calls, _deleted = _bind_store(monkeypatch, tmp_path / "chroma")
    ingest_knowledge(root.parent)
    assert len(upsert_calls) == 1
    assert len(upsert_calls[0]) == len(docs)
    upsert_calls.clear()
    again = ingest_knowledge(root.parent)
    assert again.written == 0
    assert upsert_calls == []
    assert again.docs == len(docs)


def test_removed_file_is_deleted(tmp_path: Path, monkeypatch):
    root = _tree(tmp_path / "knowledge")
    store, _upserts, deleted = _bind_store(monkeypatch, tmp_path / "chroma")
    ingest_knowledge(root)
    (root / "prompts" / "2026-09-29-pour.md").unlink()
    result = ingest_knowledge(root)
    assert result.docs == 1
    assert len(store) == 1
    assert store[0]["metadata"]["category"] == "failures"
    assert any("prompts/2026-09-29-pour.md" in doc_id for doc_id in deleted)


def test_symlink_outside_root_is_not_ingested(tmp_path: Path):
    root = tmp_path / "knowledge"
    (root / "prompts").mkdir(parents=True)
    outside = tmp_path / "secret.md"
    outside.write_text("- **status:** recorded\napi_key: sk-outsidekey1234567890\n", encoding="utf-8")
    link = root / "prompts" / "2026-09-29-link.md"
    link.symlink_to(outside)
    docs, skipped = collect_knowledge_docs(root)
    assert docs == []
    assert ("2026-09-29-link.md", "outside") in skipped


def test_empty_tree_does_not_open_chroma(tmp_path: Path, monkeypatch):
    root = tmp_path / "knowledge"
    root.mkdir()
    (root / "README.md").write_text("# schema\n", encoding="utf-8")
    (root / "prompts").mkdir()
    (root / "prompts" / "EXAMPLE-seed.md").write_text("status: example\n", encoding="utf-8")
    monkeypatch.setattr(ingest, "CHROMA_DIR", tmp_path / "no-store")

    def _boom():
        raise AssertionError("chroma should stay closed")

    monkeypatch.setattr(ingest, "kb_available", _boom)
    result = ingest_knowledge(root)
    assert result == KnowledgeSync(0, 2, 0, 0)


def test_missing_dir_does_not_wipe(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(ingest, "CHROMA_DIR", tmp_path / "chroma")
    called = {"list": 0}

    def _list(_collection):
        called["list"] += 1
        return [{"id": "knowledge:failures/old.md:abc", "metadata": {"source": "knowledge", "path": "failures/old.md"}}]

    monkeypatch.setattr(ingest, "list_doc_metas", _list)
    monkeypatch.setattr(ingest, "kb_available", lambda: True)
    result = ingest_knowledge(tmp_path / "missing-knowledge")
    assert result.docs == 0
    assert called["list"] == 0


def test_ensure_reruns_when_stamp_changes(tmp_path: Path, monkeypatch):
    calls: list[Path | None] = []

    def _ingest(root=None):
        calls.append(root)
        return KnowledgeSync(1, 0, 1, 0)

    stamps = iter(["aaa", "aaa", "bbb"])
    monkeypatch.setattr(ingest, "ingest_knowledge", _ingest)
    monkeypatch.setattr(ingest, "_disk_stamp", lambda _root: next(stamps))
    monkeypatch.setattr(ingest, "KNOWLEDGE_DIR", tmp_path)
    assert ingest.ensure_knowledge_ingested() == 1
    assert ingest.ensure_knowledge_ingested() == 1
    assert ingest.ensure_knowledge_ingested() == 1
    assert len(calls) == 2


def test_recall_lists_failures_ahead_of_closer_notes(monkeypatch):
    monkeypatch.setattr("master_agent.kb.ingest.ensure_knowledge_ingested", lambda: 0)

    def fake_search(_collection, _query, k=3, where=None):
        if where == {"category": "failures"}:
            return [
                {
                    "id": "fail",
                    "text": "do not queue length 8",
                    "metadata": {
                        "category": "failures",
                        "priority": "high",
                        "path": "failures/2026-09-29-length-8.md",
                    },
                    "distance": 0.8,
                }
            ]
        return [
            {
                "id": "prompt",
                "text": "glass pour",
                "metadata": {
                    "category": "prompts",
                    "priority": "normal",
                    "path": "prompts/2026-09-29-pour.md",
                },
                "distance": 0.05,
            }
        ]

    monkeypatch.setattr("master_agent.kb.recall.search", fake_search)
    block = recall_knowledge("length 8 on an LTX graph", k=3)
    assert block.index("failures") < block.index("prompts")
    assert "[failures high]" in block
    assert "do not queue length 8" in block
