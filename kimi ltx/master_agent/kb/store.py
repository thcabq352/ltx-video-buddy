"""Knowledge base — ChromaDB store with local Ollama embeddings.

Two collections:
- ``workflows`` — digests of the workflow JSON templates (what each is for)
- ``runs``      — every orchestrator/pipeline run record (self-learning seed)

Embeddings come from Ollama (nomic-embed-text) — no cloud, no extra model
downloads beyond the ollama pull. All functions degrade to no-ops when
KB_ENABLED=0 or chromadb/ollama is unavailable.
"""

from __future__ import annotations

from typing import Any, Optional

from master_agent.config import CHROMA_DIR, KB_ENABLED, KB_EMBED_MODEL, OLLAMA_URL

COLLECTION_WORKFLOWS = "workflows"
COLLECTION_RUNS = "runs"
COLLECTION_CHARACTERS = "characters"
COLLECTION_LORA_RUNS = "lora_runs"


def kb_available() -> bool:
    if not KB_ENABLED:
        return False
    try:
        import chromadb  # noqa: F401

        return True
    except ImportError:
        return False


class OllamaEmbedding:
    """ChromaDB embedding function backed by Ollama /api/embed."""

    def __init__(self, model: str | None = None):
        self.model = model or KB_EMBED_MODEL

    def __call__(self, input: list[str]) -> list[list[float]]:  # chroma protocol
        import httpx

        resp = httpx.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": self.model, "input": input},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]

    # chromadb >=1.x embedding-function protocol
    @staticmethod
    def name() -> str:
        return "ollama"

    def get_config(self) -> dict:
        return {"model": self.model}

    @classmethod
    def build_from_config(cls, config: dict) -> "OllamaEmbedding":
        return cls(model=config.get("model"))

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return self(input)

    def embed_query(self, input: str | list[str]) -> list[list[float]]:
        texts = [input] if isinstance(input, str) else input
        return self(texts)


_client = None


def get_client():
    global _client
    if _client is None:
        import chromadb

        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _client


def get_collection(name: str):
    return get_client().get_or_create_collection(
        name=name,
        embedding_function=OllamaEmbedding(),
        metadata={"hnsw:space": "cosine"},
    )


def upsert_docs(
    collection: str,
    ids: list[str],
    texts: list[str],
    metadatas: list[dict[str, Any]],
) -> int:
    """Upsert docs; returns count written. 0 when KB unavailable."""
    if not (ids and kb_available()):
        return 0
    try:
        coll = get_collection(collection)
        # chroma metadata values must be scalar
        clean = [
            {k: v for k, v in (m or {}).items() if isinstance(v, (str, int, float, bool))}
            for m in metadatas
        ]
        coll.upsert(ids=ids, documents=texts, metadatas=clean)
        return len(ids)
    except Exception:
        return 0


def search(
    collection: str,
    query: str,
    *,
    k: int = 3,
    where: Optional[dict] = None,
) -> list[dict[str, Any]]:
    """Semantic search; [{id, text, metadata, distance}]. [] on any failure."""
    if not kb_available():
        return []
    try:
        coll = get_collection(collection)
        kwargs: dict[str, Any] = {"query_texts": [query], "n_results": k}
        if where:
            kwargs["where"] = where
        res = coll.query(**kwargs)
        out = []
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        for i, doc_id in enumerate(ids):
            out.append(
                {
                    "id": doc_id,
                    "text": docs[i] if i < len(docs) else "",
                    "metadata": metas[i] if i < len(metas) else {},
                    "distance": dists[i] if i < len(dists) else None,
                }
            )
        return out
    except Exception:
        return []


def collection_count(collection: str) -> int:
    if not kb_available():
        return 0
    try:
        return get_collection(collection).count()
    except Exception:
        return 0
