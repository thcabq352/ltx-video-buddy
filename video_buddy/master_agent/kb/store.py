"""Knowledge base — ChromaDB store with local embeddings.

Two collections:
- ``workflows`` — digests of the workflow JSON templates (what each is for)
- ``runs``      — every orchestrator/pipeline run record (self-learning seed)

Embeddings come from the active local LLM backend:
- Ollama: native ``POST {OLLAMA_URL}/api/embed``
- llama.cpp: OpenAI-compat ``POST {LLAMACPP_URL}/v1/embeddings``

No cloud, no extra model downloads beyond the local server. All functions
degrade to no-ops when KB_ENABLED=0 or chromadb / embeddings are unavailable.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from master_agent.config import CHROMA_DIR, KB_ENABLED, KB_EMBED_MODEL, OLLAMA_URL
from master_agent.llm import (
    active_local_backend,
    embeddings_endpoint,
    normalize_provider_name,
)

log = logging.getLogger(__name__)

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


def embed_texts(
    texts: list[str],
    *,
    backend: str | None = None,
    model: str | None = None,
) -> list[list[float]]:
    """Embed via Ollama ``/api/embed`` or llama.cpp ``/v1/embeddings``.

    Raises on missing backend or HTTP failure so callers can degrade.
    """
    chosen = normalize_provider_name(backend or "") if backend else active_local_backend()
    if chosen not in ("ollama", "llamacpp"):
        raise RuntimeError(
            "no local embedding backend (start Ollama or llama.cpp, or set LLM_PROVIDER)"
        )
    name = model or KB_EMBED_MODEL
    import httpx

    url = embeddings_endpoint(chosen)
    if chosen == "llamacpp":
        resp = httpx.post(
            url,
            json={"model": name, "input": texts},
            timeout=120,
        )
        resp.raise_for_status()
        payload = resp.json()
        rows = payload.get("data")
        if not isinstance(rows, list):
            raise RuntimeError(
                f"llama.cpp embeddings missing data[] at {url} "
                "(server needs OpenAI-compat /v1/embeddings)"
            )
        ordered = sorted(rows, key=lambda r: int((r or {}).get("index") or 0))
        out = [list((r or {}).get("embedding") or []) for r in ordered]
        if not out or not out[0]:
            raise RuntimeError(f"llama.cpp embeddings empty at {url}")
        return out

    resp = httpx.post(
        url or f"{OLLAMA_URL}/api/embed",
        json={"model": name, "input": texts},
        timeout=120,
    )
    resp.raise_for_status()
    embeds = resp.json().get("embeddings")
    if not embeds:
        raise RuntimeError(f"Ollama /api/embed returned no embeddings at {url}")
    return embeds


class LocalEmbedding:
    """ChromaDB embedding function: Ollama or llama.cpp, picked at call time."""

    def __init__(self, model: str | None = None, backend: str | None = None):
        self.model = model or KB_EMBED_MODEL
        self.backend = backend

    def __call__(self, input: list[str]) -> list[list[float]]:  # chroma protocol
        try:
            return embed_texts(input, backend=self.backend, model=self.model)
        except Exception as exc:
            log.warning(
                "KB embeddings unavailable via %s (%s); this write/search is a no-op",
                self.backend or active_local_backend() or "none",
                exc,
            )
            raise

    @staticmethod
    def name() -> str:
        return "local-llm"

    def get_config(self) -> dict:
        return {"model": self.model, "backend": self.backend or ""}

    @classmethod
    def build_from_config(cls, config: dict) -> "LocalEmbedding":
        return cls(model=config.get("model"), backend=config.get("backend") or None)

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return self(input)

    def embed_query(self, input: str | list[str]) -> list[list[float]]:
        texts = [input] if isinstance(input, str) else input
        return self(texts)


class OllamaEmbedding(LocalEmbedding):
    """ChromaDB embedding function backed by Ollama /api/embed (legacy name)."""

    def __init__(self, model: str | None = None):
        super().__init__(model=model, backend="ollama")

    @staticmethod
    def name() -> str:
        return "ollama"

    @classmethod
    def build_from_config(cls, config: dict) -> "OllamaEmbedding":
        return cls(model=config.get("model"))


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
        embedding_function=LocalEmbedding(),
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
        log.debug("kb upsert failed for %s", collection, exc_info=True)
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
