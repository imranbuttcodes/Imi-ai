# ==============================================================
# infrastructure/retrieval/vector_store.py — Retriever Factory
# ==============================================================
# Builds the hybrid retriever: BM25 (keyword) + ChromaDB (semantic)
# combined via EnsembleRetriever.
#
# Usage:
#   from infrastructure.retrieval.vector_store import get_retriever
#   retriever = get_retriever()
#   docs = retriever.invoke("What is the Q3 budget?")
# ==============================================================

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from infrastructure.retrieval.embeddings import get_embeddings
from core.config import settings


def get_vector_store(db_path: str | None = None) -> Chroma:
    """Returns the ChromaDB vector store instance."""
    return Chroma(
        embedding_function=get_embeddings(),
        persist_directory=db_path or settings.chroma_db_path
    )


def get_retriever(db_path: str | None = None) -> EnsembleRetriever:
    """
    Builds and returns a Hybrid Retriever.

    Combines:
    - BM25 (keyword matching, 40% weight) — exact term matching
    - ChromaDB (semantic similarity, 60% weight) — meaning-based

    Together they miss fewer relevant documents than either alone.
    """
    k = settings.retrieval_k

    # ── Vector Retriever ───────────────────────────────────────
    vector_store = get_vector_store(db_path)
    vector_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    # ── BM25 Retriever (built from same ChromaDB data) ─────────
    store_data = vector_store.get(include=["documents", "metadatas"])
    all_chunks = [
        Document(page_content=content, metadata=meta or {})
        for content, meta in zip(
            store_data["documents"],
            store_data["metadatas"]
        )
    ]

    if not all_chunks:
        # If DB is empty, BM25 crashes on init. Just return vector retriever.
        return vector_retriever

    bm25_retriever = BM25Retriever.from_documents(all_chunks)
    bm25_retriever.k = k

    # ── Ensemble: keyword + semantic ───────────────────────────
    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6]
    )
