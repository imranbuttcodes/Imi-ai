# ==============================================================
# infrastructure/retrieval/embeddings.py — Embedding Factory
# ==============================================================
# Provider-agnostic embedding model factory.
# Set EMBEDDING_PROVIDER in .env to switch providers.
#
# Usage:
#   from infrastructure.retrieval.embeddings import get_embeddings
#   embeddings = get_embeddings()           # uses .env default
#   embeddings = get_embeddings("openai")   # override
# ==============================================================

from langchain_core.embeddings import Embeddings
from core.config import settings


def get_embeddings(provider: str | None = None) -> Embeddings:
    """
    Returns a LangChain-compatible embedding model.

    IMPORTANT: Use the same provider for both ingestion and retrieval.
    Mixing providers produces incompatible vectors and garbage results.
    """
    _provider = provider or settings.embedding_provider

    if _provider == "cohere":
        try:
            from langchain_cohere import CohereEmbeddings
            return CohereEmbeddings(
                model="embed-english-v3.0",
                cohere_api_key=settings.cohere_api_key
            )
        except (ImportError, ModuleNotFoundError):
            print("[Embeddings Warning] 'langchain_cohere' not installed. Falling back to HuggingFace Embeddings.")
            _provider = "huggingface"

    if _provider == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=settings.openai_api_key
            )
        except (ImportError, ModuleNotFoundError):
            print("[Embeddings Warning] 'langchain_openai' not installed. Falling back to HuggingFace Embeddings.")
            _provider = "huggingface"

    if _provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    raise ValueError(
        f"Unknown embedding provider: '{_provider}'. "
        f"Supported: 'cohere', 'openai', 'huggingface'."
    )