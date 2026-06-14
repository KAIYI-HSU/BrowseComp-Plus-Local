from __future__ import annotations

from .openai_embedding_faiss_searcher import (
    DEFAULT_EMBEDDING_BASE_URL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_QWEN3_QUERY_PREFIX,
    OpenAIEmbeddingFaissSearcher,
    normalize_query_vector,
    resolve_api_key,
)


FireworksFaissSearcher = OpenAIEmbeddingFaissSearcher

__all__ = [
    "DEFAULT_EMBEDDING_BASE_URL",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_QWEN3_QUERY_PREFIX",
    "FireworksFaissSearcher",
    "OpenAIEmbeddingFaissSearcher",
    "normalize_query_vector",
    "resolve_api_key",
]
