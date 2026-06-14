"""
Searchers package for different search implementations.
"""

from enum import Enum

from .base import BaseSearcher
from .custom_searcher import CustomSearcher
from .openai_embedding_faiss_searcher import OpenAIEmbeddingFaissSearcher


def _unavailable_searcher(name: str, import_error: Exception):
    class UnavailableSearcher(BaseSearcher):
        @classmethod
        def parse_args(cls, parser):
            raise RuntimeError(
                f"{name} searcher is unavailable because an optional dependency "
                f"could not be imported: {import_error}"
            )

        def __init__(self, args):
            raise RuntimeError(
                f"{name} searcher is unavailable because an optional dependency "
                f"could not be imported: {import_error}"
            )

        def search(self, query: str, k: int = 10):
            raise RuntimeError(f"{name} searcher is unavailable")

        def get_document(self, docid: str):
            raise RuntimeError(f"{name} searcher is unavailable")

        @property
        def search_type(self) -> str:
            return name

    return UnavailableSearcher


try:
    from .bm25_searcher import BM25Searcher
except Exception as exc:  # pragma: no cover - depends on optional local install
    BM25Searcher = _unavailable_searcher("BM25", exc)

try:
    from .faiss_searcher import FaissSearcher, ReasonIrSearcher
except Exception as exc:  # pragma: no cover - depends on optional local install
    FaissSearcher = _unavailable_searcher("FAISS", exc)
    ReasonIrSearcher = _unavailable_searcher("ReasonIR", exc)


class SearcherType(Enum):
    """Enum for managing available searcher types and their CLI mappings."""

    BM25 = ("bm25", BM25Searcher)
    FAISS = ("faiss", FaissSearcher)
    OPENAI_EMBEDDING_FAISS = ("openai-embedding-faiss", OpenAIEmbeddingFaissSearcher)
    FIREWORKS_FAISS = ("fireworks-faiss", OpenAIEmbeddingFaissSearcher)
    REASONIR = ("reasonir", ReasonIrSearcher)
    CUSTOM = (
        "custom",
        CustomSearcher,
    )  # Your custom searcher class, yet to be implemented

    def __init__(self, cli_name, searcher_class):
        self.cli_name = cli_name
        self.searcher_class = searcher_class

    @classmethod
    def get_choices(cls):
        """Get list of CLI choices for argument parser."""
        return [searcher_type.cli_name for searcher_type in cls]

    @classmethod
    def get_searcher_class(cls, cli_name):
        """Get searcher class by CLI name."""
        for searcher_type in cls:
            if searcher_type.cli_name == cli_name:
                return searcher_type.searcher_class
        raise ValueError(f"Unknown searcher type: {cli_name}")


__all__ = ["BaseSearcher", "SearcherType"]
