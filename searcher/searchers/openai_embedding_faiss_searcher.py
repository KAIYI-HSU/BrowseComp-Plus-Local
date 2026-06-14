from __future__ import annotations

import glob
import logging
import os
import pickle
from typing import Any, Dict, List, Optional

import numpy as np
from dotenv import load_dotenv

from .base import BaseSearcher


logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_BASE_URL = "http://localhost:8000/v1"
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_QWEN3_QUERY_PREFIX = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
    "Query:"
)


def normalize_query_vector(vector: list[float], *, normalize: bool) -> np.ndarray:
    query = np.asarray(vector, dtype=np.float32).reshape(1, -1)
    if not normalize:
        return query

    norms = np.linalg.norm(query, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (query / norms).astype(np.float32)


def resolve_api_key(*, api_key: Optional[str], api_key_env: Optional[str]) -> str:
    load_dotenv()
    if api_key_env:
        env_value = os.getenv(api_key_env)
        if not env_value:
            raise RuntimeError(
                f"{api_key_env} is not set in the environment or .env"
            )
        return env_value
    return api_key or "EMPTY"


class OpenAIEmbeddingFaissSearcher(BaseSearcher):
    """FAISS searcher that embeds queries through an OpenAI-compatible endpoint."""

    @classmethod
    def parse_args(cls, parser):
        parser.add_argument(
            "--index-path",
            required=True,
            help="Glob pattern for corpus pickle files, e.g. indexes/qwen3-embedding-0.6b/corpus.shard*.pkl",
        )
        parser.add_argument(
            "--embedding-model",
            default=DEFAULT_EMBEDDING_MODEL,
            help="Embedding model id exposed by the OpenAI-compatible endpoint",
        )
        parser.add_argument(
            "--embedding-base-url",
            default=DEFAULT_EMBEDDING_BASE_URL,
            help="OpenAI-compatible embeddings API base URL",
        )
        parser.add_argument(
            "--embedding-api-key",
            default="EMPTY",
            help="Literal API key for local endpoints such as vLLM",
        )
        parser.add_argument(
            "--embedding-api-key-env",
            default=None,
            help="Optional environment variable containing the embeddings API key",
        )
        parser.add_argument(
            "--embedding-dimensions",
            type=int,
            default=None,
            help="Optional dimensions parameter for embeddings APIs that support it",
        )
        parser.add_argument(
            "--normalize",
            action="store_true",
            default=False,
            help="Normalize query embeddings before FAISS search",
        )
        parser.add_argument(
            "--dataset-name",
            default="Tevatron/browsecomp-plus-corpus",
            help="Dataset name for document text lookup",
        )
        parser.add_argument(
            "--task-prefix",
            default=DEFAULT_QWEN3_QUERY_PREFIX,
            help="Instruction prefix prepended to every query before embedding",
        )

    def __init__(self, args):
        self.args = args
        self.index = None
        self.lookup: list[str] = []
        self.docid_to_text: dict[str, str] = {}
        self.client = self._build_client()
        self._load_faiss_index()
        self._load_dataset()

    def _build_client(self):
        import openai

        api_key = resolve_api_key(
            api_key=self.args.embedding_api_key,
            api_key_env=self.args.embedding_api_key_env,
        )
        return openai.OpenAI(api_key=api_key, base_url=self.args.embedding_base_url)

    def _load_faiss_index(self) -> None:
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError(
                "faiss is required for openai-embedding-faiss search. Install the project dependencies with `uv sync`."
            ) from exc

        index_files = sorted(glob.glob(self.args.index_path))
        if not index_files:
            raise ValueError(f"No files found matching pattern: {self.args.index_path}")

        for path in index_files:
            with open(path, "rb") as f:
                reps, lookup = pickle.load(f)
            reps = np.asarray(reps, dtype=np.float32)
            if reps.ndim != 2:
                raise ValueError(f"{path} did not contain a 2D embedding matrix")
            if self.index is None:
                self.index = faiss.IndexFlatIP(reps.shape[1])
            self.index.add(reps)
            self.lookup.extend(str(docid) for docid in lookup)

        logger.info(
            "Loaded %s FAISS vectors from %s shards", len(self.lookup), len(index_files)
        )

    def _load_dataset(self) -> None:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "datasets is required for openai-embedding-faiss document lookup. Install the project dependencies with `uv sync`."
            ) from exc

        dataset_cache = os.getenv("HF_DATASETS_CACHE") or None
        ds = load_dataset(self.args.dataset_name, split="train", cache_dir=dataset_cache)
        self.docid_to_text = {str(row["docid"]): row["text"] for row in ds}

    def _embed_query(self, query: str) -> np.ndarray:
        request = {
            "model": self.args.embedding_model,
            "input": self.args.task_prefix + query,
        }
        if self.args.embedding_dimensions:
            request["dimensions"] = self.args.embedding_dimensions

        response = self.client.embeddings.create(**request)
        if not response.data:
            raise RuntimeError("Embeddings response contained no vectors")
        return normalize_query_vector(
            response.data[0].embedding, normalize=bool(self.args.normalize)
        )

    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        if self.index is None:
            raise RuntimeError("FAISS index is not initialized")

        q_reps = self._embed_query(query)
        scores, indices = self.index.search(q_reps, k)

        results = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            docid = str(self.lookup[int(index)])
            results.append(
                {
                    "docid": docid,
                    "score": float(score),
                    "text": self.docid_to_text.get(docid, "Text not found"),
                }
            )
        return results

    def get_document(self, docid: str) -> Optional[Dict[str, Any]]:
        text = self.docid_to_text.get(str(docid))
        if text is None:
            return None
        return {"docid": str(docid), "text": text}

    @property
    def search_type(self) -> str:
        return "OpenAIEmbeddingFAISS"
