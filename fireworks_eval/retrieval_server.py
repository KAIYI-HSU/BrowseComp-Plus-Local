"""Local retrieval HTTP endpoint (the swappable "data source").

Wraps the repo's existing dense retriever (`searcher.searchers.FaissSearcher`,
Qwen3-Embedding) and serves a clean, documented contract:

  GET  /health                          -> readiness + config
  POST /search       {query, k?}        -> {results: [{docid, score, title, snippet}], k}
  POST /get_document {docid}            -> {docid, title, text}  (404 if missing)

Run standalone (the pipeline does this automatically):

  python -m fireworks_eval.retrieval_server \\
    --backend faiss --index-path "indexes/qwen3-embedding-0.6b/corpus.shard*.pkl" \\
    --model-name Qwen/Qwen3-Embedding-0.6B --normalize --k 5 \\
    --snippet-max-tokens 512 --port 8123
"""

from __future__ import annotations

import os

# faiss and torch each bundle an OpenMP runtime; on macOS loading both in one
# process segfaults unless we tolerate the duplicate runtime AND pin OpenMP to a
# single thread. Set before faiss/torch are imported (they load lazily below).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import logging
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [retrieval] %(message)s")
logger = logging.getLogger(__name__)

_FRONTMATTER_TITLE_RE = re.compile(r"^title:\s*(.*)$", re.MULTILINE)
DEFAULT_TASK_PREFIX = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
)


# Defined at module level so FastAPI can resolve these annotations under
# `from __future__ import annotations` (local classes would not resolve).
class SearchRequest(BaseModel):
    query: str
    k: Optional[int] = None


class GetDocRequest(BaseModel):
    docid: str


def split_title_body(text: str) -> tuple[str, str]:
    """Split a corpus document into (title, body), handling YAML frontmatter."""
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            front = text[3:end]
            match = _FRONTMATTER_TITLE_RE.search(front)
            title = match.group(1).strip().strip('"') if match else ""
            body = text[end + 4 :].lstrip("\n")
            return title, body
    head, _, rest = text.partition("\n")
    return head.strip().strip('"'), rest


def _resolve_dtype(torch_dtype: str) -> str:
    if torch_dtype != "auto":
        return torch_dtype
    try:
        import torch

        return "float16" if torch.cuda.is_available() else "float32"
    except Exception:
        return "float32"


def _import_dense_searchers():
    """Import FaissSearcher/ReasonIrSearcher WITHOUT running the searcher package
    ``__init__``, which eagerly imports the BM25 searcher (pyserini -> jnius ->
    Java). Java is not required for dense retrieval, so we load the two needed
    modules directly while pre-registering stub parent packages so that the
    relative ``from .base import ...`` still resolves.
    """
    import importlib.util
    import types

    searchers_dir = REPO_ROOT / "searcher" / "searchers"

    if "searcher" not in sys.modules:
        pkg = types.ModuleType("searcher")
        pkg.__path__ = [str(REPO_ROOT / "searcher")]
        sys.modules["searcher"] = pkg
    if "searcher.searchers" not in sys.modules:
        subpkg = types.ModuleType("searcher.searchers")
        subpkg.__path__ = [str(searchers_dir)]
        sys.modules["searcher.searchers"] = subpkg

    def _load(module_name: str, filename: str):
        if module_name in sys.modules and getattr(sys.modules[module_name], "__file__", None):
            return sys.modules[module_name]
        spec = importlib.util.spec_from_file_location(module_name, searchers_dir / filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    _load("searcher.searchers.base", "base.py")
    faiss_mod = _load("searcher.searchers.faiss_searcher", "faiss_searcher.py")

    class EagerFaissSearcher(faiss_mod.FaissSearcher):
        """FaissSearcher that loads the embedding model with eager attention.

        The upstream loader defaults to FlashAttention-2, which is unavailable
        on CPU/Mac. We override only model loading to force ``eager`` attention.
        """

        def _load_model(self) -> None:  # noqa: D401
            fm = faiss_mod
            cache_dir = fm.os.getenv("HF_HOME") or None
            model_args = fm.ModelArguments(
                model_name_or_path=self.args.model_name,
                normalize=self.args.normalize,
                pooling=self.args.pooling,
                cache_dir=cache_dir,
            )
            dtype = {
                "float16": fm.torch.float16,
                "bfloat16": fm.torch.bfloat16,
                "float32": fm.torch.float32,
            }[self.args.torch_dtype]
            self.model = fm.DenseModel.load(
                model_args.model_name_or_path,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                lora_name_or_path=model_args.lora_name_or_path,
                cache_dir=model_args.cache_dir,
                torch_dtype=dtype,
                attn_implementation="eager",
            )
            device = "cuda" if fm.torch.cuda.is_available() else "cpu"
            self.model = self.model.to(device)
            self.model.eval()
            self.tokenizer = fm.AutoTokenizer.from_pretrained(
                model_args.tokenizer_name or model_args.model_name_or_path,
                cache_dir=model_args.cache_dir,
                padding_side="left",
            )

    return EagerFaissSearcher, faiss_mod.ReasonIrSearcher


def build_searcher(args: argparse.Namespace):
    """Instantiate the dense searcher from the repo's searcher package."""
    FaissSearcher, ReasonIrSearcher = _import_dense_searchers()

    searcher_args = SimpleNamespace(
        model_name=args.model_name,
        index_path=args.index_path,
        normalize=args.normalize,
        pooling=args.pooling,
        torch_dtype=_resolve_dtype(args.torch_dtype),
        dataset_name=args.dataset_name,
        task_prefix=args.task_prefix,
        max_length=args.max_length,
    )
    cls = ReasonIrSearcher if args.backend == "reasonir" else FaissSearcher
    logger.info("Loading %s (model=%s, dtype=%s)...", cls.__name__, args.model_name, searcher_args.torch_dtype)
    return cls(searcher_args)


def _load_snippet_tokenizer(snippet_max_tokens: int):
    if snippet_max_tokens is None or snippet_max_tokens <= 0:
        return None
    import transformers

    logger.info("Loading snippet tokenizer Qwen/Qwen3-0.6B...")
    return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")


def _truncate(text: str, tokenizer, max_tokens: int) -> str:
    if tokenizer is None or max_tokens is None or max_tokens <= 0:
        return text
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)


def create_app(searcher, tokenizer, args: argparse.Namespace):
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title=f"{searcher.search_type} retrieval endpoint")

    def _format(results: list[dict]) -> list[dict]:
        formatted = []
        for r in results:
            title, body = split_title_body(r.get("text", ""))
            snippet = _truncate(body, tokenizer, args.snippet_max_tokens)
            item = {"docid": str(r["docid"]), "title": title, "snippet": snippet}
            if r.get("score") is not None:
                item["score"] = float(r["score"])
            formatted.append(item)
        return formatted

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "ready": True,
            "searcher": searcher.search_type,
            "backend": args.backend,
            "k": args.k,
            "snippet_max_tokens": args.snippet_max_tokens,
            "get_document": args.get_document,
        }

    @app.post("/search")
    def search(request: SearchRequest):
        k = request.k or args.k
        results = searcher.search(request.query, k=k)
        return {"results": _format(results), "k": k}

    @app.post("/get_document")
    def get_document(request: GetDocRequest):
        doc = searcher.get_document(request.docid)
        if doc is None:
            raise HTTPException(status_code=404, detail=f"docid {request.docid} not found")
        title, body = split_title_body(doc.get("text", ""))
        return {"docid": str(doc["docid"]), "title": title, "text": body}

    # Backwards-compatible alias for the repo's search_r1 contract.
    @app.post("/retrieve")
    def retrieve(request: SearchRequest):
        results = searcher.search(request.query, k=request.k or args.k)
        return {"result": [{"document": {"title": t, "text": b}, "docid": d}
                           for d, t, b in ((str(r["docid"]), *split_title_body(r.get("text", ""))) for r in results)]}

    return app


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local dense retrieval endpoint")
    parser.add_argument("--backend", choices=["faiss", "reasonir"], default="faiss")
    parser.add_argument("--index-path", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--normalize", action="store_true", default=False)
    parser.add_argument("--pooling", default="eos")
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--dataset-name", default="Tevatron/browsecomp-plus-corpus")
    parser.add_argument("--task-prefix", default=DEFAULT_TASK_PREFIX)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--snippet-max-tokens", type=int, default=512)
    parser.add_argument("--get-document", action="store_true", default=False)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8123)
    return parser


def main() -> None:
    import uvicorn

    args = build_arg_parser().parse_args()
    searcher = build_searcher(args)
    tokenizer = _load_snippet_tokenizer(args.snippet_max_tokens)
    app = create_app(searcher, tokenizer, args)
    logger.info("Retrieval endpoint ready on http://%s:%d (k=%d, snippet_max_tokens=%d)",
                args.host, args.port, args.k, args.snippet_max_tokens)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
