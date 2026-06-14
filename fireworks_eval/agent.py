"""The built-in Fireworks reference Deep-Research agent.

Uses the OpenAI **chat.completions** API with function calling — the portable
lingua franca across providers — so switching model/provider is just a
``base_url``/``model`` change. Emits run files conforming to the contract in
``agent_protocol``. Your own agent can replace this entirely (run-file or
Python-adapter contract); this one makes the pipeline work out of the box.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

from .agent_protocol import (
    STATUS_COMPLETED,
    STATUS_ERROR,
    STATUS_INCOMPLETE,
    ResultItem,
    RetrievalBackend,
    RunRecord,
    write_run_file,
)
from .config import AgentConfig
from .errors import AgentError
from .fireworks_client import chat_completion, make_client

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)

_KNOWN_TEMPLATES = {
    "QUERY_TEMPLATE",
    "QUERY_TEMPLATE_NO_GET_DOCUMENT",
    "QUERY_TEMPLATE_NO_GET_DOCUMENT_NO_CITATION",
}


def _search_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Search the knowledge base for relevant documents. Returns the "
                "top-k results, each with docid, title, score, and a snippet of "
                "the document's contents."
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "The search query."}},
                "required": ["query"],
            },
        },
    }


def _get_document_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "get_document",
            "description": "Retrieve the full text of a document by its docid.",
            "parameters": {
                "type": "object",
                "properties": {"docid": {"type": "string", "description": "The document id."}},
                "required": ["docid"],
            },
        },
    }


def _format_user_content(question: str, template: Optional[str]) -> str:
    """Render the user prompt from a known template name or a literal template."""
    if template is None:
        return question
    if template in _KNOWN_TEMPLATES:
        from search_agent.prompts import format_query

        return format_query(question, template)
    if "{Question}" in template:
        return template.format(Question=question)
    if "{question}" in template:
        return template.format(question=question)
    return f"{template}\n\nQuestion: {question}"


def _add_unique(ordered: list[str], docid: str) -> None:
    if docid not in ordered:
        ordered.append(docid)


def _extract_reasoning(message: Any) -> Optional[str]:
    """Best-effort reasoning extraction (provider-dependent; optional)."""
    for attr in ("reasoning_content", "reasoning"):
        val = getattr(message, attr, None)
        if isinstance(val, str) and val.strip():
            return val
    extra = getattr(message, "model_extra", None) or {}
    if isinstance(extra, dict):
        val = extra.get("reasoning") or extra.get("reasoning_content")
        if isinstance(val, str) and val.strip():
            return val
    return None


def _assistant_message_dict(message: Any) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": message.content,
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in (message.tool_calls or [])
        ],
    }


class FireworksDeepResearchAgent:
    """A minimal, portable deep-research agent over the retrieval endpoint."""

    def __init__(self, cfg: AgentConfig, *, data_source_name: str = "", experiment: str = ""):
        self.cfg = cfg
        self.data_source_name = data_source_name
        self.experiment = experiment
        self.client = make_client(cfg.base_url, cfg.api_key_env, cfg.request_timeout_s)

    def _execute_tool(
        self,
        name: str,
        arguments: str,
        retrieval: RetrievalBackend,
        retrieved: list[str],
        tool_counts: dict[str, int],
    ) -> str:
        try:
            args = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            args = {}

        if name == "search":
            query = args.get("query") or args.get("user_query") or ""
            results = retrieval.search(query)
            for r in results:
                _add_unique(retrieved, str(r.get("docid")))
            tool_counts["search"] = tool_counts.get("search", 0) + 1
            payload = [
                {
                    "docid": str(r.get("docid")),
                    "score": r.get("score"),
                    "title": r.get("title"),
                    "snippet": r.get("snippet"),
                }
                for r in results
            ]
            return json.dumps(payload, ensure_ascii=False)

        if name == "get_document":
            docid = str(args.get("docid", ""))
            tool_counts["get_document"] = tool_counts.get("get_document", 0) + 1
            doc = retrieval.get_document(docid)
            if doc is None:
                return json.dumps({"error": f"docid {docid} not found"}, ensure_ascii=False)
            _add_unique(retrieved, docid)
            return json.dumps(doc, ensure_ascii=False)

        return json.dumps({"error": f"unknown tool {name}"}, ensure_ascii=False)

    def run(self, query_id: str, question: str, retrieval: RetrievalBackend) -> RunRecord:
        cfg = self.cfg
        start = time.monotonic()

        tools = [_search_tool()]
        tool_counts: dict[str, int] = {"search": 0}
        if cfg.query_template == "QUERY_TEMPLATE":
            tools.append(_get_document_tool())
            tool_counts["get_document"] = 0

        messages: list[dict[str, Any]] = []
        if cfg.system_prompt:
            messages.append({"role": "system", "content": cfg.system_prompt})
        messages.append({"role": "user", "content": _format_user_content(question, cfg.query_template)})

        result_items: list[ResultItem] = []
        retrieved: list[str] = []
        usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        status = STATUS_INCOMPLETE
        final_text = ""

        try:
            for _ in range(cfg.max_iterations):
                kwargs: dict[str, Any] = {
                    "model": cfg.model,
                    "messages": messages,
                    "tools": tools,
                    "tool_choice": "auto",
                    "max_tokens": cfg.max_tokens,
                }
                if cfg.temperature is not None:
                    kwargs["temperature"] = cfg.temperature
                if cfg.top_p is not None:
                    kwargs["top_p"] = cfg.top_p
                if cfg.reasoning_effort:
                    kwargs["extra_body"] = {"reasoning_effort": cfg.reasoning_effort}

                response = chat_completion(self.client, **kwargs)
                _accumulate_usage(usage, getattr(response, "usage", None))

                message = response.choices[0].message
                reasoning = _extract_reasoning(message)
                if reasoning:
                    result_items.append(ResultItem.reasoning(reasoning))

                tool_calls = message.tool_calls or []
                if tool_calls:
                    messages.append(_assistant_message_dict(message))
                    for tc in tool_calls:
                        args_str = tc.function.arguments or "{}"
                        output = self._execute_tool(
                            tc.function.name, args_str, retrieval, retrieved, tool_counts
                        )
                        result_items.append(ResultItem.tool_call(tc.function.name, args_str, output))
                        messages.append({"role": "tool", "tool_call_id": tc.id, "content": output})
                    continue

                final_text = message.content or ""
                status = STATUS_COMPLETED if final_text.strip() else STATUS_INCOMPLETE
                break
        except Exception as exc:  # noqa: BLE001 - one bad query must not kill the run
            logger.error("Agent failed on query %s: %s", query_id, exc)
            result_items.append(ResultItem.reasoning(f"agent error: {exc}"))
            status = STATUS_ERROR

        result_items.append(ResultItem.output_text(final_text))

        return RunRecord(
            query_id=str(query_id),
            status=status,
            result=tuple(result_items),
            retrieved_docids=tuple(retrieved),
            tool_call_counts=tool_counts,
            usage=usage,
            metadata={
                "model": cfg.model,
                "base_url": cfg.base_url,
                "query_template": cfg.query_template,
                "has_system_prompt": bool(cfg.system_prompt),
                "reasoning_effort": cfg.reasoning_effort,
                "data_source": self.data_source_name,
                "experiment": self.experiment,
                "elapsed_s": round(time.monotonic() - start, 2),
            },
        )


def _accumulate_usage(usage: dict[str, int], resp_usage: Any) -> None:
    if resp_usage is None:
        return
    usage["input_tokens"] += getattr(resp_usage, "prompt_tokens", 0) or 0
    usage["output_tokens"] += getattr(resp_usage, "completion_tokens", 0) or 0
    usage["total_tokens"] += getattr(resp_usage, "total_tokens", 0) or 0


def build_agent(cfg: AgentConfig, *, data_source_name: str = "", experiment: str = ""):
    """Construct the agent named by ``cfg.type``.

    - ``fireworks_reference`` -> FireworksDeepResearchAgent
    - ``python:<module>:<factory>`` -> import and call factory(cfg)
    - ``external`` is handled by the pipeline (no agent object).
    """
    if cfg.type == "fireworks_reference":
        return FireworksDeepResearchAgent(cfg, data_source_name=data_source_name, experiment=experiment)
    if cfg.type.startswith("python:"):
        try:
            _, module_name, factory_name = cfg.type.split(":", 2)
        except ValueError as exc:
            raise AgentError(f"agent.type must look like 'python:<module>:<factory>': {cfg.type}") from exc
        import importlib

        module = importlib.import_module(module_name)
        factory = getattr(module, factory_name)
        return factory(cfg)
    raise AgentError(f"build_agent cannot construct agent.type={cfg.type!r}")


def _run_file_exists(runs_dir: Path, query_id: str) -> bool:
    return any(runs_dir.glob(f"run_*_{query_id}.json"))


def run_agent_over_queries(
    agent: Any,
    queries: list[tuple[str, str]],
    retrieval: RetrievalBackend,
    runs_dir: str | Path,
    *,
    num_threads: int = 4,
    skip_existing: bool = True,
) -> list[Path]:
    """Run ``agent`` over (query_id, question) pairs; write run files.

    Resumable: queries whose run file already exists are skipped.
    """
    out_dir = Path(runs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pending = [
        (qid, q) for qid, q in queries if not (skip_existing and _run_file_exists(out_dir, qid))
    ]
    skipped = len(queries) - len(pending)
    if skipped:
        logger.info("Skipping %d queries with existing run files", skipped)
    if not pending:
        return sorted(out_dir.glob("run_*.json"))

    written: list[Path] = []

    def _one(qid: str, question: str) -> Optional[Path]:
        try:
            record = agent.run(qid, question, retrieval)
            return write_run_file(record, out_dir)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to produce run for query %s: %s", qid, exc)
            error_record = RunRecord(
                query_id=str(qid),
                status=STATUS_ERROR,
                result=(ResultItem.reasoning(f"pipeline error: {exc}"), ResultItem.output_text("")),
                metadata={"error": str(exc)},
            )
            try:
                return write_run_file(error_record, out_dir)
            except Exception:  # noqa: BLE001
                return None

    workers = max(1, num_threads)
    logger.info("Running agent over %d queries (%d threads)...", len(pending), workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_one, qid, q): qid for qid, q in pending}
        for fut in as_completed(futures):
            path = fut.result()
            if path is not None:
                written.append(path)
                logger.info("  run written: %s", path.name)

    return sorted(out_dir.glob("run_*.json"))
