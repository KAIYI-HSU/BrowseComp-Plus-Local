from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import openai
from dotenv import load_dotenv
from rich import print as rprint
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from search_agent.prompts import format_query
from search_agent.utils import extract_retrieved_docids_from_result


FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
DEFAULT_FIREWORKS_AGENT_MODEL = "accounts/fireworks/models/gpt-oss-120b"
CHAT_COMPLETION_METADATA_KEYS = {"api_base_url"}


def to_chat_tool_definitions(response_tool_definitions: list[dict]) -> list[dict]:
    """Convert the repo's Responses-style tool schema to Chat Completions schema."""
    chat_tools = []
    for tool in response_tool_definitions:
        if "function" in tool:
            chat_tools.append(tool)
            continue

        converted = {
            "type": tool.get("type", "function"),
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {}),
            },
        }
        if "strict" in tool:
            converted["function"]["strict"] = tool["strict"]
        chat_tools.append(converted)
    return chat_tools


def _message_to_dict(message: Any) -> dict:
    if isinstance(message, dict):
        return {k: v for k, v in message.items() if v is not None}
    if hasattr(message, "model_dump"):
        return message.model_dump(mode="python", exclude_none=True)
    result = {
        "role": getattr(message, "role", "assistant"),
        "content": getattr(message, "content", None),
        "reasoning_content": getattr(message, "reasoning_content", None),
        "tool_calls": getattr(message, "tool_calls", None),
    }
    return {k: v for k, v in result.items() if v is not None}


def _tool_call_to_dict(tool_call: Any) -> dict:
    if isinstance(tool_call, dict):
        return tool_call
    if hasattr(tool_call, "model_dump"):
        return tool_call.model_dump(mode="python", exclude_none=True)
    function = getattr(tool_call, "function", None)
    if hasattr(function, "model_dump"):
        function_dict = function.model_dump(mode="python", exclude_none=True)
    else:
        function_dict = {
            "name": getattr(function, "name", None),
            "arguments": getattr(function, "arguments", None),
        }
    return {
        "id": getattr(tool_call, "id", None),
        "type": getattr(tool_call, "type", "function"),
        "function": {k: v for k, v in function_dict.items() if v is not None},
    }


class SearchToolHandler:
    def __init__(
        self,
        searcher,
        snippet_max_tokens: Optional[int] = None,
        k: int = 5,
        include_get_document: bool = True,
    ):
        self.searcher = searcher
        self.snippet_max_tokens = snippet_max_tokens
        self.k = k
        self.include_get_document = include_get_document
        self.tokenizer = None

        if snippet_max_tokens and snippet_max_tokens > 0:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    def execute_tool(self, tool_name: str, arguments: dict):
        if tool_name == "local_knowledge_base_retrieval":
            return self._search(arguments["user_query"])
        if tool_name == "get_document":
            return self._get_document(arguments["docid"])
        raise ValueError(f"Unknown tool: {tool_name}")

    def get_response_tool_definitions(self) -> list[dict]:
        tools = [
            {
                "type": "function",
                "name": "local_knowledge_base_retrieval",
                "description": self.searcher.search_description(self.k),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_query": {
                            "type": "string",
                            "description": "Query to search the local knowledge base for relevant information",
                        }
                    },
                    "required": ["user_query"],
                    "additionalProperties": False,
                },
                "strict": True,
            }
        ]

        if self.include_get_document:
            tools.append(
                {
                    "type": "function",
                    "name": "get_document",
                    "description": self.searcher.get_document_description(),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "docid": {
                                "type": "string",
                                "description": "Document ID to retrieve",
                            }
                        },
                        "required": ["docid"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                }
            )

        return tools

    def get_chat_tool_definitions(self) -> list[dict]:
        return to_chat_tool_definitions(self.get_response_tool_definitions())

    def _truncate_text(self, text: str) -> str:
        if not self.snippet_max_tokens or self.snippet_max_tokens <= 0 or not self.tokenizer:
            return text
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= self.snippet_max_tokens:
            return text
        truncated_tokens = tokens[: self.snippet_max_tokens]
        return self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)

    def _search(self, query: str):
        candidates = self.searcher.search(query, self.k)
        results = []
        for cand in candidates:
            text = cand.get("text", cand.get("snippet", ""))
            item = {"docid": str(cand["docid"]), "snippet": self._truncate_text(text)}
            if cand.get("score") is not None:
                item["score"] = cand["score"]
            results.append(item)
        return json.dumps(results, indent=2, ensure_ascii=False)

    def _get_document(self, docid: str):
        result = self.searcher.get_document(docid)
        if result is None:
            return json.dumps({"error": f"Document with docid '{docid}' not found"})
        return json.dumps(result, indent=2, ensure_ascii=False)


def run_conversation_with_tools(
    client: openai.OpenAI,
    initial_request: dict,
    tool_handler: SearchToolHandler,
    max_iterations: int = 100,
    verbose: bool = False,
):
    tool_usage: dict[str, int] = {}
    messages = list(initial_request["messages"])

    for _ in range(max_iterations):
        request = {
            key: value
            for key, value in initial_request.items()
            if key not in CHAT_COMPLETION_METADATA_KEYS
        }
        request["messages"] = messages

        try:
            response = client.chat.completions.create(**request)
        except Exception as exc:
            if verbose:
                print(f"Error calling model: {exc}")
                rprint(request)
            raise

        if not response.choices:
            return messages, tool_usage, "incomplete"

        choice = response.choices[0]
        assistant_message = _message_to_dict(choice.message)
        tool_calls = [
            _tool_call_to_dict(tool_call)
            for tool_call in assistant_message.get("tool_calls", []) or []
        ]
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        messages.append(assistant_message)

        if not tool_calls:
            content = assistant_message.get("content")
            finish_reason = getattr(choice, "finish_reason", None)
            if finish_reason == "length":
                return messages, tool_usage, "incomplete"
            if not isinstance(content, str) or not content.strip():
                return messages, tool_usage, "incomplete"
            return messages, tool_usage, "completed"

        for tool_call in tool_calls:
            function = tool_call.get("function", {}) or {}
            tool_name = function.get("name")
            try:
                arguments = json.loads(function.get("arguments") or "{}")
                result = tool_handler.execute_tool(tool_name, arguments)
                tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1
            except Exception as exc:
                result = f"Error executing {tool_name}: {exc}"

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id"),
                    "content": result,
                }
            )

    return messages, tool_usage, "incomplete"


def _normalized_tool_counts(tool_usage: dict) -> dict[str, int]:
    normalized_tool_counts: dict[str, int] = {}
    for tool_name, count in (tool_usage or {}).items():
        normalized_name = (
            "search"
            if isinstance(tool_name, str)
            and ("retrieval" in tool_name.lower() or "search" in tool_name.lower())
            else str(tool_name)
        )
        normalized_tool_counts[normalized_name] = normalized_tool_counts.get(
            normalized_name, 0
        ) + int(count or 0)
    return normalized_tool_counts


def normalize_chat_messages_for_run(messages: list[dict]) -> list[dict]:
    tool_output_by_id: dict[str, str | dict | None] = {}
    for item in messages or []:
        if isinstance(item, dict) and item.get("role") == "tool":
            tool_output_by_id[str(item.get("tool_call_id"))] = item.get("content")

    normalized_results: list[dict] = []
    for item in messages or []:
        if not isinstance(item, dict) or item.get("role") != "assistant":
            continue

        reasoning = item.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning.strip():
            normalized_results.append(
                {
                    "type": "reasoning",
                    "tool_name": None,
                    "arguments": None,
                    "output": [reasoning.strip()],
                }
            )

        for tool_call in item.get("tool_calls", []) or []:
            call = _tool_call_to_dict(tool_call)
            function = call.get("function", {}) or {}
            call_id = str(call.get("id"))
            normalized_results.append(
                {
                    "type": "tool_call",
                    "tool_name": function.get("name"),
                    "arguments": function.get("arguments"),
                    "output": tool_output_by_id.get(call_id),
                }
            )

        content = item.get("content")
        if isinstance(content, str) and content.strip():
            normalized_results.append(
                {
                    "type": "output_text",
                    "tool_name": None,
                    "arguments": None,
                    "output": content.strip(),
                }
            )

    return normalized_results


def persist_chat_response(
    out_dir: str | Path,
    initial_request: dict,
    messages: list[dict],
    tool_usage: dict,
    status: str,
    *,
    query_id: Optional[str] = None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    normalized_results = normalize_chat_messages_for_run(messages)
    normalized_record = {
        "metadata": {
            "model": initial_request.get("model"),
            "reasoning_effort": initial_request.get("reasoning_effort"),
            "api_base_url": initial_request.get("api_base_url"),
            "output_dir": str(out_dir),
        },
        "query_id": query_id,
        "tool_call_counts": _normalized_tool_counts(tool_usage),
        "status": status,
        "retrieved_docids": extract_retrieved_docids_from_result(normalized_results),
        "result": normalized_results,
    }

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    suffix = f"_{query_id}" if query_id else ""
    path = out_dir / f"run_{ts}{suffix}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(normalized_record, f, indent=2, ensure_ascii=False, default=str)
    return path


def _process_tsv_dataset(
    tsv_path: str, client: openai.OpenAI, args, tool_handler: SearchToolHandler
) -> None:
    dataset_path = Path(tsv_path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"TSV file not found: {tsv_path}")

    out_dir = Path(args.output_dir).expanduser().resolve()
    queries = []
    with dataset_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            queries.append((row[0].strip(), row[1].strip()))

    processed_ids = set()
    if out_dir.exists():
        for json_path in out_dir.glob("run_*.json"):
            try:
                with json_path.open("r", encoding="utf-8") as jf:
                    meta = json.load(jf)
                    qid_saved = meta.get("query_id")
                    if qid_saved:
                        processed_ids.add(str(qid_saved))
            except Exception:
                continue

    remaining = [(qid, qtext) for qid, qtext in queries if qid not in processed_ids]
    print(
        f"Processing {len(remaining)} remaining queries "
        f"(skipping {len(processed_ids)}) from {dataset_path} ..."
    )

    def _handle_single_query(qid: str, qtext: str):
        initial_request = {
            "model": args.model,
            "messages": [
                {"role": "user", "content": format_query(qtext, args.query_template)}
            ],
            "tools": tool_handler.get_chat_tool_definitions(),
            "tool_choice": "auto",
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "api_base_url": args.api_base_url,
        }
        if args.reasoning_effort:
            initial_request["reasoning_effort"] = args.reasoning_effort

        messages, tool_usage, status = run_conversation_with_tools(
            client, initial_request, tool_handler, args.max_iterations, args.verbose
        )
        path = persist_chat_response(
            out_dir, initial_request, messages, tool_usage, status, query_id=qid
        )
        if args.verbose:
            print("Saved response to", path, "| tool call counts:", tool_usage)

    if args.num_threads <= 1:
        for qid, qtext in tqdm(remaining, desc="Queries", unit="query"):
            _handle_single_query(qid, qtext)
    else:
        with (
            ThreadPoolExecutor(max_workers=args.num_threads) as executor,
            tqdm(total=len(remaining), desc="Queries", unit="query") as pbar,
        ):
            futures = [executor.submit(_handle_single_query, qid, qtext) for qid, qtext in remaining]
            for future in as_completed(futures):
                future.result()
                pbar.update(1)


def build_client(*, api_key_env: str, api_base_url: str) -> openai.OpenAI:
    load_dotenv()
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} is not set in the environment or .env")
    return openai.OpenAI(api_key=api_key, base_url=api_base_url)


def main() -> None:
    from searcher.searchers import SearcherType

    parser = argparse.ArgumentParser(
        description="Run a Fireworks/OpenAI-compatible Deep Research agent with local BrowseComp-Plus search tools.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--query",
        default="topics-qrels/queries.tsv",
        help="User query or TSV file path with query_id<TAB>query rows",
    )
    parser.add_argument(
        "--model", default=DEFAULT_FIREWORKS_AGENT_MODEL, help="Fireworks model path"
    )
    parser.add_argument(
        "--api-base-url",
        default=FIREWORKS_BASE_URL,
        help="OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--api-key-env",
        default="FIREWORKS_API_KEY",
        help="Environment variable containing the API key",
    )
    parser.add_argument("--max-tokens", type=int, default=10000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--reasoning-effort", choices=["low", "medium", "high"], default="high"
    )
    parser.add_argument(
        "--output-dir", default="runs/fireworks", help="Directory to store run JSON files"
    )
    parser.add_argument(
        "--query-template",
        choices=[
            "QUERY_TEMPLATE",
            "QUERY_TEMPLATE_NO_GET_DOCUMENT",
            "QUERY_TEMPLATE_NO_GET_DOCUMENT_NO_CITATION",
        ],
        default="QUERY_TEMPLATE_NO_GET_DOCUMENT",
    )
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--snippet-max-tokens", type=int, default=512)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--get-document", action="store_true", help="Register the get_document tool"
    )

    parser.add_argument(
        "--searcher-type",
        choices=SearcherType.get_choices(),
        required=True,
        help=f"Type of searcher to use: {', '.join(SearcherType.get_choices())}",
    )
    parser.add_argument("--hf-token", type=str)
    parser.add_argument("--hf-home", type=str)

    temp_args, _ = parser.parse_known_args()
    searcher_class = SearcherType.get_searcher_class(temp_args.searcher_type)
    searcher_class.parse_args(parser)
    args = parser.parse_args()

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = args.hf_token
    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home

    client = build_client(api_key_env=args.api_key_env, api_base_url=args.api_base_url)
    searcher = searcher_class(args)
    tool_handler = SearchToolHandler(
        searcher=searcher,
        snippet_max_tokens=args.snippet_max_tokens,
        k=args.k,
        include_get_document=args.get_document,
    )

    qstr = str(args.query).strip()
    potential_path = Path(qstr)
    if qstr.lower().endswith(".tsv") and potential_path.is_file():
        _process_tsv_dataset(str(potential_path), client, args, tool_handler)
        return

    initial_request = {
        "model": args.model,
        "messages": [
            {"role": "user", "content": format_query(args.query, args.query_template)}
        ],
        "tools": tool_handler.get_chat_tool_definitions(),
        "tool_choice": "auto",
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "api_base_url": args.api_base_url,
    }
    if args.reasoning_effort:
        initial_request["reasoning_effort"] = args.reasoning_effort

    messages, tool_usage, status = run_conversation_with_tools(
        client, initial_request, tool_handler, args.max_iterations, args.verbose
    )
    path = persist_chat_response(
        args.output_dir, initial_request, messages, tool_usage, status, query_id=None
    )
    print("Saved response to", path, "| tool call counts:", tool_usage)
    rprint(messages)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
