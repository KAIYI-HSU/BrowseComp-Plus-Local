"""The agent output contract (run files) and the agent/retrieval protocols.

The *run file* is the universal boundary between a Deep-Research agent and the
evaluator: any agent, in any language, that writes a conforming run file can be
scored. The built-in Fireworks agent and the Python adapter both produce these.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

from .errors import RunFileSchemaError

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"
STATUS_COMPLETED = "completed"
STATUS_INCOMPLETE = "incomplete"
STATUS_ERROR = "error"
VALID_STATUSES = (STATUS_COMPLETED, STATUS_INCOMPLETE, STATUS_ERROR)

RESULT_REASONING = "reasoning"
RESULT_TOOL_CALL = "tool_call"
RESULT_OUTPUT_TEXT = "output_text"


@dataclass(frozen=True)
class ResultItem:
    """One step in the agent trajectory.

    The final answer is the LAST item, with ``type == "output_text"`` — this is
    what the judge reads as the agent's response.
    """

    type: str
    output: Any = None
    tool_name: Optional[str] = None
    arguments: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "output": self.output,
        }

    @classmethod
    def reasoning(cls, text: str) -> "ResultItem":
        return cls(type=RESULT_REASONING, output=text)

    @classmethod
    def tool_call(cls, tool_name: str, arguments: str, output: str) -> "ResultItem":
        return cls(type=RESULT_TOOL_CALL, tool_name=tool_name, arguments=arguments, output=output)

    @classmethod
    def output_text(cls, text: str) -> "ResultItem":
        return cls(type=RESULT_OUTPUT_TEXT, output=text)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResultItem":
        return cls(
            type=data.get("type", ""),
            output=data.get("output"),
            tool_name=data.get("tool_name"),
            arguments=data.get("arguments"),
        )


@dataclass(frozen=True)
class RunRecord:
    """A single query's run, serialized to ``run_<ts>_<qid>.json``."""

    query_id: str
    status: str
    result: tuple[ResultItem, ...]
    retrieved_docids: tuple[str, ...] = ()
    tool_call_counts: dict[str, int] = field(default_factory=dict)
    usage: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    @property
    def final_answer(self) -> str:
        if self.result and self.result[-1].type == RESULT_OUTPUT_TEXT:
            return str(self.result[-1].output or "")
        return ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "query_id": str(self.query_id),
            "status": self.status,
            "retrieved_docids": list(self.retrieved_docids),
            "tool_call_counts": dict(self.tool_call_counts),
            "result": [item.to_dict() for item in self.result],
            "usage": dict(self.usage),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunRecord":
        return cls(
            query_id=str(data.get("query_id")),
            status=data.get("status", STATUS_ERROR),
            result=tuple(ResultItem.from_dict(r) for r in data.get("result", [])),
            retrieved_docids=tuple(str(d) for d in data.get("retrieved_docids", [])),
            tool_call_counts=dict(data.get("tool_call_counts", {})),
            usage=dict(data.get("usage", {})),
            metadata=dict(data.get("metadata", {})),
            schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
        )

    def validate(self) -> None:
        """Raise RunFileSchemaError if the record violates the contract."""
        if not self.query_id or self.query_id == "None":
            raise RunFileSchemaError("run record missing query_id")
        if self.status not in VALID_STATUSES:
            raise RunFileSchemaError(
                f"status must be one of {VALID_STATUSES}, got {self.status!r}"
            )
        if not self.result:
            raise RunFileSchemaError(f"query {self.query_id}: result must be non-empty")
        if self.status == STATUS_COMPLETED:
            last = self.result[-1]
            if last.type != RESULT_OUTPUT_TEXT or not str(last.output or "").strip():
                raise RunFileSchemaError(
                    f"query {self.query_id}: a completed run must end with a non-empty "
                    "output_text result item"
                )


def _timestamp() -> str:
    """UTC timestamp matching the repo's existing run-file naming."""
    now = datetime.now(timezone.utc)
    return now.strftime("%Y%m%dT%H%M%S") + f"{now.microsecond:06d}Z"


def write_run_file(record: RunRecord, runs_dir: str | Path) -> Path:
    """Validate and write a run record; returns the file path."""
    record.validate()
    out_dir = Path(runs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"run_{_timestamp()}_{record.query_id}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(record.to_dict(), f, indent=2, ensure_ascii=False)
    return path


def read_run_file(path: str | Path) -> RunRecord:
    """Read a run file into a RunRecord (lenient about extra fields)."""
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunFileSchemaError(f"cannot read run file {p}: {exc}") from exc
    return RunRecord.from_dict(data)


def validate_run_file(path: str | Path) -> RunRecord:
    """Read and validate a run file, raising RunFileSchemaError on violation."""
    record = read_run_file(path)
    record.validate()
    return record


@runtime_checkable
class RetrievalBackend(Protocol):
    """What an agent uses to retrieve. Implemented by retrieval_client."""

    def search(self, query: str, k: Optional[int] = None) -> list[dict[str, Any]]:
        """Return [{docid, score, title, snippet}, ...]."""
        ...

    def get_document(self, docid: str) -> Optional[dict[str, Any]]:
        """Return {docid, title, text} or None."""
        ...


@runtime_checkable
class DeepResearchAgent(Protocol):
    """The Python adapter contract for an agent under test.

    Implement this and expose a factory ``def make_agent(agent_cfg) -> DeepResearchAgent``
    referenced as ``agent.type: python:<module>:make_agent``.
    """

    def run(self, query_id: str, question: str, retrieval: RetrievalBackend) -> RunRecord:
        """Answer one question using ``retrieval``; return a RunRecord."""
