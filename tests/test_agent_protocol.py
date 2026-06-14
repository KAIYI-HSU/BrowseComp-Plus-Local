"""Tests for the run-file contract (RunRecord)."""

from __future__ import annotations

import pytest

from fireworks_eval.agent_protocol import (
    ResultItem,
    RunRecord,
    read_run_file,
    validate_run_file,
    write_run_file,
)
from fireworks_eval.errors import RunFileSchemaError


def _completed_record(qid="769"):
    return RunRecord(
        query_id=qid,
        status="completed",
        result=(
            ResultItem.tool_call("search", '{"query": "x"}', "[]"),
            ResultItem.output_text("Exact Answer: Foo\nConfidence: 80%"),
        ),
        retrieved_docids=("1", "2"),
        tool_call_counts={"search": 1},
    )


def test_roundtrip_to_from_dict():
    record = _completed_record()
    restored = RunRecord.from_dict(record.to_dict())
    assert restored.query_id == "769"
    assert restored.status == "completed"
    assert restored.final_answer.startswith("Exact Answer: Foo")
    assert restored.retrieved_docids == ("1", "2")
    assert restored.tool_call_counts == {"search": 1}


def test_validate_completed_requires_output_text_last():
    bad = RunRecord(
        query_id="1",
        status="completed",
        result=(ResultItem.tool_call("search", "{}", "[]"),),  # no output_text
    )
    with pytest.raises(RunFileSchemaError):
        bad.validate()


def test_validate_completed_rejects_empty_answer():
    bad = RunRecord(query_id="1", status="completed", result=(ResultItem.output_text("   "),))
    with pytest.raises(RunFileSchemaError):
        bad.validate()


def test_incomplete_allows_empty_answer():
    rec = RunRecord(query_id="1", status="incomplete", result=(ResultItem.output_text(""),))
    rec.validate()  # must not raise


def test_write_and_read_run_file(tmp_path):
    path = write_run_file(_completed_record("770"), tmp_path)
    assert path.exists()
    assert path.name.startswith("run_") and path.name.endswith("_770.json")
    restored = validate_run_file(path)
    assert restored.query_id == "770"


def test_read_invalid_path_raises(tmp_path):
    with pytest.raises(RunFileSchemaError):
        read_run_file(tmp_path / "does_not_exist.json")
