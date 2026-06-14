"""Unit tests for scoring primitives and metric aggregation."""

from __future__ import annotations

from fireworks_eval.scoring import (
    aggregate_summary,
    compute_citation_metrics,
    extract_citations_from_response,
    parse_judge_response,
)


def test_parse_judge_yes_with_confidence():
    text = "extracted_final_answer: Paris\nreasoning: equivalent\ncorrect: yes\nconfidence: 90%"
    result = parse_judge_response(text)
    assert result["correct"] is True
    assert result["confidence"] == 90.0
    assert result["parse_error"] is False
    assert result["extracted_final_answer"] == "Paris"


def test_parse_judge_no_bold_variant():
    result = parse_judge_response("**extracted_final_answer:** X\n**correct:** no\n**confidence:** 12%")
    assert result["correct"] is False
    assert result["confidence"] == 12.0


def test_parse_judge_error_when_unparseable():
    result = parse_judge_response("this is not a verdict")
    assert result["parse_error"] is True
    assert result["correct"] is None


def test_parse_judge_empty():
    assert parse_judge_response("")["parse_error"] is True


def test_extract_citations_square_and_fullwidth():
    assert set(extract_citations_from_response("a [20]. b [30, 31].")) == {"20", "30", "31"}
    assert set(extract_citations_from_response("see 【5】 and 【6, 7】")) == {"5", "6", "7"}
    assert extract_citations_from_response("") == []


def test_compute_citation_metrics():
    m = compute_citation_metrics(["1", "2"], ["2", "3"])
    assert m["num_citations"] == 2
    assert m["num_relevant"] == 2
    assert m["precision"] == 0.5
    assert m["recall"] == 0.5
    assert compute_citation_metrics([], ["2"])["precision"] == 0.0


def _eval(qid, correct, conf, recall, search, cited):
    return {
        "query_id": qid,
        "is_completed": True,
        "judge_result": {"correct": correct, "confidence": conf, "parse_error": False},
        "retrieval": {"recall": recall},
        "tool_call_counts": {"search": search},
        "citations": {"cited_docids": cited, "metrics": {"precision": 1.0 if cited else 0.0, "recall": 0.5 if cited else 0.0}},
    }


def test_aggregate_summary_basic():
    results = [
        _eval("1", True, 90, 0.5, 3, ["1"]),
        _eval("2", False, 10, 0.0, 1, []),
    ]
    summary = aggregate_summary(
        results, llm_name="m", judge_model="j", agent_model="a", data_source="d", experiment="e"
    )
    assert summary["Accuracy (%)"] == 50.0
    assert summary["Recall (%)"] == 25.0
    assert summary["avg_tool_stats"]["search"] == 2.0
    assert summary["Calibration Error (%)"] is None  # < 100 samples
    assert summary["num_queries"] == 2
    assert len(summary["per_query_metrics"]) == 2


def test_aggregate_summary_per_domain():
    results = [
        _eval("1", True, 90, 0.5, 3, ["1"]),
        _eval("2", False, 10, 0.0, 1, []),
    ]
    tags = {"1": "History", "2": "Science & Nature"}
    summary = aggregate_summary(
        results, llm_name="m", judge_model="j", agent_model="a",
        data_source="d", experiment="e", tags=tags,
    )
    assert summary["per_domain"]["History"]["accuracy_%"] == 100.0
    assert summary["per_domain"]["Science & Nature"]["accuracy_%"] == 0.0
    assert all("domain" in q for q in summary["per_query_metrics"])


def test_aggregate_summary_recall_skips_none():
    results = [
        _eval("1", True, 90, 0.5, 3, ["1"]),
        {"query_id": "2", "is_completed": True,
         "judge_result": {"correct": True, "confidence": 50, "parse_error": False},
         "retrieval": {"recall": None}, "tool_call_counts": {"search": 1}, "citations": None},
    ]
    summary = aggregate_summary(
        results, llm_name="m", judge_model="j", agent_model="a", data_source="d", experiment="e"
    )
    assert summary["Recall (%)"] == 50.0  # only the non-None recall counts
