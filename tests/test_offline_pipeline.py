"""Offline test of the judge stage end-to-end (no network).

The judge's LLM call is monkeypatched so we exercise the real loading,
recall computation, citation metrics, eval-file writing, summary
aggregation, and CSV emission against synthetic run files.
"""

from __future__ import annotations

import json

from fireworks_eval.agent_protocol import ResultItem, RunRecord, write_run_file
from fireworks_eval.config import JudgeConfig
from fireworks_eval.judge import FireworksJudge, evaluate_runs


def _fake_judge(self, prompt: str) -> str:
    # Run responses embed a marker so the verdict is deterministic offline.
    if "ANSWER_OK" in prompt:
        return "extracted_final_answer: Right\nreasoning: equivalent\ncorrect: yes\nconfidence: 95%"
    return "extracted_final_answer: Wrong\nreasoning: differs\ncorrect: no\nconfidence: 10%"


def test_evaluate_runs_offline(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir()
    evals = tmp_path / "evals"

    write_run_file(
        RunRecord(
            query_id="1",
            status="completed",
            result=(
                ResultItem.tool_call("search", '{"query": "q"}', "[]"),
                ResultItem.output_text("Explanation: see [101].\nExact Answer: Right ANSWER_OK\nConfidence: 95%"),
            ),
            retrieved_docids=("101", "102"),
            tool_call_counts={"search": 2},
            metadata={"model": "test-agent-model"},
        ),
        runs,
    )
    write_run_file(
        RunRecord(
            query_id="2",
            status="completed",
            result=(ResultItem.output_text("Exact Answer: Wrong ANSWER_BAD\nConfidence: 50%"),),
            retrieved_docids=("999",),
            tool_call_counts={"search": 1},
            metadata={"model": "test-agent-model"},
        ),
        runs,
    )

    ground_truth = {
        "1": {"question": "q1", "answer": "Right"},
        "2": {"question": "q2", "answer": "Right"},
    }
    qrel_evidence = {"1": ["101", "103"], "2": ["201"]}

    monkeypatch.setattr(FireworksJudge, "judge", _fake_judge)
    judge = FireworksJudge(JudgeConfig())
    summary, summary_path = evaluate_runs(
        judge, runs, evals, ground_truth, qrel_evidence,
        agent_model="test-agent-model", data_source="ds", experiment="t",
    )

    assert summary["num_queries"] == 2
    assert summary["Accuracy (%)"] == 50.0          # one yes, one no
    assert summary["Recall (%)"] == 25.0            # mean(0.5, 0.0)
    assert summary["avg_tool_stats"]["search"] == 1.5
    assert summary["LLM"] == "test-agent-model"

    assert summary_path.is_file()
    assert (evals / "detailed_judge_results.csv").is_file()
    assert len(list(evals.glob("*_eval.json"))) == 2

    # Per-query correctness recorded.
    correctness = {q["query_id"]: q["correct"] for q in summary["per_query_metrics"]}
    assert correctness == {"1": True, "2": False}


def test_evaluate_runs_resumes(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir()
    evals = tmp_path / "evals"
    write_run_file(
        RunRecord(
            query_id="1", status="completed",
            result=(ResultItem.output_text("Exact Answer: Right ANSWER_OK"),),
            retrieved_docids=("101",), tool_call_counts={"search": 1},
        ),
        runs,
    )
    gt = {"1": {"question": "q1", "answer": "Right"}}
    qrel = {"1": ["101"]}

    calls = {"n": 0}

    def _counting_judge(self, prompt):
        calls["n"] += 1
        return "correct: yes\nconfidence: 90%"

    monkeypatch.setattr(FireworksJudge, "judge", _counting_judge)
    judge = FireworksJudge(JudgeConfig())
    evaluate_runs(judge, runs, evals, gt, qrel)
    assert calls["n"] == 1
    # Second run reuses existing eval files (no extra judge calls).
    evaluate_runs(judge, runs, evals, gt, qrel)
    assert calls["n"] == 1
