"""Tests for experiment comparison (regression detection)."""

from __future__ import annotations

from fireworks_eval.compare import compare_summaries


def _summary(name, acc, recall, search, pqm, per_domain=None):
    s = {
        "experiment": name,
        "Accuracy (%)": acc,
        "Recall (%)": recall,
        "Calibration Error (%)": None,
        "avg_tool_stats": {"search": search},
        "per_query_metrics": pqm,
    }
    if per_domain:
        s["per_domain"] = per_domain
    return s


def test_compare_detects_improvements_and_regressions():
    base = _summary("a", 50.0, 40.0, 10.0, [
        {"query_id": "1", "correct": True},
        {"query_id": "2", "correct": False},
        {"query_id": "3", "correct": True},
    ])
    cand = _summary("b", 66.67, 45.0, 8.0, [
        {"query_id": "1", "correct": True},
        {"query_id": "2", "correct": True},   # ✗ -> ✓ improvement
        {"query_id": "3", "correct": False},  # ✓ -> ✗ regression
    ])
    cmp = compare_summaries(base, cand)
    assert cmp["deltas"]["accuracy_%"] == 16.67
    assert cmp["deltas"]["recall_%"] == 5.0
    assert cmp["deltas"]["avg_search_calls"] == -2.0
    assert [i["query_id"] for i in cmp["improvements"]] == ["2"]
    assert [r["query_id"] for r in cmp["regressions"]] == ["3"]
    assert cmp["unchanged"] == 1
    assert "mixed" in cmp["verdict"]


def test_compare_no_change():
    s = _summary("a", 100.0, 66.67, 11.0, [{"query_id": "1", "correct": True}])
    cmp = compare_summaries(s, s)
    assert cmp["verdict"] == "no-change"
    assert cmp["unchanged"] == 1
    assert cmp["improvements"] == []
    assert cmp["regressions"] == []


def test_compare_clean_improvement():
    base = _summary("a", 0.0, 0.0, 5.0, [{"query_id": "1", "correct": False}])
    cand = _summary("b", 100.0, 0.0, 5.0, [{"query_id": "1", "correct": True}])
    cmp = compare_summaries(base, cand)
    assert cmp["verdict"] == "improved"
    assert len(cmp["improvements"]) == 1
    assert not cmp["regressions"]


def test_compare_per_domain_deltas():
    base = _summary("a", 50, 40, 10, [{"query_id": "1", "correct": True}],
                    per_domain={"History": {"accuracy_%": 50.0, "recall_%": 40.0, "n": 2}})
    cand = _summary("b", 75, 50, 9, [{"query_id": "1", "correct": True}],
                    per_domain={"History": {"accuracy_%": 75.0, "recall_%": 50.0, "n": 2}})
    cmp = compare_summaries(base, cand)
    assert cmp["per_domain_deltas"]["History"]["accuracy_%"] == 25.0
    assert cmp["per_domain_deltas"]["History"]["recall_%"] == 10.0
