"""Scoring primitives and metric aggregation.

The grader prompt, judge-response parsing, calibration error, and citation
metrics are faithful replications of
``scripts_evaluation/evaluate_run.py`` so that results are directly comparable
to the official BrowseComp-Plus evaluator. We replicate (not import) because
that module hard-imports ``vllm`` at module load, which is impractical here.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from datetime import date
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Canonical grader template (evaluate_run.py:17-38).
GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

[correct_answer]: {correct_answer}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response].

[correct_answer]: Repeat the [correct_answer] given above.

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], in the context of this [question]. You should judge whether the extracted_final_answer is semantically equivalent to [correct_answer], allowing the extracted_final_answer to be string variations of [correct_answer]. You should also allow the extracted_final_answer to be more precise or verbose than [correct_answer], as long as its additional details are correct. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers are semantically equivalent.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\\%| and 100|\\%| from [response]. Put 100 if there is no confidence score available.
""".strip()


def create_judge_prompt(question: str, response: str, correct_answer: str) -> str:
    return GRADER_TEMPLATE.format(
        question=question, response=response, correct_answer=correct_answer
    )


def parse_judge_response(judge_response: str) -> dict[str, Any]:
    """Parse the grader's free-text output into structured fields.

    Replicates evaluate_run.py:60-149 (bold/plain variants, yes/no, confidence).
    """
    result: dict[str, Any] = {
        "extracted_final_answer": None,
        "reasoning": None,
        "correct": None,
        "confidence": None,
        "parse_error": False,
    }
    if not judge_response:
        result["parse_error"] = True
        return result

    answer_match = re.search(
        r"\*\*extracted_final_answer:\*\*\s*(.*?)(?=\n|$)", judge_response, re.IGNORECASE | re.DOTALL
    )
    if not answer_match:
        answer_match = re.search(
            r"\*\*extracted_final_answer\*\*:\s*(.*?)(?=\n|$)", judge_response, re.IGNORECASE | re.DOTALL
        )
    if not answer_match:
        answer_match = re.search(
            r"extracted_final_answer:\s*(.*?)(?=\n|$)", judge_response, re.IGNORECASE | re.DOTALL
        )
    if answer_match:
        result["extracted_final_answer"] = answer_match.group(1).strip()

    reasoning_match = re.search(
        r"\*\*reasoning:\*\*\s*(.*?)(?=\n\*\*correct:\*\*|\n\*\*correct\*\*:|\ncorrect:|$)",
        judge_response, re.IGNORECASE | re.DOTALL,
    )
    if not reasoning_match:
        reasoning_match = re.search(
            r"\*\*reasoning\*\*:\s*(.*?)(?=\n\*\*correct:\*\*|\n\*\*correct\*\*:|\ncorrect:|$)",
            judge_response, re.IGNORECASE | re.DOTALL,
        )
    if not reasoning_match:
        reasoning_match = re.search(
            r"reasoning:\s*(.*?)(?=\ncorrect:|$)", judge_response, re.IGNORECASE | re.DOTALL
        )
    if reasoning_match:
        result["reasoning"] = reasoning_match.group(1).strip()

    correct_match = re.search(r"\*\*correct:\*\*\s*(yes|no)", judge_response, re.IGNORECASE)
    if not correct_match:
        correct_match = re.search(r"\*\*correct\*\*:\s*(yes|no)", judge_response, re.IGNORECASE)
    if not correct_match:
        correct_match = re.search(r"correct:\s*(yes|no)", judge_response, re.IGNORECASE)
    if correct_match:
        result["correct"] = correct_match.group(1).lower() == "yes"

    confidence_match = re.search(
        r"\*\*confidence:\*\*\s*(\d+(?:\.\d+)?)\s*%?", judge_response, re.IGNORECASE
    )
    if not confidence_match:
        confidence_match = re.search(
            r"\*\*confidence\*\*:\s*(\d+(?:\.\d+)?)\s*%?", judge_response, re.IGNORECASE
        )
    if not confidence_match:
        confidence_match = re.search(
            r"confidence:\s*(\d+(?:\.\d+)?)\s*%?", judge_response, re.IGNORECASE
        )
    if confidence_match:
        result["confidence"] = float(confidence_match.group(1))
        if result["confidence"] > 100:
            result["confidence"] = 100

    if result["correct"] is None:
        result["parse_error"] = True
    return result


# Calibration: Hendrycks et al. (evaluate_run.py:152-197).
def calib_err(confidence, correct, p: str = "2", beta: int = 100):
    idxs = np.argsort(confidence)
    confidence = confidence[idxs]
    correct = correct[idxs]
    bins = [[i * beta, (i + 1) * beta] for i in range(len(confidence) // beta)]
    bins[-1] = [bins[-1][0], len(confidence)]

    cerr = 0
    total_examples = len(confidence)
    for i in range(len(bins) - 1):
        bin_confidence = confidence[bins[i][0] : bins[i][1]]
        bin_correct = correct[bins[i][0] : bins[i][1]]
        num_examples_in_bin = len(bin_confidence)
        if num_examples_in_bin > 0:
            difference = np.abs(np.nanmean(bin_confidence) - np.nanmean(bin_correct))
            if p == "2":
                cerr += num_examples_in_bin / total_examples * np.square(difference)
            elif p == "1":
                cerr += num_examples_in_bin / total_examples * difference
            elif p in ("infty", "infinity", "max"):
                cerr = np.maximum(cerr, difference)
            else:
                raise ValueError("p must be '1', '2', or 'infty'")
    if p == "2":
        cerr = np.sqrt(cerr)
    return cerr


def calculate_calibration_error(
    confidences: list[float], correctness: list[bool], beta: int = 100
) -> float:
    assert len(confidences) == len(correctness)
    assert len(confidences) > 0
    confidence = np.array(confidences) / 100.0
    correct = np.array(correctness, dtype=float)
    return float(calib_err(confidence, correct, p="2", beta=beta) * 100)


def extract_citations_from_response(response_text: str) -> list[str]:
    """Extract cited docids: [id], [id1, id2], and full-width brackets."""
    if not response_text:
        return []
    single_matches = re.findall(r"\[(\d+)\]", response_text)
    multi_matches = re.findall(r"\[([^\[\]]*?)\]", response_text)
    single_fullwidth_matches = re.findall(r"【(\d+)】", response_text)
    multi_fullwidth_matches = re.findall(r"【([^【】]*?)】", response_text)

    all_docids: set[str] = set()
    all_docids.update(single_matches)
    all_docids.update(single_fullwidth_matches)
    for match in multi_matches:
        if match in single_matches:
            continue
        all_docids.update(re.findall(r"\d+", match))
    for match in multi_fullwidth_matches:
        if match in single_fullwidth_matches:
            continue
        all_docids.update(re.findall(r"\d+", match))
    return list(all_docids)


def compute_citation_metrics(
    cited_docids: list[str], relevant_docids: list[str]
) -> dict[str, float]:
    metrics = {
        "num_citations": len(cited_docids),
        "num_relevant": len(relevant_docids),
        "precision": 0.0,
        "recall": 0.0,
    }
    if not cited_docids:
        return metrics
    cited_set, relevant_set = set(cited_docids), set(relevant_docids)
    relevant_cited = cited_set & relevant_set
    metrics["precision"] = len(relevant_cited) / len(cited_docids)
    if relevant_docids:
        metrics["recall"] = len(relevant_cited) / len(relevant_docids)
    return metrics


def _mean_pct(values: list[float]) -> Optional[float]:
    return round(float(np.mean(values)) * 100.0, 2) if values else None


def aggregate_summary(
    results: list[dict[str, Any]],
    *,
    llm_name: str,
    judge_model: str,
    agent_model: str,
    data_source: str,
    experiment: str,
    retriever_label: str = "change me when submitting",
    tags: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """Aggregate per-query eval dicts into a leaderboard-compatible summary
    (plus citation, per-domain, and provenance extensions)."""
    total = len(results)
    if total == 0:
        return {"LLM": llm_name, "Accuracy (%)": 0.0, "num_queries": 0}

    # Average tool calls per query.
    tool_counts: dict[str, float] = defaultdict(float)
    for r in results:
        for name, count in (r.get("tool_call_counts") or {}).items():
            tool_counts[name] += count
    avg_tool_stats = {name: count / total for name, count in tool_counts.items()}

    # Accuracy.
    correct_count = sum(1 for r in results if (r.get("judge_result") or {}).get("correct"))
    accuracy_percent = round(correct_count / total * 100.0, 2)

    # Retrieval recall (skip queries without evidence qrels → recall is None).
    recalls = [
        r["retrieval"]["recall"]
        for r in results
        if isinstance(r.get("retrieval"), dict) and r["retrieval"].get("recall") is not None
    ]
    recall_percent = _mean_pct(recalls)

    # Calibration (needs >= 100 graded confidences).
    confidences, correctness = [], []
    for r in results:
        jr = r.get("judge_result") or {}
        if not jr.get("parse_error", False) and jr.get("correct") is not None and jr.get("confidence") is not None:
            confidences.append(jr["confidence"])
            correctness.append(jr["correct"])
    if len(confidences) >= 100:
        calibration_percent = round(calculate_calibration_error(confidences, correctness), 2)
    else:
        calibration_percent = None

    # Citation summary.
    with_cites = [
        r for r in results
        if isinstance(r.get("citations"), dict) and r["citations"].get("cited_docids")
    ]
    n_cite = len(with_cites)
    citation = {
        "coverage_%": round(n_cite / total * 100.0, 2),
        "avg_citations": round(sum(len(r["citations"]["cited_docids"]) for r in with_cites) / n_cite, 2) if n_cite else 0.0,
        "precision_%": round(sum((r["citations"].get("metrics") or {}).get("precision", 0) for r in with_cites) / n_cite * 100.0, 2) if n_cite else 0.0,
        "recall_%": round(sum((r["citations"].get("metrics") or {}).get("recall", 0) for r in with_cites) / n_cite * 100.0, 2) if n_cite else 0.0,
    }

    parse_errors = sum(1 for r in results if (r.get("judge_result") or {}).get("parse_error"))
    completed = sum(1 for r in results if r.get("is_completed"))

    per_query = []
    for r in results:
        qid = str(r.get("query_id"))
        recall_val = (r.get("retrieval") or {}).get("recall")
        entry = {
            "query_id": qid,
            "correct": bool((r.get("judge_result") or {}).get("correct", False)),
            "recall": round(recall_val * 100.0, 2) if isinstance(recall_val, (int, float)) else None,
        }
        if tags:
            entry["domain"] = tags.get(qid, "untagged")
        per_query.append(entry)

    summary = {
        "LLM": llm_name,
        "Accuracy (%)": accuracy_percent,
        "Recall (%)": recall_percent,
        "avg_tool_stats": dict(avg_tool_stats),
        "Calibration Error (%)": calibration_percent,
        "Retriever": retriever_label,
        "Link": "change me when submitting",
        "Evaluation Date": date.today().isoformat(),
        "Citation": citation,
        "num_queries": total,
        "num_completed": completed,
        "num_parse_errors": parse_errors,
        "experiment": experiment,
        "data_source": data_source,
        "agent_model": agent_model,
        "judge_model": judge_model,
        "per_query_metrics": per_query,
    }

    if tags:
        summary["per_domain"] = _per_domain_breakdown(results, tags)
    return summary


def _per_domain_breakdown(
    results: list[dict[str, Any]], tags: dict[str, str]
) -> dict[str, dict[str, Any]]:
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        domain = tags.get(str(r.get("query_id")), "untagged")
        by_domain[domain].append(r)

    breakdown: dict[str, dict[str, Any]] = {}
    for domain, items in sorted(by_domain.items()):
        n = len(items)
        correct = sum(1 for r in items if (r.get("judge_result") or {}).get("correct"))
        recalls = [
            r["retrieval"]["recall"]
            for r in items
            if isinstance(r.get("retrieval"), dict) and r["retrieval"].get("recall") is not None
        ]
        breakdown[domain] = {
            "n": n,
            "accuracy_%": round(correct / n * 100.0, 2) if n else 0.0,
            "recall_%": _mean_pct(recalls),
        }
    return breakdown
