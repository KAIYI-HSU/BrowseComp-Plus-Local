"""Fireworks LLM-as-judge over a runs directory.

Produces per-query ``*_eval.json``, an ``evaluation_summary.json``, and a
``detailed_judge_results.csv`` matching the schema of the official
``scripts_evaluation/evaluate_run.py`` — but judging through an
OpenAI-compatible endpoint (Fireworks) instead of local vLLM, and threaded.
"""

from __future__ import annotations

import csv
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

from .agent_protocol import STATUS_COMPLETED, read_run_file
from .config import JudgeConfig
from .errors import JudgeError
from .fireworks_client import chat_completion, make_client
from .scoring import (
    aggregate_summary,
    compute_citation_metrics,
    create_judge_prompt,
    extract_citations_from_response,
    parse_judge_response,
)

logger = logging.getLogger(__name__)

_CSV_FIELDS = [
    "query_id", "predicted_answer", "correct_answer", "judge_correct", "confidence",
    "is_completed", "parse_error", "json_path", "num_citations",
    "precision_positives", "recall_positives", "domain",
]


class FireworksJudge:
    """Calls the grader model through an OpenAI-compatible endpoint."""

    def __init__(self, cfg: JudgeConfig):
        self.cfg = cfg
        self.client = make_client(cfg.base_url, cfg.api_key_env, cfg.request_timeout_s)

    def judge(self, prompt: str) -> str:
        kwargs: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.cfg.max_output_tokens,
            "temperature": self.cfg.temperature,
        }
        if self.cfg.reasoning_effort:
            kwargs["extra_body"] = {"reasoning_effort": self.cfg.reasoning_effort}
        response = chat_completion(self.client, **kwargs)
        return response.choices[0].message.content or ""


def _retrieval_recall(retrieved: set[str], positives: list[str]) -> Optional[float]:
    if not positives:
        return None
    return len(retrieved & set(positives)) / float(len(positives))


def _incomplete_eval(
    run_path: Path, query_id: str, response: str, correct_answer: str,
    is_completed: bool, tool_call_counts: dict, retrieval: dict, judge_model: str,
    max_output_tokens: int,
) -> dict[str, Any]:
    return {
        "json_path": str(run_path),
        "query_id": query_id,
        "response": response,
        "correct_answer": correct_answer,
        "is_completed": is_completed,
        "judge_prompt": None,
        "judge_response": None,
        "judge_result": {"parse_error": True, "error": "Response incomplete or cannot be parsed"},
        "tool_call_counts": tool_call_counts,
        "citations": None,
        "retrieval": retrieval,
        "model_info": {"judge_model": judge_model, "max_output_tokens": max_output_tokens},
    }


def evaluate_runs(
    judge: FireworksJudge,
    runs_dir: str | Path,
    eval_dir: str | Path,
    ground_truth: dict[str, dict[str, str]],
    qrel_evidence: dict[str, list[str]],
    *,
    agent_model: str = "",
    data_source: str = "",
    experiment: str = "",
    tags: Optional[dict[str, str]] = None,
    force: bool = False,
) -> tuple[dict[str, Any], Path]:
    """Judge every run file and write per-query evals + summary + CSV."""
    runs_path = Path(runs_dir)
    out_dir = Path(eval_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_files = sorted(runs_path.glob("run_*.json"))
    if not run_files:
        raise JudgeError(f"No run files found in {runs_path}")

    llm_name = _detect_model(run_files[0]) or agent_model or "unknown"

    results: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []

    for run_path in run_files:
        eval_path = out_dir / f"{run_path.stem}_eval.json"
        if eval_path.exists() and not force:
            try:
                results.append(json.loads(eval_path.read_text(encoding="utf-8")))
                continue
            except (OSError, json.JSONDecodeError):
                pass  # re-evaluate

        record = read_run_file(run_path)
        qid = str(record.query_id)
        if qid not in ground_truth:
            logger.warning("No ground truth for query_id %s (%s); skipping", qid, run_path.name)
            continue

        correct_answer = ground_truth[qid]["answer"]
        question = ground_truth[qid]["question"]
        is_completed = record.status == STATUS_COMPLETED
        retrieved = set(record.retrieved_docids)
        positives = qrel_evidence.get(qid, [])
        retrieval = {
            "retrieved_docids": sorted(retrieved),
            "recall": _retrieval_recall(retrieved, positives),
        }
        response = record.final_answer

        if not response or not is_completed:
            eval_obj = _incomplete_eval(
                run_path, qid, response, correct_answer, is_completed,
                dict(record.tool_call_counts), retrieval, judge.cfg.model, judge.cfg.max_output_tokens,
            )
            if tags:
                eval_obj["domain"] = tags.get(qid, "untagged")
            _write_json(eval_path, eval_obj)
            results.append(eval_obj)
            continue

        pending.append({
            "run_path": run_path,
            "eval_path": eval_path,
            "query_id": qid,
            "question": question,
            "correct_answer": correct_answer,
            "response": response,
            "tool_call_counts": dict(record.tool_call_counts),
            "retrieval": retrieval,
            "judge_prompt": create_judge_prompt(question, response, correct_answer),
        })

    _judge_pending(judge, pending, qrel_evidence, tags, results)

    summary = aggregate_summary(
        results,
        llm_name=llm_name,
        judge_model=judge.cfg.model,
        agent_model=agent_model or llm_name,
        data_source=data_source,
        experiment=experiment,
        tags=tags,
    )
    summary_path = out_dir / "evaluation_summary.json"
    _write_json(summary_path, summary)
    _write_csv(results, out_dir / "detailed_judge_results.csv")
    logger.info(
        "Judged %d queries | Accuracy %.2f%% | Recall %s",
        summary.get("num_queries", 0),
        summary.get("Accuracy (%)", 0.0),
        f"{summary['Recall (%)']:.2f}%" if summary.get("Recall (%)") is not None else "N/A",
    )
    return summary, summary_path


def _judge_pending(judge, pending, qrel_evidence, tags, results) -> None:
    if not pending:
        return
    workers = max(1, judge.cfg.num_threads)
    logger.info("Judging %d completed responses (%d threads)...", len(pending), workers)

    def _one(item: dict[str, Any]) -> dict[str, Any]:
        judge_text = judge.judge(item["judge_prompt"])
        judge_result = parse_judge_response(judge_text)
        cited = extract_citations_from_response(item["response"])
        citation_metrics = compute_citation_metrics(cited, qrel_evidence.get(item["query_id"], []))
        eval_obj = {
            "json_path": str(item["run_path"]),
            "query_id": item["query_id"],
            "question": item["question"],
            "response": item["response"],
            "correct_answer": item["correct_answer"],
            "is_completed": True,
            "judge_prompt": item["judge_prompt"],
            "judge_response": judge_text,
            "judge_result": judge_result,
            "tool_call_counts": item["tool_call_counts"],
            "citations": {"cited_docids": cited, "metrics": citation_metrics},
            "retrieval": item["retrieval"],
            "model_info": {"judge_model": judge.cfg.model, "max_output_tokens": judge.cfg.max_output_tokens},
        }
        if tags:
            eval_obj["domain"] = tags.get(item["query_id"], "untagged")
        _write_json(item["eval_path"], eval_obj)
        return eval_obj

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_one, item): item["query_id"] for item in pending}
        for fut in as_completed(futures):
            qid = futures[fut]
            try:
                results.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                logger.error("Judge failed for query %s: %s", qid, exc)


def _detect_model(run_path: Path) -> Optional[str]:
    try:
        data = json.loads(run_path.read_text(encoding="utf-8"))
        model = (data.get("metadata") or {}).get("model")
        return str(model) if model else None
    except (OSError, json.JSONDecodeError):
        return None


def _write_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _write_csv(results: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for r in results:
            jr = r.get("judge_result") or {}
            citations = r.get("citations") if isinstance(r.get("citations"), dict) else {}
            metrics = (citations or {}).get("metrics") or {}
            predicted = jr.get("extracted_final_answer") or ""
            if not predicted:
                full = r.get("response", "") or ""
                predicted = full[:200] + "..." if len(full) > 200 else full
            writer.writerow({
                "query_id": r.get("query_id", ""),
                "predicted_answer": predicted,
                "correct_answer": r.get("correct_answer", ""),
                "judge_correct": jr.get("correct", ""),
                "confidence": jr.get("confidence", ""),
                "is_completed": r.get("is_completed", ""),
                "parse_error": jr.get("parse_error", False),
                "json_path": r.get("json_path", ""),
                "num_citations": len((citations or {}).get("cited_docids", [])),
                "precision_positives": metrics.get("precision", 0),
                "recall_positives": metrics.get("recall", 0),
                "domain": r.get("domain", ""),
            })
