"""End-to-end smoke test: 3 queries through the full pipeline + validation.

Runs the real pipeline (managed retrieval endpoint -> Fireworks agent ->
Fireworks judge -> report) on a tiny sample, then asserts the outputs are
well-formed and sane. Critical checks gate PASS/FAIL; advisory checks warn.

    python -m fireworks_eval.smoke_test                  # ids 769,770,771
    python -m fireworks_eval.smoke_test --sample 3 --seed 7
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

from .agent_protocol import STATUS_COMPLETED, read_run_file
from .config import ExperimentConfig
from .pipeline import PipelineResult, run_pipeline

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SMOKE_IDS = ("769", "770", "771")
SEVERITY_CRITICAL = "critical"
SEVERITY_ADVISORY = "advisory"


@dataclass(frozen=True)
class Check:
    name: str
    severity: str
    passed: bool
    detail: str = ""


def build_smoke_config(
    *,
    experiment_name: str = "e2e_smoke_claude",
    query_ids: tuple[str, ...] = DEFAULT_SMOKE_IDS,
    sample: Optional[int] = None,
    seed: int = 42,
    port: int = 8123,
    max_tokens: int = 8000,
    max_iterations: int = 30,
    threads: int = 3,
    reasoning_effort: str = "medium",
    tagging: bool = False,
) -> ExperimentConfig:
    dataset: dict[str, Any] = {"sample": sample, "seed": seed} if sample else {"query_ids": list(query_ids)}
    raw = {
        "experiment_name": experiment_name,
        "description": "E2E smoke test (sample of queries through the full pipeline)",
        "dataset": dataset,
        "data_source": {"name": "qwen3-0.6b-snippet512", "k": 5, "snippet_max_tokens": 512, "port": port},
        "agent": {
            "max_tokens": max_tokens,
            "max_iterations": max_iterations,
            "num_threads": threads,
            "reasoning_effort": reasoning_effort,
        },
        "judge": {"num_threads": threads},
        "tagging": {"enabled": tagging},
    }
    return ExperimentConfig.from_dict(raw)


def _run_files_for(runs_dir: Path, qid: str) -> list[Path]:
    return sorted(runs_dir.glob(f"run_*_{qid}.json"))


def check_results(result: PipelineResult) -> list[Check]:
    checks: list[Check] = []
    runs_dir, summary = result.runs_dir, result.summary
    expected = result.query_ids

    # 1. one valid run file per selected query.
    for qid in expected:
        files = _run_files_for(runs_dir, qid)
        if not files:
            checks.append(Check(f"run_file[{qid}]", SEVERITY_CRITICAL, False, "no run file produced"))
            continue
        try:
            record = read_run_file(files[-1])
            record.validate()
            checks.append(Check(f"run_file[{qid}]", SEVERITY_CRITICAL, True, f"{files[-1].name} ({record.status})"))
        except Exception as exc:  # noqa: BLE001
            checks.append(Check(f"run_file[{qid}]", SEVERITY_CRITICAL, False, f"invalid: {exc}"))

    # 2. summary present + metric ranges.
    checks.append(Check("summary_file", SEVERITY_CRITICAL, result.summary_path.is_file(),
                        str(result.summary_path)))
    acc = summary.get("Accuracy (%)")
    checks.append(Check("accuracy_in_range", SEVERITY_CRITICAL,
                        isinstance(acc, (int, float)) and 0 <= acc <= 100, f"Accuracy={acc}"))
    rec = summary.get("Recall (%)")
    checks.append(Check("recall_in_range", SEVERITY_CRITICAL,
                        rec is None or (isinstance(rec, (int, float)) and 0 <= rec <= 100), f"Recall={rec}"))
    pqm = summary.get("per_query_metrics") or []
    checks.append(Check("per_query_count", SEVERITY_CRITICAL, len(pqm) == len(expected),
                        f"{len(pqm)} metrics for {len(expected)} queries"))

    # 3. report present.
    checks.append(Check("report_file", SEVERITY_CRITICAL, result.report_path.is_file(),
                        str(result.report_path)))

    # 4. completed runs: judge produced a clean verdict.
    eval_files = list(result.eval_dir.glob("*_eval.json"))
    parse_errors = 0
    completed = 0
    for ef in eval_files:
        try:
            obj = json.loads(ef.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if obj.get("is_completed"):
            completed += 1
            if (obj.get("judge_result") or {}).get("parse_error"):
                parse_errors += 1
    checks.append(Check("judge_parse_clean", SEVERITY_CRITICAL, parse_errors == 0,
                        f"{parse_errors} parse errors among {completed} completed"))

    # Advisory sanity checks on completed runs.
    checks.append(Check("at_least_one_completed", SEVERITY_ADVISORY, completed >= 1,
                        f"{completed} completed"))
    for qid in expected:
        files = _run_files_for(runs_dir, qid)
        if not files:
            continue
        try:
            record = read_run_file(files[-1])
        except Exception:  # noqa: BLE001
            continue
        if record.status != STATUS_COMPLETED:
            checks.append(Check(f"completed[{qid}]", SEVERITY_ADVISORY, False, f"status={record.status}"))
            continue
        ans_ok = bool(record.final_answer.strip())
        search_ok = record.tool_call_counts.get("search", 0) >= 1
        docs_ok = len(record.retrieved_docids) >= 1
        checks.append(Check(f"sanity[{qid}]", SEVERITY_ADVISORY, ans_ok and search_ok and docs_ok,
                            f"answer={ans_ok}, searches={record.tool_call_counts.get('search', 0)}, docs={len(record.retrieved_docids)}"))

    return checks


def format_checks(checks: list[Check]) -> str:
    lines = ["", "Smoke test checks:", "-" * 60]
    for c in checks:
        mark = "PASS" if c.passed else ("FAIL" if c.severity == SEVERITY_CRITICAL else "WARN")
        lines.append(f"  [{mark}] ({c.severity}) {c.name}: {c.detail}")
    lines.append("-" * 60)
    return "\n".join(lines)


def smoke_passed(checks: list[Check]) -> bool:
    return all(c.passed for c in checks if c.severity == SEVERITY_CRITICAL)


def run_smoke(cfg: Optional[ExperimentConfig] = None, **kwargs: Any) -> tuple[PipelineResult, list[Check]]:
    cfg = cfg or build_smoke_config(**kwargs)
    result = run_pipeline(cfg)
    checks = check_results(result)
    return result, checks


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    parser = argparse.ArgumentParser(description="E2E smoke test for fireworks_eval")
    parser.add_argument("--ids", default=",".join(DEFAULT_SMOKE_IDS),
                        help="Comma-separated query ids (default 769,770,771)")
    parser.add_argument("--sample", type=int, help="Random sample N instead of fixed ids")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment-name", default="e2e_smoke_claude")
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--max-tokens", type=int, default=8000)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--reasoning-effort", default="medium", help="low|medium|high")
    parser.add_argument("--tagging", action="store_true", help="Also run per-domain tagging")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    ids = tuple(q.strip() for q in args.ids.split(",") if q.strip())
    cfg = build_smoke_config(
        experiment_name=args.experiment_name, query_ids=ids, sample=args.sample,
        seed=args.seed, port=args.port, max_tokens=args.max_tokens,
        threads=args.threads, reasoning_effort=args.reasoning_effort, tagging=args.tagging,
    )
    result, checks = run_smoke(cfg)

    print(format_checks(checks))
    passed = smoke_passed(checks)
    print(f"\nSMOKE TEST: {'PASS' if passed else 'FAIL'}")
    print(f"Report: {result.report_path}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
