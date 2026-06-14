"""CLI: run a full evaluation experiment from a config file.

    python -m fireworks_eval.run_experiment --config configs/experiment.example.yaml
    python -m fireworks_eval.run_experiment --config configs/smoke.yaml --sample 5 --tagging
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from .config import ExperimentConfig
from .pipeline import run_pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_raw(path: str) -> dict[str, Any]:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        return json.loads(text)
    return yaml.safe_load(text) or {}


def _apply_overrides(raw: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.experiment_name:
        raw["experiment_name"] = args.experiment_name
    if args.sample is not None:
        raw.setdefault("dataset", {})["sample"] = args.sample
    if args.query_ids:
        raw.setdefault("dataset", {})["query_ids"] = [q.strip() for q in args.query_ids.split(",") if q.strip()]
    if args.tagging is not None:
        raw.setdefault("tagging", {})["enabled"] = args.tagging
    return raw


def _print_summary(summary: dict[str, Any], result) -> None:
    recall = summary.get("Recall (%)")
    avg_search = (summary.get("avg_tool_stats") or {}).get("search")
    print("\n" + "=" * 60)
    print(f"Experiment : {summary.get('experiment')}")
    print(f"Agent model: {summary.get('agent_model')}")
    print(f"Judge model: {summary.get('judge_model')}")
    print(f"Data source: {summary.get('data_source')}")
    print("-" * 60)
    print(f"Queries        : {summary.get('num_queries')} (completed {summary.get('num_completed')})")
    print(f"Accuracy (%)   : {summary.get('Accuracy (%)')}")
    print(f"Recall (%)     : {recall if recall is not None else 'N/A'}")
    print(f"Avg search     : {round(avg_search, 2) if avg_search is not None else 'N/A'}")
    print(f"Calibration (%): {summary.get('Calibration Error (%)')}")
    print(f"Parse errors   : {summary.get('num_parse_errors')}")
    if summary.get("per_domain"):
        print("-" * 60)
        print("Per-domain accuracy:")
        for domain, stats in summary["per_domain"].items():
            print(f"  {domain:20s} n={stats['n']:<4d} acc={stats['accuracy_%']}%")
    print("-" * 60)
    print(f"Runs   : {result.runs_dir}")
    print(f"Evals  : {result.eval_dir}")
    print(f"Report : {result.report_path}")
    print("=" * 60)


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    parser = argparse.ArgumentParser(description="Run a fireworks_eval experiment")
    parser.add_argument("--config", required=True, help="Path to experiment config (YAML/JSON)")
    parser.add_argument("--experiment-name", help="Override experiment_name")
    parser.add_argument("--sample", type=int, help="Override dataset.sample")
    parser.add_argument("--query-ids", help="Comma-separated query ids (override subset)")
    parser.add_argument("--tagging", dest="tagging", action="store_true", default=None,
                        help="Enable per-domain tagging")
    parser.add_argument("--no-tagging", dest="tagging", action="store_false",
                        help="Disable per-domain tagging")
    parser.add_argument("--force-agent", action="store_true", help="Re-run agent even if run files exist")
    parser.add_argument("--force-judge", action="store_true", help="Re-judge even if eval files exist")
    parser.add_argument("--rebuild-gt-cache", action="store_true", help="Rebuild slim ground-truth cache")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    raw = _apply_overrides(_load_raw(args.config), args)
    cfg = ExperimentConfig.from_dict(raw)
    result = run_pipeline(
        cfg,
        force_agent=args.force_agent,
        force_judge=args.force_judge,
        rebuild_gt_cache=args.rebuild_gt_cache,
    )
    _print_summary(result.summary, result)


if __name__ == "__main__":
    main()
