from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.append(str(Path(__file__).parent.parent))

from scripts_evaluation.evaluate_with_fireworks import (
    DEFAULT_FIREWORKS_JUDGE_MAX_OUTPUT_TOKENS,
    DEFAULT_FIREWORKS_JUDGE_MODEL,
    FIREWORKS_BASE_URL,
    FireworksJudge,
    build_client,
)
from scripts_evaluation.evaluation_core import evaluate_directory


def load_manifest(path: Path | str) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    scenarios = manifest.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("manifest must contain a non-empty scenarios list")

    for i, scenario in enumerate(scenarios):
        if not isinstance(scenario, dict):
            raise ValueError(f"scenarios[{i}] must be an object")
        if not scenario.get("name"):
            raise ValueError(f"scenarios[{i}] is missing name")
        if not scenario.get("run_dir"):
            raise ValueError(f"scenarios[{i}] is missing run_dir")

    return manifest


def _merge_judge_config(manifest: dict, scenario: dict) -> dict:
    config = {
        "api_key_env": "FIREWORKS_API_KEY",
        "base_url": FIREWORKS_BASE_URL,
        "model": DEFAULT_FIREWORKS_JUDGE_MODEL,
        "max_output_tokens": DEFAULT_FIREWORKS_JUDGE_MAX_OUTPUT_TOKENS,
        "temperature": 0.0,
        "reasoning_effort": None,
    }
    config.update(manifest.get("judge", {}) or {})
    config.update(scenario.get("judge", {}) or {})
    return config


def _run_command(command: str | list[str], *, cwd: Optional[str] = None) -> None:
    if isinstance(command, str):
        subprocess.run(command, shell=True, cwd=cwd, check=True)
    else:
        subprocess.run(command, cwd=cwd, check=True)


def _metric(summary: dict, key: str) -> Optional[float]:
    value = summary.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _search_calls(summary: dict) -> Optional[float]:
    tool_stats = summary.get("avg_tool_stats", {})
    if not isinstance(tool_stats, dict):
        return None
    value = tool_stats.get("search")
    return float(value) if isinstance(value, (int, float)) else None


def _delta(value: Optional[float], baseline: Optional[float]) -> Optional[float]:
    if value is None or baseline is None:
        return None
    return round(value - baseline, 4)


def build_comparison(scenario_reports: list[dict]) -> dict:
    if not scenario_reports:
        raise ValueError("scenario_reports must be non-empty")

    baseline = scenario_reports[0]
    baseline_summary = baseline["summary"]
    baseline_accuracy = _metric(baseline_summary, "Accuracy (%)")
    baseline_recall = _metric(baseline_summary, "Recall (%)")
    baseline_calibration = _metric(baseline_summary, "Calibration Error (%)")
    baseline_search_calls = _search_calls(baseline_summary)

    scenarios = []
    for report in scenario_reports:
        summary = report["summary"]
        accuracy = _metric(summary, "Accuracy (%)")
        recall = _metric(summary, "Recall (%)")
        calibration = _metric(summary, "Calibration Error (%)")
        search_calls = _search_calls(summary)
        scenarios.append(
            {
                "name": report["name"],
                "run_dir": report["run_dir"],
                "summary_path": report["summary_path"],
                "accuracy": accuracy,
                "recall": recall,
                "avg_search_calls": search_calls,
                "calibration_error": calibration,
                "delta_accuracy": _delta(accuracy, baseline_accuracy),
                "delta_recall": _delta(recall, baseline_recall),
                "delta_search_calls": _delta(search_calls, baseline_search_calls),
                "delta_calibration_error": _delta(calibration, baseline_calibration),
            }
        )

    return {
        "baseline": baseline["name"],
        "scenarios": scenarios,
    }


def run_pipeline(
    manifest_path: Path | str,
    *,
    skip_agent_runs: bool = False,
    force_eval: bool = False,
    only: Optional[list[str]] = None,
) -> dict:
    manifest = load_manifest(manifest_path)
    selected = set(only or [])

    scenario_reports: list[dict[str, Any]] = []
    for scenario in manifest["scenarios"]:
        if selected and scenario["name"] not in selected:
            continue

        command = scenario.get("agent_command")
        if command and not skip_agent_runs:
            _run_command(command, cwd=scenario.get("cwd"))

        judge_config = _merge_judge_config(manifest, scenario)
        client = build_client(
            api_key_env=judge_config["api_key_env"],
            base_url=judge_config["base_url"],
        )
        judge = FireworksJudge(
            client=client,
            model=judge_config["model"],
            max_output_tokens=int(judge_config["max_output_tokens"]),
            temperature=float(judge_config.get("temperature", 0.0)),
            reasoning_effort=judge_config.get("reasoning_effort"),
        )

        report = evaluate_directory(
            input_dir=Path(scenario["run_dir"]),
            ground_truth_path=Path(
                scenario.get("ground_truth", manifest.get("ground_truth", "data/browsecomp_plus_decrypted.jsonl"))
            ),
            eval_dir=Path(scenario.get("eval_dir", manifest.get("eval_dir", "evals"))),
            qrel_evidence_path=Path(
                scenario.get("qrel_evidence", manifest.get("qrel_evidence", "topics-qrels/qrel_evidence.txt"))
            ),
            judge=judge,
            judge_model=judge_config["model"],
            max_output_tokens=int(judge_config["max_output_tokens"]),
            force=force_eval,
        )
        scenario_reports.append(
            {
                "name": scenario["name"],
                "run_dir": scenario["run_dir"],
                "summary_path": str(report.summary_path),
                "summary": report.summary,
            }
        )

    comparison = build_comparison(scenario_reports)
    comparison_path = Path(
        manifest.get(
            "comparison_path",
            str(Path(manifest.get("eval_dir", "evals")) / "comparison_summary.json"),
        )
    )
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    with comparison_path.open("w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    comparison["comparison_path"] = str(comparison_path)
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run and compare BrowseComp-Plus evaluation scenarios.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", required=True, help="Pipeline manifest JSON")
    parser.add_argument(
        "--skip-agent-runs",
        action="store_true",
        help="Evaluate existing run directories without executing agent commands",
    )
    parser.add_argument(
        "--force-eval",
        action="store_true",
        help="Re-evaluate even if per-query eval files already exist",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        help="Run only the named scenario. Repeat for multiple scenarios.",
    )
    args = parser.parse_args()

    comparison = run_pipeline(
        args.manifest,
        skip_agent_runs=args.skip_agent_runs,
        force_eval=args.force_eval,
        only=args.only,
    )
    print(f"Comparison saved to {comparison['comparison_path']}")
    for scenario in comparison["scenarios"]:
        print(
            "{name}: accuracy={accuracy} delta={delta_accuracy} "
            "recall={recall} search_calls={avg_search_calls}".format(**scenario)
        )


if __name__ == "__main__":
    main()
