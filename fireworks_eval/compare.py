"""Compare two experiment summaries to detect improvement or side-effects.

Reports metric deltas, per-query correctness flips (improvements vs
regressions), and per-domain deltas, plus a one-word verdict. This is the
check developers run after each optimization.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from .errors import ConfigError

logger = logging.getLogger(__name__)


def load_summary(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if p.is_dir():
        p = p / "evaluation_summary.json"
    if not p.is_file():
        raise ConfigError(f"Summary not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _delta(candidate: Optional[float], baseline: Optional[float]) -> Optional[float]:
    if candidate is None or baseline is None:
        return None
    return round(candidate - baseline, 2)


def _correctness_map(summary: dict[str, Any]) -> dict[str, bool]:
    return {
        str(q.get("query_id")): bool(q.get("correct"))
        for q in summary.get("per_query_metrics", [])
    }


def compare_summaries(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    base_search = (baseline.get("avg_tool_stats") or {}).get("search")
    cand_search = (candidate.get("avg_tool_stats") or {}).get("search")

    deltas = {
        "accuracy_%": _delta(candidate.get("Accuracy (%)"), baseline.get("Accuracy (%)")),
        "recall_%": _delta(candidate.get("Recall (%)"), baseline.get("Recall (%)")),
        "avg_search_calls": _delta(cand_search, base_search),
        "calibration_%": _delta(candidate.get("Calibration Error (%)"), baseline.get("Calibration Error (%)")),
    }

    base_correct = _correctness_map(baseline)
    cand_correct = _correctness_map(candidate)
    shared = sorted(set(base_correct) & set(cand_correct), key=lambda q: (len(q), q))

    improvements, regressions, unchanged = [], [], 0
    for qid in shared:
        was, now = base_correct[qid], cand_correct[qid]
        if was == now:
            unchanged += 1
        elif now and not was:
            improvements.append({"query_id": qid})
        else:
            regressions.append({"query_id": qid})

    per_domain_deltas = _per_domain_deltas(baseline.get("per_domain"), candidate.get("per_domain"))

    verdict = _verdict(deltas["accuracy_%"], len(regressions), len(improvements))

    return {
        "baseline": baseline.get("experiment", "baseline"),
        "candidate": candidate.get("experiment", "candidate"),
        "baseline_summary": {k: baseline.get(k) for k in ("Accuracy (%)", "Recall (%)", "Calibration Error (%)")},
        "candidate_summary": {k: candidate.get(k) for k in ("Accuracy (%)", "Recall (%)", "Calibration Error (%)")},
        "deltas": deltas,
        "improvements": improvements,
        "regressions": regressions,
        "unchanged": unchanged,
        "shared_queries": len(shared),
        "per_domain_deltas": per_domain_deltas,
        "verdict": verdict,
    }


def _per_domain_deltas(base: Optional[dict], cand: Optional[dict]) -> dict[str, Any]:
    if not base or not cand:
        return {}
    out = {}
    for domain in sorted(set(base) & set(cand)):
        out[domain] = {
            "accuracy_%": _delta(cand[domain].get("accuracy_%"), base[domain].get("accuracy_%")),
            "recall_%": _delta(cand[domain].get("recall_%"), base[domain].get("recall_%")),
            "n": cand[domain].get("n"),
        }
    return out


def _verdict(acc_delta: Optional[float], n_regressions: int, n_improvements: int) -> str:
    if acc_delta is None:
        return "unknown"
    if acc_delta > 0 and n_regressions == 0:
        return "improved"
    if acc_delta > 0 and n_regressions > 0:
        return "mixed (net gain, with regressions)"
    if acc_delta < 0:
        return "regressed"
    if n_improvements or n_regressions:
        return "mixed (no net accuracy change)"
    return "no-change"


def render_comparison_md(cmp: dict[str, Any]) -> str:
    d = cmp["deltas"]
    lines = [
        f"# Comparison: `{cmp['candidate']}` vs `{cmp['baseline']}`",
        "",
        f"**Verdict: {cmp['verdict']}**",
        "",
        "| Metric | Baseline | Candidate | Δ |",
        "|---|---|---|---|",
        _metric_row("Accuracy (%)", cmp["baseline_summary"].get("Accuracy (%)"), cmp["candidate_summary"].get("Accuracy (%)"), d["accuracy_%"]),
        _metric_row("Recall (%)", cmp["baseline_summary"].get("Recall (%)"), cmp["candidate_summary"].get("Recall (%)"), d["recall_%"]),
        _metric_row("Calibration (%)", cmp["baseline_summary"].get("Calibration Error (%)"), cmp["candidate_summary"].get("Calibration Error (%)"), d["calibration_%"]),
        f"| Avg search calls | — | — | {_signed(d['avg_search_calls'])} |",
        "",
        f"- Shared queries: {cmp['shared_queries']}",
        f"- Improvements (✗→✓): **{len(cmp['improvements'])}** {[q['query_id'] for q in cmp['improvements']]}",
        f"- Regressions (✓→✗): **{len(cmp['regressions'])}** {[q['query_id'] for q in cmp['regressions']]}",
        f"- Unchanged: {cmp['unchanged']}",
        "",
    ]
    if cmp.get("per_domain_deltas"):
        lines.append("## Per-domain Δ")
        lines.append("")
        lines.append("| Domain | n | ΔAccuracy | ΔRecall |")
        lines.append("|---|---|---|---|")
        for domain, dd in cmp["per_domain_deltas"].items():
            lines.append(f"| {domain} | {dd.get('n')} | {_signed(dd.get('accuracy_%'))} | {_signed(dd.get('recall_%'))} |")
        lines.append("")
    return "\n".join(lines)


def _metric_row(name: str, base: Optional[float], cand: Optional[float], delta: Optional[float]) -> str:
    b = "N/A" if base is None else f"{base:.2f}"
    c = "N/A" if cand is None else f"{cand:.2f}"
    return f"| {name} | {b} | {c} | {_signed(delta)} |"


def _signed(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"+{value:.2f}" if value > 0 else f"{value:.2f}"


def write_comparison(
    baseline_path: str | Path, candidate_path: str | Path, out_dir: str | Path
) -> tuple[dict[str, Any], Path]:
    baseline = load_summary(baseline_path)
    candidate = load_summary(candidate_path)
    cmp = compare_summaries(baseline, candidate)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(json.dumps(cmp, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path = out / "comparison.md"
    md_path.write_text(render_comparison_md(cmp), encoding="utf-8")
    logger.info("Comparison written to %s (verdict: %s)", md_path, cmp["verdict"])
    return cmp, md_path


def main() -> None:
    """CLI: compare two experiment summaries.

        python -m fireworks_eval.compare --baseline evals/exp_a --candidate evals/exp_b
    """
    import argparse

    parser = argparse.ArgumentParser(description="Compare two fireworks_eval experiments")
    parser.add_argument("--baseline", required=True, help="Baseline eval dir or summary json")
    parser.add_argument("--candidate", required=True, help="Candidate eval dir or summary json")
    parser.add_argument("--out", default="reports/comparison", help="Output directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    cmp, md_path = write_comparison(args.baseline, args.candidate, args.out)
    print(render_comparison_md(cmp))
    print(f"\nWritten to {md_path}")


if __name__ == "__main__":
    main()
