"""Render a human-readable Markdown report from an evaluation summary."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _fmt(value: Optional[float], suffix: str = "") -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.2f}{suffix}"
    return f"{value}{suffix}"


def render_report(summary: dict[str, Any]) -> str:
    avg_tools = summary.get("avg_tool_stats", {}) or {}
    avg_search = avg_tools.get("search")
    citation = summary.get("Citation", {}) or {}

    lines: list[str] = []
    lines.append(f"# Evaluation Report — {summary.get('experiment', 'experiment')}")
    lines.append("")
    lines.append(f"- **Agent model**: `{summary.get('agent_model', '?')}`")
    lines.append(f"- **Judge model**: `{summary.get('judge_model', '?')}`")
    lines.append(f"- **Data source**: `{summary.get('data_source', '?')}`")
    lines.append(f"- **Date**: {summary.get('Evaluation Date', '?')}")
    lines.append(
        f"- **Queries**: {summary.get('num_queries', 0)} "
        f"(completed {summary.get('num_completed', 0)}, parse errors {summary.get('num_parse_errors', 0)})"
    )
    lines.append("")
    lines.append("## Overall metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Accuracy | {_fmt(summary.get('Accuracy (%)'), '%')} |")
    lines.append(f"| Evidence Recall | {_fmt(summary.get('Recall (%)'), '%')} |")
    lines.append(f"| Avg. search calls | {_fmt(avg_search)} |")
    lines.append(f"| Calibration Error | {_fmt(summary.get('Calibration Error (%)'), '%')} |")
    lines.append(f"| Citation coverage | {_fmt(citation.get('coverage_%'), '%')} |")
    lines.append(f"| Citation precision | {_fmt(citation.get('precision_%'), '%')} |")
    lines.append(f"| Citation recall | {_fmt(citation.get('recall_%'), '%')} |")
    lines.append("")

    per_domain = summary.get("per_domain")
    if per_domain:
        lines.append("## Per-domain")
        lines.append("")
        lines.append("| Domain | n | Accuracy | Recall |")
        lines.append("|---|---|---|---|")
        for domain, stats in per_domain.items():
            lines.append(
                f"| {domain} | {stats.get('n', 0)} | "
                f"{_fmt(stats.get('accuracy_%'), '%')} | {_fmt(stats.get('recall_%'), '%')} |"
            )
        lines.append("")

    per_query = summary.get("per_query_metrics") or []
    if per_query:
        has_domain = any("domain" in q for q in per_query)
        lines.append("## Per-query")
        lines.append("")
        header = "| query_id | correct | recall |" + (" domain |" if has_domain else "")
        sep = "|---|---|---|" + ("---|" if has_domain else "")
        lines.append(header)
        lines.append(sep)
        for q in per_query:
            row = f"| {q.get('query_id')} | {'✓' if q.get('correct') else '✗'} | {_fmt(q.get('recall'), '%')} |"
            if has_domain:
                row += f" {q.get('domain', '')} |"
            lines.append(row)
        lines.append("")

    return "\n".join(lines)


def write_report(summary: dict[str, Any], report_dir: str | Path) -> Path:
    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_md = render_report(summary)
    report_path = out_dir / "report.md"
    report_path.write_text(report_md, encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Report written to %s", report_path)
    return report_path
