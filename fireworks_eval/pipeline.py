"""End-to-end orchestration: one config in, scored report out (100% automated).

Stages: select queries -> (optional) tag domains -> ensure local retrieval
endpoint -> run agent -> judge -> report. Every stage is resumable and writes
to disk, so a failed run can be re-entered cheaply.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .agent import build_agent, run_agent_over_queries
from .config import ExperimentConfig
from .dataset import load_ground_truth, load_qrels, load_tags, select_query_ids
from .errors import ConfigError
from .judge import FireworksJudge, evaluate_runs
from .report import write_report
from .retrieval_client import managed_retrieval
from .tagging import tag_queries

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineResult:
    summary: dict[str, Any]
    summary_path: Path
    runs_dir: Path
    eval_dir: Path
    report_path: Path
    query_ids: list[str]
    tags: Optional[dict[str, str]] = field(default=None)


def _resolve_tags(
    cfg: ExperimentConfig, queries: list[tuple[str, str]]
) -> Optional[dict[str, str]]:
    if cfg.tagging.enabled:
        logger.info("Per-domain tagging enabled.")
        return tag_queries(cfg.tagging, queries)
    if cfg.dataset.tags_file:
        all_tags = load_tags(cfg.dataset.tags_file)
        return {qid: all_tags.get(qid, "untagged") for qid, _ in queries}
    return None


def _produce_runs(
    cfg: ExperimentConfig, queries: list[tuple[str, str]], force_agent: bool
) -> Path:
    """Return the directory containing run files (built or external)."""
    if cfg.agent.type == "external":
        runs_dir = Path(cfg.agent.external_runs_dir or "")
        if not runs_dir.is_dir():
            raise ConfigError(f"external_runs_dir does not exist: {runs_dir}")
        present = {p.name.rsplit("_", 1)[-1].removesuffix(".json") for p in runs_dir.glob("run_*.json")}
        missing = [qid for qid, _ in queries if qid not in present]
        if missing:
            logger.warning("external runs missing %d of %d queries: %s",
                           len(missing), len(queries), missing[:10])
        logger.info("Using external run files in %s", runs_dir)
        return runs_dir

    runs_dir = Path(cfg.output.runs_dir)
    with managed_retrieval(cfg.data_source) as retrieval:
        agent = build_agent(
            cfg.agent, data_source_name=cfg.data_source.name, experiment=cfg.experiment_name
        )
        run_agent_over_queries(
            agent, queries, retrieval, runs_dir,
            num_threads=cfg.agent.num_threads, skip_existing=not force_agent,
        )
    return runs_dir


def run_pipeline(
    cfg: ExperimentConfig,
    *,
    force_agent: bool = False,
    force_judge: bool = False,
    rebuild_gt_cache: bool = False,
) -> PipelineResult:
    logger.info("=== Experiment: %s ===", cfg.experiment_name)

    ground_truth = load_ground_truth(
        cfg.dataset.ground_truth, cfg.dataset.slim_cache, rebuild=rebuild_gt_cache
    )
    query_ids = select_query_ids(
        ground_truth, cfg.dataset.query_ids, cfg.dataset.sample, cfg.dataset.seed
    )
    queries = [(qid, ground_truth[qid]["question"]) for qid in query_ids]
    logger.info("Selected %d queries: %s", len(queries), query_ids if len(query_ids) <= 20 else f"{query_ids[:20]}...")

    tags = _resolve_tags(cfg, queries)

    runs_dir = _produce_runs(cfg, queries, force_agent)

    qrel_evidence = load_qrels(cfg.dataset.qrel_evidence)
    judge = FireworksJudge(cfg.judge)
    summary, summary_path = evaluate_runs(
        judge, runs_dir, cfg.output.evals_dir, ground_truth, qrel_evidence,
        agent_model=cfg.agent.model, data_source=cfg.data_source.name,
        experiment=cfg.experiment_name, tags=tags, force=force_judge,
    )

    report_path = write_report(summary, cfg.output.report_dir)

    return PipelineResult(
        summary=summary,
        summary_path=summary_path,
        runs_dir=Path(runs_dir),
        eval_dir=Path(cfg.output.evals_dir),
        report_path=report_path,
        query_ids=query_ids,
        tags=tags,
    )
