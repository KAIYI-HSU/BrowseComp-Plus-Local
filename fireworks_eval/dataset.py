"""Dataset access: ground truth, subset selection, qrels, and domain tags.

The decrypted dataset (`data/browsecomp_plus_decrypted.jsonl`) is ~2 GB because
each record inlines its gold/evidence/negative documents. The judge only needs
`query_id`, `query`, and `answer`, so we build a slim cache once and reuse it.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Optional

from .errors import ConfigError

logger = logging.getLogger(__name__)

_SLIM_FIELDS = ("query_id", "query", "answer")


def build_slim_ground_truth(full_path: str | Path, slim_path: str | Path) -> int:
    """Stream the large dataset and write a slim {query_id, query, answer} JSONL.

    Returns the number of records written.
    """
    full = Path(full_path)
    slim = Path(slim_path)
    if not full.is_file():
        raise ConfigError(f"Ground-truth dataset not found: {full}")
    slim.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    logger.info("Building slim ground-truth cache from %s (one-time)...", full)
    with full.open("r", encoding="utf-8") as src, slim.open("w", encoding="utf-8") as dst:
        for line_no, line in enumerate(src, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed line %d: %s", line_no, exc)
                continue
            slim_obj = {
                "query_id": str(obj["query_id"]),
                "query": obj["query"],
                "answer": obj["answer"],
            }
            dst.write(json.dumps(slim_obj, ensure_ascii=False) + "\n")
            written += 1
            if written % 100 == 0:
                logger.info("  ...%d records", written)
    logger.info("Slim cache written: %s (%d records)", slim, written)
    return written


def _load_slim(slim_path: Path) -> dict[str, dict[str, str]]:
    gt: dict[str, dict[str, str]] = {}
    with slim_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            gt[str(obj["query_id"])] = {"question": obj["query"], "answer": obj["answer"]}
    return gt


def _load_full(full_path: Path) -> dict[str, dict[str, str]]:
    gt: dict[str, dict[str, str]] = {}
    with full_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            gt[str(obj["query_id"])] = {"question": obj["query"], "answer": obj["answer"]}
    return gt


def load_ground_truth(
    full_path: str | Path,
    slim_path: Optional[str | Path] = None,
    rebuild: bool = False,
) -> dict[str, dict[str, str]]:
    """Load {query_id: {question, answer}}, using/creating the slim cache."""
    slim = Path(slim_path) if slim_path else None
    if slim is not None and slim.is_file() and not rebuild:
        logger.info("Loading ground truth from slim cache %s", slim)
        return _load_slim(slim)

    full = Path(full_path)
    if full.is_file():
        if slim is not None:
            build_slim_ground_truth(full, slim)
            return _load_slim(slim)
        return _load_full(full)

    if slim is not None and slim.is_file():
        return _load_slim(slim)

    raise ConfigError(
        f"Neither ground truth ({full}) nor slim cache ({slim}) exists. "
        "Run scripts_build_index/decrypt_dataset.py first."
    )


def _sorted_qids(qids: list[str]) -> list[str]:
    """Sort numerically when all ids are integers, else lexicographically."""
    try:
        return sorted(qids, key=lambda q: int(q))
    except ValueError:
        return sorted(qids)


def select_query_ids(
    ground_truth: dict[str, dict[str, str]],
    query_ids: tuple[str, ...] | list[str] = (),
    sample: Optional[int] = None,
    seed: int = 42,
) -> list[str]:
    """Resolve the subset of query ids to evaluate.

    Precedence: explicit ``query_ids`` > random ``sample`` > all queries.
    """
    if query_ids:
        missing = [q for q in query_ids if str(q) not in ground_truth]
        if missing:
            raise ConfigError(f"query_ids not present in ground truth: {missing[:10]}")
        return [str(q) for q in query_ids]

    all_ids = _sorted_qids(list(ground_truth.keys()))
    if sample is not None and sample < len(all_ids):
        rng = random.Random(seed)
        return _sorted_qids(rng.sample(all_ids, sample))
    return all_ids


def load_qrels(path: str | Path) -> dict[str, list[str]]:
    """Load a TREC qrels file (``qid Q0 docid rel``) → {qid: [docid, ...]}."""
    qrels: dict[str, list[str]] = {}
    p = Path(path)
    if not p.is_file():
        logger.warning("qrels file not found: %s (recall metrics will be 0)", p)
        return qrels
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 4:
                logger.warning("Skipping malformed qrels line: %s", line)
                continue
            qid, _q0, docid, _rel = parts
            qrels.setdefault(qid, []).append(docid)
    return qrels


def load_tags(path: str | Path) -> dict[str, str]:
    """Load a {query_id: domain} JSON map (string-keyed)."""
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"Tags file not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigError(f"Tags file must be a JSON object {{query_id: domain}}: {p}")
    return {str(k): str(v) for k, v in data.items()}
