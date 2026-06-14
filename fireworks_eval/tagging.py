"""Optional per-domain tagging of queries via Fireworks (cached).

The BrowseComp-Plus dataset has no domain labels. To enable per-domain
capability analysis we classify each query into a fixed taxonomy with an LLM
and cache the result, so tagging runs at most once per query.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from .config import TaggingConfig
from .errors import ConfigError
from .fireworks_client import chat_completion, make_client

logger = logging.getLogger(__name__)

_CLASSIFY_TEMPLATE = (
    "Classify the following research question into exactly ONE domain from this list:\n"
    "{domains}\n\n"
    "Respond with ONLY the domain name, nothing else.\n\n"
    "Question: {question}"
)


def load_taxonomy(path: str | Path) -> list[str]:
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"Taxonomy file not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    domains = data.get("domains") if isinstance(data, dict) else data
    if not isinstance(domains, list) or not domains:
        raise ConfigError(f"Taxonomy must be a non-empty list of domains: {p}")
    return [str(d) for d in domains]


def _normalize(raw: str, taxonomy: list[str]) -> str:
    text = (raw or "").strip().strip(".").lower()
    lookup = {d.lower(): d for d in taxonomy}
    if text in lookup:
        return lookup[text]
    for lower, canonical in lookup.items():
        if lower in text or text in lower:
            return canonical
    return "Other" if "Other" in taxonomy else taxonomy[-1]


def _load_cache(cache_path: Optional[str]) -> dict[str, str]:
    if not cache_path:
        return {}
    p = Path(cache_path)
    if not p.is_file():
        return {}
    try:
        return {str(k): str(v) for k, v in json.loads(p.read_text(encoding="utf-8")).items()}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_cache(cache_path: Optional[str], tags: dict[str, str]) -> None:
    if not cache_path:
        return
    p = Path(cache_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(tags, indent=2, ensure_ascii=False), encoding="utf-8")


def tag_queries(
    cfg: TaggingConfig,
    queries: list[tuple[str, str]],
    *,
    force: bool = False,
) -> dict[str, str]:
    """Return {query_id: domain} for ``queries``, classifying missing ones."""
    taxonomy = load_taxonomy(cfg.taxonomy)
    cache = {} if force else _load_cache(cfg.cache)

    pending = [(qid, q) for qid, q in queries if qid not in cache]
    if not pending:
        return {qid: cache[qid] for qid, _ in queries}

    client = make_client(cfg.base_url, cfg.api_key_env)
    domains_block = "\n".join(f"- {d}" for d in taxonomy)
    logger.info("Tagging %d queries into %d domains...", len(pending), len(taxonomy))

    def _classify(qid: str, question: str) -> tuple[str, str]:
        prompt = _CLASSIFY_TEMPLATE.format(domains=domains_block, question=question)
        try:
            resp = chat_completion(
                client, model=cfg.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=32, temperature=0.0,
            )
            return qid, _normalize(resp.choices[0].message.content or "", taxonomy)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tagging failed for query %s: %s", qid, exc)
            return qid, "Other" if "Other" in taxonomy else taxonomy[-1]

    with ThreadPoolExecutor(max_workers=max(1, cfg.num_threads)) as pool:
        futures = [pool.submit(_classify, qid, q) for qid, q in pending]
        for fut in as_completed(futures):
            qid, domain = fut.result()
            cache[qid] = domain

    _save_cache(cfg.cache, cache)
    return {qid: cache.get(qid, "Other") for qid, _ in queries}
