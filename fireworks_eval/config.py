"""Experiment configuration — the single, declarative input to the pipeline.

One experiment == one config file (YAML or JSON). Configs are immutable
(frozen dataclasses) so a run is fully described by its config, and two
experiments differ exactly by their config diff. Reproducing or comparing a
run means diffing two files.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

from .errors import ConfigError
from .fireworks_client import DEFAULT_API_KEY_ENV, DEFAULT_BASE_URL

logger = logging.getLogger(__name__)

# --- Defaults grounded in the existing repo layout -------------------------
DEFAULT_GROUND_TRUTH = "data/browsecomp_plus_decrypted.jsonl"
DEFAULT_SLIM_CACHE = "data/ground_truth_slim.jsonl"
DEFAULT_QREL_EVIDENCE = "topics-qrels/qrel_evidence.txt"
DEFAULT_QREL_GOLD = "topics-qrels/qrel_golds.txt"
DEFAULT_INDEX_PATH = "indexes/qwen3-embedding-0.6b/corpus.shard*.pkl"
DEFAULT_EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_CORPUS = "Tevatron/browsecomp-plus-corpus"
DEFAULT_AGENT_MODEL = "accounts/fireworks/models/gpt-oss-120b"
DEFAULT_JUDGE_MODEL = "accounts/fireworks/models/gpt-oss-120b"
DEFAULT_QUERY_TEMPLATE = "QUERY_TEMPLATE_NO_GET_DOCUMENT"
VALID_BACKENDS = ("faiss", "reasonir")
VALID_DTYPES = ("auto", "float16", "bfloat16", "float32")


@dataclass(frozen=True)
class DatasetConfig:
    ground_truth: str = DEFAULT_GROUND_TRUTH
    slim_cache: str = DEFAULT_SLIM_CACHE
    query_ids: tuple[str, ...] = ()
    sample: Optional[int] = None
    seed: int = 42
    qrel_evidence: str = DEFAULT_QREL_EVIDENCE
    qrel_gold: str = DEFAULT_QREL_GOLD
    tags_file: Optional[str] = None


@dataclass(frozen=True)
class DataSourceConfig:
    """The swappable retrieval profile. Summary mode = small snippet_max_tokens;
    full-text mode = snippet_max_tokens<=0 and/or get_document=True."""

    name: str = "qwen3-0.6b-snippet512"
    backend: str = "faiss"
    index_path: str = DEFAULT_INDEX_PATH
    model_name: str = DEFAULT_EMBED_MODEL
    normalize: bool = True
    pooling: str = "eos"
    torch_dtype: str = "auto"
    dataset_name: str = DEFAULT_CORPUS
    k: int = 5
    snippet_max_tokens: int = 512
    get_document: bool = False
    endpoint: str = "auto"
    host: str = "127.0.0.1"
    port: int = 8123
    startup_timeout_s: int = 900


@dataclass(frozen=True)
class AgentConfig:
    """The deep-research agent under test.

    type:
      - ``fireworks_reference``: built-in chat.completions agent (default).
      - ``external``: your agent already wrote run files to ``external_runs_dir``.
      - ``python:<module>:<factory>``: import a factory returning a
        DeepResearchAgent (Python adapter).
    """

    type: str = "fireworks_reference"
    model: str = DEFAULT_AGENT_MODEL
    base_url: str = DEFAULT_BASE_URL
    api_key_env: str = DEFAULT_API_KEY_ENV
    query_template: Optional[str] = DEFAULT_QUERY_TEMPLATE
    system_prompt: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    reasoning_effort: Optional[str] = "high"
    max_tokens: int = 10000
    max_iterations: int = 50
    num_threads: int = 4
    request_timeout_s: float = 600.0
    external_runs_dir: Optional[str] = None


@dataclass(frozen=True)
class JudgeConfig:
    model: str = DEFAULT_JUDGE_MODEL
    base_url: str = DEFAULT_BASE_URL
    api_key_env: str = DEFAULT_API_KEY_ENV
    max_output_tokens: int = 1024
    temperature: float = 0.0
    reasoning_effort: Optional[str] = None
    num_threads: int = 4
    request_timeout_s: float = 600.0
    cross_judge_model: Optional[str] = None


@dataclass(frozen=True)
class TaggingConfig:
    enabled: bool = False
    taxonomy: str = "configs/taxonomy/default.json"
    model: str = DEFAULT_AGENT_MODEL
    base_url: str = DEFAULT_BASE_URL
    api_key_env: str = DEFAULT_API_KEY_ENV
    cache: Optional[str] = None
    num_threads: int = 4


@dataclass(frozen=True)
class OutputConfig:
    runs_dir: str = "runs/{experiment_name}"
    evals_dir: str = "evals/{experiment_name}"
    report_dir: str = "reports/{experiment_name}"


_SUBCONFIGS = {
    "dataset": DatasetConfig,
    "data_source": DataSourceConfig,
    "agent": AgentConfig,
    "judge": JudgeConfig,
    "tagging": TaggingConfig,
    "output": OutputConfig,
}


def _build_subconfig(cls: type, data: Any, section: str):
    """Build a (frozen) sub-config, ignoring unknown keys with a warning."""
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise ConfigError(f"Config section {section!r} must be a mapping, got {type(data).__name__}")
    known = {f.name for f in fields(cls)}
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        if key not in known:
            logger.warning("Ignoring unknown key %r in config section %r", key, section)
            continue
        kwargs[key] = value
    if "query_ids" in kwargs and kwargs["query_ids"] is not None:
        kwargs["query_ids"] = tuple(str(q) for q in kwargs["query_ids"])
    return cls(**kwargs)


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str
    description: str = ""
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    data_source: DataSourceConfig = field(default_factory=DataSourceConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    tagging: TaggingConfig = field(default_factory=TaggingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        if not isinstance(data, dict):
            raise ConfigError("Top-level config must be a mapping")
        name = data.get("experiment_name")
        if not name or not isinstance(name, str):
            raise ConfigError("Config must define a non-empty string 'experiment_name'")

        subs = {sec: _build_subconfig(c, data.get(sec), sec) for sec, c in _SUBCONFIGS.items()}

        # Resolve {experiment_name} placeholders in output paths.
        out = subs["output"]
        out = OutputConfig(
            runs_dir=out.runs_dir.format(experiment_name=name),
            evals_dir=out.evals_dir.format(experiment_name=name),
            report_dir=out.report_dir.format(experiment_name=name),
        )
        subs["output"] = out

        # Default the tagging cache path to the experiment name.
        tag = subs["tagging"]
        if tag.cache is None:
            tag = TaggingConfig(**{**_as_kwargs(tag), "cache": f"data/tags/{name}.json"})
            subs["tagging"] = tag

        cfg = cls(
            experiment_name=name,
            description=str(data.get("description", "")),
            **subs,
        )
        cfg.validate()
        return cfg

    @classmethod
    def from_file(cls, path: str | Path) -> "ExperimentConfig":
        p = Path(path)
        if not p.is_file():
            raise ConfigError(f"Config file not found: {p}")
        text = p.read_text(encoding="utf-8")
        try:
            if p.suffix.lower() in (".yaml", ".yml"):
                data = yaml.safe_load(text)
            elif p.suffix.lower() == ".json":
                data = json.loads(text)
            else:
                data = yaml.safe_load(text)  # tolerate extensionless YAML
        except (yaml.YAMLError, json.JSONDecodeError) as exc:
            raise ConfigError(f"Failed to parse config {p}: {exc}") from exc
        return cls.from_dict(data or {})

    def validate(self) -> None:
        if self.data_source.backend not in VALID_BACKENDS:
            raise ConfigError(
                f"data_source.backend must be one of {VALID_BACKENDS}, got {self.data_source.backend!r}. "
                "(bm25 requires Java/pyserini and is not enabled here.)"
            )
        if self.data_source.torch_dtype not in VALID_DTYPES:
            raise ConfigError(f"data_source.torch_dtype must be one of {VALID_DTYPES}")
        if self.data_source.k <= 0:
            raise ConfigError("data_source.k must be > 0")
        if self.dataset.sample is not None and self.dataset.sample <= 0:
            raise ConfigError("dataset.sample must be > 0 when set")
        if self.agent.type == "external" and not self.agent.external_runs_dir:
            raise ConfigError("agent.type 'external' requires agent.external_runs_dir")
        if not (
            self.agent.type in ("fireworks_reference", "external")
            or self.agent.type.startswith("python:")
        ):
            raise ConfigError(
                "agent.type must be 'fireworks_reference', 'external', or 'python:<module>:<factory>'"
            )


def _as_kwargs(obj: Any) -> dict[str, Any]:
    """Shallow field->value mapping for a frozen dataclass instance."""
    if not is_dataclass(obj):
        raise TypeError(f"{obj!r} is not a dataclass instance")
    return {f.name: getattr(obj, f.name) for f in fields(obj)}
