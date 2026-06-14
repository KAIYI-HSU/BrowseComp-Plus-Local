"""fireworks_eval — a 100%-automated evaluation pipeline for Deep-Research agents.

Built on the BrowseComp-Plus benchmark. Routes all LLM calls through an
OpenAI-compatible endpoint (Fireworks by default) and runs retrieval locally
(Qwen3-Embedding-0.6B + FAISS) behind a managed local HTTP endpoint.

See ``docs/fireworks_eval.md`` for the full I/O contract and usage guide.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .config import ExperimentConfig
from .errors import (
    AgentError,
    ConfigError,
    FireworksEvalError,
    JudgeError,
    RetrievalServerError,
    RunFileSchemaError,
)

__all__ = [
    "ExperimentConfig",
    "FireworksEvalError",
    "ConfigError",
    "RetrievalServerError",
    "AgentError",
    "JudgeError",
    "RunFileSchemaError",
    "__version__",
]
