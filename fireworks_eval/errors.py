"""Typed exceptions for the fireworks_eval pipeline.

A small, explicit exception hierarchy makes error handling at each pipeline
stage unambiguous and keeps user-facing messages actionable.
"""

from __future__ import annotations


class FireworksEvalError(Exception):
    """Base class for all errors raised by the fireworks_eval pipeline."""


class ConfigError(FireworksEvalError):
    """Raised when an experiment configuration is missing or invalid."""


class RetrievalServerError(FireworksEvalError):
    """Raised when the local retrieval endpoint cannot start or stay healthy."""


class AgentError(FireworksEvalError):
    """Raised when the deep-research agent fails irrecoverably for a query."""


class JudgeError(FireworksEvalError):
    """Raised when the LLM-as-judge stage fails irrecoverably."""


class RunFileSchemaError(FireworksEvalError):
    """Raised when a run file violates the agent output contract."""
