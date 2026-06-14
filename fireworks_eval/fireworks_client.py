"""OpenAI-compatible client factory for Fireworks (or any compatible provider).

All LLM access in the pipeline goes through :func:`make_client`, so switching
provider is a single ``base_url`` / ``api_key_env`` change in the experiment
config. :func:`chat_completion` wraps ``client.chat.completions.create`` with
bounded retries on transient errors.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .errors import ConfigError

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.fireworks.ai/inference/v1"
DEFAULT_API_KEY_ENV = "FIREWORKS_API_KEY"
DEFAULT_TIMEOUT_S = 600.0
MAX_RETRY_ATTEMPTS = 5

# Errors worth retrying: transient network / provider-side conditions.
RETRYABLE_ERRORS = (
    APITimeoutError,
    APIConnectionError,
    RateLimitError,
    InternalServerError,
)


def make_client(
    base_url: str = DEFAULT_BASE_URL,
    api_key_env: str = DEFAULT_API_KEY_ENV,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> OpenAI:
    """Create an OpenAI client pointed at an OpenAI-compatible endpoint.

    Args:
        base_url: Provider base URL (Fireworks by default).
        api_key_env: Environment variable name holding the API key.
        timeout_s: Per-request timeout in seconds.

    Raises:
        ConfigError: If the API key environment variable is not set.
    """
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise ConfigError(
            f"Environment variable {api_key_env!r} is not set. "
            f"Add it to your .env (e.g. {api_key_env}=...) before running."
        )
    return OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s)


@retry(
    reraise=True,
    stop=stop_after_attempt(MAX_RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    retry=retry_if_exception_type(RETRYABLE_ERRORS),
    before_sleep=lambda state: logger.warning(
        "Retrying chat_completion (attempt %d) after %s",
        state.attempt_number,
        state.outcome.exception() if state.outcome else "unknown error",
    ),
)
def chat_completion(client: OpenAI, **kwargs: Any):
    """Call ``chat.completions.create`` with retries on transient errors.

    All keyword arguments are forwarded verbatim to the SDK.
    """
    return client.chat.completions.create(**kwargs)
