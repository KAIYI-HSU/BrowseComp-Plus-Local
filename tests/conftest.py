"""Shared pytest fixtures. All tests here are offline (no network/GPU)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(autouse=True)
def _dummy_api_key(monkeypatch):
    """Provide a dummy key so client construction never reads a real secret."""
    monkeypatch.setenv("FIREWORKS_API_KEY", "dummy-key-for-tests")
