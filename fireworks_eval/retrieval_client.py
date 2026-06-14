"""Client for the local retrieval endpoint, plus lifecycle management.

``managed_retrieval`` is a context manager that, for ``endpoint: auto``, starts
``retrieval_server.py`` as a subprocess, waits for ``/health``, yields a client,
and tears the server down on exit. An already-running endpoint is reused. A
configured URL is used as-is (no lifecycle management).
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

import httpx

from .config import DataSourceConfig
from .errors import RetrievalServerError

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
HEALTH_POLL_INTERVAL_S = 2.0
HEALTH_TIMEOUT_S = 5.0


class RetrievalEndpoint:
    """HTTP client for the retrieval endpoint; satisfies RetrievalBackend."""

    def __init__(self, base_url: str, default_k: int = 5, timeout_s: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.default_k = default_k
        self._client = httpx.Client(timeout=timeout_s)

    def health(self) -> dict[str, Any]:
        resp = self._client.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()

    def search(self, query: str, k: Optional[int] = None) -> list[dict[str, Any]]:
        resp = self._client.post(
            f"{self.base_url}/search", json={"query": query, "k": k or self.default_k}
        )
        resp.raise_for_status()
        return resp.json().get("results", [])

    def get_document(self, docid: str) -> Optional[dict[str, Any]]:
        resp = self._client.post(f"{self.base_url}/get_document", json={"docid": docid})
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        self._client.close()


def _is_healthy(base_url: str) -> bool:
    try:
        resp = httpx.get(f"{base_url.rstrip('/')}/health", timeout=HEALTH_TIMEOUT_S)
        return resp.status_code == 200
    except (httpx.HTTPError, OSError):
        return False


def _build_command(cfg: DataSourceConfig) -> list[str]:
    cmd = [
        sys.executable, "-m", "fireworks_eval.retrieval_server",
        "--backend", cfg.backend,
        "--index-path", cfg.index_path,
        "--model-name", cfg.model_name,
        "--pooling", cfg.pooling,
        "--torch-dtype", cfg.torch_dtype,
        "--dataset-name", cfg.dataset_name,
        "--k", str(cfg.k),
        "--snippet-max-tokens", str(cfg.snippet_max_tokens),
        "--host", cfg.host,
        "--port", str(cfg.port),
    ]
    if cfg.normalize:
        cmd.append("--normalize")
    if cfg.get_document:
        cmd.append("--get-document")
    return cmd


def _spawn(cfg: DataSourceConfig) -> tuple[subprocess.Popen, Path]:
    log_dir = REPO_ROOT / "tmp"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"retrieval_server_{cfg.port}.log"
    log_file = log_path.open("w", encoding="utf-8")
    cmd = _build_command(cfg)
    logger.info("Starting retrieval endpoint: %s", " ".join(cmd))
    logger.info("Server logs -> %s", log_path)
    proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=log_file, stderr=subprocess.STDOUT)
    return proc, log_path


def _log_tail(log_path: Path, n: int = 25) -> str:
    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()
        return "\n".join(lines[-n:])
    except OSError:
        return "(no logs captured)"


def _wait_for_health(base_url: str, proc: subprocess.Popen, timeout_s: float, log_path: Path) -> None:
    start = time.monotonic()
    while time.monotonic() - start < timeout_s:
        if proc.poll() is not None:
            raise RetrievalServerError(
                f"retrieval server exited early (code {proc.returncode}). Logs:\n{_log_tail(log_path)}"
            )
        if _is_healthy(base_url):
            logger.info("Retrieval endpoint healthy at %s (%.0fs)", base_url, time.monotonic() - start)
            return
        time.sleep(HEALTH_POLL_INTERVAL_S)
    raise RetrievalServerError(
        f"retrieval server not healthy within {timeout_s:.0f}s. Logs:\n{_log_tail(log_path)}"
    )


def _terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        logger.warning("Retrieval server did not stop gracefully; killing.")
        proc.kill()


@contextmanager
def managed_retrieval(cfg: DataSourceConfig) -> Iterator[RetrievalEndpoint]:
    """Yield a healthy RetrievalEndpoint, managing the server when ``endpoint==auto``."""
    if cfg.endpoint != "auto":
        base_url = cfg.endpoint.rstrip("/")
        if not _is_healthy(base_url):
            raise RetrievalServerError(f"configured retrieval endpoint not reachable: {base_url}")
        logger.info("Using external retrieval endpoint %s", base_url)
        endpoint = RetrievalEndpoint(base_url, cfg.k)
        try:
            yield endpoint
        finally:
            endpoint.close()
        return

    base_url = f"http://{cfg.host}:{cfg.port}"
    if _is_healthy(base_url):
        logger.info("Reusing already-running retrieval endpoint %s", base_url)
        endpoint = RetrievalEndpoint(base_url, cfg.k)
        try:
            yield endpoint
        finally:
            endpoint.close()
        return

    proc, log_path = _spawn(cfg)
    try:
        _wait_for_health(base_url, proc, cfg.startup_timeout_s, log_path)
        endpoint = RetrievalEndpoint(base_url, cfg.k)
        try:
            yield endpoint
        finally:
            endpoint.close()
    finally:
        _terminate(proc)
