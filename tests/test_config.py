"""Tests for experiment configuration loading and validation."""

from __future__ import annotations

import pytest

from fireworks_eval.config import ExperimentConfig
from fireworks_eval.errors import ConfigError


def test_minimal_config_defaults():
    cfg = ExperimentConfig.from_dict({"experiment_name": "exp1"})
    assert cfg.experiment_name == "exp1"
    assert cfg.agent.type == "fireworks_reference"
    assert cfg.agent.model == "accounts/fireworks/models/gpt-oss-120b"
    assert cfg.data_source.backend == "faiss"
    assert cfg.judge.model == "accounts/fireworks/models/gpt-oss-120b"


def test_output_paths_templated():
    cfg = ExperimentConfig.from_dict({"experiment_name": "myexp"})
    assert cfg.output.runs_dir == "runs/myexp"
    assert cfg.output.evals_dir == "evals/myexp"
    assert cfg.output.report_dir == "reports/myexp"
    assert cfg.tagging.cache == "data/tags/myexp.json"


def test_query_ids_coerced_to_str():
    cfg = ExperimentConfig.from_dict({"experiment_name": "x", "dataset": {"query_ids": [769, 770]}})
    assert cfg.dataset.query_ids == ("769", "770")


def test_missing_experiment_name_raises():
    with pytest.raises(ConfigError):
        ExperimentConfig.from_dict({})


def test_invalid_backend_raises():
    with pytest.raises(ConfigError):
        ExperimentConfig.from_dict({"experiment_name": "x", "data_source": {"backend": "bm25"}})


def test_external_requires_runs_dir():
    with pytest.raises(ConfigError):
        ExperimentConfig.from_dict({"experiment_name": "x", "agent": {"type": "external"}})


def test_external_with_runs_dir_ok():
    cfg = ExperimentConfig.from_dict({
        "experiment_name": "x",
        "agent": {"type": "external", "external_runs_dir": "runs/mine"},
    })
    assert cfg.agent.external_runs_dir == "runs/mine"


def test_unknown_keys_ignored():
    cfg = ExperimentConfig.from_dict({"experiment_name": "x", "data_source": {"bogus_key": 1, "k": 7}})
    assert cfg.data_source.k == 7


def test_sample_must_be_positive():
    with pytest.raises(ConfigError):
        ExperimentConfig.from_dict({"experiment_name": "x", "dataset": {"sample": 0}})


def test_from_file_yaml(tmp_path):
    path = tmp_path / "exp.yaml"
    path.write_text("experiment_name: fromfile\ndataset:\n  sample: 5\n", encoding="utf-8")
    cfg = ExperimentConfig.from_file(path)
    assert cfg.experiment_name == "fromfile"
    assert cfg.dataset.sample == 5
