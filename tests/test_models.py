from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from models.train import (
    DatasetSplits,
    ExperimentTrainingConfig,
    ModelSpec,
    TrainingConfig,
    build_comparison,
    build_model_spec,
    evaluate_model,
    load_experiment_training_config,
    train_and_log_model,
)


class DummyClassifier:
    def __init__(self, params: dict) -> None:
        self.params = params
        self.was_fit = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.was_fit = True

    @staticmethod
    def predict(X: pd.DataFrame):
        return [0, 1, 0, 1]

    @staticmethod
    def predict_proba(X: pd.DataFrame):
        return np.array(
            [
                [0.9, 0.1],
                [0.2, 0.8],
                [0.7, 0.3],
                [0.1, 0.9],
            ]
        )


class DummyRun:
    def __init__(self, run_id: str = "run-123") -> None:
        self.info = SimpleNamespace(run_id=run_id)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def return_model_builder(params: dict) -> dict:
    return params


def return_dummy_run(run_name: str) -> DummyRun:
    return DummyRun()


def return_global_training_config() -> dict:
    return {
        "seed": 42,
        "data": {"target_col": "Exited"},
        "split": {"test_size": 0.2},
        "mlflow": {
            "tracking_uri": "file:./mlruns",
            "experiment_name": "global-exp",
            "owner": "team",
            "phase": "dev",
            "dataset_name": "dataset",
        },
    }


def return_experiment_training_config(_: str) -> dict:
    return {
        "experiment": {
            "name": "random_forest_candidate",
            "run_name": "rf_candidate",
            "algorithm": "random_forest",
            "flavor": "sklearn",
        },
        "dataset": {
            "target_col": "Exited",
            "feature_set": "processed_v1",
        },
        "training": {"params": {"n_estimators": 200}},
        "inference": {"threshold": 0.5},
        "artifacts": {"model_path": "artifacts/model.pkl"},
        "mlflow": {
            "experiment_name": "candidate-exp",
            "tags": {"candidate_type": "current"},
        },
        "registry": {"enabled": False},
    }


def test_build_model_spec_extracts_name_params_and_metadata() -> None:
    spec = build_model_spec(
        model_cfg={"name": "baseline", "n_estimators": 100, "max_depth": 10},
        role="baseline",
        builder=return_model_builder,
        output_path=Path("artifacts/model.pkl"),
    )

    assert spec.name == "baseline"
    assert spec.role == "baseline"
    assert spec.params == {"n_estimators": 100, "max_depth": 10}
    assert spec.output_path == Path("artifacts/model.pkl")


def test_evaluate_model_returns_expected_metrics() -> None:
    model = DummyClassifier(params={})
    X_test = pd.DataFrame({"f1": [1, 2, 3, 4]})
    y_test = pd.Series([0, 1, 0, 1])

    metrics = evaluate_model(model, X_test, y_test)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert 0.0 <= metrics["auc"] <= 1.0


def test_build_comparison_calculates_metric_delta() -> None:
    comparison = build_comparison(
        baseline_metrics={"accuracy": 0.8, "auc": 0.7},
        challenger_metrics={"accuracy": 0.9, "auc": 0.75},
    )

    assert comparison["baseline"]["accuracy"] == 0.8  # noqa: PLR2004
    assert comparison["challenger"]["accuracy"] == 0.9  # noqa: PLR2004
    assert comparison["delta"]["accuracy"] == pytest.approx(0.1)
    assert comparison["delta"]["auc"] == pytest.approx(0.05)


def test_load_experiment_training_config_merges_global_and_experiment(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "models.train.load_global_config",
        return_global_training_config,
    )
    monkeypatch.setattr(
        "models.train.load_training_experiment_config",
        return_experiment_training_config,
    )

    cfg = load_experiment_training_config("configs/training/model_current.yaml")

    assert isinstance(cfg, ExperimentTrainingConfig)
    assert cfg.algorithm == "random_forest"
    assert cfg.run_name == "rf_candidate"
    assert cfg.model_params == {"n_estimators": 200}
    assert cfg.model_path == Path("artifacts/model.pkl")
    assert cfg.mlflow_cfg["tracking_uri"] == "file:./mlruns"
    assert cfg.mlflow_cfg["experiment_name"] == "candidate-exp"
    assert cfg.mlflow_cfg["tags"]["owner"] == "team"
    assert cfg.mlflow_cfg["tags"]["candidate_type"] == "current"


def test_train_and_log_model_trains_logs_and_saves(
    monkeypatch,
    tmp_path: Path,
) -> None:
    datasets = DatasetSplits(
        X_train=pd.DataFrame({"f1": [1, 2, 3, 4]}),
        y_train=pd.Series([0, 1, 0, 1]),
        X_test=pd.DataFrame({"f1": [5, 6, 7, 8]}),
        y_test=pd.Series([0, 1, 0, 1]),
    )
    cfg = TrainingConfig(
        seed=42,
        target_col="Exited",
        test_size=0.2,
        baseline_cfg={},
        challenger_cfg={},
        mlflow_cfg={
            "owner": "team",
            "phase": "dev",
            "dataset_name": "dataset",
            "tracking_uri": "file:./mlruns",
            "experiment_name": "exp",
        },
    )
    spec = ModelSpec(
        name="baseline",
        params={"n_estimators": 10},
        role="baseline",
        builder=DummyClassifier,
        output_path=tmp_path / "baseline_model.pkl",
    )

    monkeypatch.setattr("models.train.mlflow.start_run", return_dummy_run)
    log_metrics_mock = Mock()
    log_model_mock = Mock()
    dump_mock = Mock()
    metadata_mock = Mock()

    monkeypatch.setattr("models.train.mlflow.log_metrics", log_metrics_mock)
    monkeypatch.setattr("models.train.mlflow.sklearn.log_model", log_model_mock)
    monkeypatch.setattr("models.train.dump", dump_mock)
    monkeypatch.setattr("models.train.log_run_metadata", metadata_mock)

    metrics = train_and_log_model(spec, cfg, datasets)

    metadata_mock.assert_called_once_with(spec.params, spec.role, cfg, datasets)
    log_metrics_mock.assert_called_once()
    log_model_mock.assert_called_once()
    dump_mock.assert_called_once()
    assert metrics["accuracy"] == 1.0
