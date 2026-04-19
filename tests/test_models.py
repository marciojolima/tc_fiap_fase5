from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from models.catalog import build_model
from models.train import (
    DatasetSplits,
    ExperimentTrainingConfig,
    ModelSpec,
    RetrainingMlflowContext,
    build_model_spec,
    compute_training_data_version,
    evaluate_model,
    load_experiment_training_config,
    log_run_metadata,
    resolve_git_nearest_tag,
    resolve_git_sha,
    resolve_git_tag,
    resolve_runtime_model_params,
    run_training,
    train_and_log_model,
)


class DummyClassifier:
    def __init__(self, dummy_param: int = 1, **params) -> None:
        self.params = {"dummy_param": dummy_param, **params}
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


def build_experiment_training_config(model_path: Path) -> ExperimentTrainingConfig:
    return ExperimentTrainingConfig(
        seed=42,
        target_col="Exited",
        test_size=0.2,
        algorithm="random_forest",
        flavor="sklearn",
        experiment_name="random_forest_candidate",
        run_name="rf_candidate",
        model_version="0.2.0",
        model_params={"n_estimators": 200},
        threshold=0.5,
        feature_set="processed_v1",
        feature_service_name="customer_churn_rf_v2",
        model_path=model_path,
        training_data_version="data-hash-123",
        git_sha="abc123",
        git_tag="post_release_commits",
        git_nearest_tag="v0.2.0",
        risk_level="high",
        fairness_checked=False,
        mlflow_cfg={
            "tracking_uri": "file:./mlruns",
            "experiment_name": "candidate-exp",
            "tags": {
                "owner": "team",
                "phase": "dev",
                "dataset_name": "dataset",
                "candidate_type": "current",
            },
        },
        registry_cfg={"enabled": False},
    )


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
            "version": "0.2.0",
            "algorithm": "random_forest",
            "flavor": "sklearn",
        },
        "dataset": {
            "target_col": "Exited",
            "feature_set": "processed_v1",
        },
        "training": {"params": {"n_estimators": 200}},
        "inference": {"threshold": 0.5},
        "feast": {"feature_service_name": "customer_churn_rf_v2"},
        "artifacts": {"model_path": "artifacts/models/model.pkl"},
        "mlflow": {
            "experiment_name": "candidate-exp",
            "tags": {"candidate_type": "current"},
        },
        "registry": {"enabled": False},
        "governance": {
            "risk_level": "high",
            "fairness_checked": False,
        },
    }


def return_dummy_run(run_name: str) -> DummyRun:
    _START_RUN_NAMES.append(run_name)
    return DummyRun()


def return_dummy_classifier(algorithm: str, params: dict) -> DummyClassifier:
    return DummyClassifier(**params)


def return_logged_metrics(metrics: dict[str, float]) -> None:
    _METRICS_LOG.append(metrics)


def return_logged_model(model, name: str) -> None:
    _MODEL_LOG.append((model, name))


def return_dump_call(model, output_path: Path) -> None:
    _DUMP_LOG.append((model, output_path))


def return_logged_metadata(
    params: dict,
    cfg: ExperimentTrainingConfig,
    datasets: DatasetSplits,
    retraining_context=None,
) -> None:
    _METADATA_LOG.append((params, cfg, datasets, retraining_context))


def return_train_call(
    spec: ModelSpec,
    cfg: ExperimentTrainingConfig,
    datasets: DatasetSplits,
    retraining_context=None,
) -> dict[str, float]:
    _TRAIN_CALLS.append((spec, cfg, datasets, retraining_context))
    return {"accuracy": 1.0}


def return_datasets_stub(target_col: str) -> DatasetSplits:
    return DatasetSplits(
        X_train=pd.DataFrame({"f1": [1, 2, 3, 4]}),
        y_train=pd.Series([0, 1, 0, 1]),
        X_test=pd.DataFrame({"f1": [5, 6, 7, 8]}),
        y_test=pd.Series([0, 1, 0, 1]),
    )


def return_experiment_cfg_for_run(
    experiment_config_path: str,
) -> ExperimentTrainingConfig:
    return build_experiment_training_config(Path("artifacts/models/model.pkl"))


_METRICS_LOG: list[dict[str, float]] = []
_MODEL_LOG: list[tuple[object, str]] = []
_DUMP_LOG: list[tuple[object, Path]] = []
_METADATA_LOG: list[
    tuple[dict, ExperimentTrainingConfig, DatasetSplits, object]
] = []
_TRAIN_CALLS: list[
    tuple[ModelSpec, ExperimentTrainingConfig, DatasetSplits, object]
] = []
_PARAM_LOG: list[tuple[str, object]] = []
_TAG_LOG: list[tuple[str, str]] = []
_ARTIFACT_LOG: list[tuple[str, str]] = []
_START_RUN_NAMES: list[str] = []
EXPECTED_NEG_POS_RATIO = 3.0
EXPECTED_N_ESTIMATORS = 300
EXPECTED_HIGH_THRESHOLD_ACCURACY = 0.75
EXPECTED_HIGH_THRESHOLD_RECALL = 0.5


def return_data_hash() -> str:
    return "data-hash-123"


def return_git_sha() -> str:
    return "abc123"


def return_git_tag() -> str:
    return "post_release_commits"


def return_git_nearest_tag() -> str:
    return "v0.2.0"


def ignore_logged_params(params: dict) -> None:
    return None


def return_logged_param(key: str, value: object) -> None:
    _PARAM_LOG.append((key, value))


def return_logged_tag(key: str, value: str) -> None:
    _TAG_LOG.append((key, value))


def return_logged_artifact(local_path: str, artifact_path: str) -> None:
    _ARTIFACT_LOG.append((Path(local_path).name, artifact_path))


def test_build_model_returns_supported_sklearn_estimator() -> None:
    model = build_model("random_forest", {"n_estimators": 10, "random_state": 42})

    assert model.__class__.__name__ == "RandomForestClassifier"


def test_resolve_runtime_model_params_supports_neg_pos_ratio_token() -> None:
    resolved_params = resolve_runtime_model_params(
        {
            "scale_pos_weight": "__neg_pos_ratio__",
            "n_estimators": EXPECTED_N_ESTIMATORS,
        },
        pd.Series([0, 0, 0, 1]),
    )

    assert resolved_params["scale_pos_weight"] == EXPECTED_NEG_POS_RATIO
    assert resolved_params["n_estimators"] == EXPECTED_N_ESTIMATORS


def test_build_model_spec_uses_experiment_contract() -> None:
    cfg = build_experiment_training_config(Path("artifacts/models/model.pkl"))

    spec = build_model_spec(cfg)

    assert spec.name == "random_forest_candidate"
    assert spec.run_name == "rf_candidate"
    assert spec.algorithm == "random_forest"
    assert spec.params == {"n_estimators": 200}
    assert spec.output_path == Path("artifacts/models/model.pkl")


def test_evaluate_model_returns_expected_metrics() -> None:
    model = DummyClassifier()
    X_test = pd.DataFrame({"f1": [1, 2, 3, 4]})
    y_test = pd.Series([0, 1, 0, 1])

    metrics = evaluate_model(model, X_test, y_test, threshold=0.5)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert 0.0 <= metrics["auc"] <= 1.0


def test_evaluate_model_respects_configured_threshold() -> None:
    model = DummyClassifier()
    X_test = pd.DataFrame({"f1": [1, 2, 3, 4]})
    y_test = pd.Series([0, 1, 0, 1])

    metrics = evaluate_model(model, X_test, y_test, threshold=0.85)

    assert metrics["accuracy"] == EXPECTED_HIGH_THRESHOLD_ACCURACY
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == EXPECTED_HIGH_THRESHOLD_RECALL
    assert metrics["f1"] == 2 / 3


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
    monkeypatch.setattr(
        "models.train.compute_training_data_version",
        return_data_hash,
    )
    monkeypatch.setattr(
        "models.train.resolve_git_sha",
        return_git_sha,
    )
    monkeypatch.setattr(
        "models.train.resolve_git_tag",
        return_git_tag,
    )
    monkeypatch.setattr(
        "models.train.resolve_git_nearest_tag",
        return_git_nearest_tag,
    )

    cfg = load_experiment_training_config("configs/training/model_current.yaml")

    assert isinstance(cfg, ExperimentTrainingConfig)
    assert cfg.algorithm == "random_forest"
    assert cfg.run_name == "rf_candidate"
    assert cfg.model_version == "0.2.0"
    assert cfg.model_params == {"n_estimators": 200}
    assert cfg.feature_service_name == "customer_churn_rf_v2"
    assert cfg.model_path == Path("artifacts/models/model.pkl")
    assert cfg.training_data_version == "data-hash-123"
    assert cfg.git_sha == "abc123"
    assert cfg.git_tag == "post_release_commits"
    assert cfg.git_nearest_tag == "v0.2.0"
    assert cfg.risk_level == "high"
    assert cfg.fairness_checked is False
    assert cfg.mlflow_cfg["tracking_uri"] == "file:./mlruns"
    assert cfg.mlflow_cfg["experiment_name"] == "candidate-exp"
    assert cfg.mlflow_cfg["tags"]["owner"] == "team"
    assert cfg.mlflow_cfg["tags"]["candidate_type"] == "current"


def test_resolve_git_sha_returns_string() -> None:
    git_sha = resolve_git_sha()

    assert isinstance(git_sha, str)
    assert len(git_sha) > 0  # noqa: PLR2004


def test_resolve_git_tag_returns_string() -> None:
    git_tag = resolve_git_tag()

    assert isinstance(git_tag, str)
    assert len(git_tag) > 0  # noqa: PLR2004


def test_resolve_git_nearest_tag_returns_string() -> None:
    git_nearest_tag = resolve_git_nearest_tag()

    assert isinstance(git_nearest_tag, str)
    assert len(git_nearest_tag) > 0  # noqa: PLR2004


def test_compute_training_data_version_returns_hash(tmp_path: Path) -> None:
    train_path = tmp_path / "train.parquet"
    test_path = tmp_path / "test.parquet"
    train_path.write_bytes(b"train-data")
    test_path.write_bytes(b"test-data")

    data_version = compute_training_data_version(train_path, test_path)

    assert isinstance(data_version, str)
    assert len(data_version) == 32  # noqa: PLR2004


def test_log_run_metadata_registers_required_metadata(monkeypatch) -> None:
    cfg = build_experiment_training_config(Path("artifacts/models/model.pkl"))
    datasets = DatasetSplits(
        X_train=pd.DataFrame({"f1": [1, 2, 3, 4]}),
        y_train=pd.Series([0, 1, 0, 1]),
        X_test=pd.DataFrame({"f1": [5, 6, 7, 8]}),
        y_test=pd.Series([0, 1, 0, 1]),
    )
    _PARAM_LOG.clear()
    _TAG_LOG.clear()

    monkeypatch.setattr(
        "models.train.mlflow.log_params",
        ignore_logged_params,
    )
    monkeypatch.setattr(
        "models.train.mlflow.log_param",
        return_logged_param,
    )
    monkeypatch.setattr(
        "models.train.mlflow.set_tag",
        return_logged_tag,
    )
    monkeypatch.setattr(
        "models.train.mlflow.log_artifact",
        return_logged_artifact,
    )

    log_run_metadata(cfg.model_params, cfg, datasets)

    assert ("fairness_checked", False) in _PARAM_LOG
    assert ("feature_service_name", "customer_churn_rf_v2") in _PARAM_LOG
    assert ("model_name", "random_forest_candidate") in _TAG_LOG
    assert ("model_version", "0.2.0") in _TAG_LOG
    assert ("feature_service_name", "customer_churn_rf_v2") in _TAG_LOG
    assert ("training_data_version", "data-hash-123") in _TAG_LOG
    assert ("git_sha", "abc123") in _TAG_LOG
    assert ("git_tag", "post_release_commits") in _TAG_LOG
    assert ("git_nearest_tag", "v0.2.0") in _TAG_LOG
    assert ("risk_level", "high") in _TAG_LOG
    assert ("retrain", "false") in _TAG_LOG


def test_log_run_metadata_registers_retraining_context(monkeypatch) -> None:
    cfg = build_experiment_training_config(Path("artifacts/models/model.pkl"))
    datasets = DatasetSplits(
        X_train=pd.DataFrame({"f1": [1, 2, 3, 4]}),
        y_train=pd.Series([0, 1, 0, 1]),
        X_test=pd.DataFrame({"f1": [5, 6, 7, 8]}),
        y_test=pd.Series([0, 1, 0, 1]),
    )
    retraining_context = RetrainingMlflowContext(
        request_id="req-123",
        reason="critical_data_or_prediction_drift",
        trigger_mode="auto_train_manual_promote",
        promotion_policy="manual_approval_required",
        drift_status="critical",
        max_feature_psi=0.25,
        prediction_psi=0.14,
        drifted_features=["Age"],
        reference_row_count=8000,
        current_row_count=4,
    )
    _PARAM_LOG.clear()
    _TAG_LOG.clear()
    _ARTIFACT_LOG.clear()

    monkeypatch.setattr("models.train.mlflow.log_params", ignore_logged_params)
    monkeypatch.setattr("models.train.mlflow.log_param", return_logged_param)
    monkeypatch.setattr("models.train.mlflow.set_tag", return_logged_tag)
    monkeypatch.setattr("models.train.mlflow.log_artifact", return_logged_artifact)

    log_run_metadata(
        cfg.model_params,
        cfg,
        datasets,
        retraining_context=retraining_context,
    )

    assert ("retrain", "true") in _TAG_LOG
    assert ("training_trigger", "drift_monitoring") in _TAG_LOG
    assert ("retraining_request_id", "req-123") in _TAG_LOG
    assert ("drift_status", "critical") in _TAG_LOG
    assert ("drift_max_feature_psi", 0.25) in _PARAM_LOG
    assert ("drift_prediction_psi", 0.14) in _PARAM_LOG
    assert ("drifted_feature_count", 1) in _PARAM_LOG
    assert ("drift_reference_row_count", 8000) in _PARAM_LOG
    assert ("drift_current_row_count", 4) in _PARAM_LOG
    assert ("retraining_context.json", "retraining") in _ARTIFACT_LOG


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
    cfg = build_experiment_training_config(tmp_path / "candidate_model.pkl")
    spec = build_model_spec(cfg)

    _METRICS_LOG.clear()
    _MODEL_LOG.clear()
    _DUMP_LOG.clear()
    _METADATA_LOG.clear()
    _START_RUN_NAMES.clear()

    monkeypatch.setattr("models.train.mlflow.start_run", return_dummy_run)
    monkeypatch.setattr("models.train.mlflow.log_metrics", return_logged_metrics)
    monkeypatch.setattr("models.train.mlflow.sklearn.log_model", return_logged_model)
    monkeypatch.setattr("models.train.dump", return_dump_call)
    monkeypatch.setattr("models.train.log_run_metadata", return_logged_metadata)
    monkeypatch.setattr("models.train.build_model", return_dummy_classifier)

    metrics = train_and_log_model(spec, cfg, datasets)

    assert _METADATA_LOG == [(spec.params, cfg, datasets, None)]
    assert len(_METRICS_LOG) == 1
    assert len(_MODEL_LOG) == 1
    assert len(_DUMP_LOG) == 1
    assert _START_RUN_NAMES == ["rf_candidate"]
    assert metrics["accuracy"] == 1.0


def test_train_and_log_model_uses_retrain_suffix_in_mlflow_run_name(
    monkeypatch,
    tmp_path: Path,
) -> None:
    datasets = DatasetSplits(
        X_train=pd.DataFrame({"f1": [1, 2, 3, 4]}),
        y_train=pd.Series([0, 1, 0, 1]),
        X_test=pd.DataFrame({"f1": [5, 6, 7, 8]}),
        y_test=pd.Series([0, 1, 0, 1]),
    )
    cfg = build_experiment_training_config(tmp_path / "candidate_model.pkl")
    spec = build_model_spec(cfg)
    retraining_context = RetrainingMlflowContext(
        request_id="req-123",
        reason="critical_data_or_prediction_drift",
        trigger_mode="auto_train_manual_promote",
        promotion_policy="manual_approval_required",
        drift_status="critical",
        max_feature_psi=0.25,
        prediction_psi=0.14,
        drifted_features=["Age"],
        reference_row_count=8000,
        current_row_count=4,
    )

    _START_RUN_NAMES.clear()

    monkeypatch.setattr("models.train.mlflow.start_run", return_dummy_run)
    monkeypatch.setattr("models.train.mlflow.log_metrics", return_logged_metrics)
    monkeypatch.setattr("models.train.mlflow.sklearn.log_model", return_logged_model)
    monkeypatch.setattr("models.train.dump", return_dump_call)
    monkeypatch.setattr("models.train.log_run_metadata", return_logged_metadata)
    monkeypatch.setattr("models.train.build_model", return_dummy_classifier)

    train_and_log_model(
        spec,
        cfg,
        datasets,
        retraining_context=retraining_context,
    )

    assert _START_RUN_NAMES == ["rf_candidate_retrain"]


def test_run_training_executes_single_experiment(monkeypatch) -> None:
    seed_calls = []
    mlflow_cfg_calls = []
    _TRAIN_CALLS.clear()

    monkeypatch.setattr(
        "models.train.load_experiment_training_config",
        return_experiment_cfg_for_run,
    )
    monkeypatch.setattr("models.train.set_global_seed", seed_calls.append)
    monkeypatch.setattr("models.train.load_training_data", return_datasets_stub)
    monkeypatch.setattr("models.train.configure_mlflow", mlflow_cfg_calls.append)
    monkeypatch.setattr("models.train.train_and_log_model", return_train_call)

    run_training("configs/training/model_current.yaml")

    assert seed_calls == [42]
    assert len(mlflow_cfg_calls) == 1
    assert len(_TRAIN_CALLS) == 1
    assert _TRAIN_CALLS[0][3] is None
