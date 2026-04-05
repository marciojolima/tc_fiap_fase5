from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from inference.what_if import (
    WhatIfResult,
    build_scenario,
    load_scenario_suite,
    log_what_if_run,
    run_what_if_prediction,
    run_what_if_suite,
    WhatIfConfig,
)
from serving.pipeline import ServingConfig


class DummyRun:
    def __init__(self, run_id: str = "what-if-run") -> None:
        self.info = SimpleNamespace(run_id=run_id)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def return_serving_config(experiment_config_path: str) -> ServingConfig:
    return ServingConfig(
        target_col="Exited",
        leakage_columns=["Exited", "Complain", "Satisfaction Score"],
        model_path=Path("artifacts/random_forest_v2.pkl"),
        preprocessor_path=Path("artifacts/preprocessor.joblib"),
        threshold=0.5,
        model_name="random_forest_v2",
        run_name="random_forest_v2",
    )


def return_prediction(
    df_feat,
    cfg: ServingConfig,
) -> tuple[float, int]:
    return 0.69, 1


def test_run_what_if_prediction_returns_standardized_output(monkeypatch) -> None:
    scenario = build_scenario(
        name="high_churn_candidate",
        payload={
            "Age": 60,
            "Balance": 0,
            "Card Type": "SILVER",
            "CreditScore": 400,
            "EstimatedSalary": 30000,
            "Gender": "Female",
            "Geography": "Germany",
            "HasCrCard": 0,
            "IsActiveMember": 0,
            "NumOfProducts": 1,
            "Point Earned": 100,
            "Tenure": 1,
        },
    )

    monkeypatch.setattr("inference.what_if.build_serving_config", return_serving_config)
    monkeypatch.setattr(
        "inference.what_if.predict_from_dataframe_with_config",
        return_prediction,
    )

    result = run_what_if_prediction(
        scenario,
        "configs/training/experiments/random_forest_v2.yaml",
    )

    assert isinstance(result, WhatIfResult)
    assert result.churn_probability == 0.69  # noqa: PLR2004
    assert result.churn_prediction == 1
    assert result.threshold == 0.5  # noqa: PLR2004
    assert result.model_name == "random_forest_v2"


def test_load_scenario_suite_supports_dict_wrapper(tmp_path: Path) -> None:
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(
        (
            '{"scenarios": ['
            '{"name": "high_risk", "payload": {"Age": 60, "Balance": 0, '
            '"Card Type": "SILVER", "CreditScore": 400, '
            '"EstimatedSalary": 30000, "Gender": "Female", '
            '"Geography": "Germany", "HasCrCard": 0, '
            '"IsActiveMember": 0, "NumOfProducts": 1, '
            '"Point Earned": 100, "Tenure": 1}}'
            "]}"
        ),
        encoding="utf-8",
    )

    scenarios = load_scenario_suite(str(suite_path))

    assert len(scenarios) == 1
    assert scenarios[0].name == "high_risk"


def return_suite_result(
    scenario,
    experiment_config_path: str,
) -> WhatIfResult:
    probability = 0.91 if scenario.name == "high_risk" else 0.08
    prediction = 1 if scenario.name == "high_risk" else 0
    return WhatIfResult(
        scenario_name=scenario.name,
        churn_probability=probability,
        churn_prediction=prediction,
        threshold=0.5,
        model_name="random_forest_v2",
        run_name="random_forest_v2",
    )


def return_dummy_run(run_name: str) -> DummyRun:
    return DummyRun()


def ignore_tracking_uri(uri: str) -> None:
    return None


def ignore_experiment_name(name: str) -> None:
    return None


def return_logged_param(key: str, value: object) -> None:
    _PARAM_LOG.append((key, value))


def return_logged_metric(key: str, value: float) -> None:
    _METRIC_LOG.append((key, value))


def return_logged_tag(key: str, value: str) -> None:
    _TAG_LOG.append((key, value))


def return_json_artifact(
    data: dict,
    filename: str,
    artifact_dir: str,
) -> None:
    _ARTIFACT_LOG.append((filename, artifact_dir))


_PARAM_LOG: list[tuple[str, object]] = []
_METRIC_LOG: list[tuple[str, float]] = []
_TAG_LOG: list[tuple[str, str]] = []
_ARTIFACT_LOG: list[tuple[str, str]] = []


def test_run_what_if_suite_executes_all_scenarios(
    monkeypatch,
    tmp_path: Path,
) -> None:
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(
        (
            "["
            '{"name": "high_risk", "payload": {"Age": 60, "Balance": 0, '
            '"Card Type": "SILVER", "CreditScore": 400, '
            '"EstimatedSalary": 30000, "Gender": "Female", '
            '"Geography": "Germany", "HasCrCard": 0, '
            '"IsActiveMember": 0, "NumOfProducts": 1, '
            '"Point Earned": 100, "Tenure": 1}},'
            '{"name": "low_risk", "payload": {"Age": 35, "Balance": 70000, '
            '"Card Type": "DIAMOND", "CreditScore": 780, '
            '"EstimatedSalary": 120000, "Gender": "Male", '
            '"Geography": "France", "HasCrCard": 1, '
            '"IsActiveMember": 1, "NumOfProducts": 2, '
            '"Point Earned": 800, "Tenure": 8}}'
            "]"
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("inference.what_if.run_what_if_scenario", return_suite_result)

    results = run_what_if_suite(str(suite_path))

    assert len(results) == 2
    assert [result.scenario_name for result in results] == ["high_risk", "low_risk"]


def test_log_what_if_run_registers_mlflow_data(monkeypatch) -> None:
    scenario = build_scenario(
        name="high_risk_profile",
        payload={
            "Age": 60,
            "Balance": 0,
            "Card Type": "SILVER",
            "CreditScore": 400,
            "EstimatedSalary": 30000,
            "Gender": "Female",
            "Geography": "Germany",
            "HasCrCard": 0,
            "IsActiveMember": 0,
            "NumOfProducts": 1,
            "Point Earned": 100,
            "Tenure": 1,
        },
    )
    result = WhatIfResult(
        scenario_name="high_risk_profile",
        churn_probability=0.69,
        churn_prediction=1,
        threshold=0.5,
        model_name="random_forest_v2",
        run_name="random_forest_v2",
    )
    cfg = WhatIfConfig(
        tracking_uri="file:./mlruns",
        mlflow_experiment_name="datathon-churn-what-if",
        experiment_config_path="configs/training/model_current.yaml",
    )

    _PARAM_LOG.clear()
    _METRIC_LOG.clear()
    _TAG_LOG.clear()
    _ARTIFACT_LOG.clear()

    monkeypatch.setattr("inference.what_if.mlflow.set_tracking_uri", ignore_tracking_uri)
    monkeypatch.setattr("inference.what_if.mlflow.set_experiment", ignore_experiment_name)
    monkeypatch.setattr("inference.what_if.mlflow.start_run", return_dummy_run)
    monkeypatch.setattr(
        "inference.what_if.mlflow.log_param",
        return_logged_param,
    )
    monkeypatch.setattr(
        "inference.what_if.mlflow.log_metric",
        return_logged_metric,
    )
    monkeypatch.setattr(
        "inference.what_if.mlflow.set_tag",
        return_logged_tag,
    )
    monkeypatch.setattr(
        "inference.what_if._log_json_artifact",
        return_json_artifact,
    )

    log_what_if_run(scenario, result, cfg)

    assert ("threshold", 0.5) in _PARAM_LOG  # noqa: PLR2004
    assert ("churn_probability", 0.69) in _METRIC_LOG  # noqa: PLR2004
    assert ("flow", "what_if") in _TAG_LOG
    assert ("payload.json", "what_if/high_risk_profile") in _ARTIFACT_LOG
