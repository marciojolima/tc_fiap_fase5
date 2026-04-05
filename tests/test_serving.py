from __future__ import annotations

from unittest.mock import Mock

import pandas as pd
from fastapi.testclient import TestClient

from serving.app import app
from serving.pipeline import ServingConfig, prepare_inference_dataframe
from serving.schemas import ChurnPredictionRequest


def return_route_config() -> ServingConfig:
    return ServingConfig(
        target_col="Exited",
        leakage_columns=["Exited"],
        model_path=Mock(),
        preprocessor_path=Mock(),
        threshold=0.5,
        model_name="random_forest_current",
        run_name="random_forest_current",
    )


def return_route_dataframe(payload, cfg) -> pd.DataFrame:
    return pd.DataFrame([{"feature": 1.0}])


def return_route_prediction(df_feat) -> tuple[float, int]:
    return 0.81, 1


def test_prepare_inference_dataframe_removes_leakage_columns() -> None:
    payload = ChurnPredictionRequest(
        **{
            "CreditScore": 650,
            "Geography": "France",
            "Gender": "Female",
            "Age": 35,
            "Tenure": 4,
            "Balance": 2000.0,
            "NumOfProducts": 2,
            "HasCrCard": 1,
            "IsActiveMember": 1,
            "EstimatedSalary": 50000.0,
            "Card Type": "GOLD",
            "Point Earned": 200,
        }
    )
    cfg = ServingConfig(
        target_col="Exited",
        leakage_columns=["Exited", "Complain", "Satisfaction Score"],
        model_path=Mock(),
        preprocessor_path=Mock(),
        threshold=0.5,
        model_name="random_forest_current",
        run_name="random_forest_current",
    )

    df_feat = prepare_inference_dataframe(payload, cfg)

    assert isinstance(df_feat, pd.DataFrame)
    assert "Exited" not in df_feat.columns
    assert "Complain" not in df_feat.columns
    assert "Satisfaction Score" not in df_feat.columns
    assert "BalancePerProduct" in df_feat.columns
    assert "PointsPerSalary" in df_feat.columns


def test_predict_route_returns_prediction_payload(monkeypatch) -> None:
    client = TestClient(app)

    monkeypatch.setattr(
        "serving.routes.load_serving_config",
        return_route_config,
    )
    monkeypatch.setattr(
        "serving.routes.prepare_inference_dataframe",
        return_route_dataframe,
    )
    monkeypatch.setattr(
        "serving.routes.predict_from_dataframe",
        return_route_prediction,
    )

    response = client.post(
        "/predict",
        json={
            "CreditScore": 600,
            "Geography": "Germany",
            "Gender": "Female",
            "Age": 40,
            "Tenure": 3,
            "Balance": 60000.0,
            "NumOfProducts": 2,
            "HasCrCard": 1,
            "IsActiveMember": 1,
            "EstimatedSalary": 50000.0,
            "Card Type": "DIAMOND",
            "Point Earned": 450,
        },
    )

    assert response.status_code == 200  # noqa: PLR2004
    assert response.json() == {
        "churn_probability": 0.81,
        "churn_prediction": 1,
        "model_name": "random_forest_current",
        "threshold": 0.5,
    }
