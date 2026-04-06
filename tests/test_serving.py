from __future__ import annotations

from unittest.mock import Mock, patch

import pandas as pd
from fastapi.testclient import TestClient

from serving.app import app
from serving.pipeline import ServingConfig, prepare_inference_dataframe
from serving.schemas import ChurnPredictionRequest


class DummyFeaturePipeline:
    @staticmethod
    def transform(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Card Type": 1.0,
                    "Gender": 0.0,
                    "Geo_Germany": 0.0,
                    "Geo_Spain": 0.0,
                    "CreditScore": 0.0,
                    "Age": 0.0,
                    "Tenure": 0.0,
                    "Balance": 0.0,
                    "NumOfProducts": 0.0,
                    "HasCrCard": 1.0,
                    "IsActiveMember": 1.0,
                    "EstimatedSalary": 0.0,
                    "Point Earned": 0.0,
                    "BalancePerProduct": 1000.0,
                    "PointsPerSalary": 0.004,
                }
            ]
        )


def return_route_config() -> ServingConfig:
    return ServingConfig(
        target_col="Exited",
        leakage_columns=["Exited"],
        drop_columns=["RowNumber", "CustomerId", "Surname"],
        governed_columns=["Geography"],
        model_path=Mock(),
        feature_pipeline_path=Mock(),
        threshold=0.5,
        model_name="random_forest_current",
        run_name="random_forest_current",
    )


def return_route_dataframe(payload, cfg) -> pd.DataFrame:
    return pd.DataFrame([{"feature": 1.0}])


def return_route_prediction(df_feat) -> tuple[float, int]:
    return 0.81, 1


def test_prepare_inference_dataframe_uses_feature_pipeline(monkeypatch) -> None:
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
        drop_columns=["RowNumber", "CustomerId", "Surname"],
        governed_columns=["Geography"],
        model_path=Mock(),
        feature_pipeline_path=Mock(),
        threshold=0.5,
        model_name="random_forest_current",
        run_name="random_forest_current",
    )
    monkeypatch.setattr(
        "serving.pipeline.load_feature_pipeline",
        lambda _: DummyFeaturePipeline(),
    )

    transformed_features = prepare_inference_dataframe(payload, cfg)

    assert isinstance(transformed_features, pd.DataFrame)
    assert "BalancePerProduct" in transformed_features.columns
    assert "PointsPerSalary" in transformed_features.columns
    assert "Geo_Germany" in transformed_features.columns
    assert "Geo_Spain" in transformed_features.columns


def test_prepare_inference_dataframe_logs_lgpd_governance(monkeypatch) -> None:
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
        drop_columns=["RowNumber", "CustomerId", "Surname"],
        governed_columns=["Geography"],
        model_path=Mock(),
        feature_pipeline_path=Mock(),
        threshold=0.5,
        model_name="random_forest_current",
        run_name="random_forest_current",
    )
    monkeypatch.setattr(
        "serving.pipeline.load_feature_pipeline",
        lambda _: DummyFeaturePipeline(),
    )

    with patch("serving.pipeline.logger.info") as mock_info:
        prepare_inference_dataframe(payload, cfg)

    mock_info.assert_any_call(
        "LGPD: inferência preparada sem identificadores diretos; colunas vedadas por "
        "política: %s",
        ["RowNumber", "CustomerId", "Surname"],
    )
    mock_info.assert_any_call(
        "LGPD: colunas utilizadas sob governança para predição em produção: %s",
        ["Geography"],
    )


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
