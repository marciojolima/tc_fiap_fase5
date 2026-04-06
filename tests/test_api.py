from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from serving.pipeline import (
    ServingConfig,
    load_prediction_model,
    predict_from_dataframe,
    prepare_inference_dataframe,
)
from serving.schemas import ChurnPredictionRequest


class DummyFeaturePipeline:
    @staticmethod
    def transform(df: pd.DataFrame) -> pd.DataFrame:
        assert list(df.columns) == [
            "CreditScore",
            "Geography",
            "Gender",
            "Age",
            "Tenure",
            "Balance",
            "NumOfProducts",
            "HasCrCard",
            "IsActiveMember",
            "EstimatedSalary",
            "Card Type",
            "Point Earned",
        ]

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
                    "PointsPerSalary": 0.0033332778,
                }
            ]
        )


class DummyModel:
    @staticmethod
    def predict_proba(features: pd.DataFrame):
        assert list(features.columns) == [
            "Card Type",
            "Gender",
            "Geo_Germany",
            "Geo_Spain",
            "CreditScore",
            "Age",
            "Tenure",
            "Balance",
            "NumOfProducts",
            "HasCrCard",
            "IsActiveMember",
            "EstimatedSalary",
            "Point Earned",
            "BalancePerProduct",
            "PointsPerSalary",
        ]
        return np.array([[0.2, 0.8]])


def build_request() -> ChurnPredictionRequest:
    return ChurnPredictionRequest(
        CreditScore=650,
        Geography="France",
        Gender="Female",
        Age=35,
        Tenure=5,
        Balance=2000.0,
        NumOfProducts=2,
        HasCrCard=1,
        IsActiveMember=1,
        EstimatedSalary=60000.0,
        **{"Card Type": "GOLD", "Point Earned": 200},
    )


def build_serving_config() -> ServingConfig:
    return ServingConfig(
        target_col="Exited",
        leakage_columns=["Exited", "Complain", "Satisfaction Score"],
        drop_columns=["RowNumber", "CustomerId", "Surname"],
        governed_columns=["Geography"],
        model_path=Path("artifacts/random_forest_current.pkl"),
        feature_pipeline_path=Path("artifacts/feature_pipeline.joblib"),
        threshold=0.5,
        model_name="random_forest_current",
        run_name="random_forest_current",
    )


def return_dummy_feature_pipeline() -> DummyFeaturePipeline:
    return DummyFeaturePipeline()


def return_dummy_model() -> DummyModel:
    return DummyModel()


def return_test_config() -> ServingConfig:
    return build_serving_config()


def test_prepare_inference_dataframe_creates_derived_features(monkeypatch) -> None:
    monkeypatch.setattr(
        "serving.pipeline.load_feature_pipeline",
        lambda _: return_dummy_feature_pipeline(),
    )

    transformed_features = prepare_inference_dataframe(
        build_request(),
        build_serving_config(),
    )

    assert "BalancePerProduct" in transformed_features.columns
    assert "PointsPerSalary" in transformed_features.columns
    assert "Geo_Germany" in transformed_features.columns
    assert "Geo_Spain" in transformed_features.columns


def test_predict_from_dataframe_returns_probability_and_prediction(
    monkeypatch,
) -> None:
    load_prediction_model.cache_clear()

    monkeypatch.setattr(
        "serving.pipeline.load_feature_pipeline",
        lambda _: return_dummy_feature_pipeline(),
    )
    monkeypatch.setattr(
        "serving.pipeline.load_prediction_model",
        return_dummy_model,
    )
    monkeypatch.setattr(
        "serving.pipeline.load_serving_config",
        return_test_config,
    )

    probability, prediction = predict_from_dataframe(
        prepare_inference_dataframe(build_request(), build_serving_config())
    )

    assert probability == 0.8  # noqa: PLR2004
    assert prediction == 1  # noqa: PLR2004
