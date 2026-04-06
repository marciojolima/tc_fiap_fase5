from __future__ import annotations

from pathlib import Path

import numpy as np

from serving.pipeline import (
    ServingConfig,
    load_prediction_model,
    load_preprocessor,
    predict_from_dataframe,
    prepare_inference_dataframe,
)
from serving.schemas import ChurnPredictionRequest


class DummyPreprocessor:
    def __init__(self):
        self.is_fitted = True

    def transform(self, df):
        if self.is_fitted:
            return df.to_numpy()

        return None

    @staticmethod
    def get_feature_names_out():
        return np.array(
            [
                "Card Type",
                "Gender",
                "Geography_Germany",
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
        )


class DummyModel:
    @staticmethod
    def predict_proba(X):
        assert list(X.columns) == [
            "Card Type",
            "Gender",
            "Geo_Germany",
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
        model_path=Path("artifacts/random_forest_current.pkl"),
        preprocessor_path=Path("artifacts/preprocessor.joblib"),
        threshold=0.5,
        model_name="random_forest_current",
        run_name="random_forest_current",
    )


def return_dummy_preprocessor() -> DummyPreprocessor:
    return DummyPreprocessor()


def return_dummy_model() -> DummyModel:
    return DummyModel()


def return_test_config() -> ServingConfig:
    return build_serving_config()


def test_prepare_inference_dataframe_creates_derived_features() -> None:
    df_feat = prepare_inference_dataframe(build_request(), build_serving_config())

    assert "BalancePerProduct" in df_feat.columns
    assert "PointsPerSalary" in df_feat.columns
    assert "Complain" not in df_feat.columns
    assert "Satisfaction Score" not in df_feat.columns


def test_predict_from_dataframe_returns_probability_and_prediction(
    monkeypatch,
) -> None:
    load_preprocessor.cache_clear()
    load_prediction_model.cache_clear()

    monkeypatch.setattr(
        "serving.pipeline.load_preprocessor",
        return_dummy_preprocessor,
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
