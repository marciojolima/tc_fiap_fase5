import numpy as np

from serving.app import (
    ChurnPredictionRequest,
    ServingConfig,
    healthcheck,
    load_prediction_model,
    load_preprocessor,
    predict_churn,
    predict_from_dataframe,
    prepare_inference_dataframe,
)


class DummyPreprocessor:
    def transform(self, df):
        return df.to_numpy()


class DummyModel:
    def predict_proba(self, X):
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


def test_healthcheck_returns_ok() -> None:
    assert healthcheck() == {"status": "ok"}


def test_prepare_inference_dataframe_creates_derived_features() -> None:
    cfg = ServingConfig(
        target_col="Exited",
        leakage_columns=["Exited", "Complain", "Satisfaction Score"],
        model_path=None,
        preprocessor_path=None,
        threshold=0.5,
    )

    df_feat = prepare_inference_dataframe(build_request(), cfg)

    assert "BalancePerProduct" in df_feat.columns
    assert "PointsPerSalary" in df_feat.columns
    assert "Complain" not in df_feat.columns
    assert "Satisfaction Score" not in df_feat.columns


def test_predict_from_dataframe_returns_probability_and_prediction(
    monkeypatch,
) -> None:
    load_preprocessor.cache_clear()
    load_prediction_model.cache_clear()

    monkeypatch.setattr("serving.app.load_preprocessor", lambda: DummyPreprocessor())
    monkeypatch.setattr("serving.app.load_prediction_model", lambda: DummyModel())
    monkeypatch.setattr(
        "serving.app.load_serving_config",
        lambda: ServingConfig(
            target_col="Exited",
            leakage_columns=["Exited", "Complain", "Satisfaction Score"],
            model_path=None,
            preprocessor_path=None,
            threshold=0.5,
        ),
    )

    probability, prediction = predict_from_dataframe(
        prepare_inference_dataframe(
            build_request(),
            ServingConfig(
                target_col="Exited",
                leakage_columns=["Exited", "Complain", "Satisfaction Score"],
                model_path=None,
                preprocessor_path=None,
                threshold=0.5,
            ),
        )
    )

    assert probability == 0.8
    assert prediction == 1


def test_predict_churn_returns_business_response(monkeypatch) -> None:
    monkeypatch.setattr(
        "serving.app.load_serving_config",
        lambda: ServingConfig(
            target_col="Exited",
            leakage_columns=["Exited", "Complain", "Satisfaction Score"],
            model_path=None,
            preprocessor_path=None,
            threshold=0.5,
        ),
    )
    monkeypatch.setattr(
        "serving.app.predict_from_dataframe",
        lambda df_feat: (0.73, 1),
    )

    response = predict_churn(build_request())

    assert response.churn_probability == 0.73
    assert response.churn_prediction == 1
    assert response.model_name == "challenger_model"
    assert response.threshold == 0.5
