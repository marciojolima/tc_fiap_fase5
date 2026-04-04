import pandas as pd

from features.feature_engineering import (
    FeatureEngineeringConfig,
    build_features,
    create_features,
    drop_leakage_from_features,
    preprocess_features,
)


def test_create_features_adds_expected_columns(
    churn_dataframe: pd.DataFrame,
) -> None:
    base_df = churn_dataframe.copy()

    featured_df = create_features(base_df)

    assert "BalancePerProduct" in featured_df.columns
    assert "PointsPerSalary" in featured_df.columns
    assert "BalancePerProduct" not in base_df.columns
    assert featured_df.loc[0, "BalancePerProduct"] == 1000.0  # noqa: PLR2004


def test_drop_leakage_from_features_removes_only_feature_columns(
    churn_dataframe: pd.DataFrame,
) -> None:
    X_train = churn_dataframe.iloc[:6].drop(columns=["Exited"])
    X_test = churn_dataframe.iloc[6:].drop(columns=["Exited"])

    train_clean, test_clean = drop_leakage_from_features(
        X_train=X_train,
        X_test=X_test,
        leakage_columns=["Exited", "Complain", "Satisfaction Score"],
        target_col="Exited",
    )

    assert "Complain" not in train_clean.columns
    assert "Satisfaction Score" not in train_clean.columns
    assert "Complain" not in test_clean.columns
    assert "Satisfaction Score" not in test_clean.columns
    assert "CreditScore" in train_clean.columns


def test_preprocess_features_returns_expected_encoded_columns(
    churn_dataframe: pd.DataFrame,
) -> None:
    X_train = churn_dataframe.iloc[:6].drop(
        columns=["Exited", "Complain", "Satisfaction Score"]
    )
    X_test = churn_dataframe.iloc[6:].drop(
        columns=["Exited", "Complain", "Satisfaction Score"]
    )

    train_df, test_df, feature_cols, preprocessor = preprocess_features(X_train, X_test)

    assert train_df.shape[1] == len(feature_cols)
    assert test_df.shape[1] == len(feature_cols)
    assert "Geo_Germany" in feature_cols
    assert "Geo_Spain" in feature_cols
    assert "Geography" not in feature_cols
    assert preprocessor is not None


def test_build_features_returns_train_and_test_without_leakage(
    churn_dataframe: pd.DataFrame,
    monkeypatch,
) -> None:
    cfg = FeatureEngineeringConfig(
        seed=42,
        target_col="Exited",
        leakage_columns=["Exited", "Complain", "Satisfaction Score"],
        test_size=0.2,
        random_state=42,
        stratify=True,
    )
    monkeypatch.setattr(
        "features.feature_engineering.load_feature_engineering_config",
        lambda: cfg,
    )

    artifacts = build_features(churn_dataframe)

    assert len(artifacts.train_df) + len(artifacts.test_df) == len(churn_dataframe)
    assert artifacts.train_df.columns[-1] == "Exited"
    assert artifacts.test_df.columns[-1] == "Exited"
    assert "Complain" not in artifacts.feature_cols
    assert "Satisfaction Score" not in artifacts.feature_cols
    assert "Geo_Germany" in artifacts.feature_cols
