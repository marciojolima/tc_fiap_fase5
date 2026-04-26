from unittest.mock import patch

import pandas as pd

from feature_engineering.feature_engineering import (
    FeatureEngineeringConfig,
    build_features,
    clean_interim_data,
    create_features,
    drop_leakage_from_features,
    preprocess_features,
    validate_lgpd_exclusions,
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


def test_clean_interim_data_removes_duplicates_and_missing_values(
    churn_dataframe: pd.DataFrame,
) -> None:
    dirty_df = pd.concat(
        [churn_dataframe, churn_dataframe.iloc[[0]]], ignore_index=True
    )
    dirty_df.loc[1, "Balance"] = None

    cleaned_df = clean_interim_data(dirty_df)

    assert len(cleaned_df) == len(churn_dataframe) - 1
    assert cleaned_df.isna().sum().sum() == 0


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

    train_df, test_df, feature_cols, feature_pipeline = preprocess_features(
        X_train,
        X_test,
    )

    assert train_df.shape[1] == len(feature_cols)
    assert test_df.shape[1] == len(feature_cols)
    assert "Geo_Germany" in feature_cols
    assert "Geo_Spain" in feature_cols
    assert "Geography" not in feature_cols
    assert feature_pipeline is not None


def test_build_features_returns_train_and_test_without_leakage(
    churn_dataframe: pd.DataFrame,
    monkeypatch,
) -> None:
    cfg = FeatureEngineeringConfig(
        seed=42,
        target_column="Exited",
        direct_identifier_columns=["RowNumber", "CustomerId", "Surname"],
        leakage_feature_columns=["Exited", "Complain", "Satisfaction Score"],
        test_size=0.2,
        random_state=42,
        stratify=True,
    )
    monkeypatch.setattr(
        "feature_engineering.feature_engineering.load_feature_engineering_config",
        lambda: cfg,
    )

    artifacts = build_features(churn_dataframe)

    assert len(artifacts.train_df) + len(artifacts.test_df) == len(churn_dataframe)
    assert artifacts.train_df.columns[-1] == "Exited"
    assert artifacts.test_df.columns[-1] == "Exited"
    assert "Complain" not in artifacts.feature_cols
    assert "Satisfaction Score" not in artifacts.feature_cols
    assert "Geo_Germany" in artifacts.feature_cols
    assert artifacts.train_df["Exited"].isna().sum() == 0
    assert artifacts.test_df["Exited"].isna().sum() == 0


def test_validate_lgpd_exclusions_logs_success() -> None:
    df = pd.DataFrame({"CreditScore": [600], "Geography": ["France"]})

    with patch("feature_engineering.feature_engineering.logger.info") as mock_info:
        validate_lgpd_exclusions(df, ["RowNumber", "CustomerId", "Surname"])

    mock_info.assert_any_call(
        "LGPD: exclusão de identificadores validada com sucesso: %s",
        ["RowNumber", "CustomerId", "Surname"],
    )


def test_build_features_logs_lgpd_governance_for_geography(
    churn_dataframe: pd.DataFrame,
    monkeypatch,
) -> None:
    cfg = FeatureEngineeringConfig(
        seed=42,
        target_column="Exited",
        direct_identifier_columns=["RowNumber", "CustomerId", "Surname"],
        leakage_feature_columns=["Exited", "Complain", "Satisfaction Score"],
        test_size=0.2,
        random_state=42,
        stratify=True,
    )
    monkeypatch.setattr(
        "feature_engineering.feature_engineering.load_feature_engineering_config",
        lambda: cfg,
    )

    with patch("feature_engineering.feature_engineering.logger.info") as mock_info:
        build_features(churn_dataframe)

    mock_info.assert_any_call(
        "LGPD: colunas de exclusão obrigatória configuradas: %s",
        ["RowNumber", "CustomerId", "Surname"],
    )
    mock_info.assert_any_call(
        "LGPD: colunas mantidas sob governança para predição e fairness: %s",
        ["Geography"],
    )
