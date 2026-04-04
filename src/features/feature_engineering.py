import json
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from common.config_loader import load_config
from common.data_loader import load_raw_data
from common.logger import get_logger
from common.seed import set_global_seed
from features.schema_validation import validate_schema

logger = get_logger(__name__)

CARD_CATEGORIES = [["SILVER", "GOLD", "PLATINUM", "DIAMOND"]]
GENDER_CATEGORIES = [["Female", "Male"]]
GEOGRAPHY_CATEGORIES = [["France", "Germany", "Spain"]]


def _build_preprocessor(numeric_features: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "card_type",
                OrdinalEncoder(
                    categories=CARD_CATEGORIES,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                ["Card Type"],
            ),
            (
                "geography",
                OneHotEncoder(
                    categories=GEOGRAPHY_CATEGORIES,
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                ["Geography"],
            ),
            (
                "gender",
                OrdinalEncoder(
                    categories=GENDER_CATEGORIES,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                ["Gender"],
            ),
            ("numeric", StandardScaler(), numeric_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _normalize_feature_names(feature_names: list[str]) -> list[str]:
    return [name.replace("Geography_", "Geo_") for name in feature_names]


def build_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], ColumnTransformer]:
    config = load_config()

    seed = config["seed"]
    target_col = config["data"]["target_col"]
    leakage_columns = config["features"]["leakage_columns"]

    set_global_seed(seed)

    df_feat = df.copy()

    logger.info("Target: %s | Excluídas por leakage: %s", target_col, leakage_columns)

    # 4) Novas features
    df_feat["BalancePerProduct"] = df_feat["Balance"] / df_feat[
        "NumOfProducts"
    ].replace(0, 1)
    logger.info(
        "Feature criada: BalancePerProduct — media=%.2f | std=%.2f",
        df_feat["BalancePerProduct"].mean(),
        df_feat["BalancePerProduct"].std(),
    )

    df_feat["PointsPerSalary"] = df_feat["Point Earned"] / (
        df_feat["EstimatedSalary"] + 1
    )
    logger.info(
        "Feature criada: PointsPerSalary — media=%.6f | std=%.6f",
        df_feat["PointsPerSalary"].mean(),
        df_feat["PointsPerSalary"].std(),
    )

    raw_feature_cols = [c for c in df_feat.columns if c not in leakage_columns]
    X = df_feat[raw_feature_cols].copy()
    y = df_feat[target_col].copy()

    # Split antes do escalonamento para evitar data leakage
    test_size = config["split"]["test_size"]
    random_state = config["split"]["random_state"]
    stratify = y if config["split"]["stratify"] else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    logger.info("Split realizado antes do pré-processamento para evitar data leakage")

    numeric_features = [
        c for c in X_train.columns if c not in {"Card Type", "Geography", "Gender"}
    ]
    preprocessor = _build_preprocessor(numeric_features)

    logger.info(
        "Card Type — Ordinal Encoding configurado: %s",
        dict(zip(CARD_CATEGORIES[0], range(1, len(CARD_CATEGORIES[0]) + 1))),
    )
    logger.info(
        "Gender — Ordinal Encoding configurado: %s",
        dict(zip(GENDER_CATEGORIES[0], range(len(GENDER_CATEGORIES[0])))),
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    feature_cols = _normalize_feature_names(
        preprocessor.get_feature_names_out().tolist()
    )
    geo_cols = [c for c in feature_cols if c.startswith("Geo_")]

    X_train_final = pd.DataFrame(
        X_train_processed,
        columns=feature_cols,
        index=X_train.index,
    )
    X_test_final = pd.DataFrame(
        X_test_processed,
        columns=feature_cols,
        index=X_test.index,
    )
    logger.info("Geography — One-Hot Encoding aplicado: %s", geo_cols)
    logger.info("StandardScaler aplicado com fit no treino e transform no teste")
    logger.info("Features finais (%d): %s", len(feature_cols), feature_cols)

    train_df = X_train_final.reset_index(drop=True)
    train_df[target_col] = y_train.reset_index(drop=True)

    test_df = X_test_final.reset_index(drop=True)
    test_df[target_col] = y_test.reset_index(drop=True)

    logger.info("Split realizado — Train: %d | Test: %d", len(train_df), len(test_df))
    logger.info(
        "Churn no treino: %.1f%% | Churn no teste: %.1f%%",
        y_train.mean() * 100,
        y_test.mean() * 100,
    )

    return train_df, test_df, feature_cols, preprocessor


def save_processed_outputs(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    preprocessor: ColumnTransformer,
) -> None:
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_path = processed_dir / "train.parquet"
    test_path = processed_dir / "test.parquet"
    feature_cols_path = processed_dir / "feature_columns.json"
    schema_report_path = processed_dir / "schema_report.json"
    preprocessor_path = processed_dir / "preprocessor.joblib"

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    dump(preprocessor, preprocessor_path)

    with open(feature_cols_path, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2, ensure_ascii=False)

    with open(schema_report_path, "w", encoding="utf-8") as f:
        json.dump({"status": "validated"}, f, indent=2, ensure_ascii=False)

    logger.info("Arquivos processados salvos em data/processed/")
    logger.info("Pré-processador persistido em %s", preprocessor_path)


def main() -> None:
    df = load_raw_data()
    validate_schema(df)
    train_df, test_df, feature_cols, preprocessor = build_features(df)
    save_processed_outputs(train_df, test_df, feature_cols, preprocessor)


if __name__ == "__main__":
    main()
