import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from common.config_loader import load_config
from common.data_loader import load_raw_data
from common.logger import get_logger
from common.seed import set_global_seed
from features.schema_validation import validate_schema

logger = get_logger(__name__)


def build_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    config = load_config()

    seed = config["seed"]
    target_col = config["data"]["target_col"]
    leakage_columns = config["features"]["leakage_columns"]

    set_global_seed(seed)

    df_feat = df.copy()

    logger.info("Target: %s | Excluídas por leakage: %s", target_col, leakage_columns)

    # 1) Card Type -> ordinal encoding
    card_mapping = {"SILVER": 1, "GOLD": 2, "PLATINUM": 3, "DIAMOND": 4}
    df_feat["Card Type"] = df_feat["Card Type"].map(card_mapping)
    logger.info("Card Type — Ordinal Encoding aplicado: %s", card_mapping)

    # 2) Geography -> one-hot encoding
    df_feat = pd.get_dummies(
        df_feat,
        columns=["Geography"],
        prefix="Geo",
        drop_first=True,
    )
    geo_cols = [c for c in df_feat.columns if c.startswith("Geo_")]
    logger.info("Geography — One-Hot Encoding aplicado: %s", geo_cols)

    # 3) Gender -> label encoding
    le_gender = LabelEncoder()
    df_feat["Gender"] = le_gender.fit_transform(df_feat["Gender"])
    logger.info(
        "Gender — Label Encoding: %s",
        dict(zip(le_gender.classes_, le_gender.transform(le_gender.classes_))),
    )

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

    feature_cols = [c for c in df_feat.columns if c not in leakage_columns]
    X = df_feat[feature_cols]
    y = df_feat[target_col]

    # Escalonamento
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_final = pd.DataFrame(X_scaled, columns=feature_cols)
    logger.info("StandardScaler aplicado — média ~0 | desvio padrão ~1")
    logger.info("Features finais (%d): %s", len(feature_cols), feature_cols)

    # Split
    test_size = config["split"]["test_size"]
    random_state = config["split"]["random_state"]
    stratify = y if config["split"]["stratify"] else None

    X_train, X_test, y_train, y_test = train_test_split(
        X_final,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    train_df = X_train.copy()
    train_df[target_col] = y_train.reset_index(drop=True)

    test_df = X_test.copy()
    test_df[target_col] = y_test.reset_index(drop=True)

    logger.info("Split realizado — Train: %d | Test: %d", len(train_df), len(test_df))
    logger.info(
        "Churn no treino: %.1f%% | Churn no teste: %.1f%%",
        y_train.mean() * 100,
        y_test.mean() * 100,
    )

    return train_df, test_df, feature_cols


def save_processed_outputs(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> None:
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_path = processed_dir / "train.parquet"
    test_path = processed_dir / "test.parquet"
    feature_cols_path = processed_dir / "feature_columns.json"
    schema_report_path = processed_dir / "schema_report.json"

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    with open(feature_cols_path, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2, ensure_ascii=False)

    with open(schema_report_path, "w", encoding="utf-8") as f:
        json.dump({"status": "validated"}, f, indent=2, ensure_ascii=False)

    logger.info("Arquivos processados salvos em data/processed/")


def main() -> None:
    df = load_raw_data()
    validate_schema(df)
    train_df, test_df, feature_cols = build_features(df)
    save_processed_outputs(train_df, test_df, feature_cols)


if __name__ == "__main__":
    main()
