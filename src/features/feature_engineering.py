"""Pipeline de feature engineering para o dataset Bank Customer Churn.

Responsabilidades deste módulo:
    1. Criação de features derivadas
    2. Separação treino/teste sem data leakage
    3. Encoding de variáveis categóricas
    4. Escalonamento de variáveis numéricas
    5. Persistência dos artefatos processados

Uso:
    python -m src.features.feature_engineering
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

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

CARD_CATEGORIES: list[list[str]] = [["SILVER", "GOLD", "PLATINUM", "DIAMOND"]]
GENDER_CATEGORIES: list[list[str]] = [["Female", "Male"]]
GEOGRAPHY_CATEGORIES: list[list[str]] = [["France", "Germany", "Spain"]]

ORDINAL_COLUMNS: list[str] = ["Card Type", "Gender"]
OHE_COLUMNS: list[str] = ["Geography"]


class PipelineArtifacts(NamedTuple):
    """Artefatos produzidos pela etapa de feature engineering."""

    train_df: pd.DataFrame
    test_df: pd.DataFrame
    feature_cols: list[str]
    preprocessor: ColumnTransformer


class FeatureEngineeringConfig(NamedTuple):
    """Configuração usada no pipeline de feature engineering."""

    seed: int
    target_col: str
    leakage_columns: list[str]
    test_size: float
    random_state: int
    stratify: bool


def load_feature_engineering_config() -> FeatureEngineeringConfig:
    """Carrega apenas a configuração necessária para esta etapa."""

    config = load_config()
    return FeatureEngineeringConfig(
        seed=config["seed"],
        target_col=config["data"]["target_col"],
        leakage_columns=config["features"]["leakage_columns"],
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"],
        stratify=config["split"]["stratify"],
    )


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features derivadas de domínio sem alterar o DataFrame original."""

    required_cols = {"Balance", "NumOfProducts", "Point Earned", "EstimatedSalary"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise KeyError(
            "Colunas necessárias para criar features estão ausentes: "
            f"{sorted(missing_cols)}"
        )

    df_feat = df.copy()

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

    return df_feat


def split_train_test(
    df: pd.DataFrame,
    target_col: str,
    test_size: float,
    random_state: int,
    stratify: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Separa treino e teste antes do pré-processamento para evitar leakage."""

    if target_col not in df.columns:
        raise KeyError(
            f"Coluna target '{target_col}' não encontrada. "
            f"Colunas disponíveis: {list(df.columns)}"
        )

    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()
    stratify_col = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col,
    )

    logger.info(
        ("Split realizado antes do pré-processamento (anti-leakage) — "
        "Train: %d | Test: %d"),
        len(X_train),
        len(X_test),
    )
    logger.info(
        "Churn no treino: %.1f%% | Churn no teste: %.1f%%",
        y_train.mean() * 100,
        y_test.mean() * 100,
    )

    return X_train, X_test, y_train, y_test


def build_preprocessor(numeric_features: list[str]) -> ColumnTransformer:
    """Cria o pré-processador com encoders persistíveis e scaler."""

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
    """Padroniza nomes de colunas geradas pelo ColumnTransformer."""

    return [name.replace("Geography_", "Geo_") for name in feature_names]


def _assemble_dataframe(
    X_array,
    feature_cols: list[str],
    y: pd.Series,
    target_col: str,
) -> pd.DataFrame:
    """Reconstrói DataFrame processado com features e target."""

    df_out = pd.DataFrame(X_array, columns=feature_cols)
    df_out[target_col] = y.reset_index(drop=True)
    return df_out


def _log_preprocessing_details(feature_cols: list[str]) -> None:
    """Registra no log os detalhes das transformações configuradas."""

    geo_cols = [c for c in feature_cols if c.startswith("Geo_")]

    logger.info(
        "Card Type — Ordinal Encoding configurado: %s",
        dict(zip(CARD_CATEGORIES[0], range(1, len(CARD_CATEGORIES[0]) + 1))),
    )
    logger.info(
        "Gender — Ordinal Encoding configurado: %s",
        dict(zip(GENDER_CATEGORIES[0], range(len(GENDER_CATEGORIES[0])))),
    )
    logger.info("Geography — One-Hot Encoding aplicado: %s", geo_cols)
    logger.info("StandardScaler aplicado com fit no treino e transform no teste")
    logger.info("Features finais (%d): %s", len(feature_cols), feature_cols)


def drop_leakage_from_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    leakage_columns: list[str],
    target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove colunas de leakage apenas do espaço de features."""

    leakage_in_features = [c for c in leakage_columns if c != target_col]
    existing_leakage = [c for c in leakage_in_features if c in X_train.columns]
    missing_leakage = sorted(set(leakage_in_features) - set(existing_leakage))

    if missing_leakage:
        logger.warning(
            "Colunas marcadas como leakage não encontradas nas features: %s",
            missing_leakage,
        )

    if not existing_leakage:
        logger.info("Nenhuma coluna extra de leakage encontrada nas features")
        return X_train, X_test

    logger.info("Removendo colunas de leakage das features: %s", existing_leakage)
    return (
        X_train.drop(columns=existing_leakage),
        X_test.drop(columns=existing_leakage),
    )


def preprocess_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], ColumnTransformer]:
    """Ajusta o pré-processador no treino e transforma treino e teste."""

    categorical_cols = set(ORDINAL_COLUMNS + OHE_COLUMNS)
    numeric_features = [c for c in X_train.columns if c not in categorical_cols]
    preprocessor = build_preprocessor(numeric_features)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    feature_cols = _normalize_feature_names(
        preprocessor.get_feature_names_out().tolist()
    )

    _log_preprocessing_details(feature_cols)

    train_df = pd.DataFrame(X_train_processed, columns=feature_cols)
    test_df = pd.DataFrame(X_test_processed, columns=feature_cols)
    return train_df, test_df, feature_cols, preprocessor


def build_features(df: pd.DataFrame) -> PipelineArtifacts:
    """Orquestra o pipeline completo como uma fachada simples."""

    cfg = load_feature_engineering_config()
    set_global_seed(cfg.seed)

    logger.info(
        "Target: %s | Excluídas por leakage em X: %s",
        cfg.target_col,
        cfg.leakage_columns,
    )

    # Roteiro do pipeline:
    # 1. criar features
    # 2. split
    # 3. limpar leakage em X
    # 4. preprocessar
    # 5. montar artefatos
    df_feat = create_features(df)
    X_train, X_test, y_train, y_test = split_train_test(
        df=df_feat,
        target_col=cfg.target_col,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=cfg.stratify,
    )
    X_train, X_test = drop_leakage_from_features(
        X_train=X_train,
        X_test=X_test,
        leakage_columns=cfg.leakage_columns,
        target_col=cfg.target_col,
    )
    X_train_processed, X_test_processed, feature_cols, preprocessor = (
        preprocess_features(X_train, X_test)
    )

    train_df = _assemble_dataframe(
        X_train_processed.to_numpy(),
        feature_cols,
        y_train,
        cfg.target_col,
    )
    test_df = _assemble_dataframe(
        X_test_processed.to_numpy(),
        feature_cols,
        y_test,
        cfg.target_col,
    )

    return PipelineArtifacts(
        train_df=train_df,
        test_df=test_df,
        feature_cols=feature_cols,
        preprocessor=preprocessor,
    )


def save_artifacts(
    artifacts: PipelineArtifacts,
    output_dir: Path = Path("data/processed"),
) -> None:
    """Persiste os artefatos processados em disco com logs diagnósticos."""

    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        train_path = output_dir / "train.parquet"
        test_path = output_dir / "test.parquet"
        feature_cols_path = output_dir / "feature_columns.json"
        schema_report_path = output_dir / "schema_report.json"
        preprocessor_path = output_dir / "preprocessor.joblib"

        artifacts.train_df.to_parquet(train_path, index=False)
        artifacts.test_df.to_parquet(test_path, index=False)
        dump(artifacts.preprocessor, preprocessor_path)

        with open(feature_cols_path, "w", encoding="utf-8") as f:
            json.dump(artifacts.feature_cols, f, indent=2, ensure_ascii=False)

        with open(schema_report_path, "w", encoding="utf-8") as f:
            json.dump({"status": "validated"}, f, indent=2, ensure_ascii=False)
    except OSError as exc:
        logger.exception("Falha ao persistir artefatos em %s", output_dir)
        raise OSError(f"Não foi possível salvar artefatos em '{output_dir}'") from exc

    logger.info("Arquivos processados salvos em %s", output_dir)
    logger.info("Pré-processador persistido em %s", preprocessor_path)


def main() -> None:
    """Executa o pipeline completo de feature engineering."""

    logger.info("Iniciando pipeline de feature engineering")

    df = load_raw_data()
    validate_schema(df)

    artifacts = build_features(df)
    save_artifacts(artifacts)

    logger.info("Pipeline de feature engineering concluído com sucesso")


if __name__ == "__main__":
    main()
