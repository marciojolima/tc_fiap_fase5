"""Orquestra a preparação de datasets e artefatos de features para modelagem."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from common.config_loader import load_config
from common.data_loader import load_raw_data
from common.logger import get_logger
from common.seed import set_global_seed
from features.pipeline_components import (
    DomainFeatureEnricher,
    FeatureEncodingConfig,
    LeakageFeatureDropper,
    build_feature_preprocessor,
    build_feature_transformation_pipeline,
)
from features.schema_validation import (
    validate_interim_dataset_schema,
    validate_raw_dataset_schema,
)

logger = get_logger(__name__)


@dataclass(frozen=True)
class FeatureEngineeringStageConfig:
    """Configuração consolidada da etapa de engenharia de atributos."""

    seed: int
    target_column: str
    direct_identifier_columns: list[str]
    leakage_feature_columns: list[str]
    test_size: float
    random_state: int
    stratify: bool
    governed_columns: list[str] = field(
        default_factory=lambda: _load_default_governed_columns(load_config())
    )
    encoding_config: FeatureEncodingConfig = field(
        default_factory=lambda: _load_default_encoding_config(load_config())
    )


@dataclass(frozen=True)
class ModelingDatasetArtifacts:
    """Artefatos necessários para treino, rastreabilidade e serving."""

    train_dataset: pd.DataFrame
    test_dataset: pd.DataFrame
    feature_names: list[str]
    feature_pipeline: Pipeline

    @property
    def train_df(self) -> pd.DataFrame:
        return self.train_dataset

    @property
    def test_df(self) -> pd.DataFrame:
        return self.test_dataset

    @property
    def feature_cols(self) -> list[str]:
        return self.feature_names

    @property
    def preprocessor(self) -> Pipeline:
        return self.feature_pipeline


@dataclass(frozen=True)
class InterimDatasetArtifacts:
    """Artefatos produzidos na camada de dados intermediários."""

    interim_dataset: pd.DataFrame

    @property
    def cleaned_df(self) -> pd.DataFrame:
        return self.interim_dataset


def _load_default_encoding_config(config: dict) -> FeatureEncodingConfig:
    """Cria a configuração de encoding a partir do YAML global."""

    categorical_feature_config = config["features"]["categorical_features"]
    return FeatureEncodingConfig(
        ordinal_categories_by_column=categorical_feature_config["ordinal"],
        one_hot_categories_by_column=categorical_feature_config["one_hot"],
    )


def _load_default_governed_columns(config: dict) -> list[str]:
    """Cria a configuração padrão de colunas governadas a partir do YAML."""

    return config["features"].get("governed_columns", [])


def load_feature_engineering_stage_config() -> FeatureEngineeringStageConfig:
    """Carrega a configuração necessária para preparar os dados de modelagem."""

    global_config = load_config()
    encoding_config = _load_default_encoding_config(global_config)

    return FeatureEngineeringStageConfig(
        seed=global_config["seed"],
        target_column=global_config["data"]["target_col"],
        direct_identifier_columns=global_config["data"]["drop_columns"],
        leakage_feature_columns=global_config["features"]["leakage_columns"],
        governed_columns=global_config["features"].get("governed_columns", []),
        encoding_config=encoding_config,
        test_size=global_config["split"]["test_size"],
        random_state=global_config["split"]["random_state"],
        stratify=global_config["split"]["stratify"],
    )


def load_feature_engineering_config() -> FeatureEngineeringStageConfig:
    """Compatibilidade retroativa com o antigo loader de configuração."""

    return load_feature_engineering_stage_config()


def remove_direct_identifier_columns(
    raw_dataset: pd.DataFrame,
    direct_identifier_columns: list[str],
) -> pd.DataFrame:
    """Aplica minimização LGPD removendo identificadores diretos do dataset."""

    minimized_dataset = raw_dataset.drop(
        columns=direct_identifier_columns,
        errors="ignore",
    )
    logger.info(
        "LGPD: identificadores diretos removidos do dataset bruto: %s",
        direct_identifier_columns,
    )
    logger.info("Shape após minimização LGPD: %s", minimized_dataset.shape)
    return minimized_dataset


def validate_lgpd_exclusions(
    dataset: pd.DataFrame,
    direct_identifier_columns: list[str],
) -> None:
    """Valida exclusão de identificadores diretos segundo a política LGPD."""

    existing_identifier_columns = [
        column for column in direct_identifier_columns if column in dataset.columns
    ]
    if existing_identifier_columns:
        logger.error(
            "LGPD: colunas identificadoras ainda presentes no pipeline: %s",
            existing_identifier_columns,
        )
        raise ValueError(
            "LGPD: colunas identificadoras não foram excluídas antes da "
            f"engenharia de atributos: {existing_identifier_columns}"
        )

    logger.info(
        "LGPD: exclusão de identificadores validada com sucesso: %s",
        direct_identifier_columns,
    )


def log_governed_columns(dataset: pd.DataFrame, governed_columns: list[str]) -> None:
    """Registra o uso deliberado de colunas sob governança de risco."""

    present_governed_columns = [
        column for column in governed_columns if column in dataset.columns
    ]
    if present_governed_columns:
        logger.info(
            "LGPD: colunas mantidas sob governança para predição e fairness: %s",
            present_governed_columns,
        )


def clean_interim_data(dataset: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicidades e ausências na camada interim antes da modelagem."""

    initial_row_count = len(dataset)
    deduplicated_dataset = dataset.drop_duplicates()
    cleaned_dataset = deduplicated_dataset.dropna().reset_index(drop=True)

    removed_duplicates = initial_row_count - len(deduplicated_dataset)
    removed_missing = len(deduplicated_dataset) - len(cleaned_dataset)

    logger.info(
        (
            "Camada interim preparada — linhas iniciais: %d | "
            "duplicadas removidas: %d | linhas com valores ausentes removidas: %d"
        ),
        initial_row_count,
        removed_duplicates,
        removed_missing,
    )
    logger.info("Shape do conjunto interim: %s", cleaned_dataset.shape)
    return cleaned_dataset


def split_modeling_dataset(
    modeling_base_dataset: pd.DataFrame,
    target_column: str,
    test_size: float,
    random_state: int,
    stratify: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Separa treino e teste antes do fit do pipeline para evitar vazamento."""

    if target_column not in modeling_base_dataset.columns:
        raise KeyError(
            f"Coluna target '{target_column}' não encontrada. "
            f"Colunas disponíveis: {list(modeling_base_dataset.columns)}"
        )

    training_features = modeling_base_dataset.drop(columns=[target_column]).copy()
    target_series = modeling_base_dataset[target_column].copy()
    stratify_series = target_series if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        training_features,
        target_series,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_series,
    )

    logger.info(
        "Split realizado antes do pipeline de features — Train: %d | Test: %d",
        len(X_train),
        len(X_test),
    )
    logger.info(
        "Churn no treino: %.1f%% | Churn no teste: %.1f%%",
        y_train.mean() * 100,
        y_test.mean() * 100,
    )
    return X_train, X_test, y_train, y_test


def transform_model_inputs(
    training_features: pd.DataFrame,
    test_features: pd.DataFrame,
    stage_config: FeatureEngineeringStageConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], Pipeline]:
    """Ajusta o pipeline de features no treino e transforma treino/teste."""

    feature_pipeline = build_feature_transformation_pipeline(
        training_features=training_features,
        encoding_config=stage_config.encoding_config,
        leakage_columns=stage_config.leakage_feature_columns,
    )

    transformed_train_features = feature_pipeline.fit_transform(training_features)
    transformed_test_features = feature_pipeline.transform(test_features)
    feature_names = transformed_train_features.columns.tolist()

    _log_feature_pipeline_details(
        feature_names=feature_names,
        stage_config=stage_config,
    )
    return (
        transformed_train_features,
        transformed_test_features,
        feature_names,
        feature_pipeline,
    )


def assemble_modeling_dataset(
    transformed_features: pd.DataFrame,
    target_series: pd.Series,
    target_column: str,
) -> pd.DataFrame:
    """Reconstrói um dataset final de modelagem com features e target."""

    modeling_dataset = transformed_features.copy()
    modeling_dataset[target_column] = target_series.reset_index(drop=True)
    return modeling_dataset


def prepare_modeling_datasets(
    interim_dataset: pd.DataFrame,
) -> ModelingDatasetArtifacts:
    """Executa a etapa de feature engineering para gerar datasets de modelagem."""

    stage_config = load_feature_engineering_config()
    set_global_seed(stage_config.seed)

    logger.info(
        "Target: %s | Colunas de leakage em X: %s",
        stage_config.target_column,
        stage_config.leakage_feature_columns,
    )
    logger.info(
        "LGPD: colunas de exclusão obrigatória configuradas: %s",
        stage_config.direct_identifier_columns,
    )

    validate_lgpd_exclusions(
        dataset=interim_dataset,
        direct_identifier_columns=stage_config.direct_identifier_columns,
    )
    log_governed_columns(
        dataset=interim_dataset,
        governed_columns=stage_config.governed_columns,
    )

    (
        training_features,
        test_features,
        training_target,
        test_target,
    ) = split_modeling_dataset(
        modeling_base_dataset=interim_dataset,
        target_column=stage_config.target_column,
        test_size=stage_config.test_size,
        random_state=stage_config.random_state,
        stratify=stage_config.stratify,
    )
    (
        transformed_train_features,
        transformed_test_features,
        feature_names,
        feature_pipeline,
    ) = transform_model_inputs(
        training_features=training_features,
        test_features=test_features,
        stage_config=stage_config,
    )

    train_dataset = assemble_modeling_dataset(
        transformed_features=transformed_train_features,
        target_series=training_target,
        target_column=stage_config.target_column,
    )
    test_dataset = assemble_modeling_dataset(
        transformed_features=transformed_test_features,
        target_series=test_target,
        target_column=stage_config.target_column,
    )

    return ModelingDatasetArtifacts(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        feature_names=feature_names,
        feature_pipeline=feature_pipeline,
    )


def save_interim_artifacts(
    interim_artifacts: InterimDatasetArtifacts,
    output_dir: Path = Path("data/interim"),
) -> None:
    """Persiste o dataset interim em disco."""

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        interim_dataset_path = output_dir / "cleaned.parquet"
        interim_artifacts.interim_dataset.to_parquet(interim_dataset_path, index=False)
    except OSError as exc:
        logger.exception("Falha ao persistir dados intermediários em %s", output_dir)
        raise OSError(
            f"Não foi possível salvar dados intermediários em '{output_dir}'"
        ) from exc

    logger.info("Dados intermediários salvos em %s", interim_dataset_path)


def save_modeling_artifacts(
    modeling_artifacts: ModelingDatasetArtifacts,
    output_dir: Path = Path("data/processed"),
) -> None:
    """Persiste datasets processados e o pipeline de features reutilizável."""

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        train_dataset_path = output_dir / "train.parquet"
        test_dataset_path = output_dir / "test.parquet"
        feature_columns_path = output_dir / "feature_columns.json"
        schema_report_path = output_dir / "schema_report.json"
        feature_pipeline_path = artifacts_dir / "feature_pipeline.joblib"

        modeling_artifacts.train_dataset.to_parquet(train_dataset_path, index=False)
        modeling_artifacts.test_dataset.to_parquet(test_dataset_path, index=False)
        dump(modeling_artifacts.feature_pipeline, feature_pipeline_path)

        with open(feature_columns_path, "w", encoding="utf-8") as file_obj:
            json.dump(
                modeling_artifacts.feature_names,
                file_obj,
                indent=2,
                ensure_ascii=False,
            )

        with open(schema_report_path, "w", encoding="utf-8") as file_obj:
            json.dump(
                {"status": "validated", "pipeline": "feature_pipeline.joblib"},
                file_obj,
                indent=2,
                ensure_ascii=False,
            )
    except OSError as exc:
        logger.exception("Falha ao persistir artefatos em %s", output_dir)
        raise OSError(f"Não foi possível salvar artefatos em '{output_dir}'") from exc

    logger.info("Arquivos processados salvos em %s", output_dir)
    logger.info("Pipeline de features persistido em %s", feature_pipeline_path)


def create_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """Compatibilidade retroativa para enriquecimento de features de domínio."""

    return DomainFeatureEnricher().fit_transform(dataset)


def build_features(interim_dataset: pd.DataFrame) -> ModelingDatasetArtifacts:
    """Compatibilidade retroativa com o antigo nome da fachada principal."""

    return prepare_modeling_datasets(interim_dataset)


def save_artifacts(
    modeling_artifacts: ModelingDatasetArtifacts,
    output_dir: Path = Path("data/processed"),
) -> None:
    """Compatibilidade retroativa com o antigo nome da persistência final."""

    save_modeling_artifacts(modeling_artifacts, output_dir)


def drop_leakage_from_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    leakage_columns: list[str],
    target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compatibilidade retroativa para remoção de leakage nas features."""

    leakage_feature_columns = [
        column for column in leakage_columns if column != target_col
    ]
    leakage_dropper = LeakageFeatureDropper(leakage_feature_columns)
    leakage_dropper.fit(X_train)
    return leakage_dropper.transform(X_train), leakage_dropper.transform(X_test)


def preprocess_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], Pipeline]:
    """Compatibilidade retroativa para fit/transform do pipeline de atributos."""

    stage_config = load_feature_engineering_config()
    feature_preprocessor = build_feature_preprocessor(
        input_columns=X_train.columns.tolist(),
        encoding_config=stage_config.encoding_config,
    )
    transformed_train = feature_preprocessor.fit_transform(X_train)
    transformed_test = feature_preprocessor.transform(X_test)
    transformed_train.columns = [
        column_name.replace("Geography_", "Geo_")
        for column_name in transformed_train.columns
    ]
    transformed_test.columns = [
        column_name.replace("Geography_", "Geo_")
        for column_name in transformed_test.columns
    ]
    feature_names = transformed_train.columns.tolist()
    return transformed_train, transformed_test, feature_names, feature_preprocessor


def _log_feature_pipeline_details(
    feature_names: list[str],
    stage_config: FeatureEngineeringStageConfig,
) -> None:
    """Registra as escolhas do pipeline de transformação de features."""

    geography_feature_names = [
        column_name for column_name in feature_names if column_name.startswith("Geo_")
    ]

    logger.info(
        "Card Type — Ordinal Encoding configurado: %s",
        stage_config.encoding_config.ordinal_categories_by_column["Card Type"],
    )
    logger.info(
        "Gender — Ordinal Encoding configurado: %s",
        stage_config.encoding_config.ordinal_categories_by_column["Gender"],
    )
    logger.info(
        "Geography — One-Hot Encoding aplicado: %s",
        geography_feature_names,
    )
    logger.info("Pipeline de features ajustado com sucesso")
    logger.info("Features finais (%d): %s", len(feature_names), feature_names)


def main() -> None:
    """Executa o pipeline completo de engenharia de atributos."""

    logger.info("Iniciando pipeline de engenharia de atributos")

    stage_config = load_feature_engineering_config()
    raw_dataset = load_raw_data()
    validate_raw_dataset_schema(raw_dataset)

    minimized_dataset = remove_direct_identifier_columns(
        raw_dataset=raw_dataset,
        direct_identifier_columns=stage_config.direct_identifier_columns,
    )
    validate_lgpd_exclusions(
        dataset=minimized_dataset,
        direct_identifier_columns=stage_config.direct_identifier_columns,
    )

    interim_dataset = clean_interim_data(minimized_dataset)
    validate_interim_dataset_schema(interim_dataset)
    save_interim_artifacts(InterimDatasetArtifacts(interim_dataset=interim_dataset))

    modeling_artifacts = prepare_modeling_datasets(interim_dataset)
    save_modeling_artifacts(modeling_artifacts)

    logger.info("Pipeline de engenharia de atributos concluído com sucesso")


if __name__ == "__main__":
    main()


FeatureEngineeringConfig = FeatureEngineeringStageConfig
PipelineArtifacts = ModelingDatasetArtifacts
InterimArtifacts = InterimDatasetArtifacts
