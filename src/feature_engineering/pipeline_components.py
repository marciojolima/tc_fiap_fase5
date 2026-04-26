"""Componentes reutilizáveis do pipeline de transformação de features."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from common.logger import get_logger

logger = get_logger("feature_engineering.pipeline_components")


@dataclass(frozen=True)
class FeatureEncodingConfig:
    """Configuração das variáveis categóricas codificadas no pipeline."""

    ordinal_categories_by_column: dict[str, list[str]]
    one_hot_categories_by_column: dict[str, list[str]]

    @property
    def categorical_columns(self) -> list[str]:
        """Lista consolidada das colunas categóricas configuradas."""

        return list(self.ordinal_categories_by_column) + list(
            self.one_hot_categories_by_column
        )


class DomainFeatureEnricher(BaseEstimator, TransformerMixin):
    """Adiciona features derivadas de domínio ao conjunto de atributos."""

    required_columns = {
        "Balance",
        "NumOfProducts",
        "Point Earned",
        "EstimatedSalary",
    }

    def fit(self, X: pd.DataFrame, y=None) -> DomainFeatureEnricher:
        self._validate_required_columns(X)
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._validate_required_columns(X)
        enriched_dataframe = X.copy()

        enriched_dataframe["BalancePerProduct"] = enriched_dataframe["Balance"] / (
            enriched_dataframe["NumOfProducts"].replace(0, 1)
        )
        enriched_dataframe["PointsPerSalary"] = enriched_dataframe["Point Earned"] / (
            enriched_dataframe["EstimatedSalary"] + 1
        )

        logger.info(
            "Features de domínio aplicadas: %s",
            ["BalancePerProduct", "PointsPerSalary"],
        )
        return enriched_dataframe

    def _validate_required_columns(self, dataframe: pd.DataFrame) -> None:
        missing_columns = self.required_columns - set(dataframe.columns)
        if missing_columns:
            raise KeyError(
                "Colunas necessárias para enriquecer features estão ausentes: "
                f"{sorted(missing_columns)}"
            )


class LeakageFeatureDropper(BaseEstimator, TransformerMixin):
    """Remove colunas com vazamento do espaço de atributos."""

    def __init__(self, leakage_columns: list[str]):
        self.leakage_columns = leakage_columns

    def fit(self, X: pd.DataFrame, y=None) -> LeakageFeatureDropper:
        existing_columns = [
            column for column in self.leakage_columns if column in X.columns
        ]
        missing_columns = sorted(set(self.leakage_columns) - set(existing_columns))

        if missing_columns:
            logger.warning(
                "Colunas marcadas como leakage não encontradas nas features: %s",
                missing_columns,
            )

        self.columns_to_drop_ = existing_columns
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not getattr(self, "columns_to_drop_", []):
            logger.info("Nenhuma coluna extra de leakage encontrada nas features")
            return X.copy()

        logger.info(
            "Removendo colunas de leakage das features: %s",
            self.columns_to_drop_,
        )
        return X.drop(columns=self.columns_to_drop_, errors="ignore")


class FeatureNameCleaner(BaseEstimator, TransformerMixin):
    """Padroniza nomes de colunas após o pré-processamento."""

    def __init__(self, replacements: dict[str, str] | None = None):
        self.replacements = replacements or {"Geography_": "Geo_"}

    def fit(self, X: pd.DataFrame, y=None) -> FeatureNameCleaner:
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        renamed_dataframe = X.copy()
        renamed_dataframe.columns = self._rename_columns(renamed_dataframe.columns)
        return renamed_dataframe

    def _rename_columns(self, columns: list[str] | pd.Index) -> list[str]:
        renamed_columns = list(columns)
        for old_value, new_value in self.replacements.items():
            renamed_columns = [
                column.replace(old_value, new_value) for column in renamed_columns
            ]
        return renamed_columns


def build_feature_preprocessor(
    input_columns: list[str],
    encoding_config: FeatureEncodingConfig,
) -> ColumnTransformer:
    """Cria o transformador de encoding/scaling para as features finais."""

    transformer_steps: list[tuple[str, object, list[str]]] = []

    for column_name, categories in encoding_config.ordinal_categories_by_column.items():
        transformer_steps.append(
            (
                _sanitize_transformer_name(column_name),
                OrdinalEncoder(
                    categories=[categories],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                [column_name],
            )
        )

    for column_name, categories in encoding_config.one_hot_categories_by_column.items():
        transformer_steps.append(
            (
                _sanitize_transformer_name(column_name),
                OneHotEncoder(
                    categories=[categories],
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                [column_name],
            )
        )

    numeric_columns = [
        column_name
        for column_name in input_columns
        if column_name not in encoding_config.categorical_columns
    ]
    if numeric_columns:
        transformer_steps.append(("numeric", StandardScaler(), numeric_columns))

    column_transformer = ColumnTransformer(
        transformers=transformer_steps,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return column_transformer.set_output(transform="pandas")


def build_feature_transformation_pipeline(
    training_features: pd.DataFrame,
    encoding_config: FeatureEncodingConfig,
    leakage_columns: list[str],
) -> Pipeline:
    """Monta o pipeline declarativo completo para transformação de atributos."""

    preparation_pipeline = Pipeline(
        steps=[
            ("domain_feature_enricher", DomainFeatureEnricher()),
            ("leakage_feature_dropper", LeakageFeatureDropper(leakage_columns)),
        ]
    )
    prepared_training_features = preparation_pipeline.fit_transform(training_features)
    feature_preprocessor = build_feature_preprocessor(
        input_columns=prepared_training_features.columns.tolist(),
        encoding_config=encoding_config,
    )

    return Pipeline(
        steps=[
            ("domain_feature_enricher", DomainFeatureEnricher()),
            ("leakage_feature_dropper", LeakageFeatureDropper(leakage_columns)),
            ("feature_preprocessor", feature_preprocessor),
            ("feature_name_cleaner", FeatureNameCleaner()),
        ]
    )


def _sanitize_transformer_name(column_name: str) -> str:
    """Converte um nome de coluna em identificador legível para o sklearn."""

    return column_name.lower().replace(" ", "_")
