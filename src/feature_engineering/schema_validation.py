"""Schemas de validação para estágios distintos do pipeline de dados."""

from __future__ import annotations

import pandera.pandas as pa
from pandera.typing import Series

from common.config_loader import load_global_config
from common.logger import get_logger

logger = get_logger("feature_engineering.schema_validation")

_GLOBAL_CONFIG = load_global_config()
_FEATURE_CONFIG = _GLOBAL_CONFIG["features"]["categorical_features"]
_CARD_TYPE_CATEGORIES = _FEATURE_CONFIG["ordinal"]["Card Type"]
_GENDER_CATEGORIES = _FEATURE_CONFIG["ordinal"]["Gender"]
_GEOGRAPHY_CATEGORIES = _FEATURE_CONFIG["one_hot"]["Geography"]


class RawCustomerDatasetSchema(pa.DataFrameModel):
    """Schema do dado bruto antes da minimização LGPD."""

    RowNumber: Series[int]
    CustomerId: Series[int]
    Surname: Series[str]
    CreditScore: Series[int] = pa.Field(in_range={"min_value": 300, "max_value": 850})
    Geography: Series[str] = pa.Field(isin=_GEOGRAPHY_CATEGORIES)
    Gender: Series[str] = pa.Field(isin=_GENDER_CATEGORIES)
    Age: Series[int] = pa.Field(in_range={"min_value": 18, "max_value": 100})
    Tenure: Series[int] = pa.Field(in_range={"min_value": 0, "max_value": 10})
    Balance: Series[float] = pa.Field(ge=0)
    NumOfProducts: Series[int] = pa.Field(in_range={"min_value": 1, "max_value": 4})
    HasCrCard: Series[int] = pa.Field(isin=[0, 1])
    IsActiveMember: Series[int] = pa.Field(isin=[0, 1])
    EstimatedSalary: Series[float] = pa.Field(gt=0)
    Exited: Series[int] = pa.Field(isin=[0, 1])
    Complain: Series[int] = pa.Field(isin=[0, 1])
    SatisfactionScore: Series[int] = pa.Field(
        alias="Satisfaction Score",
        in_range={"min_value": 1, "max_value": 5},
    )
    CardType: Series[str] = pa.Field(alias="Card Type", isin=_CARD_TYPE_CATEGORIES)
    PointEarned: Series[int] = pa.Field(alias="Point Earned", ge=0)


class InterimCustomerDatasetSchema(pa.DataFrameModel):
    """Schema do dado após minimização LGPD e limpeza básica."""

    CreditScore: Series[int] = pa.Field(in_range={"min_value": 300, "max_value": 850})
    Geography: Series[str] = pa.Field(isin=_GEOGRAPHY_CATEGORIES)
    Gender: Series[str] = pa.Field(isin=_GENDER_CATEGORIES)
    Age: Series[int] = pa.Field(in_range={"min_value": 18, "max_value": 100})
    Tenure: Series[int] = pa.Field(in_range={"min_value": 0, "max_value": 10})
    Balance: Series[float] = pa.Field(ge=0)
    NumOfProducts: Series[int] = pa.Field(in_range={"min_value": 1, "max_value": 4})
    HasCrCard: Series[int] = pa.Field(isin=[0, 1])
    IsActiveMember: Series[int] = pa.Field(isin=[0, 1])
    EstimatedSalary: Series[float] = pa.Field(gt=0)
    Exited: Series[int] = pa.Field(isin=[0, 1])
    Complain: Series[int] = pa.Field(isin=[0, 1])
    SatisfactionScore: Series[int] = pa.Field(
        alias="Satisfaction Score",
        in_range={"min_value": 1, "max_value": 5},
    )
    CardType: Series[str] = pa.Field(alias="Card Type", isin=_CARD_TYPE_CATEGORIES)
    PointEarned: Series[int] = pa.Field(alias="Point Earned", ge=0)


def validate_raw_dataset_schema(raw_dataset) -> None:
    """Valida o layout bruto do dataset logo após a leitura."""

    logger.info("Validando schema do dado bruto")
    try:
        RawCustomerDatasetSchema.validate(raw_dataset)
    except pa.errors.SchemaError as exc:
        logger.error("Schema bruto inválido: %s", exc)
        raise

    logger.info(
        "Schema bruto validado com sucesso — %d linhas, %d colunas",
        *raw_dataset.shape,
    )


def validate_interim_dataset_schema(interim_dataset) -> None:
    """Valida o dataset após limpeza e minimização de identificadores."""

    logger.info("Validando schema do dado interim")
    try:
        InterimCustomerDatasetSchema.validate(interim_dataset)
    except pa.errors.SchemaError as exc:
        logger.error("Schema interim inválido: %s", exc)
        raise

    logger.info(
        "Schema interim validado com sucesso — %d linhas, %d colunas",
        *interim_dataset.shape,
    )


def validate_schema(interim_dataset) -> None:
    """Compatibilidade retroativa com a antiga validação única."""

    validate_interim_dataset_schema(interim_dataset)
