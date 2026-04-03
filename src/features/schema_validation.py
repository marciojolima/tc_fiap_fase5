import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

from common.logger import get_logger

logger = get_logger(__name__)

CHURN_SCHEMA = DataFrameSchema(
    {
        "CreditScore": Column(int, Check.between(300, 850)),
        "Geography": Column(str, Check.isin(["France", "Germany", "Spain"])),
        "Gender": Column(str, Check.isin(["Male", "Female"])),
        "Age": Column(int, Check.between(18, 100)),
        "Tenure": Column(int, Check.between(0, 10)),
        "Balance": Column(float, Check.ge(0)),
        "NumOfProducts": Column(int, Check.between(1, 4)),
        "HasCrCard": Column(int, Check.isin([0, 1])),
        "IsActiveMember": Column(int, Check.isin([0, 1])),
        "EstimatedSalary": Column(float, Check.gt(0)),
        "Exited": Column(int, Check.isin([0, 1])),
        "Complain": Column(int, Check.isin([0, 1])),
        "Satisfaction Score": Column(int, Check.between(1, 5)),
        "Card Type": Column(str, Check.isin(["DIAMOND", "GOLD", "SILVER", "PLATINUM"])),
        "Point Earned": Column(int, Check.ge(0)),
    }
)


def validate_schema(df) -> None:
    logger.info("Iniciando validação de schema...")
    try:
        CHURN_SCHEMA.validate(df)
        logger.info("Schema validado com sucesso — %d linhas, %d colunas", *df.shape)
    except pa.errors.SchemaError as e:
        logger.error("Schema inválido: %s", e)
        raise
