"""Constantes compartilhadas pela integração com Feast."""

from __future__ import annotations

from pathlib import Path

FEATURE_STORE_REPO_PATH = Path("feature_store")
FEATURE_STORE_DATA_DIR = Path("data/feature_store")
FEATURE_STORE_EXPORT_PATH = FEATURE_STORE_DATA_DIR / "customer_features.parquet"
FEATURE_STORE_EXPORT_METADATA_PATH = FEATURE_STORE_DATA_DIR / "export_metadata.json"

FEATURE_VIEW_NAME = "customer_churn_features"
FEATURE_SERVICE_NAME = "customer_churn_model_v1"
FEATURE_ENTITY_NAME = "customer"
FEATURE_ENTITY_JOIN_KEY = "customer_id"

ONLINE_FEATURE_COLUMNS = [
    "Card Type",
    "Gender",
    "Geo_Germany",
    "Geo_Spain",
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Point Earned",
    "BalancePerProduct",
    "PointsPerSalary",
]


def build_feature_references(
    feature_view_name: str = FEATURE_VIEW_NAME,
) -> list[str]:
    """Monta as referências completas das features consumidas no online serving."""

    return [f"{feature_view_name}:{column}" for column in ONLINE_FEATURE_COLUMNS]
