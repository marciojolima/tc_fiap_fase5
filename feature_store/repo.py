"""Definições do repositório Feast para o Datathon de churn bancário."""

from __future__ import annotations

import json
import sys
from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureService, FeatureView, Field, FileSource
from feast.data_format import ParquetFormat
from feast.types import Float32, Float64, Int64
from feast.value_type import ValueType

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from feast_ops.config import (  # noqa: E402
    FEATURE_ENTITY_JOIN_KEY,
    FEATURE_ENTITY_NAME,
    FEATURE_STORE_EXPORT_PATH,
    FEATURE_VIEW_NAME,
)

MODEL_CONFIG_DIR = PROJECT_ROOT / "configs" / "model_lifecycle"
CURRENT_MODEL_CONFIG_PATH = MODEL_CONFIG_DIR / "current.json"
EXPERIMENT_MODEL_CONFIG_DIR = MODEL_CONFIG_DIR / "experiments"

customer_features_source = FileSource(
    name="customer_features_source",
    path=str((PROJECT_ROOT / FEATURE_STORE_EXPORT_PATH).resolve()),
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
    file_format=ParquetFormat(),
)

customer = Entity(
    name=FEATURE_ENTITY_NAME,
    join_keys=[FEATURE_ENTITY_JOIN_KEY],
    value_type=ValueType.INT64,
    description=(
        "Cliente bancário identificado apenas pela chave técnica usada na "
        "store."
    ),
)

customer_churn_features = FeatureView(
    name=FEATURE_VIEW_NAME,
    entities=[customer],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="Card Type", dtype=Int64),
        Field(name="Gender", dtype=Int64),
        Field(name="Geo_Germany", dtype=Int64),
        Field(name="Geo_Spain", dtype=Int64),
        Field(name="CreditScore", dtype=Float32),
        Field(name="Age", dtype=Float32),
        Field(name="Tenure", dtype=Float32),
        Field(name="Balance", dtype=Float32),
        Field(name="NumOfProducts", dtype=Float32),
        Field(name="HasCrCard", dtype=Float32),
        Field(name="IsActiveMember", dtype=Float32),
        Field(name="EstimatedSalary", dtype=Float32),
        Field(name="Point Earned", dtype=Float32),
        Field(name="BalancePerProduct", dtype=Float64),
        Field(name="PointsPerSalary", dtype=Float64),
    ],
    source=customer_features_source,
    online=True,
    tags={
        "dominio": "churn_bancario",
        "origem": "pipeline_local_existente",
        "uso": "serving_online",
    },
)


def _load_model_config(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _discover_feature_service_names() -> list[str]:
    config_paths = [CURRENT_MODEL_CONFIG_PATH]
    config_paths.extend(sorted(EXPERIMENT_MODEL_CONFIG_DIR.glob("*.json")))

    feature_service_names: list[str] = []
    seen_feature_service_names: set[str] = set()

    for config_path in config_paths:
        model_config = _load_model_config(config_path)
        feature_service_name = (
            model_config.get("feast", {}).get("feature_service_name", "").strip()
        )
        if (
            not feature_service_name
            or feature_service_name in seen_feature_service_names
        ):
            continue

        seen_feature_service_names.add(feature_service_name)
        feature_service_names.append(feature_service_name)

    return feature_service_names


def _register_feature_services() -> list[FeatureService]:
    feature_services: list[FeatureService] = []

    for feature_service_name in _discover_feature_service_names():
        feature_service = FeatureService(
            name=feature_service_name,
            features=[customer_churn_features],
        )
        globals()[feature_service_name] = feature_service
        feature_services.append(feature_service)

    return feature_services


REGISTERED_FEATURE_SERVICES = _register_feature_services()
