from pathlib import Path

import pandas as pd

from feast_ops.config import FEATURE_ENTITY_JOIN_KEY, ONLINE_FEATURE_COLUMNS
from feast_ops.export import build_feature_store_export_dataframe
from features.feature_engineering import load_feature_engineering_config
from features.pipeline_components import build_feature_transformation_pipeline

EXPECTED_FIRST_CUSTOMER_ID = 1000


def test_build_feature_store_export_dataframe_reuses_pipeline(
    churn_dataframe: pd.DataFrame,
) -> None:
    raw_df = churn_dataframe.copy()
    raw_df.insert(
        0,
        "CustomerId",
        range(EXPECTED_FIRST_CUSTOMER_ID, EXPECTED_FIRST_CUSTOMER_ID + len(raw_df)),
    )
    raw_df.insert(0, "Surname", [f"Cliente{i}" for i in range(len(raw_df))])
    raw_df.insert(0, "RowNumber", range(1, len(raw_df) + 1))

    feature_input = raw_df.drop(
        columns=["RowNumber", "CustomerId", "Surname", "Exited"]
    )
    stage_config = load_feature_engineering_config()
    feature_pipeline = build_feature_transformation_pipeline(
        training_features=feature_input,
        encoding_config=stage_config.encoding_config,
        leakage_columns=["Complain", "Satisfaction Score"],
    )
    feature_pipeline.fit(feature_input)

    export_df = build_feature_store_export_dataframe(raw_df, feature_pipeline)

    assert FEATURE_ENTITY_JOIN_KEY in export_df.columns
    assert "event_timestamp" in export_df.columns
    assert "created_timestamp" in export_df.columns
    assert export_df.columns.tolist()[3:] == ONLINE_FEATURE_COLUMNS
    assert export_df[FEATURE_ENTITY_JOIN_KEY].tolist()[0] == EXPECTED_FIRST_CUSTOMER_ID
    assert export_df["event_timestamp"].is_monotonic_increasing


def test_feature_store_gitignore_file_exists() -> None:
    assert Path("data/feature_store/.gitignore").exists()
