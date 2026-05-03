import json

from feature_store import repo


def test_discover_feature_service_names_deduplicates_across_configs(
    monkeypatch,
    tmp_path,
) -> None:
    model_config_dir = tmp_path / "configs" / "model_lifecycle"
    experiments_dir = model_config_dir / "experiments"
    experiments_dir.mkdir(parents=True)

    current_config_path = model_config_dir / "current.json"
    current_config_path.write_text(
        json.dumps({"feast": {"feature_service_name": "customer_churn_gb_v4"}}),
        encoding="utf-8",
    )
    (experiments_dir / "exp_a.json").write_text(
        json.dumps({"feast": {"feature_service_name": "customer_churn_rf_v2"}}),
        encoding="utf-8",
    )
    (experiments_dir / "exp_b.json").write_text(
        json.dumps({"feast": {"feature_service_name": "customer_churn_gb_v4"}}),
        encoding="utf-8",
    )
    (experiments_dir / "exp_c.json").write_text(
        json.dumps({"feast": {"feature_service_name": "customer_churn_xgb_v1"}}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "feature_store.repo.CURRENT_MODEL_CONFIG_PATH",
        current_config_path,
    )
    monkeypatch.setattr(
        "feature_store.repo.EXPERIMENT_MODEL_CONFIG_DIR",
        experiments_dir,
    )

    feature_service_names = repo._discover_feature_service_names()

    assert feature_service_names == [
        "customer_churn_gb_v4",
        "customer_churn_rf_v2",
        "customer_churn_xgb_v1",
    ]


def test_register_feature_services_exposes_services_in_module_globals(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "feature_store.repo._discover_feature_service_names",
        lambda: ["customer_churn_test_v1", "customer_churn_test_v2"],
    )

    registered_feature_services = repo._register_feature_services()

    assert [service.name for service in registered_feature_services] == [
        "customer_churn_test_v1",
        "customer_churn_test_v2",
    ]
    assert repo.customer_churn_test_v1.name == "customer_churn_test_v1"
    assert repo.customer_churn_test_v2.name == "customer_churn_test_v2"
