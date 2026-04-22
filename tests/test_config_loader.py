from common.config_loader import load_global_config


def test_load_global_config_allows_mlflow_tracking_uri_override(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

    config = load_global_config()

    assert config["mlflow"]["tracking_uri"] == "http://127.0.0.1:5000"


def test_load_global_config_keeps_default_tracking_uri_without_override(
    monkeypatch,
) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    config = load_global_config()

    assert config["mlflow"]["tracking_uri"] == "file:./mlruns"
