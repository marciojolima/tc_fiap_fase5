from common.config_loader import (
    load_global_config,
    resolve_llm_base_url,
    resolve_llm_model_name,
    resolve_llm_provider,
)


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


def test_resolve_llm_provider_and_model_name_from_global_config() -> None:
    config = load_global_config()
    expected_provider = config["llm"]["active_provider"]
    expected_model = config["llm"]["providers"][expected_provider]["model_name"]

    assert resolve_llm_provider(config) == expected_provider
    assert resolve_llm_model_name(expected_provider, config) == expected_model


def test_resolve_llm_base_url_allows_environment_override(monkeypatch) -> None:
    monkeypatch.setenv("LLM_BASE_URL", "http://ollama:11434")

    assert resolve_llm_base_url("ollama") == "http://ollama:11434"
