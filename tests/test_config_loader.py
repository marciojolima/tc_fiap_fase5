from common.config_loader import (
    DEFAULT_CURRENT_EXPERIMENT_CONFIG_PATH,
    load_env_value,
    load_global_config,
    normalize_mlflow_tracking_uri,
    resolve_experiment_config_path,
    resolve_llm_api_key,
    resolve_llm_base_url,
    resolve_llm_model_name,
    resolve_llm_provider,
)


def test_load_global_config_allows_mlflow_tracking_uri_override(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///tmp/custom-mlflow.db")

    config = load_global_config()

    assert config["mlflow"]["tracking_uri"].endswith("/tmp/custom-mlflow.db")
    assert config["mlflow"]["tracking_uri"].startswith("sqlite:////")


def test_load_global_config_keeps_default_tracking_uri_without_override(
    monkeypatch,
) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    config = load_global_config()

    assert config["mlflow"]["tracking_uri"].endswith("/mlruns/mlflow.db")
    assert config["mlflow"]["tracking_uri"].startswith("sqlite:////")


def test_load_global_config_reads_tracking_uri_from_project_dotenv(
    monkeypatch,
    tmp_path,
) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "pipeline_global_config.yaml").write_text(
        "mlflow:\n  tracking_uri: sqlite:///mlruns/mlflow.db\n",
        encoding="utf-8",
    )
    env_path = tmp_path / ".env"
    env_path.write_text(
        "MLFLOW_TRACKING_URI=sqlite:///tmp/from-dotenv.db\n",
        encoding="utf-8",
    )

    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setattr("common.config_loader.ROOT_DIR", tmp_path)

    config = load_global_config()

    assert config["mlflow"]["tracking_uri"].startswith("sqlite:////")
    assert config["mlflow"]["tracking_uri"].endswith("/tmp/from-dotenv.db")


def test_normalize_mlflow_tracking_uri_keeps_absolute_sqlite_uri() -> None:
    tracking_uri = "sqlite:////app/mlruns/mlflow.db"

    assert normalize_mlflow_tracking_uri(tracking_uri) == tracking_uri


def test_resolve_llm_provider_and_model_name_from_global_config() -> None:
    config = load_global_config()
    expected_provider = config["llm"]["active_provider"]
    expected_model = config["llm"]["providers"][expected_provider]["model_name"]

    assert resolve_llm_provider(config) == expected_provider
    assert resolve_llm_model_name(expected_provider, config) == expected_model


def test_resolve_llm_base_url_allows_environment_override(monkeypatch) -> None:
    monkeypatch.setenv("LLM_BASE_URL", "http://ollama:11434")

    assert resolve_llm_base_url("ollama") == "http://ollama:11434"


def test_load_env_value_reads_local_dotenv_without_overriding_env(
    monkeypatch,
    tmp_path,
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("ANTHROPIC_API_KEY=from-file\n", encoding="utf-8")

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert load_env_value("ANTHROPIC_API_KEY", env_path=env_path) == "from-file"

    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
    assert load_env_value("ANTHROPIC_API_KEY", env_path=env_path) == "from-env"


def test_resolve_llm_api_key_reads_project_dotenv(monkeypatch, tmp_path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("ANTHROPIC_API_KEY=from-file\n", encoding="utf-8")

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("common.config_loader.ROOT_DIR", tmp_path)

    config = {
        "llm": {
            "active_provider": "claude",
            "providers": {
                "claude": {
                    "model_name": "fake-model",
                    "api_key_env_var": "ANTHROPIC_API_KEY",
                }
            },
        }
    }

    assert resolve_llm_api_key("claude", config) == "from-file"


def test_resolve_experiment_config_path_uses_current_as_default() -> None:
    assert resolve_experiment_config_path() == DEFAULT_CURRENT_EXPERIMENT_CONFIG_PATH
    assert (
        resolve_experiment_config_path("current")
        == DEFAULT_CURRENT_EXPERIMENT_CONFIG_PATH
    )


def test_resolve_experiment_config_path_builds_experiment_path() -> None:
    assert (
        resolve_experiment_config_path("rf_v3_recall")
        == "configs/model_lifecycle/experiments/rf_v3_recall.json"
    )
