import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_ROOT_DIR = Path(__file__).resolve().parents[2]
ROOT_DIR = Path(os.getenv("PROJECT_ROOT", DEFAULT_ROOT_DIR)).resolve()
DEFAULT_GLOBAL_CONFIG_PATH = "configs/pipeline_global_config.yaml"
DEFAULT_CURRENT_EXPERIMENT_CONFIG_PATH = "configs/training/model_current.yaml"
OLLAMA_MODEL_ENV_VAR = "OLLAMA_MODEL"


def load_config(config_path: str = DEFAULT_GLOBAL_CONFIG_PATH) -> dict[str, Any]:
    full_path = ROOT_DIR / config_path
    with open(full_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_global_config() -> dict[str, Any]:
    """Carrega a configuração global do pipeline."""

    config = load_config(DEFAULT_GLOBAL_CONFIG_PATH)
    tracking_uri_override = os.getenv("MLFLOW_TRACKING_URI")

    if tracking_uri_override:
        config["mlflow"]["tracking_uri"] = tracking_uri_override

    return config


def resolve_ollama_model(config: dict[str, Any] | None = None) -> str:
    """Resolve o modelo Ollama com prioridade para variável de ambiente."""

    raw_env_model = os.getenv(OLLAMA_MODEL_ENV_VAR, "").strip()
    if raw_env_model:
        return raw_env_model

    active_config = config if config is not None else load_global_config()
    yaml_model = str(active_config.get("llm", {}).get("model_name", "") or "").strip()
    if yaml_model:
        return yaml_model

    raise ValueError(
        "Modelo Ollama nao configurado. Defina OLLAMA_MODEL no ambiente "
        "ou llm.model_name em configs/pipeline_global_config.yaml."
    )


def load_training_experiment_config(
    config_path: str = DEFAULT_CURRENT_EXPERIMENT_CONFIG_PATH,
) -> dict[str, Any]:
    """Carrega a configuração de um experimento individual de treino."""

    return load_config(config_path)


def merge_configs(
    base_config: dict[str, Any],
    override_config: dict[str, Any],
) -> dict[str, Any]:
    """Mescla dicionários recursivamente preservando a estrutura-base."""

    merged = dict(base_config)
    for key, value in override_config.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_configs(merged[key], value)
            continue

        merged[key] = value

    return merged
