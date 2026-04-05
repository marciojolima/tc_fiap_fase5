from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_GLOBAL_CONFIG_PATH = "configs/pipeline_global_config.yaml"
DEFAULT_CURRENT_EXPERIMENT_CONFIG_PATH = "configs/training/model_current.yaml"


def load_config(config_path: str = DEFAULT_GLOBAL_CONFIG_PATH) -> dict[str, Any]:
    full_path = ROOT_DIR / config_path
    with open(full_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_global_config() -> dict[str, Any]:
    """Carrega a configuração global do pipeline."""

    return load_config(DEFAULT_GLOBAL_CONFIG_PATH)


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
