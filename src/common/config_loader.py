import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_ROOT_DIR = Path(__file__).resolve().parents[2]
ROOT_DIR = Path(os.getenv("PROJECT_ROOT", DEFAULT_ROOT_DIR)).resolve()
DEFAULT_GLOBAL_CONFIG_PATH = "configs/pipeline_global_config.yaml"
DEFAULT_CURRENT_EXPERIMENT_CONFIG_PATH = "configs/training/model_current.yaml"
OLLAMA_MODEL_ENV_VAR = "OLLAMA_MODEL"
LLM_PROVIDER_ENV_VAR = "LLM_PROVIDER"
LLM_BASE_URL_ENV_VAR = "LLM_BASE_URL"
OLLAMA_BASE_URL_ENV_VAR = "OLLAMA_BASE_URL"
OPENAI_BASE_URL_ENV_VAR = "OPENAI_BASE_URL"
ANTHROPIC_BASE_URL_ENV_VAR = "ANTHROPIC_BASE_URL"
OPENAI_API_KEY_ENV_VAR = "OPENAI_API_KEY"
ANTHROPIC_API_KEY_ENV_VAR = "ANTHROPIC_API_KEY"


def load_env_value(env_var: str, env_path: str | Path = ".env") -> str | None:
    """Carrega uma variável simples do `.env` local sem sobrescrever o ambiente."""

    if os.getenv(env_var):
        return os.getenv(env_var)

    full_path = ROOT_DIR / env_path
    if not full_path.exists():
        return None

    with open(full_path, encoding="utf-8") as file_obj:
        for raw_line in file_obj:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            if key.strip() != env_var:
                continue

            cleaned_value = value.strip().strip('"').strip("'")
            return cleaned_value or None

    return None


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


def load_llm_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Carrega a configuração LLM consolidada do pipeline global."""

    active_config = config if config is not None else load_global_config()
    llm_config = active_config.get("llm", {})
    if not isinstance(llm_config, dict):
        raise ValueError("Configuração 'llm' inválida no pipeline global.")
    return llm_config


def resolve_llm_provider(config: dict[str, Any] | None = None) -> str:
    """Resolve o provedor LLM ativo, priorizando configuração explícita."""

    raw_env_provider = os.getenv(LLM_PROVIDER_ENV_VAR, "").strip()
    if raw_env_provider:
        return raw_env_provider.lower()

    llm_config = load_llm_config(config)
    provider = str(
        llm_config.get("active_provider") or llm_config.get("provider") or ""
    ).strip()
    if provider:
        return provider.lower()

    raise ValueError(
        "Provider LLM não configurado. Defina llm.active_provider em "
        "configs/pipeline_global_config.yaml."
    )


def resolve_llm_timeout_seconds(config: dict[str, Any] | None = None) -> int:
    """Resolve timeout padrão das chamadas LLM."""

    llm_config = load_llm_config(config)
    timeout_value = llm_config.get("timeout_seconds", 45)
    return int(timeout_value)


def resolve_llm_provider_config(
    provider: str | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Retorna o bloco de configuração específico do provider."""

    llm_config = load_llm_config(config)
    active_provider = (provider or resolve_llm_provider(config)).lower()
    providers = llm_config.get("providers", {})

    if isinstance(providers, dict) and active_provider in providers:
        provider_config = providers.get(active_provider, {})
        if isinstance(provider_config, dict):
            return provider_config

    if active_provider == "ollama":
        legacy_config = {
            key: llm_config.get(key)
            for key in ("base_url", "model_name")
            if llm_config.get(key) is not None
        }
        if legacy_config:
            return legacy_config

    return {}


def resolve_llm_model_name(
    provider: str | None = None,
    config: dict[str, Any] | None = None,
) -> str:
    """Resolve o nome do modelo configurado para o provider ativo."""

    active_provider = (provider or resolve_llm_provider(config)).lower()

    if active_provider == "ollama":
        raw_env_model = os.getenv(OLLAMA_MODEL_ENV_VAR, "").strip()
        if raw_env_model:
            return raw_env_model

    provider_config = resolve_llm_provider_config(active_provider, config)
    model_name = str(provider_config.get("model_name", "") or "").strip()
    if model_name:
        return model_name

    raise ValueError(
        f"Modelo LLM não configurado para provider '{active_provider}'. "
        "Defina llm.providers.<provider>.model_name em "
        "configs/pipeline_global_config.yaml."
    )


def resolve_llm_base_url(
    provider: str | None = None,
    config: dict[str, Any] | None = None,
) -> str | None:
    """Resolve a base URL do provider, quando aplicável."""

    active_provider = (provider or resolve_llm_provider(config)).lower()
    env_var_candidates = [LLM_BASE_URL_ENV_VAR]
    provider_env_var = {
        "ollama": OLLAMA_BASE_URL_ENV_VAR,
        "openai": OPENAI_BASE_URL_ENV_VAR,
        "claude": ANTHROPIC_BASE_URL_ENV_VAR,
    }.get(active_provider)
    if provider_env_var:
        env_var_candidates.insert(0, provider_env_var)

    for env_var in env_var_candidates:
        raw_env_url = os.getenv(env_var, "").strip()
        if raw_env_url:
            return raw_env_url.rstrip("/")

    provider_config = resolve_llm_provider_config(active_provider, config)
    base_url = str(provider_config.get("base_url", "") or "").strip()
    return base_url.rstrip("/") if base_url else None


def resolve_llm_api_key(
    provider: str | None = None,
    config: dict[str, Any] | None = None,
) -> str:
    """Resolve a chave de API do provider externo a partir do ambiente."""

    active_provider = (provider or resolve_llm_provider(config)).lower()
    provider_config = resolve_llm_provider_config(active_provider, config)
    default_env_var = {
        "openai": OPENAI_API_KEY_ENV_VAR,
        "claude": ANTHROPIC_API_KEY_ENV_VAR,
    }.get(active_provider, "")
    env_var = str(provider_config.get("api_key_env_var") or default_env_var).strip()
    if not env_var:
        raise ValueError(
            f"Provider '{active_provider}' não usa api_key ou não possui "
            "variável de ambiente configurada."
        )

    api_key = (load_env_value(env_var) or "").strip()
    if api_key:
        return api_key

    raise ValueError(
        f"Chave de API não configurada para provider '{active_provider}'. "
        f"Defina {env_var} no arquivo .env."
    )


def resolve_llm_max_tokens(
    provider: str | None = None,
    config: dict[str, Any] | None = None,
) -> int | None:
    """Resolve limite opcional de tokens do provider."""

    active_provider = (provider or resolve_llm_provider(config)).lower()
    provider_config = resolve_llm_provider_config(active_provider, config)
    max_tokens = provider_config.get("max_tokens")
    if max_tokens in {None, ""}:
        return None
    return int(max_tokens)


def resolve_ollama_model(config: dict[str, Any] | None = None) -> str:
    """Resolve o modelo Ollama com prioridade para variável de ambiente."""
    return resolve_llm_model_name("ollama", config)


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
