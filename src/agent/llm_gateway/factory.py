from __future__ import annotations

from typing import Any

from agent.llm_gateway.providers.base import LLMProvider, ProviderChatConfig
from agent.llm_gateway.providers.claude import ClaudeProvider
from agent.llm_gateway.providers.ollama import OllamaProvider
from agent.llm_gateway.providers.openai import OpenAIProvider
from common.config_loader import (
    load_global_config,
    resolve_llm_api_key,
    resolve_llm_base_url,
    resolve_llm_max_tokens,
    resolve_llm_model_name,
    resolve_llm_provider,
    resolve_llm_timeout_seconds,
)


def build_llm_client(config: dict[str, Any] | None = None) -> LLMProvider:
    """Constroi o client/provider LLM ativo a partir da configuração global."""

    active_config = config if config is not None else load_global_config()
    provider = resolve_llm_provider(active_config)
    provider_config = ProviderChatConfig(
        provider=provider,
        model_name=resolve_llm_model_name(provider, active_config),
        timeout_seconds=resolve_llm_timeout_seconds(active_config),
        base_url=resolve_llm_base_url(provider, active_config),
        max_tokens=resolve_llm_max_tokens(provider, active_config),
        api_key=(
            resolve_llm_api_key(provider, active_config)
            if provider in {"openai", "claude"}
            else None
        ),
    )

    if provider == "ollama":
        return OllamaProvider(provider_config)
    if provider == "openai":
        return OpenAIProvider(provider_config)
    if provider == "claude":
        return ClaudeProvider(provider_config)

    raise ValueError(
        f"Provider LLM '{provider}' não suportado. "
        "Use ollama, openai ou claude."
    )
