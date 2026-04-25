from __future__ import annotations

from time import perf_counter

from openai import OpenAI

from agent.llm_gateway.providers.base import ProviderChatConfig
from common.logger import get_logger
from monitoring.metrics import finish_llm_chat_provider_call_for_monitor

logger = get_logger("llm.providers.openai")


class OpenAIProvider:
    """Client para modelos servidos pela API da OpenAI."""

    def __init__(self, config: ProviderChatConfig):
        self.config = config
        if not self.config.api_key:
            raise ValueError("OpenAIProvider requer api_key configurada.")
        client_kwargs: dict[str, object] = {
            "api_key": self.config.api_key,
            "timeout": float(self.config.timeout_seconds),
        }
        if self.config.base_url:
            client_kwargs["base_url"] = self.config.base_url
        self.client = OpenAI(**client_kwargs)

    def metadata(self) -> dict[str, object]:
        return {
            "provider": "openai",
            "model_name": self.config.model_name,
            "base_url": self.config.base_url,
            "timeout_seconds": self.config.timeout_seconds,
        }

    def chat(self, messages: list[dict[str, str]]) -> str:
        provider_start_time = perf_counter()
        logger.info(
            "Iniciando chamada ao provider OpenAI | model=%s | mensagens=%d",
            self.config.model_name,
            len(messages),
        )
        request_kwargs: dict[str, object] = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": 0,
        }
        if self.config.max_tokens is not None:
            request_kwargs["max_tokens"] = self.config.max_tokens
        try:
            response = self.client.chat.completions.create(**request_kwargs)
        except Exception as exc:
            logger.warning(
                "Falha ao chamar OpenAI | model=%s | erro=%s",
                self.config.model_name,
                exc,
            )
            raise RuntimeError(f"Falha ao chamar o provider LLM: {exc}") from exc
        finally:
            finish_llm_chat_provider_call_for_monitor(
                provider_start_time,
                provider="openai",
            )

        content = response.choices[0].message.content if response.choices else ""
        if not content:
            raise RuntimeError("Provider LLM retornou resposta vazia.")
        return content

    def status(self) -> dict[str, object]:
        return {
            "provider": "openai",
            "reachable": None,
            "base_url": self.config.base_url,
            "model_name": self.config.model_name,
            "api_key_configured": bool(self.config.api_key),
            "hint": (
                "Provider externo configurado. Valide OPENAI_API_KEY no .env "
                "e o model_name "
                "em llm.providers.openai.model_name."
            ),
        }
