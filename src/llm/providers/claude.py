from __future__ import annotations

from importlib import import_module
from time import perf_counter

from common.logger import get_logger
from llm.providers.base import ProviderChatConfig
from monitoring.metrics import finish_llm_chat_provider_call_for_monitor

logger = get_logger("llm.providers.claude")


def _merge_text_blocks(content_blocks: list[object]) -> str:
    parts: list[str] = []
    for block in content_blocks:
        text = getattr(block, "text", "")
        if text:
            parts.append(str(text))
    return "\n".join(parts).strip()


class ClaudeProvider:
    """Client para modelos servidos pela API da Anthropic."""

    def __init__(self, config: ProviderChatConfig):
        self.config = config
        if not self.config.api_key:
            raise ValueError("ClaudeProvider requer api_key configurada.")
        try:
            anthropic_module = import_module("anthropic")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Dependencia 'anthropic' nao instalada para usar o provider Claude."
            ) from exc
        anthropic_client_cls = getattr(anthropic_module, "Anthropic")
        client_kwargs: dict[str, object] = {"api_key": self.config.api_key}
        if self.config.base_url:
            client_kwargs["base_url"] = self.config.base_url
        self.client = anthropic_client_cls(**client_kwargs)

    def metadata(self) -> dict[str, object]:
        return {
            "provider": "claude",
            "model_name": self.config.model_name,
            "base_url": self.config.base_url,
            "timeout_seconds": self.config.timeout_seconds,
        }

    def chat(self, messages: list[dict[str, str]]) -> str:
        provider_start_time = perf_counter()
        system_messages = [
            message["content"]
            for message in messages
            if message.get("role") == "system"
        ]
        conversation_messages = [
            {
                "role": "assistant" if message.get("role") == "assistant" else "user",
                "content": message.get("content", ""),
            }
            for message in messages
            if message.get("role") != "system"
        ]
        if not conversation_messages:
            raise RuntimeError(
                "Nenhuma mensagem de usuario disponivel para o provider Claude."
            )

        request_kwargs: dict[str, object] = {
            "model": self.config.model_name,
            "messages": conversation_messages,
            "max_tokens": self.config.max_tokens or 1024,
            "temperature": 0,
        }
        system_prompt = "\n".join(system_messages).strip()
        if system_prompt:
            request_kwargs["system"] = system_prompt
        try:
            response = self.client.messages.create(**request_kwargs)
        except Exception as exc:
            logger.warning(
                "Falha ao chamar Claude | model=%s | erro=%s",
                self.config.model_name,
                exc,
            )
            raise RuntimeError(f"Falha ao chamar o provider LLM: {exc}") from exc
        finally:
            finish_llm_chat_provider_call_for_monitor(
                provider_start_time,
                provider="claude",
            )

        content = _merge_text_blocks(list(response.content))
        if not content:
            raise RuntimeError("Provider LLM retornou resposta vazia.")
        return content

    def status(self) -> dict[str, object]:
        return {
            "provider": "claude",
            "reachable": None,
            "base_url": self.config.base_url,
            "model_name": self.config.model_name,
            "api_key_configured": bool(self.config.api_key),
            "hint": (
                "Provider externo configurado. Valide ANTHROPIC_API_KEY no .env "
                "e o model_name "
                "em llm.providers.claude.model_name."
            ),
        }
