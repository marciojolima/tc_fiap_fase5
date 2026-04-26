from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ProviderChatConfig:
    provider: str
    model_name: str
    timeout_seconds: int
    base_url: str | None = None
    api_key: str | None = None
    max_tokens: int | None = None


class LLMProvider(Protocol):
    def chat(self, messages: list[dict[str, str]]) -> str:
        """Executa uma chamada de chat e retorna apenas o texto final."""

    def metadata(self) -> dict[str, object]:
        """Metadados operacionais do provider ativo."""

    def status(self) -> dict[str, object]:
        """Status diagnóstico do provider."""
