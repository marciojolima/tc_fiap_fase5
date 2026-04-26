from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from time import perf_counter

from agent.llm_gateway.providers.base import ProviderChatConfig
from common.logger import get_logger
from monitoring.metrics import finish_llm_chat_provider_call_for_monitor

logger = get_logger("llm.providers.ollama")

HTTP_NOT_FOUND = 404
MODEL_NAMES_PREVIEW = 15


def fetch_ollama_tags_json(base_url: str, timeout_seconds: int = 5) -> dict | None:
    """GET /api/tags — retorna JSON ou None se falhar."""

    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def list_model_names_from_tags(tags_payload: dict) -> list[str]:
    models = tags_payload.get("models") or []
    return [str(item.get("name", "")) for item in models if item.get("name")]


def model_is_available_in_ollama(expected: str, installed_names: list[str]) -> bool:
    """True se o nome pedido existe na lista (ex.: tag :latest)."""

    if expected in installed_names:
        return True
    base = expected.split(":", maxsplit=1)[0]
    for name in installed_names:
        if name == base or name.startswith(base + ":"):
            return True
    return False


def probe_ollama_http(base_url: str, timeout_seconds: int = 5) -> tuple[bool, str]:
    """GET /api/tags — verifica se o daemon Ollama responde nessa base URL."""

    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
            _ = response.read()
            return True, f"HTTP {response.status}"
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}: {exc.reason}"
    except urllib.error.URLError as exc:
        return False, str(exc)


class OllamaProvider:
    """Client para modelos locais/remotos servidos por Ollama."""

    def __init__(self, config: ProviderChatConfig):
        self.config = config
        if not self.config.base_url:
            raise ValueError("OllamaProvider requer base_url configurada.")

    def metadata(self) -> dict[str, object]:
        return {
            "provider": "ollama",
            "model_name": self.config.model_name,
            "base_url": self.config.base_url,
            "timeout_seconds": self.config.timeout_seconds,
        }

    def chat(self, messages: list[dict[str, str]]) -> str:
        provider_start_time = perf_counter()
        logger.info(
            "Iniciando chamada ao provider Ollama | model=%s | base_url=%s | "
            "mensagens=%d",
            self.config.model_name,
            self.config.base_url,
            len(messages),
        )
        payload = json.dumps(
            {
                "model": self.config.model_name,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            url=f"{self.config.base_url.rstrip('/')}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.config.timeout_seconds,
            ) as response:
                parsed = json.loads(response.read().decode("utf-8"))
                logger.info(
                    "Ollama respondeu | model=%s | status=%s | latency=%.3fs",
                    self.config.model_name,
                    response.status,
                    perf_counter() - provider_start_time,
                )
        except urllib.error.HTTPError as exc:
            logger.warning(
                "Falha HTTP ao chamar Ollama | model=%s | status=%s | latency=%.3fs",
                self.config.model_name,
                exc.code,
                perf_counter() - provider_start_time,
            )
            if exc.code == HTTP_NOT_FOUND:
                base_tags = self.config.base_url.rstrip("/")
                raise RuntimeError(
                    f"Modelo '{self.config.model_name}' nao encontrado no Ollama "
                    f"(HTTP {HTTP_NOT_FOUND} em /api/chat). "
                    f"Execute: ollama pull {self.config.model_name} "
                    f"ou alinhe llm.providers.ollama.model_name com um nome listado "
                    f"em GET {base_tags}/api/tags."
                ) from exc
            raise RuntimeError(f"Falha ao chamar o provider LLM: {exc}") from exc
        except urllib.error.URLError as exc:
            logger.warning(
                "Falha de conectividade ao chamar Ollama | model=%s | "
                "latency=%.3fs | erro=%s",
                self.config.model_name,
                perf_counter() - provider_start_time,
                exc,
            )
            raise RuntimeError(f"Falha ao chamar o provider LLM: {exc}") from exc
        except (TimeoutError, socket.timeout) as exc:
            logger.warning(
                "Timeout ao chamar Ollama | model=%s | timeout=%ss | latency=%.3fs",
                self.config.model_name,
                self.config.timeout_seconds,
                perf_counter() - provider_start_time,
            )
            raise RuntimeError(
                "O provider Ollama excedeu o tempo limite da requisição "
                f"({self.config.timeout_seconds}s)."
            ) from exc
        finally:
            finish_llm_chat_provider_call_for_monitor(
                provider_start_time,
                provider="ollama",
            )

        message = parsed.get("message", {})
        content = message.get("content", "")
        if not content:
            logger.warning(
                "Ollama retornou mensagem vazia | model=%s | latency=%.3fs",
                self.config.model_name,
                perf_counter() - provider_start_time,
            )
            raise RuntimeError("Provider LLM retornou resposta vazia.")
        return content

    def status(self) -> dict[str, object]:
        ok, detail = probe_ollama_http(self.config.base_url, timeout_seconds=5)
        tags_payload = fetch_ollama_tags_json(
            self.config.base_url,
            timeout_seconds=5,
        ) if ok else None
        installed = list_model_names_from_tags(tags_payload) if tags_payload else []
        model_ready = bool(tags_payload) and model_is_available_in_ollama(
            self.config.model_name,
            installed,
        )

        hint = ""
        if not ok:
            hint = (
                "Confirme que o daemon Ollama esta rodando e que "
                "llm.providers.ollama.base_url "
                "aponta para a instancia correta."
            )
        elif not model_ready:
            preview = installed[:MODEL_NAMES_PREVIEW]
            truncated = "..." if len(installed) > MODEL_NAMES_PREVIEW else ""
            hint = (
                f"O daemon responde, mas o modelo '{self.config.model_name}' "
                "nao aparece em /api/tags. "
                f"Execute: ollama pull {self.config.model_name}. "
                f"Modelos instalados agora: {preview}{truncated}"
            )

        return {
            "provider": "ollama",
            "reachable": ok,
            "base_url": self.config.base_url,
            "model_name": self.config.model_name,
            "model_ready": model_ready,
            "models_installed": installed[:30],
            "probe_detail": detail,
            "hint": hint,
        }
