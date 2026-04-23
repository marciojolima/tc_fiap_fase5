from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from time import perf_counter

from fastapi import APIRouter, HTTPException

from agent.rag_pipeline import get_rag_runtime_summary
from agent.react_agent import LLMClientProtocol, run_react_agent
from common.config_loader import load_global_config, resolve_ollama_model
from common.logger import get_logger
from monitoring.metrics import (
    finish_llm_chat_ollama_call_for_monitor,
    finish_llm_chat_request_for_monitor,
    start_llm_chat_request_for_monitor,
    start_step_timer_for_monitor,
)
from security.guardrails import InputGuardrail, OutputGuardrail
from serving.schemas import LLMChatRequest, LLMChatResponse

router = APIRouter(prefix="/llm", tags=["llm"])
logger = get_logger("serving.llm_routes")

_DEFAULT_OLLAMA_BASE = "http://127.0.0.1:11434"
HTTP_NOT_FOUND = 404
MODEL_NAMES_PREVIEW = 15


def resolve_ollama_base_url() -> str:
    """Resolve Ollama URL: env overrides YAML (Docker vs host)."""

    for key in ("LLM_BASE_URL", "OLLAMA_BASE_URL"):
        raw = os.environ.get(key, "").strip()
        if raw:
            return raw.rstrip("/")

    cfg = load_global_config().get("llm", {})
    yaml_url = str(cfg.get("base_url", "") or "").strip()
    if yaml_url:
        return yaml_url.rstrip("/")

    return _DEFAULT_OLLAMA_BASE


@dataclass(frozen=True)
class OllamaConfig:
    base_url: str
    model: str
    timeout_seconds: int


class OllamaClient(LLMClientProtocol):
    """Client for an externally served quantized model via Ollama API."""

    def __init__(self, config: OllamaConfig):
        self.config = config

    def metadata(self) -> dict[str, object]:
        return {
            "provider": "ollama",
            "model_name": self.config.model,
            "base_url": self.config.base_url,
            "timeout_seconds": self.config.timeout_seconds,
        }

    def chat(self, messages: list[dict[str, str]]) -> str:
        ollama_start_time = start_step_timer_for_monitor()
        request_start = perf_counter()
        logger.info(
            "Iniciando chamada ao Ollama | model=%s | base_url=%s | mensagens=%d",
            self.config.model,
            self.config.base_url,
            len(messages),
        )
        payload = json.dumps(
            {
                "model": self.config.model,
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
                    self.config.model,
                    response.status,
                    perf_counter() - request_start,
                )
        except urllib.error.HTTPError as exc:
            logger.warning(
                "Falha HTTP ao chamar Ollama | model=%s | status=%s | latency=%.3fs",
                self.config.model,
                exc.code,
                perf_counter() - request_start,
            )
            if exc.code == HTTP_NOT_FOUND:
                base_tags = self.config.base_url.rstrip("/")
                raise RuntimeError(
                    f"Modelo '{self.config.model}' nao encontrado no Ollama "
                    f"(HTTP {HTTP_NOT_FOUND} em /api/chat). "
                    f"No host ou no container ollama, execute: "
                    f"ollama pull {self.config.model} "
                    f"— ou alinhe OLLAMA_MODEL / pipeline_global_config llm.model_name "
                    f"com um nome listado em GET {base_tags}/api/tags."
                ) from exc
            raise RuntimeError(f"Falha ao chamar o LLM externo: {exc}") from exc
        except urllib.error.URLError as exc:
            logger.warning(
                "Falha de conectividade ao chamar Ollama | model=%s | "
                "latency=%.3fs | erro=%s",
                self.config.model,
                perf_counter() - request_start,
                exc,
            )
            raise RuntimeError(f"Falha ao chamar o LLM externo: {exc}") from exc
        except (TimeoutError, socket.timeout) as exc:
            logger.warning(
                "Timeout ao chamar Ollama | model=%s | timeout=%ss | latency=%.3fs",
                self.config.model,
                self.config.timeout_seconds,
                perf_counter() - request_start,
            )
            raise RuntimeError(
                "Ollama excedeu o tempo limite da requisição "
                f"({self.config.timeout_seconds}s)."
            ) from exc
        finally:
            finish_llm_chat_ollama_call_for_monitor(ollama_start_time)

        message = parsed.get("message", {})
        content = message.get("content", "")
        if not content:
            logger.warning(
                "Ollama retornou mensagem vazia | model=%s | latency=%.3fs",
                self.config.model,
                perf_counter() - request_start,
            )
            raise RuntimeError("LLM externo retornou resposta vazia.")
        logger.info(
            "Resposta do Ollama validada | model=%s | chars_resposta=%d",
            self.config.model,
            len(content),
        )
        return content


def _build_ollama_client() -> OllamaClient:
    cfg = load_global_config().get("llm", {})
    model = resolve_ollama_model({"llm": cfg})
    timeout_raw = os.environ.get("LLM_TIMEOUT_SECONDS", "").strip()
    timeout_seconds = (
        int(timeout_raw)
        if timeout_raw.isdigit()
        else int(cfg.get("timeout_seconds", 45))
    )
    return OllamaClient(
        OllamaConfig(
            base_url=resolve_ollama_base_url(),
            model=model,
            timeout_seconds=timeout_seconds,
        )
    )


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


@router.get("/health")
def llm_healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/status")
def llm_status() -> dict[str, object]:
    """Diagnostico: URL resolvida, modelo esperado e se o Ollama responde."""

    base_url = resolve_ollama_base_url()
    cfg = load_global_config().get("llm", {})
    model = resolve_ollama_model({"llm": cfg})
    ok, detail = probe_ollama_http(base_url, timeout_seconds=5)
    tags_payload = fetch_ollama_tags_json(base_url, timeout_seconds=5) if ok else None
    installed = list_model_names_from_tags(tags_payload) if tags_payload else []
    model_ready = bool(tags_payload) and model_is_available_in_ollama(model, installed)

    hint = ""
    if not ok:
        if "127.0.0.1" in base_url or base_url.endswith("localhost:11434"):
            hint = (
                "Se a API roda dentro do Docker, 127.0.0.1/local host apontam para o "
                "container, nao para o PC. Defina "
                "LLM_BASE_URL=http://host.docker.internal:11434 "
                "(Ollama no Windows) ou http://ollama:11434 "
                "(servico ollama no Compose)."
            )
        else:
            hint = (
                "Confirme que o Ollama esta rodando e que o modelo foi baixado "
                f"(`ollama pull {model}`). Teste no host: curl {base_url}/api/tags"
            )
    elif ok and not model_ready:
        preview = installed[:MODEL_NAMES_PREVIEW]
        truncated = "..." if len(installed) > MODEL_NAMES_PREVIEW else ""
        hint = (
            f"O daemon responde, mas o modelo '{model}' nao aparece em /api/tags. "
            f"Isso gera HTTP {HTTP_NOT_FOUND} em /api/chat. Execute: "
            f"ollama pull {model} "
            "(no mesmo ambiente do Ollama que LLM_BASE_URL aponta). "
            f"Modelos instalados agora: {preview}{truncated}"
        )

    return {
        "llm_base_url_resolved": base_url,
        "model_expected": model,
        "ollama_reachable": ok,
        "model_present_in_ollama": model_ready,
        "models_installed": installed[:30],
        "probe_detail": detail,
        "hint_if_unreachable": hint if not ok else "",
        "hint_if_model_missing": hint if ok and not model_ready else "",
        "rag": get_rag_runtime_summary(),
    }


@router.post(
    "/chat",
    response_model=LLMChatResponse,
    description=(
        "Executa uma pergunta no agente ReAct com tools de domínio e RAG sobre o "
        "repositório. Para um smoke test simples, use a pergunta "
        "`Cite pelo menos três ferramentas do agente ReAct ligadas ao domínio do "
        "datathon.`. A resposta esperada deve mencionar `rag_search`, "
        "`predict_churn`, `drift_status` e/ou `scenario_prediction`. "
        "Mantenha `include_trace=true` para depuração."
    ),
)
def chat_with_react_agent(payload: LLMChatRequest) -> LLMChatResponse:
    start_time = start_llm_chat_request_for_monitor()
    status_code = "500"
    try:
        input_guardrail = InputGuardrail()
        output_guardrail = OutputGuardrail()
        is_valid, reason = input_guardrail.validate(payload.message)
        if not is_valid:
            status_code = "400"
            logger.warning(
                "Requisicao POST /llm/chat bloqueada pelo input guardrail | "
                "chars=%d | motivo=%s",
                len(payload.message),
                reason,
            )
            raise HTTPException(status_code=400, detail=reason)

        try:
            logger.info(
                "Recebida requisicao POST /llm/chat | chars=%d | include_trace=%s",
                len(payload.message),
                payload.include_trace,
            )
            result = run_react_agent(payload.message, _build_ollama_client())
        except RuntimeError as exc:
            status_code = "503"
            logger.warning("Falha em /llm/chat | motivo=%s", exc)
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception:
            logger.exception("Falha inesperada em /llm/chat")
            raise

        safe_answer = output_guardrail.sanitize(result.answer)
        status_code = "200"
        logger.info(
            "Resposta /llm/chat pronta | tools=%s | trace_steps=%d | "
            "chars_resposta=%d",
            result.used_tools,
            len(result.trace),
            len(safe_answer),
        )
        return LLMChatResponse(
            answer=safe_answer,
            used_tools=result.used_tools,
            trace=result.trace if payload.include_trace else [],
        )
    finally:
        finish_llm_chat_request_for_monitor(
            start_time,
            method="POST",
            status_code=status_code,
        )
