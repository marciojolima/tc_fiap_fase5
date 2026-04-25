from __future__ import annotations

from fastapi import APIRouter, HTTPException

from agent.llm_gateway.factory import build_llm_client
from agent.rag_pipeline import get_rag_runtime_summary
from agent.react_agent import run_react_agent
from common.config_loader import (
    load_global_config,
    resolve_llm_model_name,
    resolve_llm_provider,
)
from common.logger import get_logger
from monitoring.metrics import (
    finish_llm_chat_request_for_monitor,
    start_llm_chat_request_for_monitor,
)
from security.guardrails import InputGuardrail, OutputGuardrail
from serving.schemas import LLMChatRequest, LLMChatResponse

router = APIRouter(prefix="/llm", tags=["llm"])
logger = get_logger("serving.llm_routes")


@router.get("/health")
def llm_healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/status")
def llm_status() -> dict[str, object]:
    """Diagnostico do provider LLM ativo e do RAG."""

    config = load_global_config()
    provider = resolve_llm_provider(config)
    model_name = resolve_llm_model_name(provider, config)
    client = build_llm_client(config)
    provider_status = client.status()

    return {
        "provider": provider,
        "model_expected": model_name,
        "provider_status": provider_status,
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
            result = run_react_agent(payload.message, build_llm_client())
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
