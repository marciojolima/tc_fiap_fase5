from __future__ import annotations

import json
import re

import pytest
from prometheus_client import generate_latest

from agent.llm_gateway.providers.base import ProviderChatConfig
from agent.llm_gateway.providers.ollama import OllamaProvider
from agent.react_agent import AgentRunResult
from agent.tools import AgentTool
from serving.app import create_app
from serving.llm_routes import chat_with_react_agent
from serving.schemas import LLMChatRequest

CANONICAL_CASES = (
    {
        "question": (
            "Quais rotas HTTP o projeto expõe especificamente para o "
            "assistente LLM e diagnóstico do provider LLM?"
        ),
        "evidence": (
            "Rotas do prefixo /llm disponíveis no projeto: /llm/health, "
            "/llm/status e /llm/chat."
        ),
        "expected_answer": "As rotas são /llm/health, /llm/status e /llm/chat.",
        "required_terms": ("/llm/health", "/llm/status", "/llm/chat"),
    },
    {
        "question": (
            "Cite pelo menos três ferramentas do agente ReAct ligadas ao "
            "domínio do datathon."
        ),
        "evidence": (
            "Tools do agente: rag_search, predict_churn, drift_status e "
            "scenario_prediction."
        ),
        "expected_answer": (
            "As tools incluem rag_search, predict_churn, drift_status e "
            "scenario_prediction."
        ),
        "required_terms": (
            "rag_search",
            "predict_churn",
            "drift_status",
            "scenario_prediction",
        ),
    },
    {
        "question": (
            "Em linhas gerais, como o RAG do projeto obtém contexto para uma "
            "pergunta?"
        ),
        "evidence": (
            "O RAG descobre conteúdo em README.md, docs/**/*.md e JSON "
            "relevantes; gera embeddings, faz busca vetorial em memória e "
            "retorna trechos com fonte."
        ),
        "expected_answer": (
            "O RAG lê README.md, docs/**/*.md e JSON relevantes, gera "
            "embeddings, faz busca vetorial em memória e retorna trechos com fonte."
        ),
        "required_terms": (
            "readme.md",
            "docs/**/*.md",
            "json",
            "embeddings",
            "busca vetorial",
            "fonte",
        ),
    },
    {
        "question": (
            "O que o monitoramento de drift busca identificar neste repositório?"
        ),
        "evidence": (
            "O monitoramento compara distribuições, calcula PSI, registra "
            "artefatos de monitoramento e apoia decisão de retreino."
        ),
        "expected_answer": (
            "Ele busca identificar drift com PSI, gerar artefatos de "
            "monitoramento e apoiar decisão de retreino."
        ),
        "required_terms": ("psi", "artefatos", "retreino"),
    },
)


class CanonicalQuestionLLMClient:
    def __init__(self, cases: tuple[dict[str, object], ...]):
        self.cases = cases
        self.current_case: dict[str, object] | None = None
        self.calls = 0

    @staticmethod
    def metadata() -> dict[str, str]:
        return {"provider": "stub", "model_name": "canonical-smoke"}

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.calls += 1
        prompt = messages[-1]["content"]
        if self.current_case is None:
            for case in self.cases:
                question = str(case["question"])
                if question in prompt:
                    self.current_case = case
                    return json.dumps(
                        {
                            "thought": "vou buscar evidências no repositório",
                            "action": "rag_search",
                            "action_input": question,
                        },
                        ensure_ascii=False,
                    )
            return (
                '{"thought":"pergunta desconhecida",'
                '"final_answer":"Não reconheci a pergunta."}'
            )

        required_terms = tuple(
            str(term) for term in self.current_case["required_terms"]
        )
        lowered_prompt = prompt.lower()
        if all(term in lowered_prompt for term in required_terms):
            final_answer = str(self.current_case["expected_answer"])
        else:
            final_answer = "Não encontrei evidências suficientes no repositório."
        self.current_case = None
        return json.dumps(
            {
                "thought": "agora consigo responder com base nas evidências",
                "final_answer": final_answer,
            },
            ensure_ascii=False,
        )


def _canonical_tools() -> list[AgentTool]:
    evidence_by_question = {
        str(case["question"]): str(case["evidence"]) for case in CANONICAL_CASES
    }

    def rag_tool_run(question: str) -> str:
        evidence = evidence_by_question[question]
        return (
            '{\n'
            '  "tool_name": "rag_search",\n'
            '  "status": "ok",\n'
            f'  "query": "{question}",\n'
            '  "evidence": [\n'
            f'    "{evidence}"\n'
            "  ],\n"
            '  "sources": ["docs/RAG_EXPLANATION.md"],\n'
            '  "confidence": "alta"\n'
            "}"
        )

    return [
        AgentTool(
            name="rag_search",
            description="Busca contexto documental do projeto.",
            run=rag_tool_run,
        ),
        AgentTool(
            name="predict_churn",
            description="Predição de churn.",
            run=lambda _: '{"tool_name":"predict_churn","status":"ok"}',
        ),
        AgentTool(
            name="drift_status",
            description="Leitura de drift.",
            run=lambda _: '{"tool_name":"drift_status","status":"ok"}',
        ),
        AgentTool(
            name="scenario_prediction",
            description="Simulação de cenário.",
            run=lambda _: '{"tool_name":"scenario_prediction","status":"ok"}',
        ),
    ]


def _stub_llm_client() -> object:
    return object()


def test_app_registers_llm_routes() -> None:
    app = create_app()
    paths = {getattr(route, "path", "") for route in app.routes}
    assert "/llm/chat" in paths
    assert "/llm/health" in paths
    assert "/llm/status" in paths


def test_chat_with_react_agent_returns_structured_response(monkeypatch) -> None:
    captured_kwargs: dict[str, object] = {}

    def fake_run_react_agent(*_args: object, **kwargs: object) -> AgentRunResult:
        captured_kwargs.update(kwargs)
        return AgentRunResult(
            answer="Resposta final.",
            trace=[{"iteration": 1, "action": "rag_search"}],
            used_tools=["rag_search"],
        )

    monkeypatch.setattr(
        "serving.llm_routes.build_llm_client",
        _stub_llm_client,
    )
    monkeypatch.setattr(
        "serving.llm_routes.run_react_agent",
        fake_run_react_agent,
    )
    response = chat_with_react_agent(
        LLMChatRequest(
            message="Explique churn",
            include_trace=True,
            answer_style="short",
        )
    )
    assert response.answer == "Resposta final."
    assert response.used_tools == ["rag_search"]
    assert response.trace
    assert captured_kwargs["answer_style"] == "short"


def test_llm_chat_request_defaults_to_medium_answer_style() -> None:
    payload = LLMChatRequest(message="Explique churn")

    assert payload.answer_style == "medium"


def test_chat_with_react_agent_updates_llm_metrics(monkeypatch) -> None:
    monkeypatch.setattr(
        "serving.llm_routes.build_llm_client",
        _stub_llm_client,
    )
    monkeypatch.setattr(
        "serving.llm_routes.run_react_agent",
        lambda *_args, **_kwargs: AgentRunResult(
            answer="Resposta final.",
            trace=[],
            used_tools=["rag_search"],
        ),
    )

    response = chat_with_react_agent(
        LLMChatRequest(message="Explique churn", include_trace=False)
    )
    metrics_payload = generate_latest().decode("utf-8")

    assert response.answer == "Resposta final."
    assert (
        'churn_serving_llm_chat_requests_total{method="POST",status_code="200"}'
        in metrics_payload
    )


def test_ollama_provider_translates_timeout_to_runtime_error(monkeypatch) -> None:
    def fake_urlopen(*_args: object, **_kwargs: object) -> object:
        raise TimeoutError("timed out")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = OllamaProvider(
        ProviderChatConfig(
            provider="ollama",
            base_url="http://ollama:11434",
            model_name="qwen2.5:3b",
            timeout_seconds=45,
        )
    )

    with pytest.raises(RuntimeError, match="tempo limite"):
        client.chat([{"role": "user", "content": "teste"}])


@pytest.mark.parametrize("case", CANONICAL_CASES)
def test_chat_with_react_agent_answers_canonical_precision_questions(
    monkeypatch,
    case: dict[str, object],
) -> None:
    monkeypatch.setattr(
        "serving.llm_routes.build_llm_client",
        lambda: CanonicalQuestionLLMClient(CANONICAL_CASES),
    )
    monkeypatch.setattr("agent.react_agent.build_default_tools", _canonical_tools)

    response = chat_with_react_agent(
        LLMChatRequest(message=str(case["question"]), include_trace=True)
    )

    for term in case["required_terms"]:
        assert str(term) in response.answer.lower()
    assert response.used_tools == ["rag_search"]
    assert response.trace
    assert response.trace[0]["question_mode"] == "documental"
    assert response.trace[0]["action"] == "rag_search"
    assert re.search(r"rag_search", response.trace[0]["raw_llm_output"])
