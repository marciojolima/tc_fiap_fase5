from agent.react_agent import (
    LLMClientProtocol,
    is_documental_question,
    run_react_agent,
)
from agent.tools import AgentTool, build_default_tools


class StubLLMClient(LLMClientProtocol):
    def __init__(self):
        self.calls = 0

    def metadata(self) -> dict[str, str]:
        _ = self.calls
        return {"provider": "stub", "model_name": "stub-react"}

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.calls += 1
        if self.calls == 1:
            return (
                '{"thought":"preciso buscar contexto",'
                '"action":"fake_tool","action_input":"churn germany"}'
            )
        return (
            '{"thought":"agora sei responder",'
            '"final_answer":"Cliente com alto risco de churn."}'
        )


def test_react_agent_executes_tool_then_returns_final_answer() -> None:
    tool = AgentTool(
        name="fake_tool",
        description="Ferramenta de teste",
        run=lambda _: "contexto encontrado",
    )
    aux_tool_1 = AgentTool(
        name="aux_tool_1",
        description="Auxiliar",
        run=lambda _: "ok",
    )
    aux_tool_2 = AgentTool(
        name="aux_tool_2",
        description="Auxiliar",
        run=lambda _: "ok",
    )
    result = run_react_agent(
        "Qual o risco desse perfil?",
        llm_client=StubLLMClient(),
        tools=[tool, aux_tool_1, aux_tool_2],
        max_iterations=3,
    )
    assert "alto risco" in result.answer.lower()
    assert result.used_tools
    assert result.trace[0]["llm_metadata"]["model_name"] == "stub-react"


class InvalidJsonThenValidLLMClient(LLMClientProtocol):
    def __init__(self):
        self.calls = 0

    def metadata(self) -> dict[str, str]:
        _ = self.calls
        return {"provider": "stub", "model_name": "stub-invalid-json"}

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.calls += 1
        if self.calls == 1:
            return '{"thought":"quebrado"}\n{"thought":"segundo json"}'
        return (
            '{"thought":"agora sim","action":"fake_tool","action_input":"churn"}'
        )


def test_react_agent_retries_when_llm_returns_invalid_json() -> None:
    tool = AgentTool(
        name="fake_tool",
        description="Ferramenta de teste",
        run=lambda _: "contexto encontrado",
    )
    aux_tool_1 = AgentTool(
        name="aux_tool_1",
        description="Auxiliar",
        run=lambda _: "ok",
    )
    aux_tool_2 = AgentTool(
        name="aux_tool_2",
        description="Auxiliar",
        run=lambda _: "ok",
    )
    result = run_react_agent(
        "Explique churn",
        llm_client=InvalidJsonThenValidLLMClient(),
        tools=[tool, aux_tool_1, aux_tool_2],
        max_iterations=2,
    )

    assert result.used_tools == ["fake_tool"]
    assert result.trace[0]["parse_status"] == "json_invalid"
    assert "raw_llm_output" in result.trace[0]


def test_default_predict_churn_tool_mentions_predict_raw_contract() -> None:
    tools = {tool.name: tool for tool in build_default_tools()}

    assert "predict_churn" in tools
    assert "/predict/raw" in tools["predict_churn"].description


def test_is_documental_question_detects_repository_grounded_prompt() -> None:
    assert is_documental_question(
        "Quais rotas HTTP o projeto expõe para o assistente LLM?"
    )
    assert not is_documental_question(
        "Avalie este payload de churn para um cliente novo."
    )


class DocumentaryQuestionLLMClient(LLMClientProtocol):
    def __init__(self):
        self.calls = 0

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.calls += 1
        if self.calls == 1:
            return (
                '{"thought":"vou tentar predição",'
                '"action":"predict_churn","action_input":"{}"}'
            )
        return (
            '{"thought":"agora respondo com base no repositório",'
            '"final_answer":"As rotas são /llm/health, /llm/status e /llm/chat."}'
        )


def test_react_agent_restricts_documental_questions_to_rag_search() -> None:
    rag_tool = AgentTool(
        name="rag_search",
        description="Busca documental",
        run=lambda _: "rotas /llm/health /llm/status /llm/chat",
    )
    predict_tool = AgentTool(
        name="predict_churn",
        description="Predição",
        run=lambda _: "não deveria executar",
    )
    aux_tool = AgentTool(
        name="scenario_prediction",
        description="Cenário",
        run=lambda _: "não deveria executar",
    )
    result = run_react_agent(
        "Quais rotas HTTP o projeto expõe especificamente para o assistente LLM?",
        llm_client=DocumentaryQuestionLLMClient(),
        tools=[rag_tool, predict_tool, aux_tool],
        max_iterations=2,
    )

    assert "llm/health" in result.answer
    assert result.used_tools == []
    assert result.trace[0]["question_mode"] == "documental"
    assert "Ferramenta 'predict_churn' inválida" in result.trace[0]["observation"]
