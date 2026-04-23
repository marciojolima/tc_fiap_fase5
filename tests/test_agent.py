import json
from types import SimpleNamespace

from agent import tools as agent_tools
from agent.react_agent import (
    LLMClientProtocol,
    is_documental_question,
    run_react_agent,
)
from agent.tools import AgentTool, build_default_tools

SECOND_CALL = 2


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
        if self.calls == SECOND_CALL:
            return (
                '{"thought":"vou buscar evidência documental",'
                '"action":"rag_search","action_input":"rotas llm"}'
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
        max_iterations=3,
    )

    assert "llm/health" in result.answer
    assert result.used_tools == ["rag_search"]
    assert result.trace[0]["question_mode"] == "documental"
    assert "Ferramenta 'predict_churn' inválida" in result.trace[0]["observation"]


class DocumentaryFinalAnswerWithoutEvidenceLLMClient(LLMClientProtocol):
    def __init__(self):
        self.calls = 0

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.calls += 1
        if self.calls == 1:
            return (
                '{"thought":"acho que já sei",'
                '"final_answer":"As ferramentas são rag_search, '
                'predict_churn e drift_status."}'
            )
        if self.calls == SECOND_CALL:
            return (
                '{"thought":"vou buscar no repositório",'
                '"action":"rag_search","action_input":"ferramentas react datathon"}'
            )
        return (
            '{"thought":"agora posso concluir",'
            '"final_answer":"As ferramentas incluem rag_search, '
            'predict_churn, drift_status e scenario_prediction."}'
        )


def test_react_agent_requires_rag_search_before_documental_final_answer() -> None:
    rag_tool = AgentTool(
        name="rag_search",
        description="Busca documental",
        run=lambda _: (
            "Ferramentas: rag_search, predict_churn, drift_status, "
            "scenario_prediction"
        ),
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
        "Cite pelo menos três ferramentas do agente ReAct ligadas ao domínio "
        "do datathon.",
        llm_client=DocumentaryFinalAnswerWithoutEvidenceLLMClient(),
        tools=[rag_tool, predict_tool, aux_tool],
        max_iterations=3,
    )

    assert "scenario_prediction" in result.answer
    assert result.used_tools == ["rag_search"]
    assert (
        result.trace[0]["parse_status"] == "missing_documentary_evidence"
    )
    assert "use rag_search antes da resposta final" in result.trace[0]["observation"]


def test_rag_search_tool_returns_structured_evidence(monkeypatch) -> None:
    monkeypatch.setattr(
        "agent.tools.retrieve_contexts",
        lambda *_args, **_kwargs: [
            (
                "README.md\nAs rotas /llm/health, /llm/status e /llm/chat "
                "fazem parte da API."
            )
        ],
    )
    payload = json.loads(
        agent_tools._rag_search_tool(
            "Quais rotas HTTP o projeto expõe especificamente para o assistente LLM?"
        )
    )

    assert payload["tool_name"] == "rag_search"
    assert payload["status"] == "ok"
    assert payload["sources"] == ["README.md"]
    assert payload["evidence"]


def test_predict_churn_tool_returns_structured_output(monkeypatch) -> None:
    cfg = SimpleNamespace(threshold=0.5, model_name="modelo_teste")
    monkeypatch.setattr("agent.tools.load_serving_config", lambda: cfg)
    monkeypatch.setattr("agent.tools.prepare_inference_dataframe", lambda *_args: "df")
    monkeypatch.setattr(
        "agent.tools.predict_from_dataframe_with_config",
        lambda *_args: (0.812345, 1),
    )
    payload = json.loads(
        agent_tools._predict_churn_tool(
            '{"CreditScore": 600, "Geography": "France", "Gender": "Female", '
            '"Age": 40, "Tenure": 5, "Balance": 0.0, "NumOfProducts": 1, '
            '"HasCrCard": 1, "IsActiveMember": 1, "EstimatedSalary": 50000.0, '
            '"Card Type": "SILVER", "Point Earned": 100}'
        )
    )

    assert payload["tool_name"] == "predict_churn"
    assert payload["status"] == "ok"
    assert payload["result"]["churn_prediction"] == 1
    assert payload["evidence"]


def test_drift_status_tool_returns_structured_output(tmp_path, monkeypatch) -> None:
    status_dir = tmp_path / "artifacts" / "monitoring" / "drift"
    status_dir.mkdir(parents=True, exist_ok=True)
    (status_dir / "drift_status.json").write_text(
        '{"status": "critical", "message": "PSI acima do limite"}',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    payload = json.loads(agent_tools._drift_status_tool(""))

    assert payload["tool_name"] == "drift_status"
    assert payload["status"] == "ok"
    assert payload["result"]["drift_status"] == "critical"
    assert payload["evidence"]


def test_scenario_prediction_tool_returns_structured_output(monkeypatch) -> None:
    monkeypatch.setattr(
        "agent.tools.run_scenario_prediction",
        lambda _scenario: type(
            "ScenarioResult",
            (),
            {
                "_asdict": lambda self: {
                    "churn_probability": 0.62,
                    "churn_prediction": 1,
                }
            },
        )(),
    )
    payload = json.loads(agent_tools._scenario_prediction_tool('{"Age": 45}'))

    assert payload["tool_name"] == "scenario_prediction"
    assert payload["status"] == "ok"
    assert payload["result"]["churn_prediction"] == 1
    assert payload["evidence"]
