import json
from types import SimpleNamespace

from agent import tools as agent_tools
from agent.react_agent import (
    LLMClientProtocol,
    is_comparative_scenario_question,
    is_documental_question,
    run_react_agent,
)
from agent.tools import AgentTool, build_default_tools

SECOND_CALL = 2
PREDICTION_THRESHOLD = 0.5


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


class StructuredActionInputLLMClient(LLMClientProtocol):
    def chat(self, messages: list[dict[str, str]]) -> str:
        _ = self
        _ = messages
        return json.dumps(
            {
                "thought": "vou simular um cenário",
                "action": "scenario_prediction",
                "action_input": {
                    "baseline_scenario": {"Age": 45, "NumOfProducts": 1},
                    "improved_scenario": {"Age": 45, "NumOfProducts": 3},
                },
            },
            ensure_ascii=False,
        )


class PromptCaptureLLMClient(LLMClientProtocol):
    def __init__(self):
        self.prompt = ""

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.prompt = messages[-1]["content"]
        return (
            '{"thought":"resposta direta",'
            '"final_answer":"Resumo curto do comportamento solicitado."}'
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


def test_react_agent_serializes_structured_action_input_as_json() -> None:
    captured_inputs: list[str] = []
    tool = AgentTool(
        name="scenario_prediction",
        description="Ferramenta de teste de cenário",
        run=lambda raw_input: captured_inputs.append(raw_input) or "ok",
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
        "Simule um cenário com mais produtos.",
        llm_client=StructuredActionInputLLMClient(),
        tools=[tool, aux_tool_1, aux_tool_2],
        max_iterations=1,
    )

    assert result.used_tools == ["scenario_prediction"]
    assert captured_inputs == [
        (
            '{"baseline_scenario": {"Age": 45, "NumOfProducts": 1}, '
            '"improved_scenario": {"Age": 45, "NumOfProducts": 3}}'
        )
    ]


def test_react_agent_includes_answer_style_instruction_in_prompt() -> None:
    client = PromptCaptureLLMClient()
    tool = AgentTool(
        name="fake_tool",
        description="Ferramenta de teste",
        run=lambda _: "ok",
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
        "Avalie este perfil de churn.",
        llm_client=client,
        tools=[tool, aux_tool_1, aux_tool_2],
        max_iterations=1,
        answer_style="short",
    )

    assert result.answer == "Resumo curto do comportamento solicitado."
    assert "Estilo da resposta final: short" in client.prompt
    assert "no máximo 3 bullets" in client.prompt
    assert result.trace[0]["answer_style"] == "short"


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


def test_is_comparative_scenario_question_detects_two_scenarios() -> None:
    assert is_comparative_scenario_question(
        "Simule dois cenários para este cliente: um com saldo maior e "
        "inatividade, outro com mais produtos e atividade."
    )
    assert not is_comparative_scenario_question(
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


class ComparativeScenarioRecoveryLLMClient(LLMClientProtocol):
    def __init__(self):
        self.calls = 0

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.calls += 1
        if self.calls == 1:
            return (
                '{"thought":"vou testar um cenário inicial",'
                '"action":"scenario_prediction","action_input":{"Age":45}}'
            )
        if self.calls == SECOND_CALL:
            return json.dumps(
                {
                    "thought": "vou comparar baseline e improved",
                    "action": "scenario_prediction",
                    "action_input": {
                        "baseline_scenario": {"Age": 45, "NumOfProducts": 1},
                        "improved_scenario": {"Age": 45, "NumOfProducts": 3},
                    },
                },
                ensure_ascii=False,
            )
        return (
            '{"thought":"agora consigo concluir",'
            '"final_answer":"O cenário ajustado reduz o risco de churn em relação '
            'ao cenário base e é melhor para retenção."}'
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


def test_react_agent_requires_comparative_scenario_evidence() -> None:
    def scenario_tool(raw_input: str) -> str:
        payload = json.loads(raw_input)
        if "baseline_scenario" in payload:
            return json.dumps(
                {
                    "tool_name": "scenario_prediction",
                    "status": "ok",
                    "result": {
                        "baseline_scenario": {
                            "churn_probability": 0.71,
                            "churn_prediction": 1,
                        },
                        "improved_scenario": {
                            "churn_probability": 0.33,
                            "churn_prediction": 0,
                        },
                        "probability_delta": -0.38,
                        "better_scenario_for_retention": "improved_scenario",
                    },
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "tool_name": "scenario_prediction",
                "status": "ok",
                "result": {
                    "churn_probability": 0.62,
                    "churn_prediction": 1,
                },
            },
            ensure_ascii=False,
        )

    scenario = AgentTool(
        name="scenario_prediction",
        description="Cenário",
        run=scenario_tool,
    )
    aux_tool_1 = AgentTool(
        name="predict_churn",
        description="Predição",
        run=lambda _: "não deveria executar",
    )
    aux_tool_2 = AgentTool(
        name="rag_search",
        description="Busca",
        run=lambda _: "não deveria executar",
    )

    result = run_react_agent(
        "Simule dois cenários para este cliente: um com saldo maior e "
        "inatividade, outro com mais produtos e atividade. Qual deles parece "
        "melhor para retenção?",
        llm_client=ComparativeScenarioRecoveryLLMClient(),
        tools=[scenario, aux_tool_1, aux_tool_2],
        max_iterations=3,
    )

    assert "melhor para retenção" in result.answer
    assert result.used_tools == ["scenario_prediction", "scenario_prediction"]
    assert "baseline_scenario" in result.trace[0]["observation"]


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
    assert payload["retrieved_contexts"]


def test_predict_churn_tool_returns_structured_output(monkeypatch) -> None:
    cfg = SimpleNamespace(threshold=0.5, model_name="modelo_teste")
    monkeypatch.setattr(
        "agent.tools.load_serving_config",
        lambda model_name="current": cfg,
    )
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
    status_dir = tmp_path / "artifacts" / "evaluation" / "model" / "drift"
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


def test_scenario_prediction_tool_compares_two_scenarios(monkeypatch) -> None:
    def fake_run_scenario_prediction(scenario):
        if scenario.name.endswith("_baseline"):
            probability = 0.71
        else:
            probability = 0.33
        return type(
            "ScenarioResult",
            (),
            {
                "_asdict": lambda self: {
                    "scenario_name": scenario.name,
                    "churn_probability": probability,
                    "churn_prediction": int(probability >= PREDICTION_THRESHOLD),
                    "threshold": PREDICTION_THRESHOLD,
                    "model_name": "modelo_teste",
                    "run_name": "run_teste",
                }
            },
        )()

    monkeypatch.setattr(
        "agent.tools.run_scenario_prediction",
        fake_run_scenario_prediction,
    )

    payload = json.loads(
        agent_tools._scenario_prediction_tool(
            json.dumps(
                {
                    "baseline_scenario": {"Age": 45, "NumOfProducts": 1},
                    "improved_scenario": {"Age": 45, "NumOfProducts": 3},
                    "comparison_description": "mais produtos e mais atividade",
                },
                ensure_ascii=False,
            )
        )
    )

    assert payload["tool_name"] == "scenario_prediction"
    assert payload["status"] == "ok"
    assert payload["result"]["baseline_scenario"]["churn_prediction"] == 1
    assert payload["result"]["improved_scenario"]["churn_prediction"] == 0
    assert payload["result"]["better_scenario_for_retention"] == "improved_scenario"


def test_scenario_prediction_tool_parses_natural_language_comparison(
    monkeypatch,
) -> None:
    expected_age = 48
    expected_credit_score = 610
    expected_balance = 125000.0
    expected_improved_products = 3
    captured_payloads: dict[str, dict[str, object]] = {}

    def fake_run_scenario_prediction(scenario):
        captured_payloads[scenario.name] = scenario.payload
        probability = 0.71 if scenario.name.endswith("_baseline") else 0.33
        return type(
            "ScenarioResult",
            (),
            {
                "_asdict": lambda self: {
                    "scenario_name": scenario.name,
                    "churn_probability": probability,
                    "churn_prediction": int(probability >= PREDICTION_THRESHOLD),
                    "threshold": PREDICTION_THRESHOLD,
                    "model_name": "modelo_teste",
                    "run_name": "run_teste",
                }
            },
        )()

    monkeypatch.setattr(
        "agent.tools.run_scenario_prediction",
        fake_run_scenario_prediction,
    )

    payload = json.loads(
        agent_tools._scenario_prediction_tool(
            (
                "Use scenario_prediction com baseline_scenario: cliente de 48 "
                "anos, Alemanha, credit score 610, saldo 125000, 1 produto e "
                "inativo. improved_scenario: mesmo cliente, mas com 3 produtos "
                "e ativo."
            )
        )
    )

    assert payload["status"] == "ok"
    assert payload["result"]["better_scenario_for_retention"] == "improved_scenario"
    baseline = captured_payloads["agent_scenario_baseline"]
    improved = captured_payloads["agent_scenario_improved"]
    assert baseline["Age"] == expected_age
    assert baseline["Geography"] == "Germany"
    assert baseline["CreditScore"] == expected_credit_score
    assert baseline["Balance"] == expected_balance
    assert baseline["NumOfProducts"] == 1
    assert baseline["IsActiveMember"] == 0
    assert improved["Age"] == expected_age
    assert improved["Geography"] == "Germany"
    assert improved["CreditScore"] == expected_credit_score
    assert improved["Balance"] == expected_balance
    assert improved["NumOfProducts"] == expected_improved_products
    assert improved["IsActiveMember"] == 1


def test_predict_churn_tool_accepts_python_literal_like_payload(monkeypatch) -> None:
    cfg = SimpleNamespace(threshold=0.5, model_name="modelo_teste")
    monkeypatch.setattr(
        "agent.tools.load_serving_config",
        lambda model_name="current": cfg,
    )
    monkeypatch.setattr("agent.tools.prepare_inference_dataframe", lambda *_args: "df")
    monkeypatch.setattr(
        "agent.tools.predict_from_dataframe_with_config",
        lambda *_args: (0.212345, 0),
    )

    payload = json.loads(
        agent_tools._predict_churn_tool(
            "{'CreditScore': 600, 'Geography': 'France', 'Gender': 'Female', "
            "'Age': 40, 'Tenure': 5, 'Balance': 0.0, 'NumOfProducts': 1, "
            "'HasCrCard': 1, 'IsActiveMember': 1, 'EstimatedSalary': 50000.0, "
            "'Card Type': 'SILVER', 'Point Earned': 100}"
        )
    )

    assert payload["tool_name"] == "predict_churn"
    assert payload["status"] == "ok"
    assert payload["result"]["churn_prediction"] == 0
