from agent.react_agent import LLMClientProtocol, run_react_agent
from agent.tools import AgentTool


class StubLLMClient(LLMClientProtocol):
    def __init__(self):
        self.calls = 0

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
