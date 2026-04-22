from agent.react_agent import AgentRunResult
from serving.app import create_app
from serving.llm_routes import chat_with_react_agent
from serving.schemas import LLMChatRequest


def test_app_registers_llm_routes() -> None:
    app = create_app()
    paths = {getattr(route, "path", "") for route in app.routes}
    assert "/llm/chat" in paths
    assert "/llm/health" in paths
    assert "/llm/status" in paths


def test_chat_with_react_agent_returns_structured_response(monkeypatch) -> None:
    monkeypatch.setattr(
        "serving.llm_routes.run_react_agent",
        lambda *_args, **_kwargs: AgentRunResult(
            answer="Resposta final.",
            trace=[{"iteration": 1, "action": "rag_search"}],
            used_tools=["rag_search"],
        ),
    )
    response = chat_with_react_agent(
        LLMChatRequest(message="Explique churn", include_trace=True)
    )
    assert response.answer == "Resposta final."
    assert response.used_tools == ["rag_search"]
    assert response.trace
