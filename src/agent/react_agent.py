"""ReAct agent with tool use for Datathon workflows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from agent.tools import AgentTool, build_default_tools
from common.config_loader import load_global_config
from common.logger import get_logger
from security.guardrails import InputGuardrail, OutputGuardrail

logger = get_logger("agent.react_agent")

REACT_SYSTEM_PROMPT = """Você é um assistente especialista no projeto de churn.
Você pode usar ferramentas para responder com precisão.

Você deve responder SEMPRE em JSON válido e sem texto extra:
1) Para chamar ferramenta:
{"thought":"curto raciocínio","action":"tool_name","action_input":"texto para tool"}
2) Para concluir:
{"thought":"curto raciocínio","final_answer":"resposta final em português"}
"""


@dataclass(frozen=True)
class AgentRunResult:
    """Structured output from the ReAct executor."""

    answer: str
    trace: list[dict[str, Any]]
    used_tools: list[str]


class LLMClientProtocol:
    """Protocol-like base for an external LLM client."""

    def chat(self, messages: list[dict[str, str]]) -> str:  # pragma: no cover - interface
        raise NotImplementedError


def _safe_parse_json(raw_text: str) -> dict[str, Any]:
    """Parse model answer into a dict expected by the ReAct loop."""

    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("LLM returned non-JSON output; forcing final answer fallback.")
        return {"final_answer": raw_text}

    if not isinstance(parsed, dict):
        return {"final_answer": raw_text}
    return parsed


def run_react_agent(
    user_input: str,
    llm_client: LLMClientProtocol,
    tools: list[AgentTool] | None = None,
    max_iterations: int | None = None,
) -> AgentRunResult:
    """Execute a compact ReAct loop with at least three tools."""

    config = load_global_config()
    agent_cfg = config.get("agent", {})
    max_steps = max_iterations or int(agent_cfg.get("max_iterations", 6))
    active_tools = tools or build_default_tools()

    if len(active_tools) < 3:
        logger.warning("Expected >=3 tools for Datathon. Received %d.", len(active_tools))

    input_guardrail = InputGuardrail()
    output_guardrail = OutputGuardrail()
    is_valid, reason = input_guardrail.validate(user_input)
    if not is_valid:
        return AgentRunResult(answer=reason, trace=[], used_tools=[])

    tool_by_name = {tool.name: tool for tool in active_tools}
    tool_descriptions = "\n".join(
        f"- {tool.name}: {tool.description}" for tool in active_tools
    )
    trace: list[dict[str, Any]] = []
    used_tools: list[str] = []
    scratchpad: list[str] = []

    for step in range(max_steps):
        prompt = (
            f"{REACT_SYSTEM_PROMPT}\n"
            f"Ferramentas disponíveis:\n{tool_descriptions}\n\n"
            f"Pergunta do usuário: {user_input}\n\n"
            f"Histórico ReAct:\n" + "\n".join(scratchpad)
        )
        llm_answer = llm_client.chat(
            [
                {"role": "system", "content": REACT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )
        parsed = _safe_parse_json(llm_answer)
        thought = str(parsed.get("thought", "")).strip()

        if "final_answer" in parsed:
            final_answer = output_guardrail.sanitize(str(parsed["final_answer"]))
            trace.append(
                {
                    "iteration": step + 1,
                    "thought": thought,
                    "final_answer": final_answer,
                }
            )
            return AgentRunResult(answer=final_answer, trace=trace, used_tools=used_tools)

        action_name = str(parsed.get("action", "")).strip()
        action_input = str(parsed.get("action_input", "")).strip()
        tool = tool_by_name.get(action_name)
        if tool is None:
            observation = (
                f"Ferramenta '{action_name}' inválida. "
                f"Use apenas: {', '.join(tool_by_name)}."
            )
        else:
            try:
                observation = tool.run(action_input)
                used_tools.append(action_name)
            except Exception as exc:  # noqa: BLE001
                observation = f"Erro ao executar ferramenta {action_name}: {exc}"

        trace.append(
            {
                "iteration": step + 1,
                "thought": thought,
                "action": action_name,
                "action_input": action_input,
                "observation": observation,
            }
        )
        scratchpad.append(f"Thought: {thought}")
        scratchpad.append(f"Action: {action_name}")
        scratchpad.append(f"Action Input: {action_input}")
        scratchpad.append(f"Observation: {observation}")

    timeout_message = output_guardrail.sanitize(
        "Não consegui concluir no limite de iterações. "
        "Tente reformular a pergunta para algo mais específico."
    )
    return AgentRunResult(answer=timeout_message, trace=trace, used_tools=used_tools)
