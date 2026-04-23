"""ReAct agent with tool use for Datathon workflows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from agent.tools import AgentTool, build_default_tools
from common.config_loader import load_global_config
from common.logger import get_logger
from security.guardrails import InputGuardrail, OutputGuardrail

logger = get_logger("agent.react_agent")

MIN_TOOLS_EXPECTED = 3
DOCUMENTAL_KEYWORDS = (
    "/llm",
    "/predict",
    "api",
    "artefato",
    "artefatos",
    "arquivo",
    "arquivos",
    "config",
    "configuração",
    "configs",
    "contrato",
    "docs",
    "documentação",
    "dvc",
    "endpoint",
    "endpoints",
    "feature store",
    "ferramenta",
    "ferramentas",
    "mlflow",
    "monitoramento",
    "métrica",
    "métricas",
    "pipeline",
    "projeto",
    "rag",
    "react",
    "readme",
    "rota",
    "rotas",
    "schema",
    "serving",
    "tool",
    "tools",
    "yaml",
)
DOCUMENTAL_PATTERNS = (
    "cite",
    "como o projeto",
    "explique",
    "o que significa",
    "onde fica",
    "onde ficam",
    "onde está",
    "onde estão",
    "quais são",
    "qual rota",
    "quais rotas",
)

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

    def chat(self, messages: list[dict[str, str]]) -> str:  # pragma: no cover
        raise NotImplementedError


def _safe_parse_json(raw_text: str) -> tuple[dict[str, Any] | None, str, str]:
    """Parse model answer into a dict expected by the ReAct loop."""

    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("LLM returned invalid JSON output; requesting retry.")
        return None, "json_invalid", "O modelo nao retornou um JSON unico e valido."

    if not isinstance(parsed, dict):
        return None, "json_non_dict", "O modelo retornou JSON, mas nao um objeto."
    return parsed, "ok", ""


def _resolve_llm_metadata(llm_client: LLMClientProtocol) -> dict[str, Any]:
    metadata_fn = getattr(llm_client, "metadata", None)
    if callable(metadata_fn):
        raw_metadata = metadata_fn()
        if isinstance(raw_metadata, dict):
            return raw_metadata
    return {}


def is_documental_question(user_input: str) -> bool:
    """Heurística leve para perguntas cuja resposta deve vir do repositório."""

    normalized = " ".join(user_input.lower().split())
    if not normalized:
        return False

    if any(keyword in normalized for keyword in DOCUMENTAL_KEYWORDS):
        return True
    return any(pattern in normalized for pattern in DOCUMENTAL_PATTERNS)


def run_react_agent(  # noqa: PLR0914, PLR0915
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
    llm_metadata = _resolve_llm_metadata(llm_client)

    if len(active_tools) < MIN_TOOLS_EXPECTED:
        logger.warning(
            "Expected >=%d tools for Datathon. Received %d.",
            MIN_TOOLS_EXPECTED,
            len(active_tools),
        )

    input_guardrail = InputGuardrail()
    output_guardrail = OutputGuardrail()
    is_valid, reason = input_guardrail.validate(user_input)
    if not is_valid:
        return AgentRunResult(answer=reason, trace=[], used_tools=[])

    question_mode = "documental" if is_documental_question(user_input) else "default"
    if question_mode == "documental":
        rag_tool = next(
            (tool for tool in active_tools if tool.name == "rag_search"),
            None,
        )
        if rag_tool is not None:
            active_tools = [rag_tool]
            logger.info(
                "Pergunta documental detectada; restringindo tools para rag_search."
            )

    tool_by_name = {tool.name: tool for tool in active_tools}
    tool_descriptions = "\n".join(
        f"- {tool.name}: {tool.description}" for tool in active_tools
    )
    trace: list[dict[str, Any]] = []
    used_tools: list[str] = []
    scratchpad: list[str] = []

    for step in range(max_steps):
        iteration_start = perf_counter()
        prompt = (
            f"{REACT_SYSTEM_PROMPT}\n"
            f"Modo da pergunta: {question_mode}\n"
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
        parsed, parse_status, parse_message = _safe_parse_json(llm_answer)
        if parsed is None:
            observation = (
                f"Saida invalida do modelo ({parse_status}). "
                "Responda com um unico objeto JSON valido e sem markdown."
            )
            trace.append(
                {
                    "iteration": step + 1,
                    "question_mode": question_mode,
                    "parse_status": parse_status,
                    "fallback_reason": parse_message,
                    "raw_llm_output": llm_answer,
                    "observation": observation,
                    "llm_metadata": llm_metadata,
                    "iteration_latency_seconds": round(
                        perf_counter() - iteration_start,
                        6,
                    ),
                }
            )
            scratchpad.append(f"Observation: {observation}")
            continue

        thought = str(parsed.get("thought", "")).strip()

        if "final_answer" in parsed:
            final_answer = output_guardrail.sanitize(str(parsed["final_answer"]))
            trace.append(
                {
                    "iteration": step + 1,
                    "question_mode": question_mode,
                    "thought": thought,
                    "final_answer": final_answer,
                    "parse_status": parse_status,
                    "llm_metadata": llm_metadata,
                    "raw_llm_output": llm_answer,
                    "iteration_latency_seconds": round(
                        perf_counter() - iteration_start,
                        6,
                    ),
                }
            )
            return AgentRunResult(
                answer=final_answer,
                trace=trace,
                used_tools=used_tools,
            )

        action_name = str(parsed.get("action", "")).strip()
        action_input = str(parsed.get("action_input", "")).strip()
        if action_name == "tool_name":
            observation = (
                "Ferramenta placeholder detectada. Substitua 'tool_name' por uma "
                f"tool real: {', '.join(tool_by_name)}."
            )
            trace.append(
                {
                    "iteration": step + 1,
                    "question_mode": question_mode,
                    "thought": thought,
                    "action": action_name,
                    "action_input": action_input,
                    "parse_status": "placeholder_action",
                    "fallback_reason": "O modelo repetiu o template do prompt.",
                    "raw_llm_output": llm_answer,
                    "observation": observation,
                    "llm_metadata": llm_metadata,
                    "iteration_latency_seconds": round(
                        perf_counter() - iteration_start,
                        6,
                    ),
                }
            )
            scratchpad.append(f"Observation: {observation}")
            continue

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
                "question_mode": question_mode,
                "thought": thought,
                "action": action_name,
                "action_input": action_input,
                "observation": observation,
                "parse_status": parse_status,
                "raw_llm_output": llm_answer,
                "llm_metadata": llm_metadata,
                "iteration_latency_seconds": round(
                    perf_counter() - iteration_start,
                    6,
                ),
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
