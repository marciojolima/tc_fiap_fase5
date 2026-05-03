"""ReAct agent with tool use for Datathon workflows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal, cast

from agent.tools import AgentTool, build_default_tools
from common.config_loader import load_global_config
from common.logger import get_logger
from security.guardrails import InputGuardrail, OutputGuardrail

logger = get_logger("agent.react_agent")

MIN_TOOLS_EXPECTED = 3
AnswerStyle = Literal["short", "medium", "long"]
DEFAULT_ANSWER_STYLE: AnswerStyle = "medium"
ANSWER_STYLE_INSTRUCTIONS: dict[AnswerStyle, str] = {
    "short": (
        "Responda de forma curta: no máximo 3 bullets ou 1 parágrafo curto, "
        "sem conclusão adicional."
    ),
    "medium": (
        "Responda de forma objetiva: até 2 parágrafos curtos ou 4 bullets, "
        "com apenas os detalhes essenciais."
    ),
    "long": (
        "Responda de forma detalhada quando útil, mantendo a resposta ancorada "
        "nas evidências observadas."
    ),
}
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
SCENARIO_COMPARISON_KEYWORDS = (
    "baseline",
    "cenário base",
    "cenario base",
    "cenário ajustado",
    "cenario ajustado",
    "compare",
    "comparar",
    "comparação",
    "comparacao",
    "dois cenários",
    "dois cenarios",
    "improved",
    "melhor para retenção",
    "melhor para retencao",
    "outro com",
    "simule dois cenários",
    "simule dois cenarios",
    "um com",
    "versus",
    "vs",
)

REACT_SYSTEM_PROMPT = """You are an AI agent specialized in the bank customer
churn project.

You can use external tools to answer questions more accurately.
Always decide whether to:
- call a tool using "action"
- or provide the final answer using "final_answer"

Tool usage rules:
- Use predict_churn only for valid raw customer payloads equivalent to
  /predict/raw.
- Use scenario_prediction to simulate changes in customer features and compare
  scenarios.
- Use rag_search to retrieve relevant project, business, API, or documentation context.
- Use drift_status only for monitoring, model health, or drift-related questions.
- For repository or documentation questions, prefer rag_search before answering.
- For scenario or prediction questions grounded in customer/business situations,
  do not prefer rag_search unless the user is explicitly asking about docs,
  routes, files, or repository behavior.
- If a scenario question omits part of the customer payload but clearly describes
  the intended change, build a reasonable representative payload using defaults
  from the serving schema and compare scenarios instead of stopping early.
- Do not invent tool results.
- If a tool is needed, call the tool instead of guessing.
- Do not claim facts about routes, tools, files, configs, artifacts, or project
  behavior unless they are supported by observed context.

You must respond ALWAYS with valid JSON and no extra text:
1) To call a tool:
{"thought":"short reasoning","action":"tool_name","action_input":"input for the tool"}

2) To conclude:
{"thought":"short reasoning","final_answer":"resposta final em português do Brasil"}

IMPORTANT:
- final_answer MUST be in Brazilian Portuguese.
- Never return text outside the JSON object.
- When action_input is structured, return it as a valid JSON object or JSON
  string with double-quoted keys and strings. Never use Python dict notation.
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


def _normalize_action_input(raw_action_input: Any) -> str:
    """Normalize structured action input into stable JSON for tool execution."""

    if raw_action_input is None:
        return ""
    if isinstance(raw_action_input, str):
        return raw_action_input.strip()
    if isinstance(raw_action_input, (dict, list, int, float, bool)):
        return json.dumps(raw_action_input, ensure_ascii=False)
    return str(raw_action_input).strip()


def _resolve_answer_style(answer_style: str | None) -> AnswerStyle:
    normalized = (answer_style or DEFAULT_ANSWER_STYLE).strip().lower()
    if normalized in ANSWER_STYLE_INSTRUCTIONS:
        return cast(AnswerStyle, normalized)
    return DEFAULT_ANSWER_STYLE


def is_documental_question(user_input: str) -> bool:
    """Heurística leve para perguntas cuja resposta deve vir do repositório."""

    normalized = " ".join(user_input.lower().split())
    if not normalized:
        return False

    if any(keyword in normalized for keyword in DOCUMENTAL_KEYWORDS):
        return True
    return any(pattern in normalized for pattern in DOCUMENTAL_PATTERNS)


def is_comparative_scenario_question(user_input: str) -> bool:
    """Detecta perguntas que pedem comparação explícita entre cenários."""

    normalized = " ".join(user_input.lower().split())
    if not normalized:
        return False

    scenario_terms = ("cenário", "cenario", "retenção", "retencao", "churn")
    has_scenario_term = any(term in normalized for term in scenario_terms)
    has_comparison_term = any(
        keyword in normalized for keyword in SCENARIO_COMPARISON_KEYWORDS
    )
    return has_scenario_term and has_comparison_term


def _scenario_observation_is_comparative(observation: str) -> bool:
    """Check whether a scenario tool observation contains two scenarios."""

    try:
        payload = json.loads(observation)
    except json.JSONDecodeError:
        return False

    if not isinstance(payload, dict):
        return False
    result = payload.get("result")
    if not isinstance(result, dict):
        return False
    return isinstance(result.get("baseline_scenario"), dict) and isinstance(
        result.get("improved_scenario"),
        dict,
    )


def run_react_agent(  # noqa: PLR0912, PLR0914, PLR0915
    user_input: str,
    llm_client: LLMClientProtocol,
    tools: list[AgentTool] | None = None,
    max_iterations: int | None = None,
    answer_style: str | None = DEFAULT_ANSWER_STYLE,
) -> AgentRunResult:
    """Execute a compact ReAct loop with at least three tools."""

    config = load_global_config()
    agent_cfg = config.get("agent", {})
    max_steps = max_iterations or int(agent_cfg.get("max_iterations", 6))
    active_tools = tools or build_default_tools()
    llm_metadata = _resolve_llm_metadata(llm_client)
    resolved_answer_style = _resolve_answer_style(answer_style)
    answer_style_instruction = ANSWER_STYLE_INSTRUCTIONS[resolved_answer_style]

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
    comparative_scenario_question = is_comparative_scenario_question(user_input)
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
    documentary_context_observed = False
    comparative_scenario_observed = False

    for step in range(max_steps):
        iteration_start = perf_counter()
        prompt = (
            f"{REACT_SYSTEM_PROMPT}\n"
            f"Modo da pergunta: {question_mode}\n"
            f"Estilo da resposta final: {resolved_answer_style}\n"
            f"Instrução de tamanho: {answer_style_instruction}\n"
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
                    "answer_style": resolved_answer_style,
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
            if question_mode == "documental" and not documentary_context_observed:
                observation = (
                    "Para perguntas documentais, use rag_search antes da "
                    "resposta final e baseie a resposta na evidência observada."
                )
                trace.append(
                    {
                        "iteration": step + 1,
                        "question_mode": question_mode,
                        "answer_style": resolved_answer_style,
                        "thought": thought,
                        "final_answer": str(parsed["final_answer"]),
                        "parse_status": "missing_documentary_evidence",
                        "fallback_reason": (
                            "Pergunta documental sem uso prévio de "
                            "rag_search."
                        ),
                        "raw_llm_output": llm_answer,
                        "observation": observation,
                        "llm_metadata": llm_metadata,
                        "iteration_latency_seconds": round(
                            perf_counter() - iteration_start,
                            6,
                        ),
                    }
                )
                scratchpad.append(f"Thought: {thought}")
                scratchpad.append(
                    "Attempted Final Answer: "
                    f"{str(parsed['final_answer']).strip()}"
                )
                scratchpad.append(f"Observation: {observation}")
                continue
            if comparative_scenario_question and not comparative_scenario_observed:
                observation = (
                    "A pergunta pede comparação entre cenários. Antes da resposta "
                    "final, use scenario_prediction com um payload JSON contendo "
                    "baseline_scenario e improved_scenario, e depois explique as "
                    "probabilidades de ambos e qual cenário é melhor para retenção."
                )
                trace.append(
                    {
                        "iteration": step + 1,
                        "question_mode": question_mode,
                        "answer_style": resolved_answer_style,
                        "thought": thought,
                        "final_answer": str(parsed["final_answer"]),
                        "parse_status": "missing_comparative_scenario_evidence",
                        "fallback_reason": (
                            "Pergunta comparativa sem comparação estruturada "
                            "observada em scenario_prediction."
                        ),
                        "raw_llm_output": llm_answer,
                        "observation": observation,
                        "llm_metadata": llm_metadata,
                        "iteration_latency_seconds": round(
                            perf_counter() - iteration_start,
                            6,
                        ),
                    }
                )
                scratchpad.append(f"Thought: {thought}")
                scratchpad.append(
                    "Attempted Final Answer: "
                    f"{str(parsed['final_answer']).strip()}"
                )
                scratchpad.append(f"Observation: {observation}")
                continue

            final_answer = output_guardrail.sanitize(str(parsed["final_answer"]))
            trace.append(
                {
                    "iteration": step + 1,
                    "question_mode": question_mode,
                    "answer_style": resolved_answer_style,
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
        action_input = _normalize_action_input(parsed.get("action_input", ""))
        if action_name == "tool_name":
            observation = (
                "Ferramenta placeholder detectada. Substitua 'tool_name' por uma "
                f"tool real: {', '.join(tool_by_name)}."
            )
            trace.append(
                {
                    "iteration": step + 1,
                    "question_mode": question_mode,
                    "answer_style": resolved_answer_style,
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
                if question_mode == "documental" and action_name == "rag_search":
                    documentary_context_observed = True
                if action_name == "scenario_prediction":
                    comparative_result = _scenario_observation_is_comparative(
                        observation
                    )
                    if comparative_result:
                        comparative_scenario_observed = True
                    elif comparative_scenario_question:
                        observation = (
                            "Pergunta comparativa detectada, mas a tool retornou "
                            "apenas um cenário. Reexecute scenario_prediction com "
                            "um JSON contendo baseline_scenario e "
                            "improved_scenario para comparar os dois casos."
                        )
            except Exception as exc:  # noqa: BLE001
                observation = f"Erro ao executar ferramenta {action_name}: {exc}"

        trace.append(
            {
                "iteration": step + 1,
                "question_mode": question_mode,
                "answer_style": resolved_answer_style,
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
