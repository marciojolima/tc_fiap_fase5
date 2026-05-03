"""Custom tools for the Datathon ReAct agent."""

from __future__ import annotations

import ast
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from agent.rag_pipeline import retrieve_contexts
from common.config_loader import load_global_config
from common.logger import get_logger
from scenario_experiments.inference_cases import (
    AnalysisScenario,
    run_scenario_prediction,
)
from serving.pipeline import (
    load_serving_config,
    predict_from_dataframe_with_config,
    prepare_inference_dataframe,
)
from serving.schemas import ChurnPredictionRequest

logger = get_logger("agent.tools")
_SCENARIO_FIELD_PATTERNS = {
    "Age": re.compile(r"\b(\d{2})\s+anos\b"),
    "CreditScore": re.compile(r"(?:credit score|score)\s*(\d{3})\b"),
    "Balance": re.compile(r"\bsaldo(?:\s+de)?\s*(\d+(?:[.,]\d+)?)\b"),
    "NumOfProducts": re.compile(r"\b(\d)\s+produt"),
    "Tenure": re.compile(
        r"(?:tenure|relacionamento|anos de conta|anos no banco)\s*(\d+)\b"
    ),
}
_GEOGRAPHY_BY_TERM = {
    "alemanha": "Germany",
    "espanha": "Spain",
    "franca": "France",
    "frança": "France",
    "germany": "Germany",
    "spain": "Spain",
    "france": "France",
}
_GENDER_BY_TERM = {
    "feminino": "Female",
    "female": "Female",
    "masculino": "Male",
    "male": "Male",
}


@dataclass(frozen=True)
class AgentTool:
    """Simple tool contract for the local ReAct executor."""

    name: str
    description: str
    run: Callable[[str], str]


def _json_tool_output(payload: dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _short_text(text: str, limit: int = 220) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _parse_structured_tool_input(raw_input: str) -> dict[str, Any]:
    """Parse JSON-like tool input with a tolerant fallback for Python literals."""

    try:
        parsed = json.loads(raw_input)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(raw_input)

    if not isinstance(parsed, dict):
        raise ValueError("O payload precisa ser um objeto JSON.")
    return parsed


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _default_scenario_payload() -> dict[str, Any]:
    return ChurnPredictionRequest().model_dump(by_alias=True)


def _extract_scenario_overrides(raw_text: str) -> dict[str, Any]:
    """Extract lightweight customer feature hints from natural language."""

    normalized = " ".join(raw_text.split())
    lowered = normalized.lower()
    overrides: dict[str, Any] = {}

    for field_name, pattern in _SCENARIO_FIELD_PATTERNS.items():
        match = pattern.search(lowered)
        if not match:
            continue
        raw_value = match.group(1).replace(",", ".")
        if field_name == "Balance":
            overrides[field_name] = float(raw_value)
        else:
            overrides[field_name] = int(raw_value)

    for term, geography in _GEOGRAPHY_BY_TERM.items():
        if term in lowered:
            overrides["Geography"] = geography
            break

    for term, gender in _GENDER_BY_TERM.items():
        if term in lowered:
            overrides["Gender"] = gender
            break

    if any(
        marker in lowered
        for marker in ("inativo", "sem atividade", "sem atividade recente")
    ):
        overrides["IsActiveMember"] = 0
    elif any(
        marker in lowered for marker in ("ativo", "atividade recente", "engajado")
    ):
        overrides["IsActiveMember"] = 1

    return overrides


def _extract_comparison_segments(raw_input: str) -> tuple[str | None, str | None]:
    """Split simple comparison prompts into baseline and improved fragments."""

    normalized = " ".join(raw_input.strip().split())
    lowered = _strip_accents(normalized.lower())

    explicit_match = re.search(
        (
            r"baseline(?:_scenario)?\s*[:=]?\s*(.*?)\s+"
            r"improved(?:_scenario)?\s*[:=]?\s*(.*)"
        ),
        lowered,
    )
    if explicit_match:
        return explicit_match.group(1).strip(), explicit_match.group(2).strip()

    paired_match = re.search(r"\bum com\b\s*(.*?)\s*\boutro com\b\s*(.*)", lowered)
    if paired_match:
        return paired_match.group(1).strip(), paired_match.group(2).strip()

    return None, None


def _parse_natural_language_scenario_input(raw_input: str) -> dict[str, Any]:
    """Build scenario payloads from natural language when JSON is absent."""

    baseline_text, improved_text = _extract_comparison_segments(raw_input)
    if baseline_text is None or improved_text is None:
        return {
            **_default_scenario_payload(),
            **_extract_scenario_overrides(raw_input),
        }

    global_overrides = _extract_scenario_overrides(raw_input)
    baseline_overrides = _extract_scenario_overrides(baseline_text)
    improved_overrides = _extract_scenario_overrides(improved_text)

    baseline_payload = {
        **_default_scenario_payload(),
        **global_overrides,
        **baseline_overrides,
    }
    improved_payload = baseline_payload.copy()

    same_customer_markers = ("mesmo cliente", "mesma cliente", "same customer")
    if not any(marker in improved_text for marker in same_customer_markers):
        improved_payload = {
            **_default_scenario_payload(),
            **global_overrides,
        }

    improved_payload.update(improved_overrides)
    return {
        "baseline_scenario": baseline_payload,
        "improved_scenario": improved_payload,
        "comparison_description": _short_text(raw_input, limit=180),
    }


def _comparison_payload(
    payload: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str]:
    """Extract optional baseline/improved scenario payloads from tool input."""

    baseline_payload = payload.get("baseline_scenario")
    improved_payload = payload.get("improved_scenario")
    if not isinstance(improved_payload, dict):
        improved_payload = payload.get("comparison_scenario")
    if not isinstance(baseline_payload, dict):
        baseline_payload = None
    if not isinstance(improved_payload, dict):
        improved_payload = None
    scenario_name = str(payload.get("scenario_name", "agent_scenario"))
    return baseline_payload, improved_payload, scenario_name


def _single_scenario_output(
    payload: dict[str, Any],
    scenario_name: str,
) -> str:
    """Run a single scenario prediction and return the standard tool payload."""

    scenario = AnalysisScenario(name=scenario_name, payload=payload)
    result = run_scenario_prediction(scenario)
    structured = result._asdict()
    probability = structured.get("churn_probability")
    prediction = structured.get("churn_prediction")
    return _json_tool_output(
        {
            "tool_name": "scenario_prediction",
            "status": "ok",
            "input_summary": "simulação de cenário com payload JSON",
            "result": structured,
            "evidence": [
                f"Probabilidade simulada: {probability}",
                f"Classe simulada: {prediction}",
            ],
            "confidence": "alta",
            "recommended_next_step": (
                "Compare o resultado com o cenário base antes de concluir impacto."
            ),
        }
    )


def _comparison_scenario_output(
    baseline_payload: dict[str, Any],
    improved_payload: dict[str, Any],
    scenario_name: str,
    comparison_description: str,
) -> str:
    """Run a baseline-vs-improved comparison and summarize retention impact."""

    baseline_result = run_scenario_prediction(
        AnalysisScenario(name=f"{scenario_name}_baseline", payload=baseline_payload)
    )
    improved_result = run_scenario_prediction(
        AnalysisScenario(name=f"{scenario_name}_improved", payload=improved_payload)
    )
    baseline_structured = baseline_result._asdict()
    improved_structured = improved_result._asdict()
    baseline_probability = float(baseline_structured["churn_probability"])
    improved_probability = float(improved_structured["churn_probability"])
    probability_delta = round(improved_probability - baseline_probability, 6)
    better_scenario = "improved_scenario"
    if baseline_probability < improved_probability:
        better_scenario = "baseline_scenario"

    return _json_tool_output(
        {
            "tool_name": "scenario_prediction",
            "status": "ok",
            "input_summary": "comparação entre cenário base e cenário ajustado",
            "comparison_description": comparison_description,
            "result": {
                "baseline_scenario": baseline_structured,
                "improved_scenario": improved_structured,
                "probability_delta": probability_delta,
                "better_scenario_for_retention": better_scenario,
            },
            "evidence": [
                (
                    "Probabilidade no cenário base: "
                    f"{round(baseline_probability, 6)}"
                ),
                (
                    "Probabilidade no cenário ajustado: "
                    f"{round(improved_probability, 6)}"
                ),
                f"O melhor cenário para retenção é {better_scenario}.",
            ],
            "confidence": "alta",
            "recommended_next_step": (
                "Explique a diferença de risco entre os cenários e explicite "
                "as premissas assumidas."
            ),
        }
    )


def _predict_churn_tool(raw_input: str) -> str:
    """Predict churn from a JSON customer payload."""

    try:
        payload = _parse_structured_tool_input(raw_input)
    except (json.JSONDecodeError, SyntaxError, ValueError) as exc:
        return _json_tool_output(
            {
                "tool_name": "predict_churn",
                "status": "error",
                "input_summary": "payload JSON bruto para /predict/raw",
                "error": f"JSON inválido para predição: {exc}",
            }
        )

    request = ChurnPredictionRequest(**payload)
    cfg = load_serving_config(model_name=request.model_name)
    features = prepare_inference_dataframe(request, cfg)
    probability, prediction = predict_from_dataframe_with_config(features, cfg)
    result = {
        "tool_name": "predict_churn",
        "status": "ok",
        "input_summary": "predição com payload bruto equivalente a /predict/raw",
        "result": {
            "churn_probability": round(probability, 6),
            "churn_prediction": prediction,
            "threshold": cfg.threshold,
            "model_name": cfg.model_name,
        },
        "evidence": [
            f"Probabilidade estimada: {round(probability, 6)}",
            f"Classe prevista: {prediction}",
            f"Threshold do serving: {cfg.threshold}",
        ],
        "confidence": "alta",
        "recommended_next_step": (
            "Use a probabilidade e o threshold para explicar o risco ao usuário."
        ),
    }
    return _json_tool_output(result)


def _rag_search_tool(query: str) -> str:
    """Retrieve relevant project context for a natural language query."""

    rag_cfg = load_global_config().get("rag", {})
    contexts = retrieve_contexts(query, top_k=int(rag_cfg.get("top_k", 4)))
    if not contexts:
        return _json_tool_output(
            {
                "tool_name": "rag_search",
                "status": "no_results",
                "query": query,
                "evidence": [],
                "sources": [],
                "confidence": "baixa",
                "recommended_next_step": (
                    "Reformule a pergunta com termos mais próximos do repositório."
                ),
            }
        )

    evidence = [_short_text(context) for context in contexts[:3]]
    sources: list[str] = []
    for context in contexts[:3]:
        header = context.splitlines()[0].strip() if context.splitlines() else ""
        sources.append(header or "fonte_indeterminada")
    return _json_tool_output(
        {
            "tool_name": "rag_search",
            "status": "ok",
            "query": query,
            "evidence": evidence,
            "retrieved_contexts": contexts,
            "sources": sources,
            "confidence": "média" if len(contexts) == 1 else "alta",
            "recommended_next_step": (
                "Responda usando apenas as evidências e cite rotas, tools ou arquivos "
                "explicitamente quando aparecerem nos trechos."
            ),
        }
    )


def _drift_status_tool(_: str) -> str:
    """Return latest drift status if available."""

    status_path = Path("artifacts/evaluation/model/drift/drift_status.json")
    if not status_path.exists():
        return _json_tool_output(
            {
                "tool_name": "drift_status",
                "status": "unavailable",
                "source": str(status_path),
                "evidence": [],
                "confidence": "baixa",
                "recommended_next_step": (
                    "Execute o monitoramento com `poetry run task mldrift`."
                ),
            }
        )
    raw = status_path.read_text(encoding="utf-8")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return _json_tool_output(
            {
                "tool_name": "drift_status",
                "status": "error",
                "source": str(status_path),
                "error": "Arquivo de drift não contém JSON válido.",
                "raw_excerpt": _short_text(raw),
            }
        )

    current_status = str(
        parsed.get("status") or parsed.get("drift_status") or "unknown"
    )
    evidence = [f"Status atual de drift: {current_status}"]
    for key in ("message", "reason", "decision", "summary"):
        value = parsed.get(key)
        if value:
            evidence.append(_short_text(f"{key}: {value}", limit=180))
            break
    return _json_tool_output(
        {
            "tool_name": "drift_status",
            "status": "ok",
            "source": str(status_path),
            "result": {
                "drift_status": current_status,
            },
            "evidence": evidence[:3],
            "confidence": "alta",
            "recommended_next_step": (
                "Explique o status operacional e mencione o arquivo de origem se útil."
            ),
        }
    )


def _scenario_prediction_tool(raw_input: str) -> str:
    """Run a scenario prediction from a payload JSON."""

    try:
        payload = _parse_structured_tool_input(raw_input)
    except (json.JSONDecodeError, SyntaxError, ValueError) as exc:
        try:
            payload = _parse_natural_language_scenario_input(raw_input)
        except (TypeError, ValueError) as parse_exc:
            return _json_tool_output(
                {
                    "tool_name": "scenario_prediction",
                    "status": "error",
                    "input_summary": "payload JSON de cenário",
                    "error": (
                        "Entrada inválida para cenário. Não foi possível "
                        f"interpretar JSON nem linguagem natural: {parse_exc}"
                    ),
                    "raw_error": str(exc),
                }
            )

    baseline_payload, improved_payload, scenario_name = _comparison_payload(payload)

    if baseline_payload is not None and improved_payload is not None:
        return _comparison_scenario_output(
            baseline_payload=baseline_payload,
            improved_payload=improved_payload,
            scenario_name=scenario_name,
            comparison_description=str(payload.get("comparison_description", "")),
        )

    if baseline_payload is not None:
        payload = baseline_payload
        scenario_name = str(payload.get("scenario_name", scenario_name))

    return _single_scenario_output(payload=payload, scenario_name=scenario_name)


def build_default_tools() -> list[AgentTool]:
    """Build the default Datathon toolset (>= 3 tools)."""

    tools = [
        AgentTool(
            name="rag_search",
            description=(
                "Busca contexto relevante na documentação e metadados "
                "do projeto."
            ),
            run=_rag_search_tool,
        ),
        AgentTool(
            name="predict_churn",
            description=(
                "Executa predição de churn para um payload JSON no mesmo "
                "formato do endpoint /predict/raw."
            ),
            run=_predict_churn_tool,
        ),
        AgentTool(
            name="drift_status",
            description="Lê o status mais recente de monitoramento de drift.",
            run=_drift_status_tool,
        ),
        AgentTool(
            name="scenario_prediction",
            description="Executa inferência de cenário com payload JSON.",
            run=_scenario_prediction_tool,
        ),
    ]
    logger.info("Default ReAct tools carregadas: %s", [tool.name for tool in tools])
    return tools
