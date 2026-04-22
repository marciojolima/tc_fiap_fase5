"""Custom tools for the Datathon ReAct agent."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from agent.rag_pipeline import retrieve_contexts
from common.logger import get_logger
from scenario_analysis.inference_cases import AnalysisScenario, run_scenario_prediction
from serving.pipeline import (
    load_serving_config,
    predict_from_dataframe_with_config,
    prepare_inference_dataframe,
)
from serving.schemas import ChurnPredictionRequest

logger = get_logger("agent.tools")


@dataclass(frozen=True)
class AgentTool:
    """Simple tool contract for the local ReAct executor."""

    name: str
    description: str
    run: Callable[[str], str]


def _predict_churn_tool(raw_input: str) -> str:
    """Predict churn from a JSON customer payload."""

    try:
        payload = json.loads(raw_input)
    except json.JSONDecodeError as exc:
        return f"JSON inválido para predição: {exc}"

    request = ChurnPredictionRequest(**payload)
    cfg = load_serving_config()
    features = prepare_inference_dataframe(request, cfg)
    probability, prediction = predict_from_dataframe_with_config(features, cfg)
    result = {
        "churn_probability": round(probability, 6),
        "churn_prediction": prediction,
        "threshold": cfg.threshold,
        "model_name": cfg.model_name,
    }
    return json.dumps(result, ensure_ascii=False)


def _rag_search_tool(query: str) -> str:
    """Retrieve relevant project context for a natural language query."""

    contexts = retrieve_contexts(query, top_k=3)
    if not contexts:
        return "Nenhum contexto relevante encontrado."
    return "\n\n---\n\n".join(contexts)


def _drift_status_tool(_: str) -> str:
    """Return latest drift status if available."""

    status_path = Path("artifacts/monitoring/drift/drift_status.json")
    if not status_path.exists():
        return (
            "Status de drift indisponível. Execute o monitoramento com "
            "`poetry run task mldrift`."
        )
    return status_path.read_text(encoding="utf-8")


def _scenario_prediction_tool(raw_input: str) -> str:
    """Run a scenario prediction from a payload JSON."""

    try:
        payload = json.loads(raw_input)
    except json.JSONDecodeError as exc:
        return f"JSON inválido para cenário: {exc}"

    scenario = AnalysisScenario(name="agent_scenario", payload=payload)
    result = run_scenario_prediction(scenario)
    return json.dumps(result._asdict(), ensure_ascii=False)


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
                "formato do endpoint /predict."
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
