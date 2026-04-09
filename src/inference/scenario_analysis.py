"""Executa análise de cenários de churn com rastreabilidade no MLflow."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, NamedTuple

import mlflow

from common.config_loader import (
    DEFAULT_CURRENT_EXPERIMENT_CONFIG_PATH,
    load_global_config,
)
from common.logger import get_logger
from serving.pipeline import (
    build_serving_config,
    predict_from_dataframe_with_config,
    prepare_inference_dataframe,
)
from serving.schemas import ChurnPredictionRequest

logger = get_logger("inference.scenario_analysis")


class ScenarioAnalysisConfig(NamedTuple):
    """Configuração necessária para rodar análise de cenários."""

    tracking_uri: str
    mlflow_experiment_name: str
    experiment_config_path: str


class AnalysisScenario(NamedTuple):
    """Cenário hipotético com nome estável e payload bruto."""

    name: str
    payload: dict[str, Any]


class ScenarioAnalysisResult(NamedTuple):
    """Resultado padronizado da análise de cenário."""

    scenario_name: str
    churn_probability: float
    churn_prediction: int
    threshold: float
    model_name: str
    run_name: str


def load_scenario_analysis_config(
    experiment_config_path: str = DEFAULT_CURRENT_EXPERIMENT_CONFIG_PATH,
) -> ScenarioAnalysisConfig:
    """Carrega a configuração global do fluxo de análise de cenários."""

    global_config = load_global_config()
    return ScenarioAnalysisConfig(
        tracking_uri=global_config["mlflow"]["tracking_uri"],
        mlflow_experiment_name=global_config["mlflow"].get(
            "scenario_analysis_experiment_name",
            "datathon-churn-scenario-analysis",
        ),
        experiment_config_path=experiment_config_path,
    )


def build_scenario(name: str, payload: dict[str, Any]) -> AnalysisScenario:
    """Monta um cenário hipotético validado em memória."""

    return AnalysisScenario(name=name, payload=payload)


def parse_payload(payload: dict[str, Any]) -> ChurnPredictionRequest:
    """Valida o payload hipotético usando o schema de serving."""

    return ChurnPredictionRequest(**payload)


def run_scenario_prediction(
    scenario: AnalysisScenario,
    experiment_config_path: str = DEFAULT_CURRENT_EXPERIMENT_CONFIG_PATH,
) -> ScenarioAnalysisResult:
    """Executa a inferência de um cenário hipotético."""

    serving_cfg = build_serving_config(experiment_config_path)
    request = parse_payload(scenario.payload)
    transformed_features = prepare_inference_dataframe(request, serving_cfg)
    probability, prediction = predict_from_dataframe_with_config(
        transformed_features,
        serving_cfg,
    )

    return ScenarioAnalysisResult(
        scenario_name=scenario.name,
        churn_probability=probability,
        churn_prediction=prediction,
        threshold=serving_cfg.threshold,
        model_name=serving_cfg.model_name,
        run_name=serving_cfg.run_name,
    )


def _normalize_param_key(raw_key: str) -> str:
    """Padroniza nomes de params para o MLflow."""

    return raw_key.lower().replace(" ", "_")


def _log_payload_params(payload: dict[str, Any]) -> None:
    """Registra o payload de entrada como parâmetros rastreáveis."""

    for key, value in payload.items():
        mlflow.log_param(f"input_{_normalize_param_key(key)}", value)


def _log_json_artifact(data: dict[str, Any], filename: str, artifact_dir: str) -> None:
    """Persiste um dicionário em JSON e registra o arquivo como artifact."""

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / filename
        with open(output_path, "w", encoding="utf-8") as file_obj:
            json.dump(data, file_obj, indent=2, ensure_ascii=False)

        mlflow.log_artifact(str(output_path), artifact_path=artifact_dir)


def log_scenario_analysis_run(
    scenario: AnalysisScenario,
    result: ScenarioAnalysisResult,
    cfg: ScenarioAnalysisConfig,
) -> None:
    """Registra uma execução de análise de cenário em experimento dedicado."""

    mlflow.set_tracking_uri(cfg.tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment_name)

    run_name = f"scenario_analysis::{result.model_name}::{scenario.name}"
    with mlflow.start_run(run_name=run_name):
        _log_payload_params(scenario.payload)
        mlflow.log_param("experiment_config_path", cfg.experiment_config_path)
        mlflow.log_param("threshold", result.threshold)
        mlflow.log_param("run_name", result.run_name)

        mlflow.log_metric("churn_probability", result.churn_probability)
        mlflow.log_metric("churn_prediction", float(result.churn_prediction))

        mlflow.set_tag("flow", "scenario_analysis")
        mlflow.set_tag("analysis_type", "hypothetical_scenario")
        mlflow.set_tag("scenario_name", scenario.name)
        mlflow.set_tag("model_name", result.model_name)

        artifact_dir = f"scenario_analysis/{scenario.name}"
        _log_json_artifact(
            scenario.payload,
            filename="payload.json",
            artifact_dir=artifact_dir,
        )
        _log_json_artifact(
            result._asdict(),
            filename="prediction.json",
            artifact_dir=artifact_dir,
        )


def run_scenario_analysis(
    scenario: AnalysisScenario,
    experiment_config_path: str = DEFAULT_CURRENT_EXPERIMENT_CONFIG_PATH,
) -> ScenarioAnalysisResult:
    """Executa um cenário único e registra a análise no MLflow."""

    cfg = load_scenario_analysis_config(experiment_config_path)
    result = run_scenario_prediction(scenario, experiment_config_path)
    log_scenario_analysis_run(scenario, result, cfg)

    logger.info(
        "Análise de cenário executada — cenário=%s | modelo=%s | prob=%.4f | pred=%d",
        result.scenario_name,
        result.model_name,
        result.churn_probability,
        result.churn_prediction,
    )
    return result


def load_scenario_suite(suite_path: str) -> list[AnalysisScenario]:
    """Carrega uma suíte versionada de cenários hipotéticos em JSON."""

    with open(suite_path, "r", encoding="utf-8") as file_obj:
        content = json.load(file_obj)

    scenario_items = content["scenarios"] if isinstance(content, dict) else content
    return [
        build_scenario(name=item["name"], payload=item["payload"])
        for item in scenario_items
    ]


def run_scenario_analysis_suite(
    suite_path: str,
    experiment_config_path: str = DEFAULT_CURRENT_EXPERIMENT_CONFIG_PATH,
) -> list[ScenarioAnalysisResult]:
    """Executa uma suíte de cenários hipotéticos de forma sequencial."""

    scenarios = load_scenario_suite(suite_path)
    results = [
        run_scenario_analysis(
            scenario=scenario,
            experiment_config_path=experiment_config_path,
        )
        for scenario in scenarios
    ]

    logger.info("Suíte de cenários concluída — cenários executados: %d", len(results))
    return results


def parse_args() -> argparse.Namespace:
    """Lê argumentos de linha de comando para cenário único ou suíte."""

    parser = argparse.ArgumentParser(
        description="Executa análise de cenários para churn com rastreio no MLflow.",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CURRENT_EXPERIMENT_CONFIG_PATH,
        help="Caminho relativo para o YAML do experimento de treino.",
    )
    parser.add_argument(
        "--scenario-name",
        default="manual_scenario",
        help="Nome do cenário usado no modo de cenário único.",
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--payload-json",
        help="Payload único em formato JSON.",
    )
    input_group.add_argument(
        "--payload-file",
        help="Caminho para um payload único em JSON.",
    )
    input_group.add_argument(
        "--suite-file",
        default=None,
        help="Caminho para uma suíte de cenários em JSON.",
    )

    return parser.parse_args()


def _load_payload_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Carrega um payload único a partir dos argumentos da CLI."""

    if args.payload_json:
        return json.loads(args.payload_json)

    with open(args.payload_file, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def main() -> None:
    """Ponto de entrada para execução local do fluxo de análise de cenários."""

    args = parse_args()

    if args.suite_file:
        results = run_scenario_analysis_suite(
            suite_path=args.suite_file,
            experiment_config_path=args.config,
        )
        print(
            json.dumps(
                [result._asdict() for result in results],
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    payload = _load_payload_from_args(args)
    scenario = build_scenario(args.scenario_name, payload)
    result = run_scenario_analysis(
        scenario=scenario,
        experiment_config_path=args.config,
    )
    print(json.dumps(result._asdict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
