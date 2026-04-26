"""Executor dedicado para solicitações de retreino auditáveis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, NamedTuple

import yaml

from common.config_loader import DEFAULT_CURRENT_EXPERIMENT_CONFIG_PATH
from common.logger import get_logger
from common.timezone import now_isoformat
from model_lifecycle.promotion import evaluate_challenger_promotion
from model_lifecycle.train import (
    RetrainingMlflowContext,
    build_metadata_output_path,
    load_experiment_training_config,
    run_training,
)

DEFAULT_RETRAIN_REQUEST_PATH = (
    "artifacts/evaluation/model/retraining/retrain_request.json"
)
DEFAULT_RETRAIN_RUN_PATH = "artifacts/evaluation/model/retraining/retrain_run.json"

logger = get_logger("model_lifecycle.retraining")


class RetrainingRequest(NamedTuple):
    """Contrato mínimo para execução de um retreino disparado por monitoramento."""

    request_id: str
    status: str
    reason: str
    model_path: str
    training_config_path: str
    trigger_mode: str
    created_at: str
    promotion_policy: str
    promotion_decision_path: str
    promotion_rules: dict[str, Any]
    drift_status: str
    max_feature_psi: float
    prediction_psi: float | None
    drifted_features: list[str]
    reference_row_count: int
    current_row_count: int


class RetrainingRunContext(NamedTuple):
    """Contexto de execução necessário para serializar o resultado do retreino."""

    status: str
    started_at: str
    completed_at: str
    training_config_path: str
    model_output_path: str
    model_version: str
    experiment_name: str
    challenger_training_config_path: str
    promotion_decision: dict[str, Any] | None = None
    metrics: dict[str, float] | None = None
    failure_reason: str | None = None


def load_retraining_request(path: str | Path) -> RetrainingRequest:
    """Carrega e valida a solicitação de retreino."""

    request_path = Path(path)
    with open(request_path, "r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)

    return RetrainingRequest(
        request_id=payload["request_id"],
        status=payload["status"],
        reason=payload["reason"],
        model_path=payload["model_path"],
        training_config_path=payload.get(
            "training_config_path",
            DEFAULT_CURRENT_EXPERIMENT_CONFIG_PATH,
        ),
        trigger_mode=payload.get("trigger_mode", "manual"),
        created_at=payload["created_at"],
        promotion_policy=payload.get(
            "promotion_policy",
            "manual_approval_required",
        ),
        promotion_decision_path=payload.get(
            "promotion_decision_path",
            "artifacts/evaluation/model/retraining/promotion_decision.json",
        ),
        promotion_rules=dict(payload.get("promotion_rules", {})),
        drift_status=payload["drift_status"],
        max_feature_psi=float(payload["max_feature_psi"]),
        prediction_psi=(
            float(payload["prediction_psi"])
            if payload.get("prediction_psi") is not None
            else None
        ),
        drifted_features=list(payload.get("drifted_features", [])),
        reference_row_count=int(payload.get("reference_row_count", 0)),
        current_row_count=int(payload.get("current_row_count", 0)),
    )


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Persiste um payload JSON garantindo criação do diretório pai."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def update_request_status(
    request_path: str | Path,
    request: RetrainingRequest,
    *,
    status: str,
    executed_at: str | None = None,
    failure_reason: str | None = None,
) -> None:
    """Atualiza o status da solicitação mantendo o contrato auditável."""

    payload = request._asdict()
    payload["status"] = status
    if executed_at is not None:
        payload["executed_at"] = executed_at
    if failure_reason is not None:
        payload["failure_reason"] = failure_reason
    write_json(request_path, payload)


def build_retraining_run_payload(
    request: RetrainingRequest,
    context: RetrainingRunContext,
) -> dict[str, Any]:
    """Monta o artifact final da execução do retreino."""

    payload: dict[str, Any] = {
        "request_id": request.request_id,
        "status": context.status,
        "started_at": context.started_at,
        "completed_at": context.completed_at,
        "reason": request.reason,
        "trigger_mode": request.trigger_mode,
        "promotion_policy": request.promotion_policy,
        "drift_status": request.drift_status,
        "max_feature_psi": request.max_feature_psi,
        "prediction_psi": request.prediction_psi,
        "drifted_features": request.drifted_features,
        "training_config_path": context.training_config_path,
        "challenger_training_config_path": context.challenger_training_config_path,
        "experiment_name": context.experiment_name,
        "model_output_path": context.model_output_path,
        "model_version": context.model_version,
    }
    if context.promotion_decision is not None:
        payload["promotion_decision"] = context.promotion_decision
    if context.metrics is not None:
        payload["metrics"] = context.metrics
    if context.failure_reason is not None:
        payload["failure_reason"] = context.failure_reason
    return payload


def create_challenger_training_config(
    request: RetrainingRequest,
) -> str:
    """Gera uma configuração temporária para treinar o challenger sem promover."""

    with open(request.training_config_path, "r", encoding="utf-8") as file_obj:
        config = yaml.safe_load(file_obj)

    request_suffix = request.request_id.split("-")[0]
    original_model_path = Path(config["artifacts"]["model_path"])
    challenger_model_path = (
        Path("artifacts/models/challengers")
        / f"{original_model_path.stem}_{request_suffix}.pkl"
    )

    config["experiment"]["run_name"] = (
        f'{config["experiment"]["run_name"]}_challenger_{request_suffix}'
    )
    config["experiment"]["version"] = (
        f'{config["experiment"]["version"]}-challenger-{request_suffix}'
    )
    config["artifacts"]["model_path"] = str(challenger_model_path)
    config.setdefault("mlflow", {}).setdefault("tags", {})
    config["mlflow"]["tags"]["candidate_type"] = "challenger"

    temp_dir = Path("artifacts/evaluation/model/retraining/generated_configs")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_config_path = temp_dir / f"retrain_{request.request_id}.yaml"
    temp_config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    return str(temp_config_path)


def run_retraining_request(
    request_path: str | Path = DEFAULT_RETRAIN_REQUEST_PATH,
    output_path: str | Path | None = DEFAULT_RETRAIN_RUN_PATH,
) -> dict[str, Any]:
    """Executa o retreino descrito em uma solicitação gerada pelo monitoramento."""

    request = load_retraining_request(request_path)
    started_at = now_isoformat()
    update_request_status(
        request_path,
        request,
        status="running",
        executed_at=started_at,
    )
    challenger_training_config_path = create_challenger_training_config(request)

    try:
        metrics = run_training(
            challenger_training_config_path,
            retraining_context=RetrainingMlflowContext(
                request_id=request.request_id,
                reason=request.reason,
                trigger_mode=request.trigger_mode,
                promotion_policy=request.promotion_policy,
                drift_status=request.drift_status,
                max_feature_psi=request.max_feature_psi,
                prediction_psi=request.prediction_psi,
                drifted_features=request.drifted_features,
                reference_row_count=request.reference_row_count,
                current_row_count=request.current_row_count,
            ),
        )
        champion_metadata_path = build_metadata_output_path(Path(request.model_path))
        challenger_training_cfg = load_experiment_training_config(
            challenger_training_config_path
        )
        challenger_metadata_path = build_metadata_output_path(
            challenger_training_cfg.model_path
        )
        promotion_decision = evaluate_challenger_promotion(
            request_id=request.request_id,
            champion_metadata_path=champion_metadata_path,
            challenger_metadata_path=challenger_metadata_path,
            output_path=request.promotion_decision_path,
            rules=request.promotion_rules,
        )
        completed_at = now_isoformat()
        result = build_retraining_run_payload(
            request,
            RetrainingRunContext(
                status="completed",
                started_at=started_at,
                completed_at=completed_at,
                training_config_path=request.training_config_path,
                challenger_training_config_path=challenger_training_config_path,
                model_output_path=str(challenger_training_cfg.model_path),
                model_version=challenger_training_cfg.model_version,
                experiment_name=challenger_training_cfg.experiment_name,
                promotion_decision=promotion_decision,
                metrics=metrics,
            ),
        )
        update_request_status(
            request_path,
            request,
            status="completed",
            executed_at=completed_at,
        )
        if output_path is not None:
            write_json(output_path, result)
        logger.info(
            "Retreino concluído com sucesso — request_id=%s | challenger=%s",
            request.request_id,
            challenger_training_cfg.experiment_name,
        )
        return result
    except Exception as exc:
        completed_at = now_isoformat()
        update_request_status(
            request_path,
            request,
            status="failed",
            executed_at=completed_at,
            failure_reason=str(exc),
        )
        result = build_retraining_run_payload(
            request,
            RetrainingRunContext(
                status="failed",
                started_at=started_at,
                completed_at=completed_at,
                training_config_path=request.training_config_path,
                challenger_training_config_path=challenger_training_config_path,
                model_output_path=request.model_path,
                model_version="unknown",
                experiment_name="unknown",
                failure_reason=str(exc),
            ),
        )
        if output_path is not None:
            write_json(output_path, result)
        logger.exception(
            "Falha ao executar retreino — request_id=%s",
            request.request_id,
        )
        raise


def parse_args() -> argparse.Namespace:
    """Lê argumentos da CLI do executor de retreino."""

    parser = argparse.ArgumentParser(
        description="Executa uma solicitação auditável de retreino do modelo.",
    )
    parser.add_argument(
        "--request",
        default=DEFAULT_RETRAIN_REQUEST_PATH,
        help="Caminho do arquivo JSON com a solicitação de retreino.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_RETRAIN_RUN_PATH,
        help="Caminho do arquivo JSON com o resultado da execução.",
    )
    return parser.parse_args()


def main() -> None:
    """Ponto de entrada do executor dedicado de retreino."""

    args = parse_args()
    run_retraining_request(
        request_path=args.request,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
