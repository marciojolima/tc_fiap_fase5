"""Helpers para persistir resultados de avaliação como artefatos auditáveis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from common.timezone import now_isoformat

RESULTS_DIR = Path("artifacts/evaluation/results")
RUNS_DIR = Path("artifacts/evaluation/runs")


def build_run_metadata() -> dict[str, str]:
    """Metadados mínimos compartilhados entre relatórios de avaliação."""

    return {
        "run_id": str(uuid4()),
        "created_at": now_isoformat(),
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Persiste JSON identado criando diretórios intermediários."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    """Acrescenta uma linha JSON em histórico append-only."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(payload, ensure_ascii=False) + "\n")


def persist_result_with_history(
    *,
    output_path: str | Path,
    history_path: str | Path,
    result_payload: dict[str, Any],
    history_payload: dict[str, Any],
) -> None:
    """Salva o resultado consolidado e registra um resumo no histórico JSONL."""

    write_json(output_path, result_payload)
    append_jsonl(history_path, history_payload)
