"""Consulta features online materializadas no Redis via Feast."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feast import FeatureStore

from common.logger import get_logger

from .config import (
    FEATURE_ENTITY_JOIN_KEY,
    FEATURE_STORE_REPO_PATH,
    build_feature_references,
)

logger = get_logger("feast_ops.demo")


def read_online_features(customer_id: int, repo_path: Path) -> dict[str, list]:
    """Lê do Redis as features online mais recentes de um cliente."""

    store = FeatureStore(repo_path=str(repo_path))
    online_response = store.get_online_features(
        features=build_feature_references(),
        entity_rows=[{FEATURE_ENTITY_JOIN_KEY: customer_id}],
    )
    return online_response.to_dict()


def build_argument_parser() -> argparse.ArgumentParser:
    """Cria a interface de linha de comando para a demo local."""

    parser = argparse.ArgumentParser(
        description="Consulta features materializadas na online store Redis.",
    )
    parser.add_argument("--customer-id", type=int, required=True)
    parser.add_argument(
        "--repo-path",
        type=Path,
        default=FEATURE_STORE_REPO_PATH,
        help="Caminho para o repositório Feast local.",
    )
    return parser


def main() -> None:
    """Executa a leitura online via Feast e imprime a resposta em JSON."""

    parser = build_argument_parser()
    args = parser.parse_args()

    response = read_online_features(
        customer_id=args.customer_id,
        repo_path=args.repo_path,
    )
    logger.info("Features online recuperadas para customer_id=%s", args.customer_id)
    print(json.dumps(response, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
