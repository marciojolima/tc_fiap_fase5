from __future__ import annotations

from pathlib import Path

import yaml

GOLDEN_PATH = Path("configs/evaluation/golden_set.yaml")
MIN_PAIRS = 20


def test_golden_set_exists_and_has_min_pairs() -> None:
    assert GOLDEN_PATH.is_file(), f"Faltando {GOLDEN_PATH}"

    data = yaml.safe_load(GOLDEN_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert data.get("schema_version") is not None

    items = data.get("items") or []
    assert len(items) >= MIN_PAIRS, f"Exigido >= {MIN_PAIRS} pares, encontrado {len(items)}"


def test_golden_set_items_have_query_and_ground_truth() -> None:
    data = yaml.safe_load(GOLDEN_PATH.read_text(encoding="utf-8"))
    for item in data.get("items") or []:
        assert "query" in item and str(item["query"]).strip()
        assert "expected_answer" in item and str(item["expected_answer"]).strip()
        assert "id" in item and str(item["id"]).strip()
