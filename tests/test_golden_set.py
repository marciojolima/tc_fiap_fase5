from __future__ import annotations

import json
from pathlib import Path

GOLDEN_PATH = Path("data/golden-set.json")
MIN_PAIRS = 20


def test_golden_set_exists_and_has_min_pairs() -> None:
    assert GOLDEN_PATH.is_file(), f"Faltando {GOLDEN_PATH}"

    data = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert data.get("schema_version") is not None

    items = data.get("items") or []
    n = len(items)
    assert n >= MIN_PAIRS, f"Exigido >= {MIN_PAIRS} pares, encontrado {n}"


def test_golden_set_items_have_query_and_ground_truth() -> None:
    data = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    for item in data.get("items") or []:
        assert "question" in item
        assert str(item["question"]).strip()
        assert "expected_answer" in item
        assert str(item["expected_answer"]).strip()
        assert "id" in item
        assert str(item["id"]).strip()
        assert isinstance(item.get("contexts"), list)
        assert isinstance(item.get("metadata"), dict)
