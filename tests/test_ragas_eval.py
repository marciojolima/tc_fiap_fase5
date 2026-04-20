from __future__ import annotations

from pathlib import Path

from evaluation.ragas_eval import load_golden_items


def test_load_golden_items_from_default_path() -> None:
    path = Path("configs/evaluation/golden_set.yaml")
    items = load_golden_items(path)
    assert len(items) >= 20
    assert all("query" in i and "expected_answer" in i for i in items)
