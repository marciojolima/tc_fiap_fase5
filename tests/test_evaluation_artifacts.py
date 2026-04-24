from __future__ import annotations

import json

from evaluation.ab_test_prompts import DEFAULT_HISTORY_OUT as PROMPT_HISTORY_OUT
from evaluation.ab_test_prompts import DEFAULT_OUT as PROMPT_OUT
from evaluation.artifacts import (
    append_jsonl,
    persist_result_with_history,
    relative_path,
    write_json,
)
from evaluation.llm_judge import DEFAULT_HISTORY_OUT as JUDGE_HISTORY_OUT
from evaluation.llm_judge import DEFAULT_OUT as JUDGE_OUT
from evaluation.ragas_eval import DEFAULT_HISTORY_OUT as RAGAS_HISTORY_OUT
from evaluation.ragas_eval import DEFAULT_OUT as RAGAS_OUT


def test_evaluation_defaults_point_to_artifacts() -> None:
    expected_paths = (
        RAGAS_OUT,
        JUDGE_OUT,
        PROMPT_OUT,
        RAGAS_HISTORY_OUT,
        JUDGE_HISTORY_OUT,
        PROMPT_HISTORY_OUT,
    )

    assert all(path.startswith("artifacts/evaluation/") for path in expected_paths)
    assert RAGAS_OUT.endswith("results/ragas_scores.json")
    assert JUDGE_OUT.endswith("results/llm_judge_scores.json")
    assert PROMPT_OUT.endswith("results/prompt_ab_results.json")
    assert RAGAS_HISTORY_OUT.endswith("runs/ragas_runs.jsonl")
    assert JUDGE_HISTORY_OUT.endswith("runs/llm_judge_runs.jsonl")
    assert PROMPT_HISTORY_OUT.endswith("runs/prompt_ab_runs.jsonl")


def test_write_json_and_append_jsonl(tmp_path) -> None:
    json_path = tmp_path / "results" / "sample.json"
    jsonl_path = tmp_path / "runs" / "sample.jsonl"

    write_json(json_path, {"status": "completed"})
    append_jsonl(jsonl_path, {"run": 1})
    append_jsonl(jsonl_path, {"run": 2})

    assert json.loads(json_path.read_text(encoding="utf-8"))["status"] == "completed"
    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["run"] for line in lines] == [1, 2]


def test_persist_result_with_history_writes_both_outputs(tmp_path) -> None:
    output_path = tmp_path / "results" / "report.json"
    history_path = tmp_path / "runs" / "history.jsonl"

    persist_result_with_history(
        output_path=output_path,
        history_path=history_path,
        result_payload={"schema": "report_v1"},
        history_payload={"schema": "history_v1"},
    )

    assert json.loads(output_path.read_text(encoding="utf-8"))["schema"] == "report_v1"
    assert (
        json.loads(history_path.read_text(encoding="utf-8"))["schema"]
        == "history_v1"
    )


def test_relative_path_keeps_project_paths_portable() -> None:
    assert (
        relative_path("configs/evaluation/golden_set.yaml")
        == "configs/evaluation/golden_set.yaml"
    )
