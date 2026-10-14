import json
from pathlib import Path

import pytest

import src.experiments.batch_runner as batch_runner
from src.experiments.batch_runner import (
    BatchExecutionSummary,
    run_experiment_batch,
)
from src.experiments.experiment_matrix import ExperimentalRunSpec, make_run_id
from src.experiments.runner import CollectionRecord, ExperimentalRunResult


def make_spec(
    replicate_index: int = 1,
    *,
    algorithm: str = "classical_nn",
) -> ExperimentalRunSpec:
    return ExperimentalRunSpec(
        point_count=5,
        replicate_index=replicate_index,
        scenario="nominal",
        algorithm=algorithm,  # type: ignore[arg-type]
        weight_profile=None if algorithm == "classical_nn" else "balanced",
        run_seed=None if algorithm in ("classical_nn", "mcnn") else 1,
    )


def make_result(spec: ExperimentalRunSpec) -> ExperimentalRunResult:
    return ExperimentalRunResult(
        point_count=spec.point_count,
        replicate_index=spec.replicate_index,
        instance_seed=5000 + spec.replicate_index,
        scenario=spec.scenario,
        algorithm=spec.algorithm,
        weight_profile=spec.weight_profile,
        run_seed=spec.run_seed,
        total_distance=123.0,
        simulated_duration=60.0,
        computational_seconds=1.5,
        final_load=5.0,
        planner_call_count=2,
        effective_optimization_count=2,
        interruption_count=0,
        collection_recalculation_count=2,
        thermal_recalculation_count=1,
        volume_recalculation_count=1,
        collection_order=(10, 20),
        collections=(CollectionRecord(10, 30.0, 2.0, 4.0),),
    )


def patch_fake_runner(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[ExperimentalRunSpec],
) -> None:
    def fake_run_experimental_case(
        point_count: int,
        replicate_index: int,
        scenario: str,
        algorithm: str,
        *,
        weight_profile: str | None,
        run_seed: int | None,
    ) -> ExperimentalRunResult:
        spec = ExperimentalRunSpec(
            point_count=point_count,
            replicate_index=replicate_index,
            scenario=scenario,  # type: ignore[arg-type]
            algorithm=algorithm,  # type: ignore[arg-type]
            weight_profile=weight_profile,  # type: ignore[arg-type]
            run_seed=run_seed,
        )
        calls.append(spec)
        return make_result(spec)

    monkeypatch.setattr(
        batch_runner,
        "run_experimental_case",
        fake_run_experimental_case,
    )


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_missing_checkpoint_executes_and_persists_multiple_specs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[ExperimentalRunSpec] = []
    patch_fake_runner(monkeypatch, calls)
    specs = (make_spec(1), make_spec(2))
    results_path = tmp_path / "results" / "runs.jsonl"
    errors_path = tmp_path / "errors" / "errors.jsonl"

    summary = run_experiment_batch(specs, results_path, errors_path)

    assert summary == BatchExecutionSummary(2, 2, 0)
    assert calls == list(specs)
    records = read_jsonl(results_path)
    assert len(records) == 2
    assert [record["run_id"] for record in records] == [
        make_run_id(spec) for spec in specs
    ]
    assert records[0]["collections"] == [
        {
            "point_id": 10,
            "completion_time": 30.0,
            "collected_volume": 2.0,
            "temperature_at_completion": 4.0,
        }
    ]


def test_second_execution_skips_all_completed_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[ExperimentalRunSpec] = []
    patch_fake_runner(monkeypatch, calls)
    specs = (make_spec(1), make_spec(2))
    results_path = tmp_path / "results.jsonl"
    errors_path = tmp_path / "errors.jsonl"

    first_summary = run_experiment_batch(specs, results_path, errors_path)
    second_summary = run_experiment_batch(specs, results_path, errors_path)

    assert first_summary == BatchExecutionSummary(2, 2, 0)
    assert second_summary == BatchExecutionSummary(2, 0, 2)
    assert calls == list(specs)
    assert len(read_jsonl(results_path)) == 2


def test_partial_checkpoint_executes_only_missing_specs_in_input_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[ExperimentalRunSpec] = []
    patch_fake_runner(monkeypatch, calls)
    first_spec = make_spec(1)
    all_specs = (first_spec, make_spec(2), make_spec(3))
    results_path = tmp_path / "results.jsonl"
    errors_path = tmp_path / "errors.jsonl"

    run_experiment_batch((first_spec,), results_path, errors_path)
    calls.clear()
    summary = run_experiment_batch(all_specs, results_path, errors_path)

    assert summary == BatchExecutionSummary(3, 2, 1)
    assert calls == list(all_specs[1:])


def test_parent_directories_are_created(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_fake_runner(monkeypatch, [])
    results_path = tmp_path / "nested" / "results" / "runs.jsonl"
    errors_path = tmp_path / "nested" / "errors" / "errors.jsonl"

    run_experiment_batch((make_spec(),), results_path, errors_path)

    assert results_path.exists()
    assert errors_path.parent.is_dir()


@pytest.mark.parametrize(
    "checkpoint_content",
    [
        "not json\n",
        '{"value": 1}\n',
        '{"run_id": "same"}\n{"run_id": "same"}\n',
    ],
)
def test_invalid_checkpoint_is_rejected(
    tmp_path: Path,
    checkpoint_content: str,
) -> None:
    results_path = tmp_path / "results.jsonl"
    results_path.write_text(checkpoint_content, encoding="utf-8")

    with pytest.raises(ValueError):
        run_experiment_batch((make_spec(),), results_path, tmp_path / "errors.jsonl")


def test_run_error_is_persisted_and_reraised(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_run(*_args: object, **_kwargs: object) -> ExperimentalRunResult:
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(batch_runner, "run_experimental_case", failing_run)
    results_path = tmp_path / "results.jsonl"
    errors_path = tmp_path / "errors.jsonl"
    spec = make_spec()

    with pytest.raises(RuntimeError, match="synthetic failure"):
        run_experiment_batch((spec,), results_path, errors_path)

    error_records = read_jsonl(errors_path)
    assert error_records == [
        {
            "run_id": make_run_id(spec),
            "point_count": 5,
            "replicate_index": 1,
            "scenario": "nominal",
            "algorithm": "classical_nn",
            "weight_profile": None,
            "run_seed": None,
            "error_type": "RuntimeError",
            "error_message": "synthetic failure",
        }
    ]
    assert not results_path.exists()


def test_summary_contains_requested_executed_and_skipped_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[ExperimentalRunSpec] = []
    patch_fake_runner(monkeypatch, calls)
    specs = (make_spec(1), make_spec(2))
    results_path = tmp_path / "results.jsonl"
    errors_path = tmp_path / "errors.jsonl"
    results_path.write_text(
        json.dumps({"run_id": make_run_id(specs[0])}) + "\n",
        encoding="utf-8",
    )

    summary = run_experiment_batch(specs, results_path, errors_path)

    assert summary.requested_count == 2
    assert summary.executed_count == 1
    assert summary.skipped_count == 1
