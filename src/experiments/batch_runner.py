"""Sequential execution with incremental JSON Lines checkpoints."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from src.experiments.experiment_matrix import ExperimentalRunSpec, make_run_id
from src.experiments.runner import run_experimental_case


@dataclass(frozen=True)
class BatchExecutionSummary:
    """Counts for one sequential batch execution."""

    requested_count: int
    executed_count: int
    skipped_count: int


def _read_completed_run_ids(results_path: Path) -> set[str]:
    """Read and validate completed run identifiers from a JSONL checkpoint."""
    if not results_path.exists():
        return set()

    completed_run_ids: set[str] = set()
    with results_path.open("r", encoding="utf-8") as results_file:
        for line_number, line in enumerate(results_file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON in results checkpoint at line {line_number}."
                ) from exc

            if not isinstance(record, dict) or "run_id" not in record:
                raise ValueError(
                    f"results checkpoint line {line_number} must contain run_id."
                )

            run_id = record["run_id"]
            if not isinstance(run_id, str) or not run_id:
                raise ValueError(
                    f"results checkpoint line {line_number} has an invalid run_id."
                )
            if run_id in completed_run_ids:
                raise ValueError(
                    f"results checkpoint contains duplicate run_id: {run_id}."
                )
            completed_run_ids.add(run_id)

    return completed_run_ids


def _append_json_line(path: Path, record: dict[str, object]) -> None:
    """Append one JSON object and flush it before returning."""
    with path.open("a", encoding="utf-8") as output_file:
        json.dump(record, output_file, ensure_ascii=False)
        output_file.write("\n")
        output_file.flush()


def _error_record(
    spec: ExperimentalRunSpec,
    run_id: str,
    error: Exception,
) -> dict[str, object]:
    """Build the checkpoint record for one failed run."""
    return {
        "run_id": run_id,
        "point_count": spec.point_count,
        "replicate_index": spec.replicate_index,
        "scenario": spec.scenario,
        "algorithm": spec.algorithm,
        "weight_profile": spec.weight_profile,
        "run_seed": spec.run_seed,
        "error_type": type(error).__name__,
        "error_message": str(error),
    }


def run_experiment_batch(
    specs: Iterable[ExperimentalRunSpec],
    results_path: Path,
    errors_path: Path,
) -> BatchExecutionSummary:
    """Execute specs sequentially with resumable JSONL result and error logs."""
    results_path.parent.mkdir(parents=True, exist_ok=True)
    errors_path.parent.mkdir(parents=True, exist_ok=True)
    completed_run_ids = _read_completed_run_ids(results_path)

    requested_count = 0
    executed_count = 0
    skipped_count = 0

    for spec in specs:
        requested_count += 1
        run_id = make_run_id(spec)
        if run_id in completed_run_ids:
            skipped_count += 1
            continue

        try:
            result = run_experimental_case(
                spec.point_count,
                spec.replicate_index,
                spec.scenario,
                spec.algorithm,
                weight_profile=spec.weight_profile,
                run_seed=spec.run_seed,
            )
        except Exception as error:
            _append_json_line(errors_path, _error_record(spec, run_id, error))
            raise

        result_record = {"run_id": run_id, **asdict(result)}
        _append_json_line(results_path, result_record)
        completed_run_ids.add(run_id)
        executed_count += 1

    return BatchExecutionSummary(
        requested_count=requested_count,
        executed_count=executed_count,
        skipped_count=skipped_count,
    )
