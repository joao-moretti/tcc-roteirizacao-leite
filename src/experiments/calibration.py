"""Pilot calibration infrastructure for the V1 SA and ACO algorithms."""

import argparse
import time
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
from math import isfinite
from statistics import mean, pstdev
from typing import Literal

from src.algorithms.ant_colony import ant_colony_optimization
from src.algorithms.nearest_neighbor import multicriteria_nearest_neighbor
from src.algorithms.simulated_annealing import simulated_annealing
from src.experiments.experimental_presets import (
    EXPERIMENTAL_T_RISK_MAX,
    EXPERIMENTAL_T_RISK_MIN,
    ThermalDisturbance,
    generate_experimental_problem_instance,
)
from src.models.candidate_route import CandidateRoute
from src.models.problem import ProblemInstance
from src.objective.logistics import dynamic_logistic_reference
from src.objective.multicriteria import weighted_objective_cost
from src.objective.occupancy import occupancy_reference
from src.objective.thermal import thermal_reference
from src.objective.weight_profiles import BALANCED_WEIGHTS
from src.simulation.snapshot import OptimizationSnapshot
from src.simulation.state import (
    build_optimization_snapshot,
    create_initial_simulation_state,
)

CalibrationAlgorithm = Literal["sa", "aco"]

CALIBRATION_POINT_COUNTS = (5, 15, 20)
CALIBRATION_INSTANCE_SEEDS_BY_POINT_COUNT = {
    5: (55001, 55002, 55003),
    15: (155001, 155002, 155003),
    20: (205001, 205002, 205003),
}
CALIBRATION_ALGORITHM_SEEDS = (9101, 9102, 9103, 9104, 9105)

SA_INITIAL_TEMPERATURES = (0.10, 0.50)
SA_COOLING_RATES = (0.95, 0.98, 0.995)
SA_ITERATIONS = (100, 300, 500)

ACO_BETAS = (1.0, 2.0, 3.0)
ACO_EVAPORATION_RATES = (0.10, 0.30, 0.50)


@dataclass(frozen=True)
class SAParameterSet:
    """One official simulated-annealing calibration configuration."""

    initial_temperature: float
    cooling_rate: float
    iterations: int


@dataclass(frozen=True)
class ACOParameterSet:
    """One official ant-colony calibration configuration."""

    beta: float
    evaporation_rate: float


SA_CALIBRATION_GRID = tuple(
    SAParameterSet(initial_temperature, cooling_rate, iterations)
    for initial_temperature, cooling_rate, iterations in product(
        SA_INITIAL_TEMPERATURES,
        SA_COOLING_RATES,
        SA_ITERATIONS,
    )
)

ACO_CALIBRATION_GRID = tuple(
    ACOParameterSet(beta, evaporation_rate)
    for beta, evaporation_rate in product(ACO_BETAS, ACO_EVAPORATION_RATES)
)


def _validate_point_count(point_count: object) -> int:
    """Validate one of the official calibration scales."""
    if (
        isinstance(point_count, bool)
        or not isinstance(point_count, int)
        or point_count not in CALIBRATION_POINT_COUNTS
    ):
        raise ValueError("point_count must be one of the official calibration scales.")
    return point_count


def _validate_non_negative_int(value: object, field_name: str) -> int:
    """Validate a non-negative integer metadata value."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer.")
    return value


def _validate_finite_non_negative(value: object, field_name: str) -> float:
    """Validate and normalize a finite non-negative metric."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be finite and non-negative.")

    normalized = float(value)
    if not isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{field_name} must be finite and non-negative.")
    return normalized


def _validate_algorithm(algorithm: object) -> CalibrationAlgorithm:
    """Validate one calibration algorithm label."""
    if algorithm not in ("sa", "aco"):
        raise ValueError("algorithm must be 'sa' or 'aco'.")
    return algorithm  # type: ignore[return-value]


@dataclass(frozen=True)
class CalibrationRun:
    """Record one timed algorithm execution in the pilot calibration."""

    algorithm: CalibrationAlgorithm
    point_count: int
    instance_seed: int
    algorithm_seed: int
    parameters: SAParameterSet | ACOParameterSet
    objective_cost: float
    elapsed_seconds: float

    def __post_init__(self) -> None:
        algorithm = _validate_algorithm(self.algorithm)
        point_count = _validate_point_count(self.point_count)
        instance_seed = _validate_non_negative_int(
            self.instance_seed,
            "instance_seed",
        )
        algorithm_seed = _validate_non_negative_int(
            self.algorithm_seed,
            "algorithm_seed",
        )
        expected_type = SAParameterSet if algorithm == "sa" else ACOParameterSet
        if not isinstance(self.parameters, expected_type):
            raise ValueError(
                f"parameters must be {expected_type.__name__} for {algorithm}."
            )

        objective_cost = _validate_finite_non_negative(
            self.objective_cost,
            "objective_cost",
        )
        elapsed_seconds = _validate_finite_non_negative(
            self.elapsed_seconds,
            "elapsed_seconds",
        )

        object.__setattr__(self, "algorithm", algorithm)
        object.__setattr__(self, "point_count", point_count)
        object.__setattr__(self, "instance_seed", instance_seed)
        object.__setattr__(self, "algorithm_seed", algorithm_seed)
        object.__setattr__(self, "objective_cost", objective_cost)
        object.__setattr__(self, "elapsed_seconds", elapsed_seconds)


@dataclass(frozen=True)
class CalibrationSummary:
    """Aggregate metrics for one calibration configuration."""

    algorithm: CalibrationAlgorithm
    point_count: int
    parameters: SAParameterSet | ACOParameterSet
    mean_objective_cost: float
    std_objective_cost: float
    mean_elapsed_seconds: float
    run_count: int

    def __post_init__(self) -> None:
        algorithm = _validate_algorithm(self.algorithm)
        point_count = _validate_point_count(self.point_count)
        expected_type = SAParameterSet if algorithm == "sa" else ACOParameterSet
        if not isinstance(self.parameters, expected_type):
            raise ValueError(
                f"parameters must be {expected_type.__name__} for {algorithm}."
            )

        mean_objective_cost = _validate_finite_non_negative(
            self.mean_objective_cost,
            "mean_objective_cost",
        )
        std_objective_cost = _validate_finite_non_negative(
            self.std_objective_cost,
            "std_objective_cost",
        )
        mean_elapsed_seconds = _validate_finite_non_negative(
            self.mean_elapsed_seconds,
            "mean_elapsed_seconds",
        )
        run_count = _validate_non_negative_int(self.run_count, "run_count")
        if run_count == 0:
            raise ValueError("run_count must be greater than zero.")

        object.__setattr__(self, "algorithm", algorithm)
        object.__setattr__(self, "point_count", point_count)
        object.__setattr__(self, "mean_objective_cost", mean_objective_cost)
        object.__setattr__(self, "std_objective_cost", std_objective_cost)
        object.__setattr__(self, "mean_elapsed_seconds", mean_elapsed_seconds)
        object.__setattr__(self, "run_count", run_count)


_CalibrationCase = tuple[
    ProblemInstance,
    OptimizationSnapshot,
    int,
    float,
    float,
    float,
    CandidateRoute,
    float,
]


def _weighted_cost(
    instance: ProblemInstance,
    snapshot: OptimizationSnapshot,
    candidate_route: CandidateRoute,
    logistic_reference_distance: float,
    thermal_reference_value: float,
    occupancy_reference_value: float,
) -> float:
    """Calculate the balanced multicriteria cost for one candidate route."""
    return weighted_objective_cost(
        instance,
        snapshot,
        candidate_route,
        logistic_reference_distance,
        thermal_reference_value,
        occupancy_reference_value,
        EXPERIMENTAL_T_RISK_MIN,
        EXPERIMENTAL_T_RISK_MAX,
        *BALANCED_WEIGHTS,
    )


def _prepare_case(
    instance: ProblemInstance,
    snapshot: OptimizationSnapshot,
    instance_seed: int,
) -> _CalibrationCase:
    """Prepare references and the shared baseline outside algorithm timing."""
    snapshot.validate(instance)
    logistic_reference_distance = dynamic_logistic_reference(instance, snapshot)
    thermal_reference_value = thermal_reference(
        instance,
        snapshot,
        EXPERIMENTAL_T_RISK_MIN,
        EXPERIMENTAL_T_RISK_MAX,
    )
    occupancy_reference_value = occupancy_reference(instance, snapshot)
    initial_route = multicriteria_nearest_neighbor(
        instance,
        snapshot,
        EXPERIMENTAL_T_RISK_MIN,
        EXPERIMENTAL_T_RISK_MAX,
        BALANCED_WEIGHTS,
    )
    baseline_cost = _weighted_cost(
        instance,
        snapshot,
        initial_route,
        logistic_reference_distance,
        thermal_reference_value,
        occupancy_reference_value,
    )
    return (
        instance,
        snapshot,
        instance_seed,
        logistic_reference_distance,
        thermal_reference_value,
        occupancy_reference_value,
        initial_route,
        baseline_cost,
    )


def _generate_pilot_instances(
    point_count: int,
) -> tuple[tuple[int, ProblemInstance, ThermalDisturbance | None], ...]:
    """Generate the three nominal pilot instances for one official scale."""
    point_count = _validate_point_count(point_count)
    pilot_instances: list[tuple[int, ProblemInstance, ThermalDisturbance | None]] = []
    for pilot_index, instance_seed in enumerate(
        CALIBRATION_INSTANCE_SEEDS_BY_POINT_COUNT[point_count],
        start=1,
    ):
        instance, disturbance = generate_experimental_problem_instance(
            instance_id=point_count * 1000 + pilot_index,
            network_id=point_count * 1000 + pilot_index,
            instance_seed=instance_seed,
            point_count=point_count,
            scenario="nominal",
        )
        pilot_instances.append((instance_seed, instance, disturbance))
    return tuple(pilot_instances)


def _prepare_pilot_cases(point_count: int) -> tuple[_CalibrationCase, ...]:
    """Generate and prepare all pilot cases for one scale."""
    cases: list[_CalibrationCase] = []
    for seed, instance, disturbance in _generate_pilot_instances(point_count):
        if disturbance is not None:
            raise ValueError(
                "nominal calibration pilot unexpectedly has a disturbance."
            )
        state = create_initial_simulation_state(instance)
        snapshot = build_optimization_snapshot(instance, state)
        cases.append(_prepare_case(instance, snapshot, seed))
    return tuple(cases)


def _validate_parameters(
    algorithm: CalibrationAlgorithm,
    parameters: SAParameterSet | ACOParameterSet,
) -> None:
    """Ensure a configuration belongs to the requested algorithm."""
    expected_type = SAParameterSet if algorithm == "sa" else ACOParameterSet
    if not isinstance(parameters, expected_type):
        raise ValueError(
            f"parameters must be {expected_type.__name__} for {algorithm}."
        )


def _evaluate_sa_case(
    case: _CalibrationCase,
    parameters: SAParameterSet,
    algorithm_seed: int,
    point_count: int,
) -> CalibrationRun:
    """Evaluate one SA configuration on one prepared pilot case."""
    (
        instance,
        snapshot,
        instance_seed,
        logistic_reference_distance,
        thermal_reference_value,
        occupancy_reference_value,
        initial_route,
        _baseline_cost,
    ) = case
    algorithm_start = time.perf_counter()
    result_route = simulated_annealing(
        instance,
        snapshot,
        initial_route,
        logistic_reference_distance,
        thermal_reference_value,
        occupancy_reference_value,
        EXPERIMENTAL_T_RISK_MIN,
        EXPERIMENTAL_T_RISK_MAX,
        BALANCED_WEIGHTS,
        parameters.initial_temperature,
        parameters.cooling_rate,
        parameters.iterations,
        algorithm_seed,
    )
    elapsed_seconds = time.perf_counter() - algorithm_start
    objective_cost = _weighted_cost(
        instance,
        snapshot,
        result_route,
        logistic_reference_distance,
        thermal_reference_value,
        occupancy_reference_value,
    )
    return CalibrationRun(
        algorithm="sa",
        point_count=point_count,
        instance_seed=instance_seed,
        algorithm_seed=algorithm_seed,
        parameters=parameters,
        objective_cost=objective_cost,
        elapsed_seconds=elapsed_seconds,
    )


def _evaluate_aco_case(
    case: _CalibrationCase,
    parameters: ACOParameterSet,
    algorithm_seed: int,
    point_count: int,
) -> CalibrationRun:
    """Evaluate one ACO configuration on one prepared pilot case."""
    (
        instance,
        snapshot,
        instance_seed,
        logistic_reference_distance,
        thermal_reference_value,
        occupancy_reference_value,
        _initial_route,
        baseline_cost,
    ) = case
    algorithm_start = time.perf_counter()
    result_route = ant_colony_optimization(
        instance,
        snapshot,
        baseline_cost,
        logistic_reference_distance,
        thermal_reference_value,
        occupancy_reference_value,
        EXPERIMENTAL_T_RISK_MIN,
        EXPERIMENTAL_T_RISK_MAX,
        BALANCED_WEIGHTS,
        parameters.beta,
        parameters.evaporation_rate,
        algorithm_seed,
    )
    elapsed_seconds = time.perf_counter() - algorithm_start
    objective_cost = _weighted_cost(
        instance,
        snapshot,
        result_route,
        logistic_reference_distance,
        thermal_reference_value,
        occupancy_reference_value,
    )
    return CalibrationRun(
        algorithm="aco",
        point_count=point_count,
        instance_seed=instance_seed,
        algorithm_seed=algorithm_seed,
        parameters=parameters,
        objective_cost=objective_cost,
        elapsed_seconds=elapsed_seconds,
    )


def evaluate_sa_configuration(
    instance: ProblemInstance,
    snapshot: OptimizationSnapshot,
    parameters: SAParameterSet,
    algorithm_seed: int,
    *,
    point_count: int,
    instance_seed: int,
) -> CalibrationRun:
    """Evaluate one SA configuration after preparing its initial snapshot."""
    point_count = _validate_point_count(point_count)
    algorithm_seed = _validate_non_negative_int(algorithm_seed, "algorithm_seed")
    instance_seed = _validate_non_negative_int(instance_seed, "instance_seed")
    _validate_parameters("sa", parameters)
    case = _prepare_case(instance, snapshot, instance_seed)
    return _evaluate_sa_case(case, parameters, algorithm_seed, point_count)


def evaluate_aco_configuration(
    instance: ProblemInstance,
    snapshot: OptimizationSnapshot,
    parameters: ACOParameterSet,
    algorithm_seed: int,
    *,
    point_count: int,
    instance_seed: int,
) -> CalibrationRun:
    """Evaluate one ACO configuration after preparing its initial snapshot."""
    point_count = _validate_point_count(point_count)
    algorithm_seed = _validate_non_negative_int(algorithm_seed, "algorithm_seed")
    instance_seed = _validate_non_negative_int(instance_seed, "instance_seed")
    _validate_parameters("aco", parameters)
    case = _prepare_case(instance, snapshot, instance_seed)
    return _evaluate_aco_case(case, parameters, algorithm_seed, point_count)


def _summarize_runs(runs: Sequence[CalibrationRun]) -> CalibrationSummary:
    """Aggregate homogeneous calibration runs into one summary."""
    normalized_runs = tuple(runs)
    if not normalized_runs:
        raise ValueError("runs must contain at least one calibration run.")

    first_run = normalized_runs[0]
    for run in normalized_runs[1:]:
        if run.algorithm != first_run.algorithm:
            raise ValueError("calibration runs must use the same algorithm.")
        if run.point_count != first_run.point_count:
            raise ValueError("calibration runs must use the same point_count.")
        if (
            type(run.parameters) is not type(first_run.parameters)
            or run.parameters != first_run.parameters
        ):
            raise ValueError("calibration runs must use the same parameters.")

    return CalibrationSummary(
        algorithm=first_run.algorithm,
        point_count=first_run.point_count,
        parameters=first_run.parameters,
        mean_objective_cost=mean(run.objective_cost for run in normalized_runs),
        std_objective_cost=pstdev(run.objective_cost for run in normalized_runs),
        mean_elapsed_seconds=mean(run.elapsed_seconds for run in normalized_runs),
        run_count=len(normalized_runs),
    )


def _summary_sort_key(summary: CalibrationSummary) -> tuple[float, float, float]:
    """Return the official presentation ranking key."""
    return (
        summary.mean_objective_cost,
        summary.std_objective_cost,
        summary.mean_elapsed_seconds,
    )


def run_sa_calibration(point_count: int) -> tuple[CalibrationSummary, ...]:
    """Run the complete SA pilot grid for one official scale."""
    point_count = _validate_point_count(point_count)
    cases = _prepare_pilot_cases(point_count)
    summaries: list[CalibrationSummary] = []

    for parameters in SA_CALIBRATION_GRID:
        runs = [
            _evaluate_sa_case(case, parameters, algorithm_seed, point_count)
            for case in cases
            for algorithm_seed in CALIBRATION_ALGORITHM_SEEDS
        ]
        if len(runs) != 15:
            raise ValueError("each calibration configuration must have 15 runs.")
        summaries.append(_summarize_runs(runs))

    return tuple(sorted(summaries, key=_summary_sort_key))


def run_aco_calibration(point_count: int) -> tuple[CalibrationSummary, ...]:
    """Run the complete ACO pilot grid for one official scale."""
    point_count = _validate_point_count(point_count)
    cases = _prepare_pilot_cases(point_count)
    summaries: list[CalibrationSummary] = []

    for parameters in ACO_CALIBRATION_GRID:
        runs = [
            _evaluate_aco_case(case, parameters, algorithm_seed, point_count)
            for case in cases
            for algorithm_seed in CALIBRATION_ALGORITHM_SEEDS
        ]
        if len(runs) != 15:
            raise ValueError("each calibration configuration must have 15 runs.")
        summaries.append(_summarize_runs(runs))

    return tuple(sorted(summaries, key=_summary_sort_key))


def _parse_arguments() -> argparse.Namespace:
    """Parse the minimal calibration command-line interface."""
    parser = argparse.ArgumentParser(description="Run V1 pilot calibration.")
    parser.add_argument("--algorithm", choices=("sa", "aco"), required=True)
    parser.add_argument(
        "--point-count",
        type=int,
        choices=CALIBRATION_POINT_COUNTS,
        required=True,
    )
    return parser.parse_args()


def _main() -> None:
    """Run calibration from the command line and print ranked summaries."""
    arguments = _parse_arguments()
    summaries = (
        run_sa_calibration(arguments.point_count)
        if arguments.algorithm == "sa"
        else run_aco_calibration(arguments.point_count)
    )
    for rank, summary in enumerate(summaries, start=1):
        print(
            f"rank={rank} parameters={summary.parameters!r} "
            f"mean_objective_cost={summary.mean_objective_cost} "
            f"std_objective_cost={summary.std_objective_cost} "
            f"mean_elapsed_seconds={summary.mean_elapsed_seconds} "
            f"run_count={summary.run_count}"
        )


if __name__ == "__main__":
    _main()
