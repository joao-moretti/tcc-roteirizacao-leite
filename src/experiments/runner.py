"""Execution and recording of one complete experimental simulation run."""

import time
from dataclasses import dataclass
from math import isfinite
from typing import Literal

from src.experiments.experimental_instances import generate_experimental_instance_pair
from src.experiments.experimental_presets import (
    EXPERIMENTAL_MINIMUM_RECALCULATION_INTERVAL,
    EXPERIMENTAL_Q_HIGH,
    EXPERIMENTAL_Q_LOW,
    EXPERIMENTAL_T_RISK_MAX,
    EXPERIMENTAL_T_RISK_MIN,
    ExperimentalScenario,
)
from src.experiments.planners import (
    EXPERIMENTAL_ALGORITHM_SEEDS,
    make_aco_planner,
    make_classical_nn_planner,
    make_multicriteria_nn_planner,
    make_sa_planner,
)
from src.models.problem import ProblemInstance
from src.objective.weight_profiles import (
    BALANCED_WEIGHTS,
    LOGISTIC_WEIGHTS,
    THERMAL_WEIGHTS,
)
from src.simulation.dynamic import DynamicSimulationResult, run_dynamic_simulation
from src.simulation.executor import NextCollectionExecution
from src.simulation.state import trajectory_sample_index

ExperimentalAlgorithm = Literal[
    "classical_nn",
    "mcnn",
    "sa",
    "aco",
]

WeightProfileName = Literal[
    "logistic",
    "balanced",
    "thermal",
]

EXPERIMENTAL_WEIGHT_PROFILES = {
    "logistic": LOGISTIC_WEIGHTS,
    "balanced": BALANCED_WEIGHTS,
    "thermal": THERMAL_WEIGHTS,
}


@dataclass(frozen=True)
class CollectionRecord:
    """Record the completion data of one collected producer."""

    point_id: int
    completion_time: float
    collected_volume: float
    temperature_at_completion: float


@dataclass(frozen=True)
class ExperimentalRunResult:
    """Immutable metrics and metadata from one experimental run."""

    point_count: int
    replicate_index: int
    instance_seed: int

    scenario: ExperimentalScenario
    algorithm: ExperimentalAlgorithm
    weight_profile: WeightProfileName | None
    run_seed: int | None

    total_distance: float
    simulated_duration: float
    computational_seconds: float
    final_load: float

    planner_call_count: int
    effective_optimization_count: int
    interruption_count: int

    collection_recalculation_count: int
    thermal_recalculation_count: int
    volume_recalculation_count: int

    collection_order: tuple[int, ...]
    collections: tuple[CollectionRecord, ...]


def _validate_algorithm(algorithm: object) -> ExperimentalAlgorithm:
    """Validate the selected experimental algorithm."""
    if algorithm not in ("classical_nn", "mcnn", "sa", "aco"):
        raise ValueError("algorithm must be one of: classical_nn, mcnn, sa, aco.")
    return algorithm  # type: ignore[return-value]


def _validate_scenario(scenario: object) -> ExperimentalScenario:
    """Validate the selected experimental scenario."""
    if scenario not in ("nominal", "disturbance"):
        raise ValueError("scenario must be 'nominal' or 'disturbance'.")
    return scenario  # type: ignore[return-value]


def _validate_weight_profile(
    algorithm: ExperimentalAlgorithm,
    weight_profile: object,
) -> WeightProfileName | None:
    """Validate profile requirements for deterministic and multicriteria runs."""
    if algorithm == "classical_nn":
        if weight_profile is not None:
            raise ValueError("classical_nn requires weight_profile=None.")
        return None

    if not isinstance(weight_profile, str) or weight_profile not in (
        "logistic",
        "balanced",
        "thermal",
    ):
        raise ValueError(
            "mcnn, sa, and aco require weight_profile to be logistic, balanced, "
            "or thermal."
        )
    return weight_profile  # type: ignore[return-value]


def _validate_run_seed(
    algorithm: ExperimentalAlgorithm,
    run_seed: object,
) -> int | None:
    """Validate deterministic and stochastic run-seed requirements."""
    if algorithm in ("classical_nn", "mcnn"):
        if run_seed is not None:
            raise ValueError(f"{algorithm} requires run_seed=None.")
        return None

    if (
        isinstance(run_seed, bool)
        or not isinstance(run_seed, int)
        or run_seed not in EXPERIMENTAL_ALGORITHM_SEEDS
    ):
        raise ValueError(
            "sa and aco require run_seed to be one of the 30 official seeds."
        )
    return run_seed


def _make_planner(
    algorithm: ExperimentalAlgorithm,
    instance: ProblemInstance,
    weights: tuple[float, float, float] | None,
    run_seed: int | None,
):
    """Build the planner selected for one experimental run."""
    if algorithm == "classical_nn":
        return make_classical_nn_planner(instance)

    if weights is None:
        raise ValueError("multicriteria algorithms require weights.")
    if algorithm == "mcnn":
        return make_multicriteria_nn_planner(weights)
    if run_seed is None:
        raise ValueError("stochastic algorithms require run_seed.")
    if algorithm == "sa":
        return make_sa_planner(weights, run_seed)
    return make_aco_planner(weights, run_seed)


def _collection_record(
    instance: ProblemInstance,
    execution: NextCollectionExecution,
) -> CollectionRecord:
    """Create one collection record from a successful execution."""
    if execution.collected_volume is None:
        raise ValueError(
            "a reached collection execution must contain collected_volume."
        )

    completion_time = execution.snapshot.current_time
    sample_index = trajectory_sample_index(instance, completion_time)
    temperature = instance.temperature_trajectories[execution.target_point_id][
        sample_index
    ]
    return CollectionRecord(
        point_id=execution.target_point_id,
        completion_time=completion_time,
        collected_volume=execution.collected_volume,
        temperature_at_completion=temperature,
    )


def _validate_non_negative_metric(value: object, field_name: str) -> float:
    """Validate and normalize a finite non-negative run metric."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be finite and non-negative.")
    normalized = float(value)
    if not isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{field_name} must be finite and non-negative.")
    return normalized


def _build_result(
    instance: ProblemInstance,
    point_count: int,
    replicate_index: int,
    instance_seed: int,
    scenario: ExperimentalScenario,
    algorithm: ExperimentalAlgorithm,
    weight_profile: WeightProfileName | None,
    run_seed: int | None,
    simulation_result: DynamicSimulationResult,
    computational_seconds: float,
) -> ExperimentalRunResult:
    """Extract and validate all experiment-level metrics from a simulation."""
    final_snapshot = simulation_result.final_snapshot
    if final_snapshot.remaining_point_ids != ():
        raise ValueError("the final simulation snapshot still has remaining points.")
    if final_snapshot.current_node_id != instance.depot.node_id:
        raise ValueError("the final simulation snapshot is not at the depot.")

    successful_executions = tuple(
        execution
        for execution in simulation_result.executions
        if execution.reached_target
    )
    collection_order = tuple(
        execution.target_point_id for execution in successful_executions
    )
    expected_point_ids = {
        collection_point.point_id for collection_point in instance.collection_points
    }
    if len(collection_order) != point_count:
        raise ValueError("collection_order does not contain all producers.")
    if set(collection_order) != expected_point_ids:
        raise ValueError("collection_order does not match the instance producers.")
    if len(set(collection_order)) != len(collection_order):
        raise ValueError("collection_order contains duplicate producers.")

    collections = tuple(
        _collection_record(instance, execution) for execution in successful_executions
    )
    if len(collections) != point_count:
        raise ValueError("collections does not contain all producers.")

    total_distance = _validate_non_negative_metric(
        simulation_result.total_distance,
        "total_distance",
    )
    simulated_duration = _validate_non_negative_metric(
        final_snapshot.current_time,
        "simulated_duration",
    )
    computational_seconds = _validate_non_negative_metric(
        computational_seconds,
        "computational_seconds",
    )
    final_load = _validate_non_negative_metric(
        final_snapshot.current_load, "final_load"
    )
    if final_load > instance.vehicle_capacity:
        raise ValueError("final_load exceeds instance.vehicle_capacity.")

    planner_call_count = simulation_result.optimization_count
    if (
        isinstance(planner_call_count, bool)
        or not isinstance(planner_call_count, int)
        or planner_call_count < 0
    ):
        raise ValueError("planner_call_count must be a non-negative integer.")

    interruption_count = sum(
        not execution.reached_target for execution in simulation_result.executions
    )
    effective_optimization_count = (
        1 if algorithm == "classical_nn" else planner_call_count
    )

    collection_recalculation_count = sum(
        "collection_completed" in execution.decision.reasons
        for execution in simulation_result.executions
    )
    thermal_recalculation_count = sum(
        "thermal_band_changed" in execution.decision.reasons
        for execution in simulation_result.executions
    )
    volume_recalculation_count = sum(
        "volume_band_changed" in execution.decision.reasons
        for execution in simulation_result.executions
    )

    if interruption_count < 0:
        raise ValueError("interruption_count must be non-negative.")

    return ExperimentalRunResult(
        point_count=point_count,
        replicate_index=replicate_index,
        instance_seed=instance_seed,
        scenario=scenario,
        algorithm=algorithm,
        weight_profile=weight_profile,
        run_seed=run_seed,
        total_distance=total_distance,
        simulated_duration=simulated_duration,
        computational_seconds=computational_seconds,
        final_load=final_load,
        planner_call_count=planner_call_count,
        effective_optimization_count=effective_optimization_count,
        interruption_count=interruption_count,
        collection_recalculation_count=collection_recalculation_count,
        thermal_recalculation_count=thermal_recalculation_count,
        volume_recalculation_count=volume_recalculation_count,
        collection_order=collection_order,
        collections=collections,
    )


def run_experimental_case(
    point_count: int,
    replicate_index: int,
    scenario: ExperimentalScenario,
    algorithm: ExperimentalAlgorithm,
    *,
    weight_profile: WeightProfileName | None = None,
    run_seed: int | None = None,
) -> ExperimentalRunResult:
    """Execute exactly one configured experimental simulation case."""
    algorithm = _validate_algorithm(algorithm)
    scenario = _validate_scenario(scenario)
    weight_profile = _validate_weight_profile(algorithm, weight_profile)
    run_seed = _validate_run_seed(algorithm, run_seed)

    pair = generate_experimental_instance_pair(point_count, replicate_index)
    instance = (
        pair.nominal_instance if scenario == "nominal" else pair.disturbance_instance
    )
    weights = (
        EXPERIMENTAL_WEIGHT_PROFILES[weight_profile]
        if weight_profile is not None
        else None
    )
    planner = _make_planner(algorithm, instance, weights, run_seed)

    # This timer deliberately excludes instance generation, scenario selection,
    # weight resolution, and planner construction.
    computation_start = time.perf_counter()
    simulation_result = run_dynamic_simulation(
        instance,
        planner,
        t_min=EXPERIMENTAL_T_RISK_MIN,
        t_max=EXPERIMENTAL_T_RISK_MAX,
        q_low=EXPERIMENTAL_Q_LOW,
        q_high=EXPERIMENTAL_Q_HIGH,
        minimum_interval=EXPERIMENTAL_MINIMUM_RECALCULATION_INTERVAL,
    )
    computational_seconds = time.perf_counter() - computation_start

    return _build_result(
        instance=instance,
        point_count=pair.point_count,
        replicate_index=pair.replicate_index,
        instance_seed=pair.instance_seed,
        scenario=scenario,
        algorithm=algorithm,
        weight_profile=weight_profile,
        run_seed=run_seed,
        simulation_result=simulation_result,
        computational_seconds=computational_seconds,
    )
