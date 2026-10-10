"""Adapters that connect the V1 algorithms to the dynamic simulator."""

import random
from typing import Final

from src.algorithms.ant_colony import ant_colony_optimization
from src.algorithms.nearest_neighbor import (
    classical_nearest_neighbor,
    multicriteria_nearest_neighbor,
)
from src.algorithms.simulated_annealing import simulated_annealing
from src.experiments.algorithm_presets import (
    ACO_BETA,
    ACO_EVAPORATION_RATE,
    SA_COOLING_RATE,
    SA_INITIAL_TEMPERATURE,
    SA_ITERATIONS,
)
from src.experiments.experimental_presets import (
    EXPERIMENTAL_T_RISK_MAX,
    EXPERIMENTAL_T_RISK_MIN,
)
from src.models.candidate_route import CandidateRoute
from src.models.problem import ProblemInstance
from src.objective.logistics import dynamic_logistic_reference
from src.objective.multicriteria import weighted_objective_cost
from src.objective.occupancy import occupancy_reference
from src.objective.thermal import thermal_reference
from src.simulation.dynamic import RoutePlanner
from src.simulation.snapshot import OptimizationSnapshot

EXPERIMENTAL_ALGORITHM_SEEDS: Final[tuple[int, ...]] = tuple(range(1, 31))
_CHILD_SEED_UPPER_BOUND = 2**63


def _validate_run_seed(run_seed: object) -> int:
    """Validate a non-negative integer seed for one experimental run."""
    if isinstance(run_seed, bool) or not isinstance(run_seed, int) or run_seed < 0:
        raise ValueError("run_seed must be a non-negative integer.")
    return run_seed


def _dynamic_references(
    instance: ProblemInstance,
    snapshot: OptimizationSnapshot,
) -> tuple[float, float, float]:
    """Calculate the three normalized-objective references for one snapshot."""
    return (
        dynamic_logistic_reference(instance, snapshot),
        thermal_reference(
            instance,
            snapshot,
            EXPERIMENTAL_T_RISK_MIN,
            EXPERIMENTAL_T_RISK_MAX,
        ),
        occupancy_reference(instance, snapshot),
    )


def make_classical_nn_planner(instance: ProblemInstance) -> RoutePlanner:
    """Create a planner with one fixed classical nearest-neighbor order."""
    initial_route = classical_nearest_neighbor(instance)
    fixed_point_order = tuple(
        stop.entity_id for stop in initial_route.stops if stop.kind == "collection"
    )

    def planner(
        current_instance: ProblemInstance,
        snapshot: OptimizationSnapshot,
    ) -> CandidateRoute:
        snapshot.validate(current_instance)
        remaining_point_ids = set(snapshot.remaining_point_ids)
        candidate_route = CandidateRoute(
            point_id
            for point_id in fixed_point_order
            if point_id in remaining_point_ids
        )
        candidate_route.validate(current_instance, snapshot)
        return candidate_route

    return planner


def make_multicriteria_nn_planner(
    weights: tuple[float, float, float],
) -> RoutePlanner:
    """Create a planner that recalculates MCNN for every snapshot."""

    def planner(
        instance: ProblemInstance,
        snapshot: OptimizationSnapshot,
    ) -> CandidateRoute:
        return multicriteria_nearest_neighbor(
            instance,
            snapshot,
            EXPERIMENTAL_T_RISK_MIN,
            EXPERIMENTAL_T_RISK_MAX,
            weights,
        )

    return planner


def make_sa_planner(
    weights: tuple[float, float, float],
    run_seed: int,
) -> RoutePlanner:
    """Create a dynamic SA planner with deterministic child seeds."""
    run_seed = _validate_run_seed(run_seed)
    seed_rng = random.Random(run_seed)

    def planner(
        instance: ProblemInstance,
        snapshot: OptimizationSnapshot,
    ) -> CandidateRoute:
        snapshot.validate(instance)
        initial_route = multicriteria_nearest_neighbor(
            instance,
            snapshot,
            EXPERIMENTAL_T_RISK_MIN,
            EXPERIMENTAL_T_RISK_MAX,
            weights,
        )
        (
            logistic_reference_distance,
            thermal_reference_value,
            occupancy_reference_value,
        ) = _dynamic_references(instance, snapshot)
        child_seed = seed_rng.randrange(0, _CHILD_SEED_UPPER_BOUND)
        return simulated_annealing(
            instance,
            snapshot,
            initial_route,
            logistic_reference_distance,
            thermal_reference_value,
            occupancy_reference_value,
            EXPERIMENTAL_T_RISK_MIN,
            EXPERIMENTAL_T_RISK_MAX,
            weights,
            SA_INITIAL_TEMPERATURE,
            SA_COOLING_RATE,
            SA_ITERATIONS,
            child_seed,
        )

    return planner


def make_aco_planner(
    weights: tuple[float, float, float],
    run_seed: int,
) -> RoutePlanner:
    """Create a dynamic ACO planner with deterministic child seeds."""
    run_seed = _validate_run_seed(run_seed)
    seed_rng = random.Random(run_seed)

    def planner(
        instance: ProblemInstance,
        snapshot: OptimizationSnapshot,
    ) -> CandidateRoute:
        snapshot.validate(instance)
        initial_route = multicriteria_nearest_neighbor(
            instance,
            snapshot,
            EXPERIMENTAL_T_RISK_MIN,
            EXPERIMENTAL_T_RISK_MAX,
            weights,
        )
        (
            logistic_reference_distance,
            thermal_reference_value,
            occupancy_reference_value,
        ) = _dynamic_references(instance, snapshot)
        baseline_cost = weighted_objective_cost(
            instance,
            snapshot,
            initial_route,
            logistic_reference_distance,
            thermal_reference_value,
            occupancy_reference_value,
            EXPERIMENTAL_T_RISK_MIN,
            EXPERIMENTAL_T_RISK_MAX,
            *weights,
        )
        child_seed = seed_rng.randrange(0, _CHILD_SEED_UPPER_BOUND)
        return ant_colony_optimization(
            instance,
            snapshot,
            baseline_cost,
            logistic_reference_distance,
            thermal_reference_value,
            occupancy_reference_value,
            EXPERIMENTAL_T_RISK_MIN,
            EXPERIMENTAL_T_RISK_MAX,
            weights,
            ACO_BETA,
            ACO_EVAPORATION_RATE,
            child_seed,
        )

    return planner
