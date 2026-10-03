"""Deterministic neighborhood infrastructure for simulated annealing."""

import random
from math import exp, isfinite

from src.models.candidate_route import CandidateRoute
from src.models.problem import ProblemInstance
from src.objective.multicriteria import weighted_objective_cost
from src.simulation.snapshot import OptimizationSnapshot


def create_algorithm_rng(algorithm_seed: int) -> random.Random:
    """Create a reproducible random generator from a non-negative seed."""
    if (
        isinstance(algorithm_seed, bool)
        or not isinstance(algorithm_seed, int)
        or algorithm_seed < 0
    ):
        raise ValueError("algorithm_seed must be a non-negative integer.")

    return random.Random(algorithm_seed)


def swap_neighbor(
    instance: ProblemInstance,
    snapshot: OptimizationSnapshot,
    candidate_route: CandidateRoute,
    rng: random.Random,
) -> CandidateRoute:
    """Return a neighbor formed by swapping two route positions."""
    candidate_route.validate(instance, snapshot)

    modified_point_ids = list(candidate_route.point_ids)
    if len(modified_point_ids) < 2:
        neighbor = CandidateRoute(tuple(modified_point_ids))
        neighbor.validate(instance, snapshot)
        return neighbor

    first_index, second_index = rng.sample(range(len(modified_point_ids)), 2)
    modified_point_ids[first_index], modified_point_ids[second_index] = (
        modified_point_ids[second_index],
        modified_point_ids[first_index],
    )

    neighbor = CandidateRoute(tuple(modified_point_ids))
    neighbor.validate(instance, snapshot)
    return neighbor


def _validate_finite_number(value: object, name: str) -> float:
    """Validate and normalize one finite numeric parameter."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite numeric value.")

    normalized = float(value)
    if not isfinite(normalized):
        raise ValueError(f"{name} must be a finite numeric value.")

    return normalized


def metropolis_accept(
    delta_cost: float,
    temperature: float,
    rng: random.Random,
) -> bool:
    """Apply the Metropolis acceptance rule to a cost difference."""
    delta_cost = _validate_finite_number(delta_cost, "delta_cost")
    temperature = _validate_finite_number(temperature, "temperature")
    if temperature <= 0.0:
        raise ValueError("temperature must be greater than zero.")

    if delta_cost <= 0.0:
        return True

    probability = exp(-delta_cost / temperature)
    return rng.random() < probability


def _validate_iterations(iterations: object) -> int:
    """Validate the number of simulated-annealing iterations."""
    if isinstance(iterations, bool) or not isinstance(iterations, int):
        raise ValueError("iterations must be a non-negative integer.")
    if iterations < 0:
        raise ValueError("iterations must be a non-negative integer.")
    return iterations


def _unpack_weights(weights: object) -> tuple[float, float, float]:
    """Unpack exactly three weights while deferring value validation."""
    try:
        w_logistic, w_thermal, w_occupancy = weights  # type: ignore[misc]
    except (TypeError, ValueError) as exc:
        raise ValueError("weights must contain exactly three values.") from exc

    return w_logistic, w_thermal, w_occupancy


def simulated_annealing(
    instance: ProblemInstance,
    snapshot: OptimizationSnapshot,
    initial_route: CandidateRoute,
    logistic_reference_distance: float,
    thermal_reference_value: float,
    occupancy_reference_value: float,
    t_min: float,
    t_max: float,
    weights: tuple[float, float, float],
    initial_temperature: float,
    cooling_rate: float,
    iterations: int,
    algorithm_seed: int,
) -> CandidateRoute:
    """Run the basic simulated-annealing loop for a candidate route."""
    initial_route.validate(instance, snapshot)
    rng = create_algorithm_rng(algorithm_seed)

    initial_temperature = _validate_finite_number(
        initial_temperature,
        "initial_temperature",
    )
    if initial_temperature <= 0.0:
        raise ValueError("initial_temperature must be greater than zero.")

    cooling_rate = _validate_finite_number(cooling_rate, "cooling_rate")
    if not 0.0 < cooling_rate < 1.0:
        raise ValueError("cooling_rate must be strictly between zero and one.")

    iterations = _validate_iterations(iterations)
    w_logistic, w_thermal, w_occupancy = _unpack_weights(weights)

    current_route = initial_route
    current_cost = weighted_objective_cost(
        instance,
        snapshot,
        current_route,
        logistic_reference_distance,
        thermal_reference_value,
        occupancy_reference_value,
        t_min,
        t_max,
        w_logistic,
        w_thermal,
        w_occupancy,
    )
    best_route = current_route
    best_cost = current_cost
    temperature = initial_temperature

    for _ in range(iterations):
        neighbor_route = swap_neighbor(instance, snapshot, current_route, rng)
        neighbor_cost = weighted_objective_cost(
            instance,
            snapshot,
            neighbor_route,
            logistic_reference_distance,
            thermal_reference_value,
            occupancy_reference_value,
            t_min,
            t_max,
            w_logistic,
            w_thermal,
            w_occupancy,
        )
        delta_cost = neighbor_cost - current_cost

        if metropolis_accept(delta_cost, temperature, rng):
            current_route = neighbor_route
            current_cost = neighbor_cost
            if current_cost < best_cost:
                best_route = current_route
                best_cost = current_cost

        temperature *= cooling_rate

    best_route.validate(instance, snapshot)
    return best_route
