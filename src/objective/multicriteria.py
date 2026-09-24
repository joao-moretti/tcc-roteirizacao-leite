"""Weighted multicriteria objective for Formulation V1."""

from math import isclose, isfinite

from src.models.candidate_route import CandidateRoute
from src.models.problem import ProblemInstance
from src.objective.logistics import normalized_dynamic_logistic_cost
from src.objective.occupancy import normalized_occupancy_cost
from src.objective.thermal import normalized_thermal_cost
from src.simulation.snapshot import OptimizationSnapshot


def _validate_weight(value: object, name: str) -> float:
    """Validate and normalize one non-negative objective weight."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{name} must be a finite number greater than or equal to zero."
        )

    normalized = float(value)
    if not isfinite(normalized) or normalized < 0.0:
        raise ValueError(
            f"{name} must be a finite number greater than or equal to zero."
        )

    return normalized


def weighted_objective_cost(
    instance: ProblemInstance,
    snapshot: OptimizationSnapshot,
    candidate_route: CandidateRoute,
    logistic_reference_distance: float,
    thermal_reference_value: float,
    occupancy_reference_value: float,
    t_min: float,
    t_max: float,
    w_logistic: float,
    w_thermal: float,
    w_occupancy: float,
) -> float:
    """Return the weighted sum of the three normalized objective components."""
    w_logistic = _validate_weight(w_logistic, "w_logistic")
    w_thermal = _validate_weight(w_thermal, "w_thermal")
    w_occupancy = _validate_weight(w_occupancy, "w_occupancy")

    if not isclose(
        w_logistic + w_thermal + w_occupancy,
        1.0,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ValueError("objective weights must sum to 1.0.")

    logistic_cost = normalized_dynamic_logistic_cost(
        instance,
        snapshot,
        candidate_route,
        logistic_reference_distance,
    )
    thermal_cost = normalized_thermal_cost(
        instance,
        snapshot,
        candidate_route,
        t_min,
        t_max,
        thermal_reference_value,
    )
    occupancy_cost = normalized_occupancy_cost(
        instance,
        snapshot,
        candidate_route,
        occupancy_reference_value,
    )

    return (
        w_logistic * logistic_cost
        + w_thermal * thermal_cost
        + w_occupancy * occupancy_cost
    )
