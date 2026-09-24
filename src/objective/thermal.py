"""Elementary thermal-risk component for Formulation V1."""

from math import isfinite

from src.models.candidate_route import CandidateRoute
from src.models.problem import ProblemInstance
from src.objective.arrival_times import arrival_times
from src.simulation.snapshot import OptimizationSnapshot


def _validate_finite_number(value: object, name: str) -> float:
    """Validate and normalize one finite numeric input."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number.")

    normalized = float(value)
    if not isfinite(normalized):
        raise ValueError(f"{name} must be a finite number.")

    return normalized


def thermal_risk(
    temperature: float,
    t_min: float,
    t_max: float,
) -> float:
    """Return the thermal risk for a temperature and two valid thresholds."""
    temperature = _validate_finite_number(temperature, "temperature")
    t_min = _validate_finite_number(t_min, "t_min")
    t_max = _validate_finite_number(t_max, "t_max")

    if t_min >= t_max:
        raise ValueError("t_min must be less than t_max.")

    if temperature <= t_min:
        return 0.0

    if temperature >= t_max:
        return 1.0

    return (temperature - t_min) / (t_max - t_min)


def thermal_cost(
    instance: ProblemInstance,
    snapshot: OptimizationSnapshot,
    candidate_route: CandidateRoute,
    t_min: float,
    t_max: float,
) -> float:
    """Return the raw thermal cost of a candidate route."""
    arrival_times_map = arrival_times(instance, snapshot, candidate_route)

    total_cost = 0.0
    for point_id in candidate_route.point_ids:
        risk = thermal_risk(snapshot.temperatures[point_id], t_min, t_max)
        total_cost += risk * arrival_times_map[point_id]

    return total_cost
