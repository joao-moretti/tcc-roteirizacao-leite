"""Elementary thermal-risk component for Formulation V1."""

from math import isfinite

from src.algorithms.shortest_path import shortest_path
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


def _point_node_id(instance: ProblemInstance, point_id: int) -> int:
    """Resolve a collection point ID to its road-network node ID."""
    for collection_point in instance.collection_points:
        if collection_point.point_id == point_id:
            return collection_point.node_id

    raise ValueError(f"collection point does not exist in instance: {point_id}.")


def thermal_mean_distance(
    instance: ProblemInstance,
    snapshot: OptimizationSnapshot,
) -> float:
    """Return the mean directed shortest-path distance for relevant nodes."""
    snapshot.validate(instance)

    if not snapshot.remaining_point_ids:
        return 0.0

    relevant_node_ids = [snapshot.current_node_id]
    relevant_node_ids.extend(
        _point_node_id(instance, point_id) for point_id in snapshot.remaining_point_ids
    )
    relevant_node_ids = list(dict.fromkeys(relevant_node_ids))

    total_distance = 0.0
    pair_count = 0
    for start_node_id in relevant_node_ids:
        for end_node_id in relevant_node_ids:
            if start_node_id == end_node_id:
                continue

            distance, _ = shortest_path(
                instance.road_network,
                start_node_id,
                end_node_id,
            )
            total_distance += distance
            pair_count += 1

    if pair_count == 0:
        return 0.0

    return total_distance / pair_count


def thermal_reference(
    instance: ProblemInstance,
    snapshot: OptimizationSnapshot,
    t_min: float,
    t_max: float,
) -> float:
    """Return the thermal reference for the current snapshot."""
    snapshot.validate(instance)
    point_count = len(snapshot.remaining_point_ids)
    if point_count == 0:
        return 0.0

    maximum_risk = max(
        thermal_risk(snapshot.temperatures[point_id], t_min, t_max)
        for point_id in snapshot.remaining_point_ids
    )
    if maximum_risk == 0.0:
        return 0.0

    mean_distance = thermal_mean_distance(instance, snapshot)
    mean_travel_time = mean_distance / instance.v_ref
    return float(maximum_risk * mean_travel_time * point_count * (point_count + 1) / 2)


def _validate_reference(value: object) -> float:
    """Validate and normalize an external thermal reference."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            "reference must be a finite number greater than or equal to zero."
        )

    normalized = float(value)
    if not isfinite(normalized) or normalized < 0.0:
        raise ValueError(
            "reference must be a finite number greater than or equal to zero."
        )

    return normalized


def normalized_thermal_cost(
    instance: ProblemInstance,
    snapshot: OptimizationSnapshot,
    candidate_route: CandidateRoute,
    t_min: float,
    t_max: float,
    reference: float,
) -> float:
    """Return the raw thermal cost divided by an explicit reference."""
    reference = _validate_reference(reference)
    raw_cost = thermal_cost(instance, snapshot, candidate_route, t_min, t_max)

    if reference == 0.0:
        if raw_cost == 0.0:
            return 0.0
        raise ValueError("reference must be greater than zero for a positive cost.")

    return raw_cost / reference
