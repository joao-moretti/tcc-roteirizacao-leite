"""Logistic distance component for a valid logical route."""

from math import isfinite

from src.algorithms.shortest_path import shortest_path
from src.models.candidate_route import CandidateRoute
from src.models.problem import ProblemInstance
from src.models.route import Route, RouteStop
from src.simulation.snapshot import OptimizationSnapshot


def _stop_node_id(instance: ProblemInstance, stop: RouteStop) -> int:
    """Resolve a logical route stop to its road-network node ID."""
    if stop.kind == "depot":
        return instance.depot.node_id

    for collection_point in instance.collection_points:
        if collection_point.point_id == stop.entity_id:
            return collection_point.node_id

    raise ValueError(f"collection point does not exist in instance: {stop.entity_id}.")


def route_distance(instance: ProblemInstance, route: Route) -> float:
    """Calculate the total shortest-path distance of a logical route."""
    route.validate(instance)

    node_ids = [_stop_node_id(instance, stop) for stop in route.stops]
    total_distance = 0.0

    for start_node_id, end_node_id in zip(node_ids, node_ids[1:]):
        distance, _ = shortest_path(instance.road_network, start_node_id, end_node_id)
        total_distance += distance

    return total_distance


def _candidate_point_node_id(instance: ProblemInstance, point_id: int) -> int:
    """Resolve a candidate collection point ID to its network node ID."""
    for collection_point in instance.collection_points:
        if collection_point.point_id == point_id:
            return collection_point.node_id

    raise ValueError(f"collection point does not exist in instance: {point_id}.")


def candidate_route_distance(
    instance: ProblemInstance,
    snapshot: OptimizationSnapshot,
    candidate_route: CandidateRoute,
) -> float:
    """Calculate directed distance from the current node through a candidate route."""
    candidate_route.validate(instance, snapshot)

    current_node_id = snapshot.current_node_id
    total_distance = 0.0

    for point_id in candidate_route.point_ids:
        point_node_id = _candidate_point_node_id(instance, point_id)
        distance, _ = shortest_path(
            instance.road_network,
            current_node_id,
            point_node_id,
        )
        total_distance += distance
        current_node_id = point_node_id

    distance, _ = shortest_path(
        instance.road_network,
        current_node_id,
        instance.depot.node_id,
    )
    total_distance += distance

    return total_distance


def logistic_distance_reference(instance: ProblemInstance) -> float:
    """Return the reference distance used to normalize route distance."""
    collection_points = instance.collection_points
    if not collection_points:
        return 0.0

    node_ids = [instance.depot.node_id] + [
        collection_point.node_id for collection_point in collection_points
    ]
    total_distance = 0.0
    pair_count = 0

    for start_node_id in node_ids:
        for end_node_id in node_ids:
            if start_node_id == end_node_id:
                continue

            distance, _ = shortest_path(
                instance.road_network, start_node_id, end_node_id
            )
            total_distance += distance
            pair_count += 1

    mean_distance = total_distance / pair_count
    return (len(collection_points) + 1) * mean_distance


def _validate_reference_distance(value: object) -> float:
    """Validate and normalize an externally supplied reference distance."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            "reference_distance must be a finite number greater than or equal to zero."
        )

    normalized = float(value)
    if not isfinite(normalized) or normalized < 0.0:
        raise ValueError(
            "reference_distance must be a finite number greater than or equal to zero."
        )

    return normalized


def normalized_logistic_cost(
    instance: ProblemInstance,
    route: Route,
    reference_distance: float,
) -> float:
    """Return the normalized logistic distance component of a valid route."""
    reference_distance = _validate_reference_distance(reference_distance)
    distance = route_distance(instance, route)

    if reference_distance == 0.0:
        if instance.collection_points:
            raise ValueError(
                "reference_distance must be greater than zero for an instance with "
                "collection points."
            )
        return 0.0

    return distance / reference_distance
