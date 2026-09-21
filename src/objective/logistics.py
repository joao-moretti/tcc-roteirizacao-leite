"""Logistic distance component for a valid logical route."""

from math import isfinite

from src.algorithms.shortest_path import shortest_path
from src.models.problem import ProblemInstance
from src.models.route import Route, RouteStop


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


def logistic_distance_upper_bound(instance: ProblemInstance) -> float:
    """Return the upper bound used to normalize route distance."""
    collection_points = instance.collection_points
    if not collection_points:
        return 0.0

    node_ids = [instance.depot.node_id] + [
        collection_point.node_id for collection_point in collection_points
    ]
    maximum_distance = 0.0

    for start_node_id in node_ids:
        for end_node_id in node_ids:
            if start_node_id == end_node_id:
                continue

            distance, _ = shortest_path(
                instance.road_network, start_node_id, end_node_id
            )
            maximum_distance = max(maximum_distance, distance)

    return (len(collection_points) + 1) * maximum_distance


def _validate_upper_bound(value: object) -> float:
    """Validate and normalize an externally supplied distance bound."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            "upper_bound must be a finite number greater than or equal to zero."
        )

    normalized = float(value)
    if not isfinite(normalized) or normalized < 0.0:
        raise ValueError(
            "upper_bound must be a finite number greater than or equal to zero."
        )

    return normalized


def normalized_logistic_cost(
    instance: ProblemInstance,
    route: Route,
    upper_bound: float,
) -> float:
    """Return the normalized logistic distance component of a valid route."""
    upper_bound = _validate_upper_bound(upper_bound)
    distance = route_distance(instance, route)

    if upper_bound == 0.0:
        if instance.collection_points:
            raise ValueError(
                "upper_bound must be greater than zero for an instance with "
                "collection points."
            )
        return 0.0

    return distance / upper_bound
