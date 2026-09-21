"""Logistic distance component for a valid logical route."""

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
