"""Classical nearest-neighbor baseline for static routing."""

from src.algorithms.shortest_path import shortest_path
from src.models.problem import ProblemInstance
from src.models.route import Route, RouteStop


def classical_nearest_neighbor(instance: ProblemInstance) -> Route:
    """Build a complete route by repeatedly selecting the nearest point."""
    current_node_id = instance.depot.node_id
    unvisited_points = list(instance.collection_points)
    stops = [RouteStop("depot", instance.depot.depot_id)]

    while unvisited_points:
        candidates: list[tuple[float, int, int]] = []
        for collection_point in unvisited_points:
            distance, _ = shortest_path(
                instance.road_network,
                current_node_id,
                collection_point.node_id,
            )
            candidates.append(
                (distance, collection_point.point_id, collection_point.node_id)
            )

        _, selected_point_id, selected_node_id = min(candidates)
        stops.append(RouteStop("collection", selected_point_id))
        current_node_id = selected_node_id
        unvisited_points = [
            point for point in unvisited_points if point.point_id != selected_point_id
        ]

    shortest_path(
        instance.road_network,
        current_node_id,
        instance.depot.node_id,
    )
    stops.append(RouteStop("depot", instance.depot.depot_id))
    route = Route(stops)
    route.validate(instance)
    return route
