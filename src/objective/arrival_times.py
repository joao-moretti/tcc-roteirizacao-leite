"""Accumulated arrival times for a candidate dynamic route."""

from src.algorithms.shortest_path import shortest_path
from src.models.candidate_route import CandidateRoute
from src.models.problem import ProblemInstance
from src.simulation.snapshot import OptimizationSnapshot


def _point_node_id(instance: ProblemInstance, point_id: int) -> int:
    """Resolve a collection point ID to its road-network node ID."""
    for collection_point in instance.collection_points:
        if collection_point.point_id == point_id:
            return collection_point.node_id

    raise ValueError(f"collection point does not exist in instance: {point_id}.")


def arrival_times(
    instance: ProblemInstance,
    snapshot: OptimizationSnapshot,
    candidate_route: CandidateRoute,
) -> dict[int, float]:
    """Return accumulated travel times from the current snapshot node."""
    candidate_route.validate(instance, snapshot)

    if not candidate_route.point_ids:
        return {}

    previous_node_id = snapshot.current_node_id
    accumulated_time = 0.0
    result: dict[int, float] = {}

    for point_id in candidate_route.point_ids:
        point_node_id = _point_node_id(instance, point_id)
        distance, _ = shortest_path(
            instance.road_network,
            previous_node_id,
            point_node_id,
        )
        accumulated_time += distance / instance.v_ref
        result[point_id] = accumulated_time
        previous_node_id = point_node_id

    return result
