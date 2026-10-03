"""Classical nearest-neighbor baseline for static routing."""

from math import isclose, isfinite

from src.algorithms.shortest_path import shortest_path
from src.models.candidate_route import CandidateRoute
from src.models.problem import ProblemInstance
from src.models.route import Route, RouteStop
from src.objective.thermal import thermal_risk
from src.simulation.snapshot import OptimizationSnapshot


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


def _point_node_id(instance: ProblemInstance, point_id: int) -> int:
    """Resolve a collection point ID to its road-network node ID."""
    for collection_point in instance.collection_points:
        if collection_point.point_id == point_id:
            return collection_point.node_id

    raise ValueError(f"collection point does not exist in instance: {point_id}.")


def _validate_multicriteria_weights(weights: object) -> tuple[float, float, float]:
    """Validate and normalize the three multicriteria weights."""
    try:
        normalized_weights = tuple(weights)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError("weights must contain exactly three numeric values.") from exc

    if len(normalized_weights) != 3:
        raise ValueError("weights must contain exactly three numeric values.")

    normalized: list[float] = []
    for index, weight in enumerate(normalized_weights):
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise ValueError(f"weight at position {index} must be a finite number.")

        normalized_weight = float(weight)
        if not isfinite(normalized_weight) or normalized_weight < 0.0:
            raise ValueError(
                f"weight at position {index} must be finite and non-negative."
            )
        normalized.append(normalized_weight)

    if not isclose(
        sum(normalized),
        1.0,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ValueError("weights must sum to 1.0.")

    return normalized[0], normalized[1], normalized[2]


def multicriteria_nearest_neighbor(
    instance: ProblemInstance,
    snapshot: OptimizationSnapshot,
    t_min: float,
    t_max: float,
    weights: tuple[float, float, float],
) -> CandidateRoute:
    """Build a candidate route using local multicriteria attractiveness."""
    w_logistic, w_thermal, w_occupancy = _validate_multicriteria_weights(weights)
    snapshot.validate(instance)

    current_node_id = snapshot.current_node_id
    unvisited_point_ids = list(snapshot.remaining_point_ids)
    selected_point_ids: list[int] = []

    while unvisited_point_ids:
        critical_point_ids = [
            point_id
            for point_id in unvisited_point_ids
            if snapshot.temperatures[point_id] >= t_max
        ]
        feasible_point_ids = critical_point_ids or unvisited_point_ids

        distances: dict[int, float] = {}
        node_ids: dict[int, int] = {}
        for point_id in feasible_point_ids:
            node_id = _point_node_id(instance, point_id)
            distance, _ = shortest_path(
                instance.road_network,
                current_node_id,
                node_id,
            )
            node_ids[point_id] = node_id
            distances[point_id] = distance

        maximum_distance = max(distances.values())
        maximum_volume = max(
            snapshot.volumes[point_id] for point_id in feasible_point_ids
        )
        attractiveness: dict[int, float] = {}

        for point_id in feasible_point_ids:
            normalized_distance = (
                0.0
                if maximum_distance == 0.0
                else distances[point_id] / maximum_distance
            )
            normalized_volume = (
                0.0
                if maximum_volume == 0.0
                else snapshot.volumes[point_id] / maximum_volume
            )
            attractiveness[point_id] = (
                w_logistic * (1.0 - normalized_distance)
                + w_thermal
                * thermal_risk(snapshot.temperatures[point_id], t_min, t_max)
                + w_occupancy * normalized_volume
            )

        selected_point_id = max(
            feasible_point_ids,
            key=lambda point_id: (attractiveness[point_id], -point_id),
        )
        selected_point_ids.append(selected_point_id)
        current_node_id = node_ids[selected_point_id]
        unvisited_point_ids.remove(selected_point_id)

    candidate_route = CandidateRoute(tuple(selected_point_ids))
    candidate_route.validate(instance, snapshot)
    return candidate_route
