"""Pheromone primitives for the V1 ant-colony formulation."""

from math import isfinite

from src.models.candidate_route import CandidateRoute
from src.models.problem import ProblemInstance
from src.simulation.snapshot import OptimizationSnapshot


def ant_count(snapshot: OptimizationSnapshot) -> int:
    """Return the number of ants prescribed for the current subproblem."""
    return min(len(snapshot.remaining_point_ids), 20)


def _validate_positive_number(value: object, field_name: str) -> float:
    """Validate and normalize a finite number greater than zero."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be finite and greater than zero.")

    normalized = float(value)
    if not isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{field_name} must be finite and greater than zero.")

    return normalized


def _validate_evaporation_rate(value: object) -> float:
    """Validate and normalize a geometric evaporation rate."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            "evaporation_rate must be finite and strictly between zero and one."
        )

    normalized = float(value)
    if not isfinite(normalized) or not 0.0 < normalized < 1.0:
        raise ValueError(
            "evaporation_rate must be finite and strictly between zero and one."
        )

    return normalized


def _point_node_ids(
    instance: ProblemInstance,
    snapshot: OptimizationSnapshot,
) -> dict[int, int]:
    """Resolve remaining point IDs to their network node IDs."""
    nodes_by_point_id = {
        point.point_id: point.node_id for point in instance.collection_points
    }
    return {
        point_id: nodes_by_point_id[point_id]
        for point_id in snapshot.remaining_point_ids
    }


def _admissible_moves(
    instance: ProblemInstance,
    snapshot: OptimizationSnapshot,
) -> set[tuple[int, int]]:
    """Return the logical directed movements for a snapshot."""
    point_node_ids = _point_node_ids(instance, snapshot)
    remaining_point_ids = tuple(snapshot.remaining_point_ids)
    moves: set[tuple[int, int]] = {
        (snapshot.current_node_id, point_node_ids[point_id])
        for point_id in remaining_point_ids
    }

    for from_point_id in remaining_point_ids:
        for to_point_id in remaining_point_ids:
            if from_point_id != to_point_id:
                moves.add(
                    (
                        point_node_ids[from_point_id],
                        point_node_ids[to_point_id],
                    )
                )

    moves.update(
        (point_node_ids[point_id], instance.depot.node_id)
        for point_id in remaining_point_ids
    )
    return moves


def initialize_pheromones(
    instance: ProblemInstance,
    snapshot: OptimizationSnapshot,
    baseline_cost: float,
) -> dict[tuple[int, int], float]:
    """Initialize every admissible logical movement with uniform pheromone."""
    snapshot.validate(instance)
    baseline_cost = _validate_positive_number(baseline_cost, "baseline_cost")

    if not snapshot.remaining_point_ids:
        return {}

    initial_pheromone = 1.0 / baseline_cost
    return {
        move: initial_pheromone
        for move in sorted(_admissible_moves(instance, snapshot))
    }


def _route_moves(
    instance: ProblemInstance,
    snapshot: OptimizationSnapshot,
    best_route: CandidateRoute,
) -> set[tuple[int, int]]:
    """Return logical movements used by a non-empty candidate route."""
    point_node_ids = _point_node_ids(instance, snapshot)
    route_nodes = [point_node_ids[point_id] for point_id in best_route.point_ids]
    nodes = [snapshot.current_node_id, *route_nodes, instance.depot.node_id]
    return set(zip(nodes, nodes[1:]))


def update_pheromones(
    instance: ProblemInstance,
    snapshot: OptimizationSnapshot,
    pheromones: dict[tuple[int, int], float],
    best_route: CandidateRoute,
    best_cost: float,
    evaporation_rate: float,
) -> dict[tuple[int, int], float]:
    """Evaporate pheromone and deposit only along the best logical route."""
    snapshot.validate(instance)
    best_route.validate(instance, snapshot)
    best_cost = _validate_positive_number(best_cost, "best_cost")
    evaporation_rate = _validate_evaporation_rate(evaporation_rate)

    if not isinstance(pheromones, dict):
        raise ValueError("pheromones must be a dictionary.")

    admissible_moves = _admissible_moves(instance, snapshot)
    if set(pheromones) != admissible_moves:
        raise ValueError(
            "pheromones keys must exactly match the admissible logical movements."
        )

    normalized_pheromones = {
        move: _validate_positive_number(value, "pheromone")
        for move, value in pheromones.items()
    }

    if not admissible_moves:
        return {}

    deposited_pheromone = 1.0 / best_cost
    route_moves = _route_moves(instance, snapshot, best_route)
    retained_fraction = 1.0 - evaporation_rate

    return {
        move: retained_fraction * pheromone
        + (deposited_pheromone if move in route_moves else 0.0)
        for move, pheromone in normalized_pheromones.items()
    }
