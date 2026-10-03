"""Pheromone primitives for the V1 ant-colony formulation."""

import random
from math import isclose, isfinite

from src.algorithms.shortest_path import shortest_path
from src.models.candidate_route import CandidateRoute
from src.models.problem import ProblemInstance
from src.objective.multicriteria import weighted_objective_cost
from src.objective.thermal import thermal_risk
from src.simulation.snapshot import OptimizationSnapshot

EPSILON = 1e-9
ACO_MAX_ITERATIONS = 100
ACO_STAGNATION_LIMIT = 20


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


def _validate_weights(weights: object) -> tuple[float, float, float]:
    """Validate and normalize the logistic, thermal, and occupancy weights."""
    try:
        w_logistic, w_thermal, w_occupancy = weights  # type: ignore[misc]
    except (TypeError, ValueError) as exc:
        raise ValueError("weights must contain exactly three values.") from exc

    normalized_weights: list[float] = []
    for name, value in (
        ("w_logistic", w_logistic),
        ("w_thermal", w_thermal),
        ("w_occupancy", w_occupancy),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"{name} must be a finite number greater than or equal to zero."
            )

        normalized = float(value)
        if not isfinite(normalized) or normalized < 0.0:
            raise ValueError(
                f"{name} must be a finite number greater than or equal to zero."
            )
        normalized_weights.append(normalized)

    if not isclose(
        sum(normalized_weights),
        1.0,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ValueError("weights must sum to 1.0.")

    return (
        normalized_weights[0],
        normalized_weights[1],
        normalized_weights[2],
    )


def _validate_beta(beta: object) -> float:
    """Validate and normalize the heuristic exponent."""
    if isinstance(beta, bool) or not isinstance(beta, (int, float)):
        raise ValueError("beta must be finite and greater than zero.")

    normalized = float(beta)
    if not isfinite(normalized) or normalized <= 0.0:
        raise ValueError("beta must be finite and greater than zero.")

    return normalized


def _validate_algorithm_seed(algorithm_seed: object) -> int:
    """Validate a non-negative integer seed for one ACO execution."""
    if (
        isinstance(algorithm_seed, bool)
        or not isinstance(algorithm_seed, int)
        or algorithm_seed < 0
    ):
        raise ValueError("algorithm_seed must be a non-negative integer.")

    return algorithm_seed


def _validate_pheromone_value(value: object) -> float:
    """Validate one pheromone value used by a candidate transition."""
    return _validate_positive_number(value, "pheromone")


def _select_candidate(
    candidates: list[int],
    transition_weights: dict[int, float],
    rng: random.Random,
) -> int:
    """Select one candidate with an explicit weighted roulette draw."""
    ordered_candidates = sorted(candidates)
    total_weight = sum(transition_weights[point_id] for point_id in ordered_candidates)
    if not isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError(
            "total transition weight must be finite and greater than zero."
        )

    draw = rng.random() * total_weight
    accumulated_weight = 0.0
    for point_id in ordered_candidates:
        accumulated_weight += transition_weights[point_id]
        if draw < accumulated_weight:
            return point_id

    return ordered_candidates[-1]


def construct_ant_route(
    instance: ProblemInstance,
    snapshot: OptimizationSnapshot,
    pheromones: dict[tuple[int, int], float],
    t_min: float,
    t_max: float,
    weights: tuple[float, float, float],
    beta: float,
    rng: random.Random,
) -> CandidateRoute:
    """Construct one candidate route using local ACO transition weights."""
    snapshot.validate(instance)
    w_logistic, w_thermal, w_occupancy = _validate_weights(weights)
    beta = _validate_beta(beta)

    point_node_ids = _point_node_ids(instance, snapshot)
    current_node_id = snapshot.current_node_id
    unvisited_point_ids = list(snapshot.remaining_point_ids)
    selected_point_ids: list[int] = []

    while unvisited_point_ids:
        risks = {
            point_id: thermal_risk(
                snapshot.temperatures[point_id],
                t_min,
                t_max,
            )
            for point_id in unvisited_point_ids
        }
        critical_unvisited = [
            point_id
            for point_id in unvisited_point_ids
            if snapshot.temperatures[point_id] >= t_max
        ]
        feasible_candidates = critical_unvisited or list(unvisited_point_ids)
        feasible_candidates.sort()

        distances: dict[int, float] = {}
        for point_id in feasible_candidates:
            distance, _ = shortest_path(
                instance.road_network,
                current_node_id,
                point_node_ids[point_id],
            )
            distances[point_id] = distance

        maximum_distance = max(distances.values())
        maximum_volume = max(
            snapshot.volumes[point_id] for point_id in feasible_candidates
        )
        transition_weights: dict[int, float] = {}

        for point_id in feasible_candidates:
            movement = (current_node_id, point_node_ids[point_id])
            if movement not in pheromones:
                raise ValueError(f"pheromones is missing logical movement: {movement}.")

            pheromone = _validate_pheromone_value(pheromones[movement])
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
            local_cost = (
                w_logistic * normalized_distance
                + w_thermal * (1.0 - risks[point_id])
                + w_occupancy * (1.0 - normalized_volume)
            )
            eta = 1.0 / (local_cost + EPSILON)

            try:
                transition_weight = pheromone * (eta**beta)
            except OverflowError as exc:
                raise ValueError(
                    "transition weight must be finite and greater than zero."
                ) from exc

            if not isfinite(transition_weight) or transition_weight <= 0.0:
                raise ValueError(
                    "transition weight must be finite and greater than zero."
                )
            transition_weights[point_id] = transition_weight

        selected_point_id = _select_candidate(
            feasible_candidates,
            transition_weights,
            rng,
        )
        selected_point_ids.append(selected_point_id)
        current_node_id = point_node_ids[selected_point_id]
        unvisited_point_ids.remove(selected_point_id)

    shortest_path(
        instance.road_network,
        current_node_id,
        instance.depot.node_id,
    )

    candidate_route = CandidateRoute(tuple(selected_point_ids))
    candidate_route.validate(instance, snapshot)
    return candidate_route


def ant_colony_optimization(
    instance: ProblemInstance,
    snapshot: OptimizationSnapshot,
    baseline_cost: float,
    logistic_reference_distance: float,
    thermal_reference_value: float,
    occupancy_reference_value: float,
    t_min: float,
    t_max: float,
    weights: tuple[float, float, float],
    beta: float,
    evaporation_rate: float,
    algorithm_seed: int,
) -> CandidateRoute:
    """Run the complete V1 ant-colony optimization loop."""
    snapshot.validate(instance)
    w_logistic, w_thermal, w_occupancy = _validate_weights(weights)
    beta = _validate_beta(beta)
    evaporation_rate = _validate_evaporation_rate(evaporation_rate)
    algorithm_seed = _validate_algorithm_seed(algorithm_seed)
    rng = random.Random(algorithm_seed)
    validated_weights = (w_logistic, w_thermal, w_occupancy)

    if not snapshot.remaining_point_ids:
        terminal_route = CandidateRoute(())
        terminal_route.validate(instance, snapshot)
        return terminal_route

    pheromones = initialize_pheromones(instance, snapshot, baseline_cost)
    number_of_ants = ant_count(snapshot)
    global_best_route: CandidateRoute | None = None
    global_best_cost: float | None = None
    stagnation_count = 0

    for _ in range(ACO_MAX_ITERATIONS):
        iteration_best_route: CandidateRoute | None = None
        iteration_best_cost: float | None = None

        for _ in range(number_of_ants):
            candidate_route = construct_ant_route(
                instance,
                snapshot,
                pheromones,
                t_min,
                t_max,
                validated_weights,
                beta,
                rng,
            )
            candidate_cost = weighted_objective_cost(
                instance,
                snapshot,
                candidate_route,
                logistic_reference_distance,
                thermal_reference_value,
                occupancy_reference_value,
                t_min,
                t_max,
                w_logistic,
                w_thermal,
                w_occupancy,
            )

            if iteration_best_cost is None or candidate_cost < iteration_best_cost:
                iteration_best_route = candidate_route
                iteration_best_cost = candidate_cost

        if iteration_best_route is None or iteration_best_cost is None:
            raise ValueError("ACO could not construct a candidate route.")

        if global_best_cost is None:
            global_best_route = iteration_best_route
            global_best_cost = iteration_best_cost
            stagnation_count = 0
        elif iteration_best_cost < global_best_cost:
            global_best_route = iteration_best_route
            global_best_cost = iteration_best_cost
            stagnation_count = 0
        else:
            stagnation_count += 1

        pheromones = update_pheromones(
            instance,
            snapshot,
            pheromones,
            iteration_best_route,
            iteration_best_cost,
            evaporation_rate,
        )

        if stagnation_count >= ACO_STAGNATION_LIMIT:
            break

    if global_best_route is None:
        raise ValueError("ACO did not produce a global best route.")

    global_best_route.validate(instance, snapshot)
    return global_best_route
