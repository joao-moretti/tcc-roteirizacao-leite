"""Reproducible generation of directed grid road networks."""

import random
from collections import deque
from math import isclose, isfinite, sqrt

from src.models.problem import CollectionPoint, Depot
from src.models.road_network import RoadArc, RoadNetwork, RoadNode

GRID_NETWORK_MAX_ATTEMPTS = 1000
POINT_PLACEMENT_MAX_ATTEMPTS = 1000


def create_instance_rng(instance_seed: int) -> random.Random:
    """Create a reproducible RNG for synthetic instance generation."""
    if (
        isinstance(instance_seed, bool)
        or not isinstance(instance_seed, int)
        or instance_seed < 0
    ):
        raise ValueError("instance_seed must be a non-negative integer.")

    return random.Random(instance_seed)


def _validate_positive_number(value: object, field_name: str) -> float:
    """Validate and normalize a finite number greater than zero."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be finite and greater than zero.")

    normalized = float(value)
    if not isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{field_name} must be finite and greater than zero.")

    return normalized


def _validate_probability(value: object, field_name: str) -> float:
    """Validate and normalize a probability in the closed unit interval."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be finite and between zero and one.")

    normalized = float(value)
    if not isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{field_name} must be finite and between zero and one.")

    return normalized


def _validate_grid_dimension(value: object, field_name: str) -> int:
    """Validate one positive integer grid dimension."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer.")

    return value


def _grid_nodes(rows: int, columns: int, spacing: float) -> list[RoadNode]:
    """Create grid nodes in row-major order."""
    return [
        RoadNode(
            node_id=row * columns + column + 1,
            x=column * spacing,
            y=row * spacing,
        )
        for row in range(rows)
        for column in range(columns)
    ]


def _neighbor_pairs(rows: int, columns: int) -> tuple[tuple[int, int], ...]:
    """Return each horizontal and vertical grid neighbor pair once."""
    pairs: list[tuple[int, int]] = []
    for row in range(rows):
        for column in range(columns):
            node_id = row * columns + column + 1
            if column + 1 < columns:
                pairs.append((node_id, node_id + 1))
            if row + 1 < rows:
                pairs.append((node_id, node_id + columns))

    return tuple(pairs)


def _is_strongly_connected(
    node_ids: tuple[int, ...],
    arcs: list[RoadArc],
) -> bool:
    """Check strong connectivity using directed and reversed reachability."""
    if not node_ids:
        return False

    adjacency: dict[int, list[int]] = {node_id: [] for node_id in node_ids}
    reverse_adjacency: dict[int, list[int]] = {node_id: [] for node_id in node_ids}
    for arc in arcs:
        adjacency[arc.from_node_id].append(arc.to_node_id)
        reverse_adjacency[arc.to_node_id].append(arc.from_node_id)

    start_node_id = node_ids[0]

    def reachable(adjacency_map: dict[int, list[int]]) -> set[int]:
        visited = {start_node_id}
        queue = deque([start_node_id])
        while queue:
            current_node_id = queue.popleft()
            for neighbor in adjacency_map[current_node_id]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return visited

    return reachable(adjacency) == set(node_ids) and reachable(
        reverse_adjacency
    ) == set(node_ids)


def _oriented_arcs(
    neighbor_pairs: tuple[tuple[int, int], ...],
    spacing: float,
    p_uni: float,
    rng: random.Random,
) -> list[RoadArc]:
    """Generate one directed orientation attempt for all physical pairs."""
    arcs: list[RoadArc] = []
    for first_node_id, second_node_id in neighbor_pairs:
        if rng.random() < p_uni:
            if rng.random() < 0.5:
                arcs.append(RoadArc(first_node_id, second_node_id, spacing))
            else:
                arcs.append(RoadArc(second_node_id, first_node_id, spacing))
        else:
            arcs.append(RoadArc(first_node_id, second_node_id, spacing))
            arcs.append(RoadArc(second_node_id, first_node_id, spacing))

    return arcs


def generate_grid_road_network(
    network_id: int,
    rows: int,
    columns: int,
    spacing: float,
    p_uni: float,
    rng: random.Random,
) -> RoadNetwork:
    """Generate a strongly connected directed rectangular grid network."""
    rows = _validate_grid_dimension(rows, "rows")
    columns = _validate_grid_dimension(columns, "columns")
    if rows * columns < 2:
        raise ValueError("the grid must contain at least two nodes.")

    spacing = _validate_positive_number(spacing, "spacing")
    p_uni = _validate_probability(p_uni, "p_uni")
    if not isinstance(rng, random.Random):
        raise ValueError("rng must be an instance of random.Random.")

    nodes = _grid_nodes(rows, columns, spacing)
    node_ids = tuple(node.node_id for node in nodes)
    neighbor_pairs = _neighbor_pairs(rows, columns)

    for _ in range(GRID_NETWORK_MAX_ATTEMPTS):
        arcs = _oriented_arcs(neighbor_pairs, spacing, p_uni, rng)
        if _is_strongly_connected(node_ids, arcs):
            return RoadNetwork(
                network_id=network_id,
                nodes=nodes,
                arcs=arcs,
            )

    raise ValueError(
        "could not generate a strongly connected directed grid "
        "with the supplied parameters."
    )


def _validate_non_negative_number(value: object, field_name: str) -> float:
    """Validate and normalize a finite number greater than or equal to zero."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be finite and non-negative.")

    normalized = float(value)
    if not isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{field_name} must be finite and non-negative.")

    return normalized


def _select_central_node(nodes: tuple[RoadNode, ...]) -> RoadNode:
    """Select the node nearest to the geometric center of the network."""
    center_x = (min(node.x for node in nodes) + max(node.x for node in nodes)) / 2.0
    center_y = (min(node.y for node in nodes) + max(node.y for node in nodes)) / 2.0

    return min(
        nodes,
        key=lambda node: (
            (node.x - center_x) ** 2 + (node.y - center_y) ** 2,
            node.node_id,
        ),
    )


def _euclidean_distance(first: RoadNode, second: RoadNode) -> float:
    """Return the geometric distance between two road nodes."""
    return sqrt((first.x - second.x) ** 2 + (first.y - second.y) ** 2)


def _has_valid_point_separation(
    selected_nodes: list[RoadNode],
    depot_node: RoadNode,
    min_separation: float,
) -> bool:
    """Check separation from the depot and between all selected nodes."""
    if any(
        _euclidean_distance(node, depot_node) < min_separation
        for node in selected_nodes
    ):
        return False

    return all(
        _euclidean_distance(first_node, second_node) >= min_separation
        for index, first_node in enumerate(selected_nodes)
        for second_node in selected_nodes[index + 1 :]
    )


def select_depot_and_collection_points(
    road_network: RoadNetwork,
    point_count: int,
    min_separation: float,
    rng: random.Random,
) -> tuple[Depot, tuple[CollectionPoint, ...]]:
    """Select a central depot and separated collection points geometrically."""
    point_count = _validate_grid_dimension(point_count, "point_count")
    min_separation = _validate_non_negative_number(
        min_separation,
        "min_separation",
    )
    if not isinstance(rng, random.Random):
        raise ValueError("rng must be an instance of random.Random.")

    if len(road_network.nodes) < point_count + 1:
        raise ValueError("road_network must contain at least point_count + 1 nodes.")

    depot_node = _select_central_node(road_network.nodes)
    depot = Depot(depot_id=0, node_id=depot_node.node_id)
    candidate_nodes = [
        node for node in road_network.nodes if node.node_id != depot_node.node_id
    ]

    for _ in range(POINT_PLACEMENT_MAX_ATTEMPTS):
        selected_nodes = rng.sample(candidate_nodes, point_count)
        if not _has_valid_point_separation(
            selected_nodes,
            depot_node,
            min_separation,
        ):
            continue

        ordered_nodes = sorted(selected_nodes, key=lambda node: node.node_id)
        collection_points = tuple(
            CollectionPoint(point_id=index, node_id=node.node_id)
            for index, node in enumerate(ordered_nodes, start=1)
        )
        return depot, collection_points

    raise ValueError(
        "could not place collection points with the supplied minimum separation."
    )


def _validate_finite_number(value: object, field_name: str) -> float:
    """Validate and normalize a finite numeric parameter."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be finite.")

    normalized = float(value)
    if not isfinite(normalized):
        raise ValueError(f"{field_name} must be finite.")

    return normalized


def _validate_probability_triplet(
    probabilities: object,
    field_name: str,
) -> tuple[float, float, float]:
    """Validate and normalize a probability triplet whose sum is one."""
    try:
        values = tuple(probabilities)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(
            f"{field_name} must contain exactly three probabilities."
        ) from exc

    if len(values) != 3:
        raise ValueError(f"{field_name} must contain exactly three probabilities.")

    normalized: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"{field_name} must contain finite probabilities between zero and one."
            )
        converted = float(value)
        if not isfinite(converted) or not 0.0 <= converted <= 1.0:
            raise ValueError(
                f"{field_name} must contain finite probabilities between zero and one."
            )
        normalized.append(converted)

    if not isclose(sum(normalized), 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"{field_name} probabilities must sum to one.")

    return normalized[0], normalized[1], normalized[2]


def _validate_strict_levels(
    levels: object,
    field_name: str,
    *,
    non_negative: bool = False,
) -> tuple[float, float, float]:
    """Validate three finite, strictly increasing levels."""
    try:
        values = tuple(levels)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"{field_name} must contain exactly three levels.") from exc

    if len(values) != 3:
        raise ValueError(f"{field_name} must contain exactly three levels.")

    normalized = tuple(_validate_finite_number(value, field_name) for value in values)
    if non_negative and any(value < 0.0 for value in normalized):
        raise ValueError(f"{field_name} levels must be non-negative.")
    if not normalized[0] < normalized[1] < normalized[2]:
        raise ValueError(f"{field_name} levels must be strictly increasing.")

    return normalized[0], normalized[1], normalized[2]


def _draw_discrete(
    values: tuple[float, float, float],
    probabilities: tuple[float, float, float],
    rng: random.Random,
) -> float:
    """Draw one value using one explicit cumulative-probability draw."""
    draw = rng.random()
    cumulative_probability = 0.0
    for value, probability in zip(values, probabilities):
        cumulative_probability += probability
        if draw < cumulative_probability:
            return value
    return values[-1]


def _validate_trajectory_parameters(
    sample_count: object,
    initial_levels: object,
    initial_probabilities: object,
    change_amount: object,
    transition_probabilities: object,
    simulation_minimum: object,
    simulation_maximum: object,
    rng: object,
    *,
    levels_field_name: str,
    change_field_name: str,
    require_non_negative_levels: bool,
) -> tuple[
    int,
    tuple[float, float, float],
    tuple[float, float, float],
    float,
    tuple[float, float, float],
    float,
    float,
    random.Random,
]:
    """Validate shared parameters for temperature and volume trajectories."""
    normalized_sample_count = _validate_grid_dimension(sample_count, "sample_count")
    normalized_levels = _validate_strict_levels(
        initial_levels,
        levels_field_name,
        non_negative=require_non_negative_levels,
    )
    normalized_initial_probabilities = _validate_probability_triplet(
        initial_probabilities,
        "initial_probabilities",
    )
    normalized_change = _validate_finite_number(change_amount, change_field_name)
    if normalized_change <= 0.0:
        raise ValueError(f"{change_field_name} must be finite and greater than zero.")

    normalized_transition_probabilities = _validate_probability_triplet(
        transition_probabilities,
        "transition_probabilities",
    )
    p_minus, p_zero, p_plus = normalized_transition_probabilities
    if p_zero <= p_minus:
        raise ValueError("transition_probabilities require p_zero > p_minus.")
    if p_plus <= p_minus:
        raise ValueError("transition_probabilities require p_plus > p_minus.")

    normalized_minimum = _validate_finite_number(
        simulation_minimum,
        "simulation_minimum",
    )
    normalized_maximum = _validate_finite_number(
        simulation_maximum,
        "simulation_maximum",
    )
    if require_non_negative_levels and normalized_minimum < 0.0:
        raise ValueError("simulation_minimum must be non-negative.")
    if normalized_minimum >= normalized_maximum:
        raise ValueError("simulation_minimum must be less than simulation_maximum.")
    if any(
        level < normalized_minimum or level > normalized_maximum
        for level in normalized_levels
    ):
        raise ValueError(
            f"{levels_field_name} levels must be within the simulation limits."
        )

    if not isinstance(rng, random.Random):
        raise ValueError("rng must be an instance of random.Random.")

    return (
        normalized_sample_count,
        normalized_levels,
        normalized_initial_probabilities,
        normalized_change,
        normalized_transition_probabilities,
        normalized_minimum,
        normalized_maximum,
        rng,
    )


def generate_temperature_trajectories(
    collection_points: tuple[CollectionPoint, ...],
    sample_count: int,
    initial_levels: tuple[float, float, float],
    initial_probabilities: tuple[float, float, float],
    delta_temperature: float,
    transition_probabilities: tuple[float, float, float],
    simulation_minimum: float,
    simulation_maximum: float,
    rng: random.Random,
) -> dict[int, tuple[float, ...]]:
    """Generate frozen discrete temperature trajectories for collection points."""
    (
        normalized_sample_count,
        normalized_levels,
        normalized_initial_probabilities,
        normalized_delta,
        normalized_transition_probabilities,
        normalized_minimum,
        normalized_maximum,
        normalized_rng,
    ) = _validate_trajectory_parameters(
        sample_count,
        initial_levels,
        initial_probabilities,
        delta_temperature,
        transition_probabilities,
        simulation_minimum,
        simulation_maximum,
        rng,
        levels_field_name="initial_levels",
        change_field_name="delta_temperature",
        require_non_negative_levels=False,
    )

    trajectories: dict[int, tuple[float, ...]] = {}
    for point in sorted(collection_points, key=lambda item: item.point_id):
        current_temperature = _draw_discrete(
            normalized_levels,
            normalized_initial_probabilities,
            normalized_rng,
        )
        values = [current_temperature]
        for _ in range(1, normalized_sample_count):
            change = _draw_discrete(
                (-normalized_delta, 0.0, normalized_delta),
                normalized_transition_probabilities,
                normalized_rng,
            )
            current_temperature = max(
                normalized_minimum,
                min(normalized_maximum, current_temperature + change),
            )
            values.append(current_temperature)
        trajectories[point.point_id] = tuple(values)

    return trajectories


def generate_volume_trajectories(
    collection_points: tuple[CollectionPoint, ...],
    sample_count: int,
    initial_levels: tuple[float, float, float],
    initial_probabilities: tuple[float, float, float],
    gamma_volume: float,
    transition_probabilities: tuple[float, float, float],
    simulation_minimum: float,
    simulation_maximum: float,
    rng: random.Random,
) -> dict[int, tuple[float, ...]]:
    """Generate frozen discrete volume trajectories for collection points."""
    (
        normalized_sample_count,
        normalized_levels,
        normalized_initial_probabilities,
        normalized_gamma,
        normalized_transition_probabilities,
        normalized_minimum,
        normalized_maximum,
        normalized_rng,
    ) = _validate_trajectory_parameters(
        sample_count,
        initial_levels,
        initial_probabilities,
        gamma_volume,
        transition_probabilities,
        simulation_minimum,
        simulation_maximum,
        rng,
        levels_field_name="initial_levels",
        change_field_name="gamma_volume",
        require_non_negative_levels=True,
    )

    trajectories: dict[int, tuple[float, ...]] = {}
    for point in sorted(collection_points, key=lambda item: item.point_id):
        current_volume = _draw_discrete(
            normalized_levels,
            normalized_initial_probabilities,
            normalized_rng,
        )
        values = [current_volume]
        for _ in range(1, normalized_sample_count):
            relative_change = _draw_discrete(
                (-normalized_gamma, 0.0, normalized_gamma),
                normalized_transition_probabilities,
                normalized_rng,
            )
            current_volume = max(
                normalized_minimum,
                min(
                    normalized_maximum,
                    current_volume * (1.0 + relative_change),
                ),
            )
            values.append(current_volume)
        trajectories[point.point_id] = tuple(values)

    return trajectories


def required_vehicle_capacity(
    volume_trajectories: dict[int, tuple[float, ...]],
) -> float:
    """Return the sum of the maximum volume required by each point."""
    return float(sum(max(trajectory) for trajectory in volume_trajectories.values()))
