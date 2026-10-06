"""Reproducible generation of directed grid road networks."""

import random
from collections import deque
from math import isfinite

from src.models.road_network import RoadArc, RoadNetwork, RoadNode

GRID_NETWORK_MAX_ATTEMPTS = 1000


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
