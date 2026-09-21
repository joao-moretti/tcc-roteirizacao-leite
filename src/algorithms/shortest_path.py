"""Shortest directed paths over a road network."""

import heapq

from src.models.road_network import RoadNetwork


def _validate_node_id(node_id: object, field_name: str) -> None:
    """Validate a non-negative integer node identifier."""
    if isinstance(node_id, bool) or not isinstance(node_id, int) or node_id < 0:
        raise ValueError(
            f"{field_name} must be a non-negative integer; got {node_id!r}."
        )


def shortest_path(
    network: RoadNetwork,
    start_node_id: int,
    end_node_id: int,
) -> tuple[float, tuple[int, ...]]:
    """Return the shortest directed path between two network nodes."""
    _validate_node_id(start_node_id, "start_node_id")
    _validate_node_id(end_node_id, "end_node_id")

    node_ids = {node.node_id for node in network.nodes}
    if start_node_id not in node_ids:
        raise ValueError(f"start node does not exist in network: {start_node_id}.")
    if end_node_id not in node_ids:
        raise ValueError(f"end node does not exist in network: {end_node_id}.")

    if start_node_id == end_node_id:
        return 0.0, (start_node_id,)

    adjacency: dict[int, list[tuple[int, float]]] = {
        node_id: [] for node_id in node_ids
    }
    for arc in network.arcs:
        adjacency[arc.from_node_id].append((arc.to_node_id, arc.distance))

    for neighbors in adjacency.values():
        neighbors.sort(key=lambda item: item[0])

    distances = {start_node_id: 0.0}
    predecessors: dict[int, int | None] = {start_node_id: None}
    priority_queue: list[tuple[float, int]] = [(0.0, start_node_id)]

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)
        if current_distance != distances[current_node]:
            continue

        if current_node == end_node_id:
            break

        for neighbor, distance in adjacency[current_node]:
            candidate_distance = current_distance + distance
            known_distance = distances.get(neighbor)
            if known_distance is not None and candidate_distance >= known_distance:
                continue

            distances[neighbor] = candidate_distance
            predecessors[neighbor] = current_node
            heapq.heappush(priority_queue, (candidate_distance, neighbor))

    if end_node_id not in distances:
        raise ValueError(
            f"no directed path exists from {start_node_id} to {end_node_id}."
        )

    path: list[int] = []
    current_node: int | None = end_node_id
    while current_node is not None:
        path.append(current_node)
        current_node = predecessors[current_node]

    path.reverse()
    return distances[end_node_id], tuple(path)
