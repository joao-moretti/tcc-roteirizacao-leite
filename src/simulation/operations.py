"""Atomic movement and collection operations for the simulation state."""

from src.models.problem import ProblemInstance
from src.simulation.state import (
    SimulationState,
    trajectory_sample_index,
)


def _validate_non_negative_int(value: object, field_name: str) -> int:
    """Validate a non-negative integer identifier."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer; got {value!r}.")

    return value


def advance_to_adjacent_node(
    instance: ProblemInstance,
    state: SimulationState,
    next_node_id: int,
) -> float:
    """Advance the state over one existing directed road arc."""
    state.validate(instance)
    _validate_non_negative_int(next_node_id, "next_node_id")

    network_node_ids = {node.node_id for node in instance.road_network.nodes}
    if next_node_id not in network_node_ids:
        raise ValueError(
            f"next_node_id does not exist in the instance road network: {next_node_id}."
        )
    if next_node_id == state.current_node_id:
        raise ValueError("next_node_id must differ from current_node_id.")

    arc = next(
        (
            candidate
            for candidate in instance.road_network.arcs
            if candidate.from_node_id == state.current_node_id
            and candidate.to_node_id == next_node_id
        ),
        None,
    )
    if arc is None:
        raise ValueError(
            "the directed arc does not exist from "
            f"{state.current_node_id} to {next_node_id}."
        )

    travel_time = arc.distance / instance.v_ref
    arrival_time = state.current_time + travel_time
    trajectory_sample_index(instance, arrival_time)

    state.current_time = arrival_time
    state.current_node_id = next_node_id
    return float(arc.distance)


def complete_collection(
    instance: ProblemInstance,
    state: SimulationState,
    point_id: int,
) -> float:
    """Complete one collection service at the producer's current node."""
    state.validate(instance)
    _validate_non_negative_int(point_id, "point_id")

    collection_point = next(
        (point for point in instance.collection_points if point.point_id == point_id),
        None,
    )
    if collection_point is None:
        raise ValueError(f"point_id does not exist in the instance: {point_id}.")
    if point_id not in state.remaining_point_ids:
        raise ValueError(f"point_id {point_id} is not among the remaining point_ids.")
    if state.current_node_id != collection_point.node_id:
        raise ValueError(
            f"current_node_id {state.current_node_id} does not match the "
            f"collection point node_id {collection_point.node_id}."
        )

    completion_time = state.current_time + instance.collection_service_duration
    sample_index = trajectory_sample_index(instance, completion_time)
    collected_volume = float(instance.volume_trajectories[point_id][sample_index])
    new_load = state.current_load + collected_volume
    if new_load > instance.vehicle_capacity:
        raise ValueError("collection would exceed instance.vehicle_capacity.")

    state.current_time = completion_time
    state.current_load = new_load
    state.remaining_point_ids = tuple(
        remaining_point_id
        for remaining_point_id in state.remaining_point_ids
        if remaining_point_id != point_id
    )
    return collected_volume
