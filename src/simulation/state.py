"""Mutable simulation state and sample-and-hold trajectory access."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import floor, isfinite

from src.models.problem import ProblemInstance
from src.simulation.snapshot import OptimizationSnapshot


def _validate_non_negative_number(value: object, field_name: str) -> float:
    """Validate and normalize a finite non-negative number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite numeric value.")

    normalized = float(value)
    if not isfinite(normalized):
        raise ValueError(f"{field_name} must be a finite numeric value.")
    if normalized < 0.0:
        raise ValueError(f"{field_name} must be greater than or equal to zero.")

    return normalized


def _validate_non_negative_int(value: object, field_name: str) -> int:
    """Validate a non-negative integer identifier."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer; got {value!r}.")

    return value


def _normalize_remaining_point_ids(
    values: Sequence[int],
) -> tuple[int, ...]:
    """Normalize remaining point IDs while preserving their order."""
    try:
        normalized = tuple(values)
    except TypeError as exc:
        raise ValueError("remaining_point_ids must be a sequence.") from exc

    point_ids: set[int] = set()
    for point_id in normalized:
        _validate_non_negative_int(point_id, "remaining_point_id")
        if point_id in point_ids:
            raise ValueError(f"duplicate remaining_point_id: {point_id}.")
        point_ids.add(point_id)

    return normalized


def _trajectory_horizon(instance: ProblemInstance) -> float | None:
    """Return the trajectory horizon, or None for an instance without points."""
    if not instance.collection_points:
        return None

    samples = next(iter(instance.temperature_trajectories.values()))
    return (len(samples) - 1) * instance.sample_period


@dataclass
class SimulationState:
    """Mutable state maintained while a simulation is running."""

    current_time: float
    current_node_id: int
    current_load: float
    remaining_point_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        current_time = _validate_non_negative_number(
            self.current_time,
            "current_time",
        )
        current_node_id = _validate_non_negative_int(
            self.current_node_id,
            "current_node_id",
        )
        current_load = _validate_non_negative_number(self.current_load, "current_load")
        remaining_point_ids = _normalize_remaining_point_ids(self.remaining_point_ids)

        self.current_time = current_time
        self.current_node_id = current_node_id
        self.current_load = current_load
        self.remaining_point_ids = remaining_point_ids

    def validate(self, instance: ProblemInstance) -> None:
        """Validate consistency with the static problem instance."""
        network_node_ids = {node.node_id for node in instance.road_network.nodes}
        if self.current_node_id not in network_node_ids:
            raise ValueError(
                "current_node_id does not exist in the instance road network: "
                f"{self.current_node_id}."
            )

        collection_point_ids = {point.point_id for point in instance.collection_points}
        unknown_point_ids = set(self.remaining_point_ids) - collection_point_ids
        if unknown_point_ids:
            raise ValueError(
                "remaining_point_ids contain unknown point_id(s): "
                f"{sorted(unknown_point_ids)}."
            )

        if self.current_load > instance.vehicle_capacity:
            raise ValueError("current_load cannot exceed instance.vehicle_capacity.")

        horizon = _trajectory_horizon(instance)
        if horizon is not None and self.current_time > horizon:
            raise ValueError(
                f"current_time exceeds the trajectory horizon: {horizon} seconds."
            )


def create_initial_simulation_state(instance: ProblemInstance) -> SimulationState:
    """Create the initial state at the depot with all points remaining."""
    state = SimulationState(
        current_time=0.0,
        current_node_id=instance.depot.node_id,
        current_load=0.0,
        remaining_point_ids=tuple(
            sorted(point.point_id for point in instance.collection_points)
        ),
    )
    state.validate(instance)
    return state


def trajectory_sample_index(
    instance: ProblemInstance,
    current_time: float,
) -> int:
    """Return the sample-and-hold trajectory index for a simulation time."""
    normalized_time = _validate_non_negative_number(current_time, "current_time")
    horizon = _trajectory_horizon(instance)
    if horizon is not None and normalized_time > horizon:
        raise ValueError(
            f"current_time exceeds the trajectory horizon: {horizon} seconds."
        )

    return floor(normalized_time / instance.sample_period)


def build_optimization_snapshot(
    instance: ProblemInstance,
    state: SimulationState,
) -> OptimizationSnapshot:
    """Build an immutable snapshot from the current mutable simulation state."""
    state.validate(instance)
    sample_index = trajectory_sample_index(instance, state.current_time)

    temperatures = {
        point_id: instance.temperature_trajectories[point_id][sample_index]
        for point_id in state.remaining_point_ids
    }
    volumes = {
        point_id: instance.volume_trajectories[point_id][sample_index]
        for point_id in state.remaining_point_ids
    }

    snapshot = OptimizationSnapshot(
        current_time=state.current_time,
        current_node_id=state.current_node_id,
        current_load=state.current_load,
        remaining_point_ids=state.remaining_point_ids,
        temperatures=temperatures,
        volumes=volumes,
    )
    snapshot.validate(instance)
    return snapshot
