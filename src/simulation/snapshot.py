"""Immutable optimization snapshot for the dynamic simulation."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType

from src.models.problem import ProblemInstance


def _validate_non_negative_int(value: object, field_name: str) -> int:
    """Validate and return a non-negative integer identifier."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer; got {value!r}.")

    return value


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


def _validate_finite_number(value: object, field_name: str) -> float:
    """Validate and normalize a finite number, including negative values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite numeric value.")

    normalized = float(value)
    if not isfinite(normalized):
        raise ValueError(f"{field_name} must be a finite numeric value.")

    return normalized


def _normalize_remaining_point_ids(
    values: Sequence[int],
) -> tuple[int, ...]:
    """Validate remaining point IDs and preserve their input order."""
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


def _normalize_snapshot_mapping(
    values: Mapping[int, float],
    remaining_point_ids: tuple[int, ...],
    field_name: str,
    *,
    allow_negative: bool,
) -> MappingProxyType:
    """Validate a snapshot mapping and protect its normalized values."""
    if not isinstance(values, Mapping):
        raise ValueError(f"{field_name} must be a mapping by point_id.")

    normalized_mapping = dict(values)
    remaining_ids = set(remaining_point_ids)

    for point_id in normalized_mapping:
        _validate_non_negative_int(point_id, f"{field_name} point_id")

    missing_point_ids = remaining_ids - normalized_mapping.keys()
    if missing_point_ids:
        raise ValueError(
            f"{field_name} is missing values for point_id(s): "
            f"{sorted(missing_point_ids)}."
        )

    extra_point_ids = normalized_mapping.keys() - remaining_ids
    if extra_point_ids:
        raise ValueError(
            f"{field_name} contains values for extra point_id(s): "
            f"{sorted(extra_point_ids)}."
        )

    if allow_negative:
        normalized = {
            point_id: _validate_finite_number(value, f"{field_name}[{point_id}]")
            for point_id, value in normalized_mapping.items()
        }
    else:
        normalized = {
            point_id: _validate_non_negative_number(value, f"{field_name}[{point_id}]")
            for point_id, value in normalized_mapping.items()
        }

    return MappingProxyType(normalized)


@dataclass(frozen=True)
class OptimizationSnapshot:
    """Immutable state known when a candidate route is optimized."""

    current_time: float
    current_node_id: int
    current_load: float
    remaining_point_ids: Sequence[int]
    temperatures: Mapping[int, float]
    volumes: Mapping[int, float]

    def __post_init__(self) -> None:
        current_time = _validate_non_negative_number(self.current_time, "current_time")
        current_node_id = _validate_non_negative_int(
            self.current_node_id, "current_node_id"
        )
        current_load = _validate_non_negative_number(self.current_load, "current_load")
        remaining_point_ids = _normalize_remaining_point_ids(self.remaining_point_ids)
        temperatures = _normalize_snapshot_mapping(
            self.temperatures,
            remaining_point_ids,
            "temperatures",
            allow_negative=True,
        )
        volumes = _normalize_snapshot_mapping(
            self.volumes,
            remaining_point_ids,
            "volumes",
            allow_negative=False,
        )

        object.__setattr__(self, "current_time", current_time)
        object.__setattr__(self, "current_node_id", current_node_id)
        object.__setattr__(self, "current_load", current_load)
        object.__setattr__(self, "remaining_point_ids", remaining_point_ids)
        object.__setattr__(self, "temperatures", temperatures)
        object.__setattr__(self, "volumes", volumes)

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

        if instance.collection_points:
            samples = next(iter(instance.temperature_trajectories.values()))
            horizon = (len(samples) - 1) * instance.sample_period
            if self.current_time > horizon:
                raise ValueError(
                    f"current_time exceeds the trajectory horizon: {horizon} seconds."
                )
