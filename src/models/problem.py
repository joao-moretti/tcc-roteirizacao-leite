"""Static domain models for a routing problem instance."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import TypeVar

from .road_network import RoadNetwork

_V1_SAMPLE_PERIOD = 30.0
_SampleT = TypeVar("_SampleT")


def _validate_non_negative_int(value: object, field_name: str) -> None:
    """Validate a non-negative integer identifier, excluding booleans."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer; got {value!r}.")


def _validate_parameter(value: object, field_name: str, *, allow_zero: bool) -> float:
    """Validate and normalize a finite non-negative or positive parameter."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite numeric value.")

    normalized = float(value)
    if not isfinite(normalized):
        raise ValueError(f"{field_name} must be a finite numeric value.")

    if allow_zero and normalized < 0:
        raise ValueError(f"{field_name} must be greater than or equal to zero.")
    if not allow_zero and normalized <= 0:
        raise ValueError(f"{field_name} must be greater than zero.")

    return normalized


def _validate_sample(value: object, field_name: str, *, allow_negative: bool) -> float:
    """Validate one finite sample, optionally allowing negative values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite numeric value.")

    normalized = float(value)
    if not isfinite(normalized):
        raise ValueError(f"{field_name} must be a finite numeric value.")

    if not allow_negative and normalized < 0:
        raise ValueError(f"{field_name} must be greater than or equal to zero.")

    return normalized


def _normalize_collection(
    values: Sequence[_SampleT], field_name: str
) -> tuple[_SampleT, ...]:
    """Normalize a sequence into a tuple with a clear domain error."""
    try:
        return tuple(values)
    except TypeError as exc:
        raise ValueError(f"{field_name} must be a sequence.") from exc


def _normalize_trajectories(
    trajectories: Mapping[int, Sequence[float]],
    point_ids: set[int],
    field_name: str,
    *,
    allow_negative: bool,
) -> dict[int, tuple[float, ...]]:
    """Validate trajectory keys and samples, returning protected values later."""
    if not isinstance(trajectories, Mapping):
        raise ValueError(f"{field_name} must be a mapping by point_id.")

    normalized_mapping = dict(trajectories)
    for point_id in normalized_mapping:
        _validate_non_negative_int(point_id, f"{field_name} point_id")

    missing_point_ids = point_ids - normalized_mapping.keys()
    if missing_point_ids:
        raise ValueError(
            f"{field_name} is missing trajectories for point_id(s): "
            f"{sorted(missing_point_ids)}."
        )

    unknown_point_ids = normalized_mapping.keys() - point_ids
    if unknown_point_ids:
        raise ValueError(
            f"{field_name} contains trajectories for unknown point_id(s): "
            f"{sorted(unknown_point_ids)}."
        )

    normalized: dict[int, tuple[float, ...]] = {}
    for point_id, samples in normalized_mapping.items():
        if not isinstance(samples, Sequence):
            raise ValueError(f"{field_name}[{point_id}] must be a sequence of samples.")

        sample_values = _normalize_collection(samples, f"{field_name}[{point_id}]")
        if not sample_values:
            raise ValueError(
                f"{field_name}[{point_id}] must contain at least one sample."
            )

        normalized[point_id] = tuple(
            _validate_sample(
                sample,
                f"{field_name}[{point_id}] sample",
                allow_negative=allow_negative,
            )
            for sample in sample_values
        )

    return normalized


@dataclass(frozen=True)
class CollectionPoint:
    """Static identity and network location of a collection point."""

    point_id: int
    node_id: int

    def __post_init__(self) -> None:
        _validate_non_negative_int(self.point_id, "point_id")
        _validate_non_negative_int(self.node_id, "node_id")


@dataclass(frozen=True)
class Depot:
    """Static identity and network location of the depot."""

    depot_id: int
    node_id: int

    def __post_init__(self) -> None:
        _validate_non_negative_int(self.depot_id, "depot_id")
        _validate_non_negative_int(self.node_id, "node_id")


@dataclass(frozen=True)
class ProblemInstance:
    """Immutable static data and frozen trajectories of one problem instance."""

    instance_id: int
    road_network: RoadNetwork
    depot: Depot
    collection_points: Sequence[CollectionPoint]
    vehicle_capacity: float
    collection_service_duration: float
    v_ref: float
    sample_period: float
    temperature_trajectories: Mapping[int, Sequence[float]]
    volume_trajectories: Mapping[int, Sequence[float]]

    def __post_init__(self) -> None:
        _validate_non_negative_int(self.instance_id, "instance_id")

        if not isinstance(self.road_network, RoadNetwork):
            raise ValueError("road_network must be a RoadNetwork.")
        if not isinstance(self.depot, Depot):
            raise ValueError("depot must be a Depot.")

        vehicle_capacity = _validate_parameter(
            self.vehicle_capacity, "vehicle_capacity", allow_zero=False
        )
        collection_service_duration = _validate_parameter(
            self.collection_service_duration,
            "collection_service_duration",
            allow_zero=True,
        )
        v_ref = _validate_parameter(self.v_ref, "v_ref", allow_zero=False)
        sample_period = _validate_parameter(
            self.sample_period, "sample_period", allow_zero=False
        )
        if sample_period != _V1_SAMPLE_PERIOD:
            raise ValueError(
                "sample_period must be 30.0 seconds in the V1 formulation."
            )

        collection_points = self._normalize_collection_points(self.collection_points)
        point_ids = {point.point_id for point in collection_points}
        network_node_ids = {node.node_id for node in self.road_network.nodes}

        if self.depot.node_id not in network_node_ids:
            raise ValueError(
                f"depot node_id does not exist in road_network: {self.depot.node_id}."
            )

        point_node_ids: set[int] = set()
        for point in collection_points:
            if point.node_id not in network_node_ids:
                raise ValueError(
                    f"collection point node_id does not exist in road_network: "
                    f"{point.node_id}."
                )
            if point.node_id in point_node_ids:
                raise ValueError(
                    f"collection points cannot share node_id: {point.node_id}."
                )
            if point.node_id == self.depot.node_id:
                raise ValueError(
                    "depot node_id cannot be shared with a collection point: "
                    f"{point.node_id}."
                )
            point_node_ids.add(point.node_id)

        temperature_trajectories = _normalize_trajectories(
            self.temperature_trajectories,
            point_ids,
            "temperature_trajectories",
            allow_negative=True,
        )
        volume_trajectories = _normalize_trajectories(
            self.volume_trajectories,
            point_ids,
            "volume_trajectories",
            allow_negative=False,
        )

        trajectory_lengths = {
            len(samples)
            for samples in (
                *temperature_trajectories.values(),
                *volume_trajectories.values(),
            )
        }
        if trajectory_lengths and len(trajectory_lengths) != 1:
            raise ValueError("all trajectories must have the same number of samples.")

        total_maximum_volume = sum(
            max(volume_trajectories[point_id]) for point_id in point_ids
        )
        if total_maximum_volume > vehicle_capacity:
            raise ValueError(
                "sum of maximum volume trajectories cannot exceed vehicle_capacity."
            )

        object.__setattr__(self, "collection_points", collection_points)
        object.__setattr__(self, "vehicle_capacity", vehicle_capacity)
        object.__setattr__(
            self, "collection_service_duration", collection_service_duration
        )
        object.__setattr__(self, "v_ref", v_ref)
        object.__setattr__(self, "sample_period", sample_period)
        object.__setattr__(
            self,
            "temperature_trajectories",
            MappingProxyType(temperature_trajectories),
        )
        object.__setattr__(
            self, "volume_trajectories", MappingProxyType(volume_trajectories)
        )

    @staticmethod
    def _normalize_collection_points(
        collection_points: Sequence[CollectionPoint],
    ) -> tuple[CollectionPoint, ...]:
        """Normalize and validate the collection point sequence."""
        try:
            normalized = tuple(collection_points)
        except TypeError as exc:
            raise ValueError("collection_points must be a sequence.") from exc

        point_ids: set[int] = set()
        for point in normalized:
            if not isinstance(point, CollectionPoint):
                raise ValueError(
                    "collection_points must contain only CollectionPoint objects."
                )
            if point.point_id in point_ids:
                raise ValueError(f"duplicate point_id: {point.point_id}.")
            point_ids.add(point.point_id)

        return normalized
