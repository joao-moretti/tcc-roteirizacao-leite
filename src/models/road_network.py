"""Structural models for the directed road network."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import TypeVar

_ItemT = TypeVar("_ItemT")


def _validate_non_negative_int(value: object, field_name: str) -> None:
    """Validate an integer identifier used by the road network models."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer; got {value!r}.")


def _validate_coordinate(value: object, field_name: str) -> None:
    """Validate a finite Cartesian coordinate expressed in meters."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{field_name} must be a finite number in meters; got {value!r}."
        )

    if not isfinite(float(value)):
        raise ValueError(
            f"{field_name} must be a finite number in meters; got {value!r}."
        )


def _validate_distance(value: object) -> None:
    """Validate a positive arc distance expressed in meters."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"distance must be a positive finite number in meters; got {value!r}."
        )

    if not isfinite(float(value)) or value <= 0:
        raise ValueError(
            f"distance must be a positive finite number in meters; got {value!r}."
        )


@dataclass(frozen=True)
class RoadNode:
    """A node of the directed road network.

    Coordinates are Cartesian coordinates in meters and are available for
    generation and visualization. They are not used to calculate operational
    travel distances.
    """

    node_id: int
    x: float
    y: float

    def __post_init__(self) -> None:
        _validate_non_negative_int(self.node_id, "node_id")
        _validate_coordinate(self.x, "x")
        _validate_coordinate(self.y, "y")

        object.__setattr__(self, "x", float(self.x))
        object.__setattr__(self, "y", float(self.y))


@dataclass(frozen=True)
class RoadArc:
    """A directed arc with a strictly positive distance in meters."""

    from_node_id: int
    to_node_id: int
    distance: float

    def __post_init__(self) -> None:
        _validate_non_negative_int(self.from_node_id, "from_node_id")
        _validate_non_negative_int(self.to_node_id, "to_node_id")

        if self.from_node_id == self.to_node_id:
            raise ValueError(
                "from_node_id and to_node_id must identify different nodes."
            )

        _validate_distance(self.distance)
        object.__setattr__(self, "distance", float(self.distance))


@dataclass(frozen=True)
class RoadNetwork:
    """An immutable collection of road nodes and directed road arcs."""

    network_id: int
    nodes: Sequence[RoadNode]
    arcs: Sequence[RoadArc]

    def __post_init__(self) -> None:
        _validate_non_negative_int(self.network_id, "network_id")

        nodes = self._normalize_items(self.nodes, RoadNode, "nodes")
        arcs = self._normalize_items(self.arcs, RoadArc, "arcs")

        node_ids: set[int] = set()
        for node in nodes:
            if node.node_id in node_ids:
                raise ValueError(f"duplicate node_id in network: {node.node_id}.")
            node_ids.add(node.node_id)

        arc_pairs: set[tuple[int, int]] = set()
        for arc in arcs:
            if arc.from_node_id not in node_ids:
                raise ValueError(
                    f"arc from_node_id references an unknown node: {arc.from_node_id}."
                )
            if arc.to_node_id not in node_ids:
                raise ValueError(
                    f"arc to_node_id references an unknown node: {arc.to_node_id}."
                )

            pair = (arc.from_node_id, arc.to_node_id)
            if pair in arc_pairs:
                raise ValueError(
                    "duplicate directed arc for ordered pair "
                    f"({arc.from_node_id}, {arc.to_node_id})."
                )
            arc_pairs.add(pair)

        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "arcs", arcs)

    @staticmethod
    def _normalize_items(
        items: Iterable[_ItemT], item_type: type[_ItemT], field_name: str
    ) -> tuple[_ItemT, ...]:
        """Convert a collection to a tuple and validate its item type."""
        try:
            normalized = tuple(items)
        except TypeError as exc:
            raise ValueError(f"{field_name} must be an iterable collection.") from exc

        for item in normalized:
            if not isinstance(item, item_type):
                raise ValueError(
                    f"{field_name} must contain only {item_type.__name__} objects; "
                    f"got {item!r}."
                )

        return normalized
