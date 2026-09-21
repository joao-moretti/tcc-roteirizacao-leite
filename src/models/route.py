"""Static logical route representation."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from .problem import ProblemInstance


def _validate_non_negative_int(value: object, field_name: str) -> None:
    """Validate a non-negative integer identifier, excluding booleans."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer; got {value!r}.")


@dataclass(frozen=True)
class RouteStop:
    """A logical stop identified as a depot or collection point."""

    kind: Literal["depot", "collection"]
    entity_id: int

    def __post_init__(self) -> None:
        if self.kind not in ("depot", "collection"):
            raise ValueError(
                f"kind must be exactly 'depot' or 'collection'; got {self.kind!r}."
            )
        _validate_non_negative_int(self.entity_id, "entity_id")


@dataclass(frozen=True)
class Route:
    """An immutable logical sequence of route stops."""

    stops: Sequence[RouteStop]

    def __post_init__(self) -> None:
        try:
            normalized = tuple(self.stops)
        except TypeError as exc:
            raise ValueError("stops must be a sequence of RouteStop objects.") from exc

        for stop in normalized:
            if not isinstance(stop, RouteStop):
                raise ValueError("stops must contain only RouteStop objects.")

        object.__setattr__(self, "stops", normalized)

    def validate(self, instance: ProblemInstance) -> None:
        """Validate the route structure against a problem instance."""
        if not isinstance(instance, ProblemInstance):
            raise ValueError("instance must be a ProblemInstance.")

        if len(self.stops) < 2:
            raise ValueError("route must contain an initial and a final depot stop.")

        first_stop = self.stops[0]
        last_stop = self.stops[-1]
        depot_id = instance.depot.depot_id

        if first_stop.kind != "depot":
            raise ValueError("route must start with a depot stop.")
        if first_stop.entity_id != depot_id:
            raise ValueError("route must start at the instance depot.")
        if last_stop.kind != "depot":
            raise ValueError("route must end with a depot stop.")
        if last_stop.entity_id != depot_id:
            raise ValueError("route must end at the instance depot.")

        expected_point_ids = {point.point_id for point in instance.collection_points}
        visited_point_ids: set[int] = set()

        for stop in self.stops[1:-1]:
            if stop.kind == "depot":
                raise ValueError("depot stops are not allowed between collections.")

            if stop.entity_id not in expected_point_ids:
                raise ValueError(
                    f"route references an unknown collection point: {stop.entity_id}."
                )
            if stop.entity_id in visited_point_ids:
                raise ValueError(
                    f"collection point appears more than once: {stop.entity_id}."
                )
            visited_point_ids.add(stop.entity_id)

        missing_point_ids = expected_point_ids - visited_point_ids
        if missing_point_ids:
            raise ValueError(
                f"route is missing collection point(s): {sorted(missing_point_ids)}."
            )
