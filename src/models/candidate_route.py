"""Representation of a candidate route for the dynamic subproblem."""

from collections.abc import Sequence
from dataclasses import dataclass

from src.models.problem import ProblemInstance
from src.simulation.snapshot import OptimizationSnapshot


def _validate_non_negative_int(value: object, field_name: str) -> int:
    """Validate a non-negative integer identifier, excluding booleans."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer; got {value!r}.")

    return value


def _normalize_point_ids(point_ids: Sequence[int]) -> tuple[int, ...]:
    """Normalize point IDs to a tuple while preserving their order."""
    try:
        normalized = tuple(point_ids)
    except TypeError as exc:
        raise ValueError("point_ids must be a sequence.") from exc

    seen_point_ids: set[int] = set()
    for point_id in normalized:
        _validate_non_negative_int(point_id, "point_id")
        if point_id in seen_point_ids:
            raise ValueError(f"duplicate point_id: {point_id}.")
        seen_point_ids.add(point_id)

    return normalized


@dataclass(frozen=True)
class CandidateRoute:
    """Ordered permutation of collection points remaining in a snapshot."""

    point_ids: Sequence[int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "point_ids", _normalize_point_ids(self.point_ids))

    def validate(
        self,
        instance: ProblemInstance,
        snapshot: OptimizationSnapshot,
    ) -> None:
        """Validate that this route is a permutation of the remaining points."""
        snapshot.validate(instance)

        expected_point_ids = set(snapshot.remaining_point_ids)
        candidate_point_ids = set(self.point_ids)

        missing_point_ids = expected_point_ids - candidate_point_ids
        extra_point_ids = candidate_point_ids - expected_point_ids
        if missing_point_ids or extra_point_ids:
            details: list[str] = []
            if missing_point_ids:
                details.append(f"missing point_id(s): {sorted(missing_point_ids)}")
            if extra_point_ids:
                details.append(f"extra point_id(s): {sorted(extra_point_ids)}")
            raise ValueError(
                "candidate route does not match remaining points: "
                + "; ".join(details)
                + "."
            )
