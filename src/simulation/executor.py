"""Execution of one directed trip to a previously selected producer."""

from dataclasses import dataclass

from src.algorithms.shortest_path import shortest_path
from src.models.problem import ProblemInstance
from src.simulation.events import (
    RecalculationDecision,
    evaluate_recalculation,
)
from src.simulation.operations import advance_to_adjacent_node, complete_collection
from src.simulation.snapshot import OptimizationSnapshot
from src.simulation.state import (
    SimulationState,
    build_optimization_snapshot,
)


@dataclass(frozen=True)
class NextCollectionExecution:
    """Result of executing movement to one selected collection point."""

    target_point_id: int
    reached_target: bool
    collected_volume: float | None
    distance_traveled: float
    snapshot: OptimizationSnapshot
    decision: RecalculationDecision


def _validate_non_negative_int(value: object, field_name: str) -> int:
    """Validate a non-negative integer identifier."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer; got {value!r}.")

    return value


def _validate_initial_snapshot(
    state: SimulationState,
    previous_snapshot: OptimizationSnapshot,
) -> None:
    """Ensure that the snapshot describes the state at execution start."""
    if (
        previous_snapshot.current_time != state.current_time
        or previous_snapshot.current_node_id != state.current_node_id
        or previous_snapshot.current_load != state.current_load
        or previous_snapshot.remaining_point_ids != state.remaining_point_ids
    ):
        raise ValueError(
            "previous_snapshot does not match the simulation state at the "
            "start of execution."
        )


def _target_collection_point(instance: ProblemInstance, target_point_id: int):
    """Return the static collection point identified by target_point_id."""
    return next(
        (
            point
            for point in instance.collection_points
            if point.point_id == target_point_id
        ),
        None,
    )


def _evaluate(
    instance: ProblemInstance,
    previous_snapshot: OptimizationSnapshot,
    current_snapshot: OptimizationSnapshot,
    *,
    t_min: float,
    t_max: float,
    q_low: float,
    q_high: float,
    last_recalculation_time: float | None,
    minimum_interval: float,
) -> RecalculationDecision:
    """Evaluate one transition between snapshots."""
    return evaluate_recalculation(
        instance,
        previous_snapshot,
        current_snapshot,
        t_min=t_min,
        t_max=t_max,
        q_low=q_low,
        q_high=q_high,
        last_recalculation_time=last_recalculation_time,
        minimum_interval=minimum_interval,
    )


def execute_to_next_collection(
    instance: ProblemInstance,
    state: SimulationState,
    target_point_id: int,
    previous_snapshot: OptimizationSnapshot,
    *,
    t_min: float,
    t_max: float,
    q_low: float,
    q_high: float,
    last_recalculation_time: float | None,
    minimum_interval: float,
) -> NextCollectionExecution:
    """Execute movement, event checks, and collection for one target point."""
    state.validate(instance)
    previous_snapshot.validate(instance)
    _validate_initial_snapshot(state, previous_snapshot)

    _validate_non_negative_int(target_point_id, "target_point_id")
    collection_point = _target_collection_point(instance, target_point_id)
    if collection_point is None:
        raise ValueError(
            f"target_point_id does not exist in the instance: {target_point_id}."
        )
    if target_point_id not in state.remaining_point_ids:
        raise ValueError(
            f"target_point_id {target_point_id} is not among the remaining point_ids."
        )

    _, path = shortest_path(
        instance.road_network,
        state.current_node_id,
        collection_point.node_id,
    )

    distance_traveled = 0.0
    for next_node_id in path[1:]:
        distance_traveled += advance_to_adjacent_node(
            instance,
            state,
            next_node_id,
        )
        current_snapshot = build_optimization_snapshot(instance, state)
        decision = _evaluate(
            instance,
            previous_snapshot,
            current_snapshot,
            t_min=t_min,
            t_max=t_max,
            q_low=q_low,
            q_high=q_high,
            last_recalculation_time=last_recalculation_time,
            minimum_interval=minimum_interval,
        )

        if (
            decision.should_recalculate
            and state.current_node_id != collection_point.node_id
        ):
            return NextCollectionExecution(
                target_point_id=target_point_id,
                reached_target=False,
                collected_volume=None,
                distance_traveled=distance_traveled,
                snapshot=current_snapshot,
                decision=decision,
            )

        previous_snapshot = current_snapshot

    collected_volume = complete_collection(instance, state, target_point_id)
    collection_snapshot = build_optimization_snapshot(instance, state)
    collection_decision = _evaluate(
        instance,
        previous_snapshot,
        collection_snapshot,
        t_min=t_min,
        t_max=t_max,
        q_low=q_low,
        q_high=q_high,
        last_recalculation_time=last_recalculation_time,
        minimum_interval=minimum_interval,
    )

    return NextCollectionExecution(
        target_point_id=target_point_id,
        reached_target=True,
        collected_volume=collected_volume,
        distance_traveled=distance_traveled,
        snapshot=collection_snapshot,
        decision=collection_decision,
    )
