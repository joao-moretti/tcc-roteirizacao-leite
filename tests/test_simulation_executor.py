import pytest

from src.models.problem import CollectionPoint, Depot, ProblemInstance
from src.models.road_network import RoadArc, RoadNetwork, RoadNode
from src.simulation.events import RecalculationDecision
from src.simulation.executor import (
    NextCollectionExecution,
    execute_to_next_collection,
)
from src.simulation.snapshot import OptimizationSnapshot
from src.simulation.state import SimulationState, build_optimization_snapshot


def make_instance(
    *,
    first_arc_distance: float = 4.0,
    second_arc_distance: float = 6.0,
    direct_arc_distance: float = 20.0,
    service_duration: float = 10.0,
    temperature_10: tuple[float, ...] = (3.0, 3.0, 3.0),
    volume_10: tuple[float, ...] = (1.0, 1.0, 1.0),
    arcs: list[RoadArc] | None = None,
) -> ProblemInstance:
    points = [
        CollectionPoint(point_id=10, node_id=2),
        CollectionPoint(point_id=20, node_id=3),
    ]
    if arcs is None:
        arcs = [
            RoadArc(1, 2, first_arc_distance),
            RoadArc(2, 3, second_arc_distance),
            RoadArc(1, 3, direct_arc_distance),
        ]

    return ProblemInstance(
        instance_id=1,
        road_network=RoadNetwork(
            network_id=1,
            nodes=[
                RoadNode(node_id=1, x=0.0, y=0.0),
                RoadNode(node_id=2, x=100.0, y=0.0),
                RoadNode(node_id=3, x=200.0, y=0.0),
            ],
            arcs=arcs,
        ),
        depot=Depot(depot_id=1, node_id=1),
        collection_points=points,
        vehicle_capacity=20.0,
        collection_service_duration=service_duration,
        v_ref=1.0,
        sample_period=30.0,
        temperature_trajectories={
            10: temperature_10,
            20: (3.0,) * len(temperature_10),
        },
        volume_trajectories={
            10: volume_10,
            20: (2.0,) * len(volume_10),
        },
    )


def make_state(
    *,
    current_time: float = 0.0,
    current_node_id: int = 1,
    current_load: float = 0.0,
    remaining_point_ids: tuple[int, ...] = (10, 20),
) -> SimulationState:
    return SimulationState(
        current_time=current_time,
        current_node_id=current_node_id,
        current_load=current_load,
        remaining_point_ids=remaining_point_ids,
    )


def make_previous_snapshot(
    instance: ProblemInstance,
    state: SimulationState,
) -> OptimizationSnapshot:
    return build_optimization_snapshot(instance, state)


def execute(
    instance: ProblemInstance,
    state: SimulationState,
    previous_snapshot: OptimizationSnapshot,
    target_point_id: int = 20,
    *,
    last_recalculation_time: float | None = None,
    minimum_interval: float = 0.0,
) -> NextCollectionExecution:
    return execute_to_next_collection(
        instance,
        state,
        target_point_id,
        previous_snapshot,
        t_min=4.0,
        t_max=7.0,
        q_low=2.0,
        q_high=5.0,
        last_recalculation_time=last_recalculation_time,
        minimum_interval=minimum_interval,
    )


def test_execute_reaches_target_via_shortest_path_and_collects() -> None:
    instance = make_instance()
    state = make_state()
    previous_snapshot = make_previous_snapshot(instance, state)

    result = execute(instance, state, previous_snapshot)

    assert result.reached_target is True
    assert result.collected_volume == 2.0
    assert result.distance_traveled == 10.0
    assert state.current_node_id == 3
    assert state.current_time == 20.0
    assert state.current_load == 2.0
    assert state.remaining_point_ids == (10,)
    assert result.snapshot.remaining_point_ids == (10,)
    assert result.decision == RecalculationDecision(
        True,
        ("collection_completed",),
    )


def test_execute_uses_indirect_shortest_path_instead_of_direct_arc() -> None:
    instance = make_instance(direct_arc_distance=50.0)
    state = make_state()
    previous_snapshot = make_previous_snapshot(instance, state)

    result = execute(instance, state, previous_snapshot)

    assert result.distance_traveled == 10.0
    assert state.current_time == 20.0


def test_execute_collects_when_vehicle_already_at_target_node() -> None:
    instance = make_instance()
    state = make_state(current_node_id=3)
    previous_snapshot = make_previous_snapshot(instance, state)

    result = execute(instance, state, previous_snapshot)

    assert result.distance_traveled == 0.0
    assert result.reached_target is True
    assert result.collected_volume == 2.0
    assert state.current_node_id == 3
    assert state.current_time == 10.0


def test_execute_stops_at_intermediate_node_for_thermal_event() -> None:
    instance = make_instance(
        first_arc_distance=30.0,
        second_arc_distance=5.0,
        direct_arc_distance=50.0,
        temperature_10=(3.0, 8.0, 8.0),
    )
    state = make_state()
    previous_snapshot = make_previous_snapshot(instance, state)

    result = execute(instance, state, previous_snapshot)

    assert result.reached_target is False
    assert result.collected_volume is None
    assert result.distance_traveled == 30.0
    assert state.current_node_id == 2
    assert state.current_time == 30.0
    assert state.remaining_point_ids == (10, 20)
    assert result.snapshot.remaining_point_ids == (10, 20)
    assert result.decision == RecalculationDecision(
        True,
        ("thermal_band_changed",),
    )


def test_execute_continues_when_thermal_event_is_blocked_by_cooldown() -> None:
    instance = make_instance(
        first_arc_distance=30.0,
        second_arc_distance=5.0,
        direct_arc_distance=50.0,
        temperature_10=(3.0, 8.0, 8.0),
    )
    state = make_state()
    previous_snapshot = make_previous_snapshot(instance, state)

    result = execute(
        instance,
        state,
        previous_snapshot,
        last_recalculation_time=0.0,
        minimum_interval=300.0,
    )

    assert result.reached_target is True
    assert result.collected_volume == 2.0
    assert result.distance_traveled == 35.0
    assert result.decision == RecalculationDecision(
        True,
        ("collection_completed",),
    )
    assert state.current_node_id == 3
    assert state.current_time == 45.0


def test_execute_stops_for_volume_event() -> None:
    instance = make_instance(
        first_arc_distance=30.0,
        second_arc_distance=5.0,
        direct_arc_distance=50.0,
        volume_10=(1.0, 6.0, 6.0),
    )
    state = make_state()
    previous_snapshot = make_previous_snapshot(instance, state)

    result = execute(instance, state, previous_snapshot)

    assert result.reached_target is False
    assert result.collected_volume is None
    assert result.distance_traveled == 30.0
    assert result.decision.reasons == ("volume_band_changed",)
    assert state.current_node_id == 2


@pytest.mark.parametrize("target_point_id", [99, True, 1.5, "20"])
def test_execute_rejects_invalid_target_point_id(target_point_id: object) -> None:
    instance = make_instance()
    state = make_state()
    previous_snapshot = make_previous_snapshot(instance, state)

    with pytest.raises(ValueError, match="target_point_id"):
        execute(
            instance,
            state,
            previous_snapshot,
            target_point_id,  # type: ignore[arg-type]
        )


def test_execute_rejects_target_that_is_not_remaining() -> None:
    instance = make_instance()
    state = make_state(remaining_point_ids=(10,))
    previous_snapshot = make_previous_snapshot(instance, state)

    with pytest.raises(ValueError, match="remaining"):
        execute(instance, state, previous_snapshot, target_point_id=20)


def test_execute_rejects_incompatible_previous_snapshot() -> None:
    instance = make_instance()
    state = make_state()
    previous_snapshot = OptimizationSnapshot(
        current_time=1.0,
        current_node_id=1,
        current_load=0.0,
        remaining_point_ids=(10, 20),
        temperatures={10: 3.0, 20: 3.0},
        volumes={10: 1.0, 20: 2.0},
    )

    with pytest.raises(ValueError, match="previous_snapshot"):
        execute(instance, state, previous_snapshot)


def test_execute_propagates_unreachable_target_error() -> None:
    instance = make_instance(arcs=[])
    state = make_state()
    previous_snapshot = make_previous_snapshot(instance, state)

    with pytest.raises(ValueError, match="no directed path"):
        execute(instance, state, previous_snapshot)


def test_execute_does_not_modify_instance_or_previous_snapshot() -> None:
    instance = make_instance()
    state = make_state()
    previous_snapshot = make_previous_snapshot(instance, state)
    instance_before = (
        dict(instance.temperature_trajectories),
        dict(instance.volume_trajectories),
    )
    snapshot_before = (
        previous_snapshot.current_time,
        previous_snapshot.current_node_id,
        previous_snapshot.current_load,
        previous_snapshot.remaining_point_ids,
        dict(previous_snapshot.temperatures),
        dict(previous_snapshot.volumes),
    )

    execute(instance, state, previous_snapshot)

    assert (
        dict(instance.temperature_trajectories),
        dict(instance.volume_trajectories),
    ) == instance_before
    assert (
        previous_snapshot.current_time,
        previous_snapshot.current_node_id,
        previous_snapshot.current_load,
        previous_snapshot.remaining_point_ids,
        dict(previous_snapshot.temperatures),
        dict(previous_snapshot.volumes),
    ) == snapshot_before
