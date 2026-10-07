import pytest

from src.models.problem import CollectionPoint, Depot, ProblemInstance
from src.models.road_network import RoadNetwork, RoadNode
from src.simulation.snapshot import OptimizationSnapshot
from src.simulation.state import (
    SimulationState,
    build_optimization_snapshot,
    create_initial_simulation_state,
    trajectory_sample_index,
)


def make_instance(
    *,
    collection_points: list[CollectionPoint] | None = None,
    trajectory_length: int = 3,
) -> ProblemInstance:
    points = collection_points
    if points is None:
        points = [
            CollectionPoint(point_id=20, node_id=2),
            CollectionPoint(point_id=10, node_id=3),
        ]

    temperatures = {
        point.point_id: tuple(
            float(point.point_id + sample_index)
            for sample_index in range(trajectory_length)
        )
        for point in points
    }
    volumes = {
        point.point_id: tuple(
            float(sample_index + 1) for sample_index in range(trajectory_length)
        )
        for point in points
    }

    return ProblemInstance(
        instance_id=1,
        road_network=RoadNetwork(
            network_id=1,
            nodes=[
                RoadNode(node_id=node_id, x=float(node_id), y=0.0)
                for node_id in range(1, 5)
            ],
            arcs=[],
        ),
        depot=Depot(depot_id=1, node_id=1),
        collection_points=points,
        vehicle_capacity=10.0,
        collection_service_duration=60.0,
        v_ref=10.0,
        sample_period=30.0,
        temperature_trajectories=temperatures,
        volume_trajectories=volumes,
    )


def test_simulation_state_normalizes_values_and_preserves_point_order() -> None:
    state = SimulationState(15, 4, 2, [20, 10])

    assert state.current_time == 15.0
    assert isinstance(state.current_time, float)
    assert state.current_load == 2.0
    assert isinstance(state.current_load, float)
    assert state.remaining_point_ids == (20, 10)


@pytest.mark.parametrize(
    "current_time",
    [-1.0, float("nan"), float("inf"), True, "1"],
)
def test_simulation_state_rejects_invalid_current_time(current_time: object) -> None:
    with pytest.raises(ValueError, match="current_time"):
        SimulationState(current_time, 1, 0.0, ())  # type: ignore[arg-type]


@pytest.mark.parametrize("current_node_id", [-1, 1.5, True, "1"])
def test_simulation_state_rejects_invalid_current_node_id(
    current_node_id: object,
) -> None:
    with pytest.raises(ValueError, match="current_node_id"):
        SimulationState(0.0, current_node_id, 0.0, ())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "current_load",
    [-1.0, float("nan"), float("inf"), True, "1"],
)
def test_simulation_state_rejects_invalid_current_load(current_load: object) -> None:
    with pytest.raises(ValueError, match="current_load"):
        SimulationState(0.0, 1, current_load, ())  # type: ignore[arg-type]


@pytest.mark.parametrize("point_id", [-1, 1.5, True, "10"])
def test_simulation_state_rejects_invalid_remaining_point_id(
    point_id: object,
) -> None:
    with pytest.raises(ValueError, match="remaining_point_id"):
        SimulationState(0.0, 1, 0.0, [point_id])  # type: ignore[list-item]


def test_simulation_state_rejects_duplicate_remaining_point_ids() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        SimulationState(0.0, 1, 0.0, [10, 10])


def test_validate_rejects_unknown_node_point_load_and_time() -> None:
    instance = make_instance()

    with pytest.raises(ValueError, match="current_node_id"):
        SimulationState(0.0, 99, 0.0, ()).validate(instance)

    with pytest.raises(ValueError, match="unknown point_id"):
        SimulationState(0.0, 1, 0.0, (99,)).validate(instance)

    with pytest.raises(ValueError, match="vehicle_capacity"):
        SimulationState(0.0, 1, 10.1, ()).validate(instance)

    with pytest.raises(ValueError, match="trajectory horizon"):
        SimulationState(60.1, 1, 0.0, ()).validate(instance)


def test_validate_accepts_time_at_trajectory_horizon() -> None:
    assert SimulationState(60.0, 1, 0.0, ()).validate(make_instance()) is None


def test_create_initial_simulation_state_starts_at_depot_and_sorts_points() -> None:
    state = create_initial_simulation_state(make_instance())

    assert state.current_time == 0.0
    assert state.current_node_id == 1
    assert state.current_load == 0.0
    assert state.remaining_point_ids == (10, 20)


@pytest.mark.parametrize(
    ("current_time", "expected_index"),
    [
        (0.0, 0),
        (29.999, 0),
        (30.0, 1),
        (59.999, 1),
        (60.0, 2),
    ],
)
def test_trajectory_sample_index_uses_sample_and_hold(
    current_time: float,
    expected_index: int,
) -> None:
    assert trajectory_sample_index(make_instance(), current_time) == expected_index


@pytest.mark.parametrize("current_time", [-1.0, float("nan"), float("inf"), True])
def test_trajectory_sample_index_rejects_invalid_time(current_time: object) -> None:
    with pytest.raises(ValueError, match="current_time"):
        trajectory_sample_index(make_instance(), current_time)  # type: ignore[arg-type]


def test_trajectory_sample_index_rejects_time_after_horizon() -> None:
    with pytest.raises(ValueError, match="trajectory horizon"):
        trajectory_sample_index(make_instance(), 60.1)


def test_build_snapshot_copies_state_and_reads_the_selected_samples() -> None:
    instance = make_instance()
    state = SimulationState(30.0, 4, 2, [20, 10])
    state_before = (
        state.current_time,
        state.current_node_id,
        state.current_load,
        state.remaining_point_ids,
    )

    snapshot = build_optimization_snapshot(instance, state)

    assert isinstance(snapshot, OptimizationSnapshot)
    assert snapshot.current_time == 30.0
    assert snapshot.current_node_id == 4
    assert snapshot.current_load == 2.0
    assert snapshot.remaining_point_ids == (20, 10)
    assert dict(snapshot.temperatures) == {20: 21.0, 10: 11.0}
    assert dict(snapshot.volumes) == {20: 2.0, 10: 2.0}
    assert state_before == (
        state.current_time,
        state.current_node_id,
        state.current_load,
        state.remaining_point_ids,
    )


def test_build_snapshot_excludes_points_removed_from_state() -> None:
    snapshot = build_optimization_snapshot(
        make_instance(),
        SimulationState(0.0, 1, 0.0, (10,)),
    )

    assert snapshot.remaining_point_ids == (10,)
    assert dict(snapshot.temperatures) == {10: 10.0}
    assert dict(snapshot.volumes) == {10: 1.0}


def test_build_snapshot_holds_values_between_samples() -> None:
    snapshot = build_optimization_snapshot(
        make_instance(),
        SimulationState(45.0, 1, 0.0, (10,)),
    )

    assert snapshot.temperatures[10] == 11.0
    assert snapshot.volumes[10] == 2.0


def test_build_snapshot_supports_terminal_state() -> None:
    snapshot = build_optimization_snapshot(
        make_instance(),
        SimulationState(0.0, 1, 0.0, ()),
    )

    assert snapshot.remaining_point_ids == ()
    assert dict(snapshot.temperatures) == {}
    assert dict(snapshot.volumes) == {}


def test_build_snapshot_does_not_modify_instance() -> None:
    instance = make_instance()
    temperatures_before = dict(instance.temperature_trajectories)
    volumes_before = dict(instance.volume_trajectories)

    build_optimization_snapshot(
        instance,
        SimulationState(45.0, 1, 0.0, (10, 20)),
    )

    assert dict(instance.temperature_trajectories) == temperatures_before
    assert dict(instance.volume_trajectories) == volumes_before
