from collections.abc import Mapping

import pytest

from src.models.problem import CollectionPoint, Depot, ProblemInstance
from src.models.road_network import RoadNetwork, RoadNode
from src.simulation.snapshot import OptimizationSnapshot


def make_instance(
    *,
    collection_points: list[CollectionPoint] | None = None,
    trajectory_length: int = 3,
) -> ProblemInstance:
    points = collection_points
    if points is None:
        points = [
            CollectionPoint(point_id=10, node_id=2),
            CollectionPoint(point_id=20, node_id=3),
        ]

    temperatures = {
        point.point_id: tuple(float(index) for index in range(trajectory_length))
        for point in points
    }
    volumes = {
        point.point_id: tuple(1.0 for _ in range(trajectory_length)) for point in points
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


def make_snapshot(
    *,
    current_time: float = 15.0,
    current_node_id: int = 4,
    current_load: float = 2.5,
    remaining_point_ids: list[int] | tuple[int, ...] = (10, 20),
    temperatures: dict[int, float] | None = None,
    volumes: dict[int, float] | None = None,
) -> OptimizationSnapshot:
    if temperatures is None:
        temperatures = {10: -1.0, 20: 5.0}
    if volumes is None:
        volumes = {10: 2.0, 20: 3.0}

    return OptimizationSnapshot(
        current_time=current_time,
        current_node_id=current_node_id,
        current_load=current_load,
        remaining_point_ids=remaining_point_ids,
        temperatures=temperatures,
        volumes=volumes,
    )


def test_snapshot_is_valid_and_normalizes_collections() -> None:
    snapshot = make_snapshot(remaining_point_ids=[20, 10])

    assert snapshot.remaining_point_ids == (20, 10)
    assert snapshot.current_time == 15.0
    assert snapshot.current_node_id == 4
    assert snapshot.current_load == 2.5
    assert snapshot.temperatures == {10: -1.0, 20: 5.0}
    assert snapshot.volumes == {10: 2.0, 20: 3.0}


def test_snapshot_mappings_are_structurally_protected() -> None:
    temperatures = {10: 1.0, 20: 2.0}
    volumes = {10: 3.0, 20: 4.0}
    snapshot = make_snapshot(temperatures=temperatures, volumes=volumes)

    temperatures[10] = 99.0
    volumes[20] = 99.0

    assert snapshot.temperatures[10] == 1.0
    assert snapshot.volumes[20] == 4.0
    assert isinstance(snapshot.temperatures, Mapping)
    assert isinstance(snapshot.volumes, Mapping)

    with pytest.raises(TypeError):
        snapshot.temperatures[10] = 99.0  # type: ignore[index]
    with pytest.raises(TypeError):
        snapshot.volumes[20] = 99.0  # type: ignore[index]


@pytest.mark.parametrize("current_time", [-1.0, float("nan"), float("inf")])
def test_snapshot_rejects_invalid_current_time(current_time: float) -> None:
    with pytest.raises(ValueError, match="current_time"):
        make_snapshot(current_time=current_time)


@pytest.mark.parametrize("current_node_id", [-1, 1.5, True, "1"])
def test_snapshot_rejects_invalid_current_node_id(current_node_id: object) -> None:
    with pytest.raises(ValueError, match="current_node_id"):
        make_snapshot(current_node_id=current_node_id)  # type: ignore[arg-type]


@pytest.mark.parametrize("current_load", [-1.0, float("nan"), float("inf")])
def test_snapshot_rejects_invalid_current_load(current_load: float) -> None:
    with pytest.raises(ValueError, match="current_load"):
        make_snapshot(current_load=current_load)


def test_snapshot_rejects_duplicate_remaining_point_ids() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        make_snapshot(remaining_point_ids=[10, 10])


@pytest.mark.parametrize("point_id", [-1, True, 1.5, "10"])
def test_snapshot_rejects_invalid_remaining_point_id(point_id: object) -> None:
    with pytest.raises(ValueError, match="remaining_point_id"):
        make_snapshot(remaining_point_ids=[point_id])  # type: ignore[list-item]


def test_snapshot_rejects_missing_temperature() -> None:
    with pytest.raises(ValueError, match="temperatures is missing"):
        make_snapshot(temperatures={10: 1.0})


def test_snapshot_rejects_extra_temperature() -> None:
    with pytest.raises(ValueError, match="temperatures contains"):
        make_snapshot(temperatures={10: 1.0, 20: 2.0, 30: 3.0})


def test_snapshot_rejects_missing_volume() -> None:
    with pytest.raises(ValueError, match="volumes is missing"):
        make_snapshot(volumes={10: 1.0})


def test_snapshot_rejects_extra_volume() -> None:
    with pytest.raises(ValueError, match="volumes contains"):
        make_snapshot(volumes={10: 1.0, 20: 2.0, 30: 3.0})


@pytest.mark.parametrize("temperature", [float("nan"), float("inf")])
def test_snapshot_rejects_non_finite_temperature(temperature: float) -> None:
    with pytest.raises(ValueError, match="temperatures"):
        make_snapshot(temperatures={10: temperature, 20: 2.0})


@pytest.mark.parametrize("volume", [-1.0, float("nan"), float("inf")])
def test_snapshot_rejects_invalid_volume(volume: float) -> None:
    with pytest.raises(ValueError, match="volumes"):
        make_snapshot(volumes={10: volume, 20: 2.0})


def test_validate_accepts_intermediate_network_node() -> None:
    snapshot = make_snapshot(current_node_id=4)

    assert snapshot.validate(make_instance()) is None


def test_validate_rejects_unknown_network_node() -> None:
    snapshot = make_snapshot(current_node_id=99)

    with pytest.raises(ValueError, match="current_node_id"):
        snapshot.validate(make_instance())


def test_validate_rejects_unknown_remaining_point() -> None:
    snapshot = make_snapshot(
        remaining_point_ids=[30],
        temperatures={30: 1.0},
        volumes={30: 2.0},
    )

    with pytest.raises(ValueError, match="unknown point_id"):
        snapshot.validate(make_instance())


def test_validate_rejects_load_above_capacity() -> None:
    snapshot = make_snapshot(current_load=10.1)

    with pytest.raises(ValueError, match="vehicle_capacity"):
        snapshot.validate(make_instance())


def test_validate_accepts_subset_of_instance_points() -> None:
    snapshot = make_snapshot(
        remaining_point_ids=[10],
        temperatures={10: 1.0},
        volumes={10: 2.0},
    )

    assert snapshot.validate(make_instance()) is None


def test_validate_accepts_no_remaining_points() -> None:
    snapshot = make_snapshot(
        remaining_point_ids=(),
        temperatures={},
        volumes={},
    )

    assert snapshot.validate(make_instance()) is None


def test_validate_accepts_snapshot_between_sample_times() -> None:
    snapshot = make_snapshot(current_time=15.0)

    assert snapshot.validate(make_instance()) is None


def test_validate_accepts_exact_end_of_trajectory_horizon() -> None:
    snapshot = make_snapshot(current_time=60.0)

    assert snapshot.validate(make_instance(trajectory_length=3)) is None


def test_validate_rejects_time_above_trajectory_horizon() -> None:
    snapshot = make_snapshot(current_time=60.1)

    with pytest.raises(ValueError, match="trajectory horizon"):
        snapshot.validate(make_instance(trajectory_length=3))


def test_validate_skips_horizon_for_instance_without_points() -> None:
    instance = make_instance(collection_points=[])
    snapshot = make_snapshot(
        current_time=1_000.0,
        remaining_point_ids=(),
        temperatures={},
        volumes={},
    )

    assert snapshot.validate(instance) is None
