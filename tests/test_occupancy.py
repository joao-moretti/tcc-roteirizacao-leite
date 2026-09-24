import pytest

from src.models.candidate_route import CandidateRoute
from src.models.problem import CollectionPoint, Depot, ProblemInstance
from src.models.road_network import RoadArc, RoadNetwork, RoadNode
from src.objective.occupancy import (
    normalized_occupancy_cost,
    occupancy_cost,
    occupancy_mean_distance,
    occupancy_reference,
)
from src.simulation.snapshot import OptimizationSnapshot


def make_instance(
    arcs: list[RoadArc],
    *,
    trajectory_volume: float = 1.0,
    collection_points: list[CollectionPoint] | None = None,
    coordinate_scale: float = 1.0,
) -> ProblemInstance:
    points = collection_points
    if points is None:
        points = [
            CollectionPoint(point_id=10, node_id=2),
            CollectionPoint(point_id=20, node_id=3),
            CollectionPoint(point_id=30, node_id=4),
        ]

    temperature_trajectory = tuple(4.0 for _ in range(20))
    volume_trajectory = tuple(trajectory_volume for _ in range(20))

    return ProblemInstance(
        instance_id=1,
        road_network=RoadNetwork(
            network_id=1,
            nodes=[
                RoadNode(node_id=node_id, x=node_id * coordinate_scale, y=0.0)
                for node_id in range(1, 6)
            ],
            arcs=arcs,
        ),
        depot=Depot(depot_id=1, node_id=1),
        collection_points=points,
        vehicle_capacity=10.0,
        collection_service_duration=60.0,
        v_ref=2.0,
        sample_period=30.0,
        temperature_trajectories={
            point.point_id: temperature_trajectory for point in points
        },
        volume_trajectories={point.point_id: volume_trajectory for point in points},
    )


def make_snapshot(
    remaining_point_ids: tuple[int, ...],
    volumes: dict[int, float],
    *,
    current_node_id: int = 1,
    current_load: float = 0.0,
    current_time: float = 300.0,
) -> OptimizationSnapshot:
    return OptimizationSnapshot(
        current_time=current_time,
        current_node_id=current_node_id,
        current_load=current_load,
        remaining_point_ids=remaining_point_ids,
        temperatures={point_id: 4.0 for point_id in remaining_point_ids},
        volumes=volumes,
    )


def test_occupancy_cost_for_one_point() -> None:
    instance = make_instance([RoadArc(1, 2, 4.0)])
    snapshot = make_snapshot((10,), {10: 2.0})

    assert occupancy_cost(instance, snapshot, CandidateRoute((10,))) == 4.0


def test_occupancy_cost_sums_two_points() -> None:
    instance = make_instance([RoadArc(1, 2, 4.0), RoadArc(2, 3, 6.0)])
    snapshot = make_snapshot((10, 20), {10: 1.0, 20: 2.0})

    assert occupancy_cost(instance, snapshot, CandidateRoute((10, 20))) == 12.0


def test_occupancy_cost_changes_with_route_order() -> None:
    instance = make_instance(
        [
            RoadArc(1, 2, 2.0),
            RoadArc(2, 3, 4.0),
            RoadArc(1, 3, 10.0),
            RoadArc(3, 2, 1.0),
        ]
    )
    snapshot = make_snapshot((10, 20), {10: 1.0, 20: 2.0})

    first_order_cost = occupancy_cost(instance, snapshot, CandidateRoute((10, 20)))
    second_order_cost = occupancy_cost(instance, snapshot, CandidateRoute((20, 10)))

    assert first_order_cost == 7.0
    assert second_order_cost == 9.5


def test_occupancy_cost_uses_snapshot_volumes() -> None:
    instance = make_instance([RoadArc(1, 2, 4.0)], trajectory_volume=1.0)
    snapshot = make_snapshot((10,), {10: 2.0})

    assert occupancy_cost(instance, snapshot, CandidateRoute((10,))) == 4.0


def test_instance_volume_trajectories_do_not_change_occupancy_cost() -> None:
    arcs = [RoadArc(1, 2, 4.0)]
    low_trajectory_instance = make_instance(arcs, trajectory_volume=1.0)
    high_trajectory_instance = make_instance(arcs, trajectory_volume=2.0)
    snapshot = make_snapshot((10,), {10: 2.0})
    candidate_route = CandidateRoute((10,))

    assert occupancy_cost(
        low_trajectory_instance, snapshot, candidate_route
    ) == occupancy_cost(high_trajectory_instance, snapshot, candidate_route)


def test_current_load_does_not_change_occupancy_cost() -> None:
    instance = make_instance([RoadArc(1, 2, 4.0)])
    empty_load_snapshot = make_snapshot((10,), {10: 2.0}, current_load=0.0)
    loaded_snapshot = make_snapshot((10,), {10: 2.0}, current_load=10.0)
    candidate_route = CandidateRoute((10,))

    assert occupancy_cost(
        instance, empty_load_snapshot, candidate_route
    ) == occupancy_cost(instance, loaded_snapshot, candidate_route)


def test_empty_candidate_route_has_zero_occupancy_cost() -> None:
    instance = make_instance([], collection_points=[])
    snapshot = make_snapshot((), {})

    assert occupancy_cost(instance, snapshot, CandidateRoute(())) == 0.0


def test_occupancy_cost_propagates_arrival_times_error() -> None:
    instance = make_instance([RoadArc(2, 1, 4.0)])
    snapshot = make_snapshot((10,), {10: 2.0})

    with pytest.raises(ValueError, match="no directed path"):
        occupancy_cost(instance, snapshot, CandidateRoute((10,)))


def test_occupancy_mean_distance_uses_directed_pairs() -> None:
    instance = make_instance([RoadArc(1, 2, 2.0), RoadArc(2, 1, 8.0)])
    snapshot = make_snapshot((10,), {10: 2.0})

    assert occupancy_mean_distance(instance, snapshot) == 5.0


def test_occupancy_mean_distance_uses_indirect_shortest_path() -> None:
    instance = make_instance(
        [
            RoadArc(1, 2, 20.0),
            RoadArc(1, 5, 3.0),
            RoadArc(5, 2, 4.0),
            RoadArc(2, 1, 8.0),
        ]
    )
    snapshot = make_snapshot((10,), {10: 2.0})

    assert occupancy_mean_distance(instance, snapshot) == 7.5


def test_occupancy_mean_distance_ignores_coordinates() -> None:
    arcs = [RoadArc(1, 2, 2.0), RoadArc(2, 1, 8.0)]
    small_coordinates = make_instance(arcs, coordinate_scale=1.0)
    large_coordinates = make_instance(arcs, coordinate_scale=1_000_000.0)
    small_snapshot = make_snapshot((10,), {10: 2.0})
    large_snapshot = make_snapshot((10,), {10: 2.0})

    assert occupancy_mean_distance(small_coordinates, small_snapshot) == (
        occupancy_mean_distance(large_coordinates, large_snapshot)
    )


def test_occupancy_reference_for_one_point() -> None:
    instance = make_instance([RoadArc(1, 2, 4.0), RoadArc(2, 1, 6.0)])
    snapshot = make_snapshot((10,), {10: 2.0})

    assert occupancy_reference(instance, snapshot) == 5.0


def test_occupancy_reference_uses_q_max_and_m_factor() -> None:
    arcs = [
        RoadArc(1, 2, 2.0),
        RoadArc(2, 1, 2.0),
        RoadArc(1, 3, 2.0),
        RoadArc(3, 1, 2.0),
        RoadArc(1, 4, 2.0),
        RoadArc(4, 1, 2.0),
        RoadArc(2, 3, 2.0),
        RoadArc(3, 2, 2.0),
        RoadArc(2, 4, 2.0),
        RoadArc(4, 2, 2.0),
        RoadArc(3, 4, 2.0),
        RoadArc(4, 3, 2.0),
    ]
    instance = make_instance(arcs)
    snapshot = make_snapshot((10, 20, 30), {10: 1.0, 20: 2.0, 30: 3.0})

    assert occupancy_reference(instance, snapshot) == 18.0


def test_occupancy_reference_is_zero_when_all_volumes_are_zero() -> None:
    instance = make_instance([RoadArc(1, 2, 4.0), RoadArc(2, 1, 4.0)])
    snapshot = make_snapshot((10,), {10: 0.0})

    assert occupancy_reference(instance, snapshot) == 0.0


def test_occupancy_reference_is_zero_without_remaining_points() -> None:
    instance = make_instance([], collection_points=[])
    snapshot = make_snapshot((), {})

    assert occupancy_reference(instance, snapshot) == 0.0


def test_normalized_occupancy_cost_divides_by_reference() -> None:
    instance = make_instance([RoadArc(1, 2, 4.0)])
    snapshot = make_snapshot((10,), {10: 2.0})

    assert (
        normalized_occupancy_cost(
            instance,
            snapshot,
            CandidateRoute((10,)),
            8.0,
        )
        == 0.5
    )


def test_normalized_occupancy_cost_can_be_greater_than_one() -> None:
    instance = make_instance([RoadArc(1, 2, 4.0)])
    snapshot = make_snapshot((10,), {10: 2.0})

    assert (
        normalized_occupancy_cost(
            instance,
            snapshot,
            CandidateRoute((10,)),
            1.0,
        )
        == 4.0
    )


def test_normalized_occupancy_cost_accepts_zero_reference_for_zero_cost() -> None:
    instance = make_instance([RoadArc(1, 2, 4.0)])
    snapshot = make_snapshot((10,), {10: 0.0})

    assert (
        normalized_occupancy_cost(
            instance,
            snapshot,
            CandidateRoute((10,)),
            0.0,
        )
        == 0.0
    )


def test_normalized_occupancy_cost_rejects_zero_reference_for_positive_cost() -> None:
    instance = make_instance([RoadArc(1, 2, 4.0)])
    snapshot = make_snapshot((10,), {10: 2.0})

    with pytest.raises(ValueError, match="reference"):
        normalized_occupancy_cost(
            instance,
            snapshot,
            CandidateRoute((10,)),
            0.0,
        )


@pytest.mark.parametrize("reference", [-1.0, float("nan"), float("inf"), True])
def test_normalized_occupancy_cost_rejects_invalid_reference(
    reference: object,
) -> None:
    instance = make_instance([RoadArc(1, 2, 4.0)])
    snapshot = make_snapshot((10,), {10: 2.0})

    with pytest.raises(ValueError, match="reference"):
        normalized_occupancy_cost(  # type: ignore[arg-type]
            instance,
            snapshot,
            CandidateRoute((10,)),
            reference,
        )


def test_occupancy_mean_distance_propagates_missing_directed_path() -> None:
    instance = make_instance([RoadArc(2, 1, 4.0)])
    snapshot = make_snapshot((10,), {10: 2.0})

    with pytest.raises(ValueError, match="no directed path"):
        occupancy_mean_distance(instance, snapshot)
