import pytest

from src.models.candidate_route import CandidateRoute
from src.models.problem import CollectionPoint, Depot, ProblemInstance
from src.models.road_network import RoadArc, RoadNetwork, RoadNode
from src.objective.arrival_times import arrival_times
from src.simulation.snapshot import OptimizationSnapshot


def make_instance(
    arcs: list[RoadArc],
    *,
    v_ref: float = 2.0,
    collection_service_duration: float = 60.0,
    collection_points: list[CollectionPoint] | None = None,
) -> ProblemInstance:
    points = collection_points
    if points is None:
        points = [
            CollectionPoint(point_id=10, node_id=2),
            CollectionPoint(point_id=20, node_id=3),
            CollectionPoint(point_id=30, node_id=4),
        ]

    trajectory = tuple(1.0 for _ in range(20))
    temperature_trajectory = tuple(4.0 for _ in range(20))

    return ProblemInstance(
        instance_id=1,
        road_network=RoadNetwork(
            network_id=1,
            nodes=[
                RoadNode(node_id=node_id, x=float(node_id), y=0.0)
                for node_id in range(1, 6)
            ],
            arcs=arcs,
        ),
        depot=Depot(depot_id=1, node_id=1),
        collection_points=points,
        vehicle_capacity=10.0,
        collection_service_duration=collection_service_duration,
        v_ref=v_ref,
        sample_period=30.0,
        temperature_trajectories={
            point.point_id: temperature_trajectory for point in points
        },
        volume_trajectories={point.point_id: trajectory for point in points},
    )


def make_snapshot(
    remaining_point_ids: tuple[int, ...],
    *,
    current_node_id: int = 1,
    current_time: float = 300.0,
) -> OptimizationSnapshot:
    return OptimizationSnapshot(
        current_time=current_time,
        current_node_id=current_node_id,
        current_load=0.0,
        remaining_point_ids=remaining_point_ids,
        temperatures={point_id: 4.0 for point_id in remaining_point_ids},
        volumes={point_id: 1.0 for point_id in remaining_point_ids},
    )


def test_arrival_times_for_one_point() -> None:
    instance = make_instance([RoadArc(1, 2, 6.0)])
    snapshot = make_snapshot((10,))
    candidate_route = CandidateRoute((10,))

    assert arrival_times(instance, snapshot, candidate_route) == {10: 3.0}


def test_arrival_times_accumulate_two_points() -> None:
    instance = make_instance([RoadArc(1, 2, 6.0), RoadArc(2, 3, 4.0)])
    snapshot = make_snapshot((10, 20))
    candidate_route = CandidateRoute((10, 20))

    assert arrival_times(instance, snapshot, candidate_route) == {
        10: 3.0,
        20: 5.0,
    }


def test_arrival_times_accumulate_three_points() -> None:
    instance = make_instance(
        [
            RoadArc(1, 2, 2.0),
            RoadArc(2, 3, 4.0),
            RoadArc(3, 4, 6.0),
        ]
    )
    snapshot = make_snapshot((10, 20, 30))
    candidate_route = CandidateRoute((10, 20, 30))

    assert arrival_times(instance, snapshot, candidate_route) == {
        10: 1.0,
        20: 3.0,
        30: 6.0,
    }


def test_arrival_times_use_shorter_indirect_path() -> None:
    instance = make_instance(
        [
            RoadArc(1, 2, 20.0),
            RoadArc(1, 5, 3.0),
            RoadArc(5, 2, 4.0),
        ]
    )
    snapshot = make_snapshot((10,))
    candidate_route = CandidateRoute((10,))

    assert arrival_times(instance, snapshot, candidate_route) == {10: 3.5}


def test_arrival_times_respect_network_direction() -> None:
    instance = make_instance([RoadArc(2, 1, 4.0)])
    snapshot = make_snapshot((10,))
    candidate_route = CandidateRoute((10,))

    with pytest.raises(ValueError, match="no directed path"):
        arrival_times(instance, snapshot, candidate_route)


def test_arrival_times_use_instance_reference_speed() -> None:
    instance = make_instance([RoadArc(1, 2, 10.0)], v_ref=5.0)
    snapshot = make_snapshot((10,))
    candidate_route = CandidateRoute((10,))

    assert arrival_times(instance, snapshot, candidate_route) == {10: 2.0}


def test_arrival_times_are_relative_to_snapshot_time() -> None:
    instance = make_instance([RoadArc(1, 2, 4.0)])
    snapshot = make_snapshot((10,), current_time=300.0)
    candidate_route = CandidateRoute((10,))

    assert arrival_times(instance, snapshot, candidate_route) == {10: 2.0}


def test_collection_service_duration_does_not_affect_arrival_times() -> None:
    instance = make_instance(
        [RoadArc(1, 2, 4.0)],
        collection_service_duration=1_000.0,
    )
    snapshot = make_snapshot((10,))
    candidate_route = CandidateRoute((10,))

    assert arrival_times(instance, snapshot, candidate_route) == {10: 2.0}


def test_empty_candidate_route_returns_empty_mapping() -> None:
    instance = make_instance([], collection_points=[])
    snapshot = make_snapshot(())
    candidate_route = CandidateRoute(())

    assert arrival_times(instance, snapshot, candidate_route) == {}


def test_candidate_route_order_changes_arrival_times() -> None:
    instance = make_instance(
        [
            RoadArc(1, 2, 2.0),
            RoadArc(2, 3, 4.0),
            RoadArc(1, 3, 10.0),
            RoadArc(3, 2, 1.0),
        ]
    )
    snapshot = make_snapshot((10, 20))

    first_order = arrival_times(instance, snapshot, CandidateRoute((10, 20)))
    second_order = arrival_times(instance, snapshot, CandidateRoute((20, 10)))

    assert first_order == {10: 1.0, 20: 3.0}
    assert second_order == {20: 3.0, 10: 3.5}


def test_arrival_times_propagate_candidate_route_validation_error() -> None:
    instance = make_instance([RoadArc(1, 2, 4.0)])
    snapshot = make_snapshot((10, 20))
    candidate_route = CandidateRoute((10,))

    with pytest.raises(ValueError, match="missing"):
        arrival_times(instance, snapshot, candidate_route)
