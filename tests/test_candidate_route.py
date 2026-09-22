import pytest

from src.models.candidate_route import CandidateRoute
from src.models.problem import CollectionPoint, Depot, ProblemInstance
from src.models.road_network import RoadNetwork, RoadNode
from src.simulation.snapshot import OptimizationSnapshot


def make_instance() -> ProblemInstance:
    points = [
        CollectionPoint(point_id=10, node_id=2),
        CollectionPoint(point_id=20, node_id=3),
    ]
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
        temperature_trajectories={10: (4.0, 4.5), 20: (5.0, 5.5)},
        volume_trajectories={10: (1.0, 1.0), 20: (1.0, 1.0)},
    )


def make_snapshot(
    remaining_point_ids: tuple[int, ...] = (10, 20),
    *,
    current_node_id: int = 4,
) -> OptimizationSnapshot:
    return OptimizationSnapshot(
        current_time=15.0,
        current_node_id=current_node_id,
        current_load=2.0,
        remaining_point_ids=remaining_point_ids,
        temperatures={point_id: 4.0 for point_id in remaining_point_ids},
        volumes={point_id: 1.0 for point_id in remaining_point_ids},
    )


def test_candidate_route_is_valid_and_normalizes_list_to_tuple() -> None:
    candidate = CandidateRoute(point_ids=[20, 10])

    assert candidate.point_ids == (20, 10)


@pytest.mark.parametrize("point_id", [-1, True, 1.5, "10"])
def test_candidate_route_rejects_invalid_point_id(point_id: object) -> None:
    with pytest.raises(ValueError, match="point_id"):
        CandidateRoute(point_ids=[point_id])  # type: ignore[list-item]


def test_candidate_route_rejects_duplicate_point_ids() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        CandidateRoute(point_ids=[10, 20, 10])


def test_validate_accepts_a_correct_permutation_in_different_order() -> None:
    instance = make_instance()
    snapshot = make_snapshot()
    candidate = CandidateRoute(point_ids=(20, 10))

    assert candidate.validate(instance, snapshot) is None


def test_validate_rejects_missing_point() -> None:
    instance = make_instance()
    snapshot = make_snapshot()
    candidate = CandidateRoute(point_ids=(10,))

    with pytest.raises(ValueError, match="missing"):
        candidate.validate(instance, snapshot)


def test_validate_rejects_extra_point() -> None:
    instance = make_instance()
    snapshot = make_snapshot()
    candidate = CandidateRoute(point_ids=(10, 20, 30))

    with pytest.raises(ValueError, match="extra"):
        candidate.validate(instance, snapshot)


def test_validate_rejects_missing_and_extra_points_together() -> None:
    instance = make_instance()
    snapshot = make_snapshot()
    candidate = CandidateRoute(point_ids=(10, 30))

    with pytest.raises(ValueError, match="missing.*extra|extra.*missing"):
        candidate.validate(instance, snapshot)


def test_empty_candidate_is_valid_when_no_points_remain() -> None:
    instance = make_instance()
    snapshot = make_snapshot(remaining_point_ids=())
    candidate = CandidateRoute(point_ids=())

    assert candidate.validate(instance, snapshot) is None


def test_empty_candidate_is_invalid_when_points_remain() -> None:
    instance = make_instance()
    snapshot = make_snapshot()
    candidate = CandidateRoute(point_ids=())

    with pytest.raises(ValueError, match="missing"):
        candidate.validate(instance, snapshot)


def test_validate_propagates_snapshot_validation_error() -> None:
    instance = make_instance()
    snapshot = make_snapshot(current_node_id=99)
    candidate = CandidateRoute(point_ids=(20, 10))

    with pytest.raises(ValueError, match="current_node_id"):
        candidate.validate(instance, snapshot)
