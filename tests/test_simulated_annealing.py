import random

import pytest

from src.algorithms.simulated_annealing import (
    create_algorithm_rng,
    swap_neighbor,
)
from src.models.candidate_route import CandidateRoute
from src.models.problem import CollectionPoint, Depot, ProblemInstance
from src.models.road_network import RoadNetwork, RoadNode
from src.simulation.snapshot import OptimizationSnapshot


def make_instance(point_count: int = 3) -> ProblemInstance:
    points = [
        CollectionPoint(point_id=10 * index, node_id=index)
        for index in range(2, point_count + 2)
    ]
    trajectory = (4.0,)
    volume = (1.0,)
    return ProblemInstance(
        instance_id=1,
        road_network=RoadNetwork(
            network_id=1,
            nodes=[
                RoadNode(node_id=node_id, x=float(node_id), y=0.0)
                for node_id in range(1, point_count + 2)
            ],
            arcs=[],
        ),
        depot=Depot(depot_id=1, node_id=1),
        collection_points=points,
        vehicle_capacity=10.0,
        collection_service_duration=60.0,
        v_ref=2.0,
        sample_period=30.0,
        temperature_trajectories={point.point_id: trajectory for point in points},
        volume_trajectories={point.point_id: volume for point in points},
    )


def make_snapshot(
    point_ids: tuple[int, ...],
) -> OptimizationSnapshot:
    return OptimizationSnapshot(
        current_time=0.0,
        current_node_id=1,
        current_load=0.0,
        remaining_point_ids=point_ids,
        temperatures={point_id: 4.0 for point_id in point_ids},
        volumes={point_id: 1.0 for point_id in point_ids},
    )


def test_create_algorithm_rng_accepts_non_negative_integer_seed() -> None:
    rng = create_algorithm_rng(123)

    assert isinstance(rng, random.Random)


def test_create_algorithm_rng_accepts_zero_seed() -> None:
    assert isinstance(create_algorithm_rng(0), random.Random)


@pytest.mark.parametrize("invalid_seed", [-1, True, 1.5, "1"])
def test_create_algorithm_rng_rejects_invalid_seed(invalid_seed: object) -> None:
    with pytest.raises(ValueError, match="algorithm_seed"):
        create_algorithm_rng(invalid_seed)  # type: ignore[arg-type]


def test_same_seed_produces_same_neighbor() -> None:
    instance = make_instance()
    snapshot = make_snapshot((20, 30, 40))
    candidate_route = CandidateRoute((20, 30, 40))

    neighbor_a = swap_neighbor(
        instance,
        snapshot,
        candidate_route,
        create_algorithm_rng(123),
    )
    neighbor_b = swap_neighbor(
        instance,
        snapshot,
        candidate_route,
        create_algorithm_rng(123),
    )

    assert neighbor_a == neighbor_b


def test_empty_route_produces_empty_neighbor() -> None:
    instance = make_instance(0)
    snapshot = make_snapshot(())

    neighbor = swap_neighbor(instance, snapshot, CandidateRoute(()), random.Random(1))

    assert neighbor == CandidateRoute(())


def test_single_point_route_remains_unchanged() -> None:
    instance = make_instance(1)
    snapshot = make_snapshot((20,))
    candidate_route = CandidateRoute((20,))

    neighbor = swap_neighbor(instance, snapshot, candidate_route, random.Random(1))

    assert neighbor == candidate_route


def test_two_point_route_swaps_both_positions() -> None:
    instance = make_instance(2)
    snapshot = make_snapshot((20, 30))

    neighbor = swap_neighbor(
        instance,
        snapshot,
        CandidateRoute((20, 30)),
        random.Random(1),
    )

    assert neighbor.point_ids == (30, 20)


def test_longer_route_changes_exactly_two_positions() -> None:
    instance = make_instance()
    snapshot = make_snapshot((20, 30, 40))
    original = (20, 30, 40)

    neighbor = swap_neighbor(
        instance,
        snapshot,
        CandidateRoute(original),
        random.Random(1),
    )

    changed_positions = sum(
        original_point_id != neighbor_point_id
        for original_point_id, neighbor_point_id in zip(original, neighbor.point_ids)
    )
    assert changed_positions == 2


def test_swap_preserves_all_point_ids_exactly_once() -> None:
    instance = make_instance()
    snapshot = make_snapshot((20, 30, 40))

    neighbor = swap_neighbor(
        instance,
        snapshot,
        CandidateRoute((20, 30, 40)),
        random.Random(1),
    )

    assert sorted(neighbor.point_ids) == [20, 30, 40]
    assert len(neighbor.point_ids) == len(set(neighbor.point_ids))


def test_original_route_remains_unchanged() -> None:
    instance = make_instance()
    snapshot = make_snapshot((20, 30, 40))
    candidate_route = CandidateRoute((20, 30, 40))

    swap_neighbor(instance, snapshot, candidate_route, random.Random(1))

    assert candidate_route.point_ids == (20, 30, 40)


def test_neighbor_is_a_validated_candidate_route() -> None:
    instance = make_instance()
    snapshot = make_snapshot((20, 30, 40))

    neighbor = swap_neighbor(
        instance,
        snapshot,
        CandidateRoute((20, 30, 40)),
        random.Random(1),
    )

    assert isinstance(neighbor, CandidateRoute)
    assert neighbor.validate(instance, snapshot) is None


def test_successive_calls_advance_the_same_rng_state() -> None:
    instance = make_instance()
    snapshot = make_snapshot((20, 30, 40))
    candidate_route = CandidateRoute((20, 30, 40))
    actual_rng = create_algorithm_rng(123)
    expected_rng = create_algorithm_rng(123)

    actual_first = swap_neighbor(instance, snapshot, candidate_route, actual_rng)
    actual_second = swap_neighbor(instance, snapshot, candidate_route, actual_rng)
    expected_first = swap_neighbor(instance, snapshot, candidate_route, expected_rng)
    expected_second = swap_neighbor(instance, snapshot, candidate_route, expected_rng)

    assert actual_first == expected_first
    assert actual_second == expected_second


def test_invalid_route_is_rejected_before_rng_is_used() -> None:
    instance = make_instance(2)
    snapshot = make_snapshot((20, 30))
    rng = create_algorithm_rng(123)
    state_before = rng.getstate()

    with pytest.raises(ValueError, match="missing"):
        swap_neighbor(instance, snapshot, CandidateRoute((20,)), rng)

    assert rng.getstate() == state_before
