import pytest

from src.models.candidate_route import CandidateRoute
from src.models.problem import CollectionPoint, Depot, ProblemInstance
from src.models.road_network import RoadArc, RoadNetwork, RoadNode
from src.models.route import Route, RouteStop
from src.objective.logistics import (
    candidate_route_distance,
    dynamic_logistic_mean_distance,
    dynamic_logistic_reference,
    logistic_distance_reference,
    normalized_dynamic_logistic_cost,
    normalized_logistic_cost,
    route_distance,
)
from src.simulation.snapshot import OptimizationSnapshot


def make_instance(
    arcs: list[RoadArc],
    collection_points: list[CollectionPoint] | None = None,
    coordinate_scale: float = 1000.0,
) -> ProblemInstance:
    points = collection_points
    if points is None:
        points = [
            CollectionPoint(point_id=10, node_id=2),
            CollectionPoint(point_id=20, node_id=3),
        ]

    return ProblemInstance(
        instance_id=1,
        road_network=RoadNetwork(
            network_id=1,
            nodes=[
                RoadNode(node_id=node_id, x=node_id * coordinate_scale, y=0.0)
                for node_id in range(1, 5)
            ],
            arcs=arcs,
        ),
        depot=Depot(depot_id=1, node_id=1),
        collection_points=points,
        vehicle_capacity=10.0,
        collection_service_duration=60.0,
        v_ref=10.0,
        sample_period=30.0,
        temperature_trajectories={point.point_id: (4.0,) for point in points},
        volume_trajectories={point.point_id: (1.0,) for point in points},
    )


def make_snapshot(
    remaining_point_ids: tuple[int, ...],
    *,
    current_node_id: int = 1,
) -> OptimizationSnapshot:
    return OptimizationSnapshot(
        current_time=0.0,
        current_node_id=current_node_id,
        current_load=0.0,
        remaining_point_ids=remaining_point_ids,
        temperatures={point_id: 4.0 for point_id in remaining_point_ids},
        volumes={point_id: 1.0 for point_id in remaining_point_ids},
    )


def test_route_distance_sums_direct_segments_and_return_to_depot() -> None:
    instance = make_instance(
        [
            RoadArc(1, 2, 4.0),
            RoadArc(2, 3, 5.0),
            RoadArc(3, 1, 6.0),
        ]
    )
    route = Route(
        stops=[
            RouteStop("depot", 1),
            RouteStop("collection", 10),
            RouteStop("collection", 20),
            RouteStop("depot", 1),
        ]
    )

    assert route_distance(instance, route) == 15.0


def test_route_distance_uses_shorter_indirect_segment() -> None:
    instance = make_instance(
        [
            RoadArc(1, 3, 10.0),
            RoadArc(1, 2, 3.0),
            RoadArc(2, 3, 4.0),
            RoadArc(3, 1, 2.0),
        ],
        collection_points=[CollectionPoint(point_id=20, node_id=3)],
    )
    route = Route(
        stops=[
            RouteStop("depot", 1),
            RouteStop("collection", 20),
            RouteStop("depot", 1),
        ]
    )

    assert route_distance(instance, route) == 9.0


def test_route_distance_respects_network_direction() -> None:
    instance = make_instance(
        [RoadArc(1, 2, 4.0)],
        collection_points=[CollectionPoint(point_id=10, node_id=2)],
    )
    route = Route(
        stops=[
            RouteStop("depot", 1),
            RouteStop("collection", 10),
            RouteStop("depot", 1),
        ]
    )

    with pytest.raises(ValueError, match="no directed path"):
        route_distance(instance, route)


def test_route_distance_for_depot_to_depot_without_collections() -> None:
    instance = make_instance([], collection_points=[])
    route = Route(stops=[RouteStop("depot", 1), RouteStop("depot", 1)])

    assert route_distance(instance, route) == 0.0


def test_route_distance_propagates_invalid_route_error() -> None:
    instance = make_instance(
        [RoadArc(1, 2, 4.0), RoadArc(2, 1, 4.0)],
    )
    route = Route(
        stops=[
            RouteStop("depot", 1),
            RouteStop("collection", 10),
            RouteStop("depot", 1),
        ]
    )

    with pytest.raises(ValueError, match="missing"):
        route_distance(instance, route)


def test_route_distance_uses_arc_distances_not_node_coordinates() -> None:
    instance = make_instance(
        [RoadArc(1, 2, 2.0), RoadArc(2, 1, 3.0)],
        collection_points=[CollectionPoint(point_id=10, node_id=2)],
    )
    route = Route(
        stops=[
            RouteStop("depot", 1),
            RouteStop("collection", 10),
            RouteStop("depot", 1),
        ]
    )

    assert route_distance(instance, route) == 5.0


def test_logistic_reference_averages_both_directed_distances() -> None:
    instance = make_instance(
        [RoadArc(1, 2, 2.0), RoadArc(2, 1, 8.0)],
        collection_points=[CollectionPoint(point_id=10, node_id=2)],
    )

    assert logistic_distance_reference(instance) == 10.0


def test_logistic_reference_uses_indirect_shortest_paths() -> None:
    instance = make_instance(
        [
            RoadArc(1, 2, 3.0),
            RoadArc(2, 3, 4.0),
            RoadArc(3, 1, 2.0),
            RoadArc(1, 3, 20.0),
        ]
    )

    assert logistic_distance_reference(instance) == 13.5


def test_logistic_reference_multiplies_mean_by_number_of_stops() -> None:
    instance = make_instance(
        [
            RoadArc(1, 2, 4.0),
            RoadArc(2, 3, 5.0),
            RoadArc(3, 1, 6.0),
        ]
    )

    assert logistic_distance_reference(instance) == 22.5


def test_normalized_logistic_cost_is_route_distance_over_reference() -> None:
    instance = make_instance(
        [
            RoadArc(1, 2, 3.0),
            RoadArc(2, 3, 4.0),
            RoadArc(3, 1, 2.0),
            RoadArc(1, 3, 20.0),
        ]
    )
    route = Route(
        stops=[
            RouteStop("depot", 1),
            RouteStop("collection", 20),
            RouteStop("collection", 10),
            RouteStop("depot", 1),
        ]
    )

    reference_distance = logistic_distance_reference(instance)
    cost = normalized_logistic_cost(instance, route, reference_distance)

    assert cost == 18.0 / 13.5
    assert cost > 1.0


def test_logistic_cost_for_instance_without_collections_is_zero() -> None:
    instance = make_instance([], collection_points=[])
    route = Route(stops=[RouteStop("depot", 1), RouteStop("depot", 1)])
    reference_distance = logistic_distance_reference(instance)

    assert reference_distance == 0.0
    assert normalized_logistic_cost(instance, route, reference_distance) == 0.0


def test_logistic_cost_rejects_zero_reference_with_collections() -> None:
    instance = make_instance(
        [RoadArc(1, 2, 2.0), RoadArc(2, 1, 3.0)],
        collection_points=[CollectionPoint(point_id=10, node_id=2)],
    )
    route = Route(
        stops=[
            RouteStop("depot", 1),
            RouteStop("collection", 10),
            RouteStop("depot", 1),
        ]
    )

    with pytest.raises(ValueError, match="reference_distance"):
        normalized_logistic_cost(instance, route, 0.0)


def test_logistic_reference_ignores_node_coordinates() -> None:
    arcs = [
        RoadArc(1, 2, 3.0),
        RoadArc(2, 3, 4.0),
        RoadArc(3, 1, 2.0),
        RoadArc(1, 3, 20.0),
    ]
    instance_with_small_coordinates = make_instance(arcs, coordinate_scale=1.0)
    instance_with_large_coordinates = make_instance(arcs, coordinate_scale=1_000_000.0)

    assert logistic_distance_reference(instance_with_small_coordinates) == (
        logistic_distance_reference(instance_with_large_coordinates)
    )


@pytest.mark.parametrize(
    "reference_distance", [-1.0, float("nan"), float("inf"), True, "1"]
)
def test_normalized_logistic_cost_rejects_invalid_reference_distance(
    reference_distance: object,
) -> None:
    instance = make_instance([RoadArc(1, 2, 2.0), RoadArc(2, 1, 3.0)])
    route = Route(
        stops=[
            RouteStop("depot", 1),
            RouteStop("collection", 10),
            RouteStop("depot", 1),
        ]
    )

    with pytest.raises(ValueError, match="reference_distance"):
        normalized_logistic_cost(  # type: ignore[arg-type]
            instance, route, reference_distance
        )


def test_candidate_route_distance_for_one_point_includes_return_to_depot() -> None:
    instance = make_instance(
        [RoadArc(1, 2, 4.0), RoadArc(2, 1, 6.0)],
    )
    snapshot = make_snapshot((10,))

    assert candidate_route_distance(instance, snapshot, CandidateRoute((10,))) == 10.0


def test_candidate_route_distance_for_two_points() -> None:
    instance = make_instance(
        [
            RoadArc(1, 2, 4.0),
            RoadArc(2, 3, 5.0),
            RoadArc(3, 1, 6.0),
        ]
    )
    snapshot = make_snapshot((10, 20))

    assert (
        candidate_route_distance(instance, snapshot, CandidateRoute((10, 20))) == 15.0
    )


def test_candidate_route_order_can_change_distance() -> None:
    instance = make_instance(
        [
            RoadArc(1, 2, 4.0),
            RoadArc(2, 3, 5.0),
            RoadArc(3, 1, 6.0),
            RoadArc(1, 3, 2.0),
            RoadArc(3, 2, 1.0),
            RoadArc(2, 1, 7.0),
        ]
    )
    snapshot = make_snapshot((10, 20))

    first_order_distance = candidate_route_distance(
        instance, snapshot, CandidateRoute((10, 20))
    )
    second_order_distance = candidate_route_distance(
        instance, snapshot, CandidateRoute((20, 10))
    )

    assert first_order_distance == 14.0
    assert second_order_distance == 10.0


def test_candidate_route_distance_uses_shorter_indirect_path() -> None:
    instance = make_instance(
        [
            RoadArc(1, 2, 10.0),
            RoadArc(1, 4, 3.0),
            RoadArc(4, 2, 4.0),
            RoadArc(2, 1, 2.0),
        ]
    )
    snapshot = make_snapshot((10,))

    assert candidate_route_distance(instance, snapshot, CandidateRoute((10,))) == 9.0


def test_candidate_route_distance_respects_network_direction() -> None:
    instance = make_instance(
        [
            RoadArc(1, 4, 2.0),
            RoadArc(4, 2, 3.0),
            RoadArc(2, 1, 5.0),
        ]
    )
    snapshot = make_snapshot((10,))

    assert candidate_route_distance(instance, snapshot, CandidateRoute((10,))) == 10.0


def test_candidate_route_distance_ignores_node_coordinates() -> None:
    arcs = [RoadArc(1, 2, 4.0), RoadArc(2, 1, 6.0)]
    small_coordinates = make_instance(arcs, coordinate_scale=1.0)
    large_coordinates = make_instance(arcs, coordinate_scale=1_000_000.0)
    small_snapshot = make_snapshot((10,))
    large_snapshot = make_snapshot((10,))

    assert candidate_route_distance(
        small_coordinates,
        small_snapshot,
        CandidateRoute((10,)),
    ) == candidate_route_distance(
        large_coordinates,
        large_snapshot,
        CandidateRoute((10,)),
    )


def test_candidate_route_distance_accepts_intermediate_current_node() -> None:
    instance = make_instance([RoadArc(4, 2, 3.0), RoadArc(2, 1, 4.0)])
    snapshot = make_snapshot((10,), current_node_id=4)

    assert candidate_route_distance(instance, snapshot, CandidateRoute((10,))) == 7.0


def test_empty_candidate_route_calculates_current_to_depot() -> None:
    instance = make_instance([RoadArc(4, 1, 9.0)], collection_points=[])
    snapshot = make_snapshot((), current_node_id=4)

    assert candidate_route_distance(instance, snapshot, CandidateRoute(())) == 9.0


def test_empty_candidate_route_at_depot_has_zero_distance() -> None:
    instance = make_instance([], collection_points=[])
    snapshot = make_snapshot((), current_node_id=1)

    assert candidate_route_distance(instance, snapshot, CandidateRoute(())) == 0.0


def test_candidate_route_distance_propagates_candidate_validation_error() -> None:
    instance = make_instance([RoadArc(1, 2, 4.0), RoadArc(2, 1, 4.0)])
    snapshot = make_snapshot((10, 20))

    with pytest.raises(ValueError, match="missing"):
        candidate_route_distance(instance, snapshot, CandidateRoute((10,)))


def test_candidate_route_distance_rejects_missing_required_path() -> None:
    instance = make_instance([RoadArc(1, 2, 4.0)])
    snapshot = make_snapshot((10,))

    with pytest.raises(ValueError, match="no directed path"):
        candidate_route_distance(instance, snapshot, CandidateRoute((10,)))


def test_dynamic_logistic_mean_distance_for_one_remaining_point() -> None:
    instance = make_instance([RoadArc(1, 2, 4.0), RoadArc(2, 1, 6.0)])
    snapshot = make_snapshot((10,))

    assert dynamic_logistic_mean_distance(instance, snapshot) == 5.0


def test_dynamic_logistic_mean_distance_changes_with_direction() -> None:
    first_instance = make_instance([RoadArc(1, 2, 4.0), RoadArc(2, 1, 6.0)])
    second_instance = make_instance([RoadArc(1, 2, 3.0), RoadArc(2, 1, 9.0)])
    first_snapshot = make_snapshot((10,))
    second_snapshot = make_snapshot((10,))

    first_mean = dynamic_logistic_mean_distance(first_instance, first_snapshot)
    second_mean = dynamic_logistic_mean_distance(second_instance, second_snapshot)

    assert first_mean == 5.0
    assert second_mean == 6.0


def test_dynamic_logistic_mean_distance_includes_remaining_to_remaining() -> None:
    instance = make_instance(
        [
            RoadArc(4, 2, 2.0),
            RoadArc(4, 3, 4.0),
            RoadArc(2, 3, 3.0),
            RoadArc(3, 2, 5.0),
            RoadArc(2, 1, 6.0),
            RoadArc(3, 1, 7.0),
        ]
    )
    snapshot = make_snapshot((10, 20), current_node_id=4)

    assert dynamic_logistic_mean_distance(instance, snapshot) == 4.5


def test_dynamic_logistic_mean_distance_does_not_complete_all_node_pairs() -> None:
    instance = make_instance(
        [
            RoadArc(4, 2, 2.0),
            RoadArc(4, 3, 4.0),
            RoadArc(2, 3, 3.0),
            RoadArc(3, 2, 5.0),
            RoadArc(2, 1, 6.0),
            RoadArc(3, 1, 7.0),
        ]
    )
    snapshot = make_snapshot((10, 20), current_node_id=4)

    assert dynamic_logistic_mean_distance(instance, snapshot) == 4.5


def test_dynamic_logistic_mean_distance_uses_indirect_shortest_path() -> None:
    instance = make_instance(
        [
            RoadArc(1, 2, 20.0),
            RoadArc(1, 4, 3.0),
            RoadArc(4, 2, 4.0),
            RoadArc(2, 1, 8.0),
        ]
    )
    snapshot = make_snapshot((10,))

    assert dynamic_logistic_mean_distance(instance, snapshot) == 7.5


def test_dynamic_logistic_mean_distance_ignores_coordinates() -> None:
    arcs = [RoadArc(1, 2, 4.0), RoadArc(2, 1, 6.0)]
    small_coordinates = make_instance(arcs, coordinate_scale=1.0)
    large_coordinates = make_instance(arcs, coordinate_scale=1_000_000.0)
    small_snapshot = make_snapshot((10,))
    large_snapshot = make_snapshot((10,))

    assert dynamic_logistic_mean_distance(small_coordinates, small_snapshot) == (
        dynamic_logistic_mean_distance(large_coordinates, large_snapshot)
    )


def test_dynamic_logistic_mean_distance_is_zero_without_remaining_points() -> None:
    instance = make_instance([], collection_points=[])
    snapshot = make_snapshot(())

    assert dynamic_logistic_mean_distance(instance, snapshot) == 0.0


def test_dynamic_logistic_reference_for_one_remaining_point() -> None:
    instance = make_instance([RoadArc(1, 2, 4.0), RoadArc(2, 1, 6.0)])
    snapshot = make_snapshot((10,))

    assert dynamic_logistic_reference(instance, snapshot) == 10.0


def test_dynamic_logistic_reference_uses_m_plus_one_for_multiple_points() -> None:
    instance = make_instance(
        [
            RoadArc(1, 2, 2.0),
            RoadArc(2, 1, 2.0),
            RoadArc(1, 3, 2.0),
            RoadArc(3, 1, 2.0),
            RoadArc(2, 3, 2.0),
            RoadArc(3, 2, 2.0),
        ]
    )
    snapshot = make_snapshot((10, 20))

    assert dynamic_logistic_reference(instance, snapshot) == 6.0


def test_dynamic_logistic_reference_is_zero_without_remaining_points() -> None:
    instance = make_instance([], collection_points=[])
    snapshot = make_snapshot(())

    assert dynamic_logistic_reference(instance, snapshot) == 0.0


def test_normalized_dynamic_logistic_cost_divides_by_reference() -> None:
    instance = make_instance([RoadArc(1, 2, 4.0), RoadArc(2, 1, 6.0)])
    snapshot = make_snapshot((10,))

    assert (
        normalized_dynamic_logistic_cost(
            instance,
            snapshot,
            CandidateRoute((10,)),
            5.0,
        )
        == 2.0
    )


def test_normalized_dynamic_logistic_cost_can_be_greater_than_one() -> None:
    instance = make_instance([RoadArc(1, 2, 4.0), RoadArc(2, 1, 6.0)])
    snapshot = make_snapshot((10,))

    result = normalized_dynamic_logistic_cost(
        instance,
        snapshot,
        CandidateRoute((10,)),
        4.0,
    )

    assert result == 2.5
    assert result > 1.0


def test_normalized_dynamic_logistic_cost_rejects_zero_reference_with_points() -> None:
    instance = make_instance([RoadArc(1, 2, 4.0), RoadArc(2, 1, 6.0)])
    snapshot = make_snapshot((10,))

    with pytest.raises(ValueError, match="reference_distance"):
        normalized_dynamic_logistic_cost(
            instance,
            snapshot,
            CandidateRoute((10,)),
            0.0,
        )


@pytest.mark.parametrize("reference_distance", [-1.0, float("nan"), float("inf"), True])
def test_normalized_dynamic_logistic_cost_rejects_invalid_reference(
    reference_distance: object,
) -> None:
    instance = make_instance([RoadArc(1, 2, 4.0), RoadArc(2, 1, 6.0)])
    snapshot = make_snapshot((10,))

    with pytest.raises(ValueError, match="reference_distance"):
        normalized_dynamic_logistic_cost(  # type: ignore[arg-type]
            instance,
            snapshot,
            CandidateRoute((10,)),
            reference_distance,
        )


def test_normalized_dynamic_logistic_cost_for_terminal_snapshot_is_zero() -> None:
    instance = make_instance([RoadArc(4, 1, 9.0)], collection_points=[])
    snapshot = make_snapshot((), current_node_id=4)

    assert (
        normalized_dynamic_logistic_cost(
            instance,
            snapshot,
            CandidateRoute(()),
            0.0,
        )
        == 0.0
    )


def test_normalized_dynamic_logistic_cost_propagates_candidate_validation_error() -> (
    None
):
    instance = make_instance([RoadArc(1, 2, 4.0), RoadArc(2, 1, 4.0)])
    snapshot = make_snapshot((10, 20))

    with pytest.raises(ValueError, match="missing"):
        normalized_dynamic_logistic_cost(
            instance,
            snapshot,
            CandidateRoute((10,)),
            10.0,
        )


def test_dynamic_logistic_mean_distance_propagates_missing_directed_path() -> None:
    instance = make_instance([RoadArc(2, 1, 4.0)])
    snapshot = make_snapshot((10,))

    with pytest.raises(ValueError, match="no directed path"):
        dynamic_logistic_mean_distance(instance, snapshot)
