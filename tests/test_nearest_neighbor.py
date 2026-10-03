import pytest

from src.algorithms.nearest_neighbor import (
    classical_nearest_neighbor,
    multicriteria_nearest_neighbor,
)
from src.models.candidate_route import CandidateRoute
from src.models.problem import CollectionPoint, Depot, ProblemInstance
from src.models.road_network import RoadArc, RoadNetwork, RoadNode
from src.models.route import RouteStop
from src.objective.weight_profiles import (
    BALANCED_WEIGHTS,
    LOGISTIC_WEIGHTS,
    THERMAL_WEIGHTS,
)
from src.simulation.snapshot import OptimizationSnapshot


def make_instance(
    arcs: list[RoadArc],
    collection_points: list[CollectionPoint] | None = None,
    *,
    coordinate_scale: float = 1.0,
) -> ProblemInstance:
    points = collection_points
    if points is None:
        points = [
            CollectionPoint(point_id=10, node_id=2),
            CollectionPoint(point_id=20, node_id=3),
            CollectionPoint(point_id=30, node_id=4),
        ]

    temperature_trajectory = (4.0,)
    volume_trajectory = (1.0,)
    return ProblemInstance(
        instance_id=1,
        road_network=RoadNetwork(
            network_id=1,
            nodes=[
                RoadNode(
                    node_id=node_id,
                    x=node_id * coordinate_scale,
                    y=0.0,
                )
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


def point_ids(route_stops: tuple[RouteStop, ...]) -> tuple[int, ...]:
    return tuple(stop.entity_id for stop in route_stops if stop.kind == "collection")


def make_snapshot(
    remaining_point_ids: tuple[int, ...],
    *,
    temperatures: dict[int, float] | None = None,
    volumes: dict[int, float] | None = None,
    current_node_id: int = 1,
) -> OptimizationSnapshot:
    return OptimizationSnapshot(
        current_time=0.0,
        current_node_id=current_node_id,
        current_load=0.0,
        remaining_point_ids=remaining_point_ids,
        temperatures=temperatures
        or {point_id: 4.0 for point_id in remaining_point_ids},
        volumes=volumes or {point_id: 1.0 for point_id in remaining_point_ids},
    )


def candidate_point_ids(candidate_route: CandidateRoute) -> tuple[int, ...]:
    return candidate_route.point_ids


def test_nearest_neighbor_returns_depot_to_depot_without_points() -> None:
    instance = make_instance([], collection_points=[])

    route = classical_nearest_neighbor(instance)

    assert route.stops == (RouteStop("depot", 1), RouteStop("depot", 1))


def test_nearest_neighbor_builds_route_for_one_point() -> None:
    instance = make_instance(
        [RoadArc(1, 2, 4.0), RoadArc(2, 1, 6.0)],
        collection_points=[CollectionPoint(point_id=10, node_id=2)],
    )

    route = classical_nearest_neighbor(instance)

    assert route.stops == (
        RouteStop("depot", 1),
        RouteStop("collection", 10),
        RouteStop("depot", 1),
    )


def test_nearest_neighbor_selects_nearest_and_updates_current_node() -> None:
    instance = make_instance(
        [
            RoadArc(1, 2, 2.0),
            RoadArc(1, 3, 5.0),
            RoadArc(1, 4, 7.0),
            RoadArc(2, 3, 1.0),
            RoadArc(2, 4, 6.0),
            RoadArc(3, 4, 1.0),
            RoadArc(4, 1, 8.0),
        ]
    )

    route = classical_nearest_neighbor(instance)

    assert point_ids(route.stops) == (10, 20, 30)


def test_nearest_neighbor_returns_to_depot_and_visits_each_point_once() -> None:
    instance = make_instance(
        [
            RoadArc(1, 2, 2.0),
            RoadArc(1, 3, 5.0),
            RoadArc(1, 4, 7.0),
            RoadArc(2, 3, 1.0),
            RoadArc(2, 4, 6.0),
            RoadArc(3, 4, 1.0),
            RoadArc(4, 1, 8.0),
        ]
    )

    route = classical_nearest_neighbor(instance)

    assert route.stops[0] == RouteStop("depot", instance.depot.depot_id)
    assert route.stops[-1] == RouteStop("depot", instance.depot.depot_id)
    assert sorted(point_ids(route.stops)) == [10, 20, 30]
    assert len(point_ids(route.stops)) == len(set(point_ids(route.stops)))


def test_nearest_neighbor_uses_shorter_indirect_path() -> None:
    instance = make_instance(
        [
            RoadArc(1, 2, 10.0),
            RoadArc(1, 5, 2.0),
            RoadArc(5, 2, 2.0),
            RoadArc(1, 3, 5.0),
            RoadArc(2, 3, 1.0),
            RoadArc(3, 1, 6.0),
        ],
        collection_points=[
            CollectionPoint(point_id=10, node_id=2),
            CollectionPoint(point_id=20, node_id=3),
        ],
    )

    route = classical_nearest_neighbor(instance)

    assert point_ids(route.stops)[0] == 10


def test_nearest_neighbor_respects_directed_distances() -> None:
    instance = make_instance(
        [
            RoadArc(1, 2, 10.0),
            RoadArc(2, 1, 1.0),
            RoadArc(1, 3, 3.0),
            RoadArc(3, 1, 1.0),
            RoadArc(3, 2, 4.0),
        ],
        collection_points=[
            CollectionPoint(point_id=10, node_id=2),
            CollectionPoint(point_id=20, node_id=3),
        ],
    )

    route = classical_nearest_neighbor(instance)

    assert point_ids(route.stops) == (20, 10)


def test_nearest_neighbor_ignores_node_coordinates() -> None:
    arcs = [
        RoadArc(1, 2, 2.0),
        RoadArc(1, 3, 5.0),
        RoadArc(2, 3, 1.0),
        RoadArc(3, 1, 6.0),
    ]
    points = [
        CollectionPoint(point_id=10, node_id=2),
        CollectionPoint(point_id=20, node_id=3),
    ]
    small_coordinates = make_instance(
        arcs,
        collection_points=points,
        coordinate_scale=1.0,
    )
    large_coordinates = make_instance(
        arcs,
        collection_points=points,
        coordinate_scale=1_000_000.0,
    )

    assert classical_nearest_neighbor(small_coordinates) == classical_nearest_neighbor(
        large_coordinates
    )


def test_nearest_neighbor_breaks_equal_distance_by_smallest_point_id() -> None:
    instance = make_instance(
        [
            RoadArc(1, 2, 4.0),
            RoadArc(1, 3, 4.0),
            RoadArc(2, 3, 1.0),
            RoadArc(3, 2, 1.0),
            RoadArc(2, 1, 6.0),
        ],
        collection_points=[
            CollectionPoint(point_id=20, node_id=2),
            CollectionPoint(point_id=10, node_id=3),
        ],
    )

    route = classical_nearest_neighbor(instance)

    assert point_ids(route.stops)[0] == 10


def test_nearest_neighbor_tie_breaking_ignores_collection_order() -> None:
    arcs = [
        RoadArc(1, 2, 4.0),
        RoadArc(1, 3, 4.0),
        RoadArc(2, 3, 1.0),
        RoadArc(3, 2, 1.0),
        RoadArc(2, 1, 6.0),
    ]
    first_instance = make_instance(
        arcs,
        collection_points=[
            CollectionPoint(point_id=20, node_id=2),
            CollectionPoint(point_id=10, node_id=3),
        ],
    )
    second_instance = make_instance(
        arcs,
        collection_points=[
            CollectionPoint(point_id=10, node_id=3),
            CollectionPoint(point_id=20, node_id=2),
        ],
    )

    first_route = classical_nearest_neighbor(first_instance)
    second_route = classical_nearest_neighbor(second_instance)

    assert first_route == second_route


def test_nearest_neighbor_propagates_unreachable_point_error() -> None:
    instance = make_instance(
        [RoadArc(1, 2, 4.0)],
        collection_points=[
            CollectionPoint(point_id=10, node_id=2),
            CollectionPoint(point_id=20, node_id=3),
        ],
    )

    with pytest.raises(ValueError, match="no directed path"):
        classical_nearest_neighbor(instance)


def test_nearest_neighbor_returns_route_validated_against_instance() -> None:
    instance = make_instance(
        [
            RoadArc(1, 2, 2.0),
            RoadArc(1, 3, 5.0),
            RoadArc(2, 3, 1.0),
            RoadArc(3, 1, 6.0),
        ],
        collection_points=[
            CollectionPoint(point_id=10, node_id=2),
            CollectionPoint(point_id=20, node_id=3),
        ],
    )

    route = classical_nearest_neighbor(instance)

    assert route.validate(instance) is None


def test_nearest_neighbor_propagates_missing_return_path() -> None:
    instance = make_instance(
        [RoadArc(1, 2, 4.0)],
        collection_points=[CollectionPoint(point_id=10, node_id=2)],
    )

    with pytest.raises(ValueError, match="no directed path"):
        classical_nearest_neighbor(instance)


def test_multicriteria_nearest_neighbor_returns_empty_route_without_points() -> None:
    instance = make_instance([], collection_points=[])
    snapshot = make_snapshot(())

    route = multicriteria_nearest_neighbor(
        instance,
        snapshot,
        2.0,
        6.0,
        LOGISTIC_WEIGHTS,
    )

    assert route == CandidateRoute(())


def test_multicriteria_nearest_neighbor_builds_route_for_one_point() -> None:
    instance = make_instance(
        [RoadArc(1, 2, 4.0)],
        collection_points=[CollectionPoint(point_id=10, node_id=2)],
    )
    snapshot = make_snapshot((10,))

    route = multicriteria_nearest_neighbor(
        instance,
        snapshot,
        2.0,
        6.0,
        LOGISTIC_WEIGHTS,
    )

    assert route == CandidateRoute((10,))


def test_multicriteria_logistic_weight_favors_nearest_point() -> None:
    instance = make_instance(
        [
            RoadArc(1, 2, 2.0),
            RoadArc(1, 3, 5.0),
            RoadArc(2, 3, 1.0),
            RoadArc(3, 2, 1.0),
        ],
        collection_points=[
            CollectionPoint(point_id=10, node_id=2),
            CollectionPoint(point_id=20, node_id=3),
        ],
    )
    snapshot = make_snapshot((10, 20))

    route = multicriteria_nearest_neighbor(
        instance,
        snapshot,
        2.0,
        6.0,
        (1.0, 0.0, 0.0),
    )

    assert candidate_point_ids(route) == (10, 20)


def test_multicriteria_thermal_risk_can_change_the_choice() -> None:
    instance = make_instance(
        [
            RoadArc(1, 2, 10.0),
            RoadArc(1, 3, 2.0),
            RoadArc(2, 3, 1.0),
            RoadArc(3, 2, 1.0),
        ],
        collection_points=[
            CollectionPoint(point_id=10, node_id=2),
            CollectionPoint(point_id=20, node_id=3),
        ],
    )
    snapshot = make_snapshot(
        (10, 20),
        temperatures={10: 2.0, 20: 5.0},
    )

    route = multicriteria_nearest_neighbor(
        instance,
        snapshot,
        2.0,
        6.0,
        (0.0, 1.0, 0.0),
    )

    assert candidate_point_ids(route)[0] == 20


def test_multicriteria_volume_can_change_the_choice() -> None:
    instance = make_instance(
        [
            RoadArc(1, 2, 10.0),
            RoadArc(1, 3, 2.0),
            RoadArc(2, 3, 1.0),
            RoadArc(3, 2, 1.0),
        ],
        collection_points=[
            CollectionPoint(point_id=10, node_id=2),
            CollectionPoint(point_id=20, node_id=3),
        ],
    )
    snapshot = make_snapshot(
        (10, 20),
        volumes={10: 1.0, 20: 5.0},
    )

    route = multicriteria_nearest_neighbor(
        instance,
        snapshot,
        2.0,
        6.0,
        (0.0, 0.0, 1.0),
    )

    assert candidate_point_ids(route)[0] == 20


@pytest.mark.parametrize(
    "weights",
    [LOGISTIC_WEIGHTS, BALANCED_WEIGHTS, THERMAL_WEIGHTS],
)
def test_multicriteria_official_profiles_are_accepted(
    weights: tuple[float, float, float],
) -> None:
    instance = make_instance(
        [
            RoadArc(1, 2, 2.0),
            RoadArc(1, 3, 4.0),
            RoadArc(2, 3, 1.0),
            RoadArc(3, 2, 1.0),
        ],
        collection_points=[
            CollectionPoint(point_id=10, node_id=2),
            CollectionPoint(point_id=20, node_id=3),
        ],
    )
    snapshot = make_snapshot((10, 20))

    route = multicriteria_nearest_neighbor(
        instance,
        snapshot,
        2.0,
        6.0,
        weights,
    )

    assert sorted(candidate_point_ids(route)) == [10, 20]


def test_multicriteria_critical_point_has_strict_precedence() -> None:
    instance = make_instance(
        [
            RoadArc(1, 2, 10.0),
            RoadArc(1, 3, 2.0),
            RoadArc(2, 3, 1.0),
            RoadArc(3, 2, 1.0),
        ],
        collection_points=[
            CollectionPoint(point_id=10, node_id=2),
            CollectionPoint(point_id=20, node_id=3),
        ],
    )
    snapshot = make_snapshot(
        (10, 20),
        temperatures={10: 6.0, 20: 2.0},
    )

    route = multicriteria_nearest_neighbor(
        instance,
        snapshot,
        2.0,
        6.0,
        LOGISTIC_WEIGHTS,
    )

    assert candidate_point_ids(route)[0] == 10


def test_multicriteria_critical_points_use_local_attractiveness_order() -> None:
    instance = make_instance(
        [
            RoadArc(1, 2, 10.0),
            RoadArc(1, 3, 2.0),
            RoadArc(3, 2, 1.0),
        ],
        collection_points=[
            CollectionPoint(point_id=10, node_id=2),
            CollectionPoint(point_id=20, node_id=3),
        ],
    )
    snapshot = make_snapshot(
        (10, 20),
        temperatures={10: 6.0, 20: 6.0},
    )

    route = multicriteria_nearest_neighbor(
        instance,
        snapshot,
        2.0,
        6.0,
        LOGISTIC_WEIGHTS,
    )

    assert candidate_point_ids(route)[0] == 20


def test_multicriteria_noncritical_points_become_eligible_after_criticals() -> None:
    instance = make_instance(
        [
            RoadArc(1, 2, 10.0),
            RoadArc(1, 3, 2.0),
            RoadArc(1, 4, 3.0),
            RoadArc(2, 3, 1.0),
            RoadArc(2, 4, 2.0),
            RoadArc(3, 4, 1.0),
        ],
        collection_points=[
            CollectionPoint(point_id=10, node_id=2),
            CollectionPoint(point_id=20, node_id=3),
            CollectionPoint(point_id=30, node_id=4),
        ],
    )
    snapshot = make_snapshot(
        (10, 20, 30),
        temperatures={10: 6.0, 20: 2.0, 30: 2.0},
    )

    route = multicriteria_nearest_neighbor(
        instance,
        snapshot,
        2.0,
        6.0,
        LOGISTIC_WEIGHTS,
    )

    assert candidate_point_ids(route) == (10, 20, 30)


def test_multicriteria_uses_snapshot_temperatures() -> None:
    instance = make_instance(
        [
            RoadArc(1, 2, 2.0),
            RoadArc(1, 3, 2.0),
            RoadArc(2, 3, 1.0),
            RoadArc(3, 2, 1.0),
        ],
        collection_points=[
            CollectionPoint(point_id=10, node_id=2),
            CollectionPoint(point_id=20, node_id=3),
        ],
    )
    snapshot = make_snapshot(
        (10, 20),
        temperatures={10: 2.0, 20: 5.0},
    )

    route = multicriteria_nearest_neighbor(
        instance,
        snapshot,
        2.0,
        6.0,
        (0.0, 1.0, 0.0),
    )

    assert candidate_point_ids(route)[0] == 20


def test_multicriteria_uses_snapshot_volumes() -> None:
    instance = make_instance(
        [
            RoadArc(1, 2, 2.0),
            RoadArc(1, 3, 2.0),
            RoadArc(2, 3, 1.0),
            RoadArc(3, 2, 1.0),
        ],
        collection_points=[
            CollectionPoint(point_id=10, node_id=2),
            CollectionPoint(point_id=20, node_id=3),
        ],
    )
    snapshot = make_snapshot(
        (10, 20),
        volumes={10: 1.0, 20: 5.0},
    )

    route = multicriteria_nearest_neighbor(
        instance,
        snapshot,
        2.0,
        6.0,
        (0.0, 0.0, 1.0),
    )

    assert candidate_point_ids(route)[0] == 20


def test_multicriteria_uses_shorter_indirect_path() -> None:
    instance = make_instance(
        [
            RoadArc(1, 2, 10.0),
            RoadArc(1, 5, 2.0),
            RoadArc(5, 2, 2.0),
            RoadArc(1, 3, 5.0),
            RoadArc(2, 3, 1.0),
        ],
        collection_points=[
            CollectionPoint(point_id=10, node_id=2),
            CollectionPoint(point_id=20, node_id=3),
        ],
    )
    snapshot = make_snapshot((10, 20))

    route = multicriteria_nearest_neighbor(
        instance,
        snapshot,
        2.0,
        6.0,
        (1.0, 0.0, 0.0),
    )

    assert candidate_point_ids(route)[0] == 10


def test_multicriteria_respects_network_direction() -> None:
    instance = make_instance(
        [
            RoadArc(1, 2, 10.0),
            RoadArc(2, 1, 1.0),
            RoadArc(1, 3, 3.0),
            RoadArc(3, 2, 4.0),
        ],
        collection_points=[
            CollectionPoint(point_id=10, node_id=2),
            CollectionPoint(point_id=20, node_id=3),
        ],
    )
    snapshot = make_snapshot((10, 20))

    route = multicriteria_nearest_neighbor(
        instance,
        snapshot,
        2.0,
        6.0,
        (1.0, 0.0, 0.0),
    )

    assert candidate_point_ids(route)[0] == 20


def test_multicriteria_ignores_node_coordinates() -> None:
    arcs = [
        RoadArc(1, 2, 2.0),
        RoadArc(1, 3, 5.0),
        RoadArc(2, 3, 1.0),
    ]
    points = [
        CollectionPoint(point_id=10, node_id=2),
        CollectionPoint(point_id=20, node_id=3),
    ]
    small_coordinates = make_instance(
        arcs,
        collection_points=points,
        coordinate_scale=1.0,
    )
    large_coordinates = make_instance(
        arcs,
        collection_points=points,
        coordinate_scale=1_000_000.0,
    )
    small_snapshot = make_snapshot((10, 20))
    large_snapshot = make_snapshot((10, 20))

    assert multicriteria_nearest_neighbor(
        small_coordinates,
        small_snapshot,
        2.0,
        6.0,
        LOGISTIC_WEIGHTS,
    ) == multicriteria_nearest_neighbor(
        large_coordinates,
        large_snapshot,
        2.0,
        6.0,
        LOGISTIC_WEIGHTS,
    )


def test_multicriteria_tie_uses_smallest_point_id() -> None:
    instance = make_instance(
        [
            RoadArc(1, 2, 4.0),
            RoadArc(1, 3, 4.0),
            RoadArc(3, 2, 1.0),
        ],
        collection_points=[
            CollectionPoint(point_id=20, node_id=2),
            CollectionPoint(point_id=10, node_id=3),
        ],
    )
    snapshot = make_snapshot((20, 10))

    route = multicriteria_nearest_neighbor(
        instance,
        snapshot,
        2.0,
        6.0,
        (1.0, 0.0, 0.0),
    )

    assert candidate_point_ids(route)[0] == 10


def test_multicriteria_tie_is_independent_of_input_order() -> None:
    arcs = [
        RoadArc(1, 2, 4.0),
        RoadArc(1, 3, 4.0),
        RoadArc(3, 2, 1.0),
    ]
    first_instance = make_instance(
        arcs,
        collection_points=[
            CollectionPoint(point_id=20, node_id=2),
            CollectionPoint(point_id=10, node_id=3),
        ],
    )
    second_instance = make_instance(
        arcs,
        collection_points=[
            CollectionPoint(point_id=10, node_id=3),
            CollectionPoint(point_id=20, node_id=2),
        ],
    )
    first_snapshot = make_snapshot((20, 10))
    second_snapshot = make_snapshot((10, 20))

    first_route = multicriteria_nearest_neighbor(
        first_instance,
        first_snapshot,
        2.0,
        6.0,
        (1.0, 0.0, 0.0),
    )
    second_route = multicriteria_nearest_neighbor(
        second_instance,
        second_snapshot,
        2.0,
        6.0,
        (1.0, 0.0, 0.0),
    )

    assert first_route == second_route


def test_multicriteria_route_contains_each_remaining_point_once() -> None:
    instance = make_instance(
        [
            RoadArc(1, 2, 2.0),
            RoadArc(1, 3, 5.0),
            RoadArc(1, 4, 7.0),
            RoadArc(2, 3, 1.0),
            RoadArc(2, 4, 2.0),
            RoadArc(3, 4, 1.0),
        ]
    )
    snapshot = make_snapshot((10, 20, 30))

    route = multicriteria_nearest_neighbor(
        instance,
        snapshot,
        2.0,
        6.0,
        BALANCED_WEIGHTS,
    )

    assert sorted(candidate_point_ids(route)) == [10, 20, 30]
    assert len(candidate_point_ids(route)) == len(set(candidate_point_ids(route)))
    assert route.validate(instance, snapshot) is None


@pytest.mark.parametrize(
    "weights",
    [
        (1.0, 0.0),
        (1.0, 0.0, 0.0, 0.0),
        (True, 0.0, 0.0),
        (float("nan"), 0.0, 0.0),
        (float("inf"), 0.0, 0.0),
        (-0.1, 0.5, 0.6),
        ("1", 0.0, 0.0),
        (0.4, 0.3, 0.2),
    ],
)
def test_multicriteria_rejects_invalid_weights(weights: object) -> None:
    instance = make_instance([RoadArc(1, 2, 4.0)])
    snapshot = make_snapshot((10,))

    with pytest.raises(ValueError, match="weight|weights"):
        multicriteria_nearest_neighbor(  # type: ignore[arg-type]
            instance,
            snapshot,
            2.0,
            6.0,
            weights,
        )


def test_multicriteria_propagates_invalid_thermal_limits() -> None:
    instance = make_instance([RoadArc(1, 2, 4.0)])
    snapshot = make_snapshot((10,))

    with pytest.raises(ValueError, match="t_min must be less than t_max"):
        multicriteria_nearest_neighbor(
            instance,
            snapshot,
            6.0,
            2.0,
            LOGISTIC_WEIGHTS,
        )


def test_multicriteria_propagates_missing_directed_path() -> None:
    instance = make_instance(
        [RoadArc(1, 2, 4.0)],
        collection_points=[
            CollectionPoint(point_id=10, node_id=2),
            CollectionPoint(point_id=20, node_id=3),
        ],
    )
    snapshot = make_snapshot((10, 20))

    with pytest.raises(ValueError, match="no directed path"):
        multicriteria_nearest_neighbor(
            instance,
            snapshot,
            2.0,
            6.0,
            LOGISTIC_WEIGHTS,
        )
