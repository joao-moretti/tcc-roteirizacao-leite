import pytest

from src.models.problem import CollectionPoint, Depot, ProblemInstance
from src.models.road_network import RoadArc, RoadNetwork, RoadNode
from src.models.route import Route, RouteStop
from src.objective.logistics import route_distance


def make_instance(
    arcs: list[RoadArc],
    collection_points: list[CollectionPoint] | None = None,
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
                RoadNode(node_id=node_id, x=node_id * 1000.0, y=0.0)
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
