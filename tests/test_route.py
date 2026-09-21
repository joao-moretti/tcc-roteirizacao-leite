import pytest

from src.models.problem import CollectionPoint, Depot, ProblemInstance
from src.models.road_network import RoadNetwork, RoadNode
from src.models.route import Route, RouteStop


def make_network() -> RoadNetwork:
    return RoadNetwork(
        network_id=1,
        nodes=[
            RoadNode(node_id=1, x=0.0, y=0.0),
            RoadNode(node_id=2, x=1.0, y=0.0),
            RoadNode(node_id=3, x=2.0, y=0.0),
        ],
        arcs=[],
    )


def make_instance() -> ProblemInstance:
    return ProblemInstance(
        instance_id=1,
        road_network=make_network(),
        depot=Depot(depot_id=1, node_id=1),
        collection_points=[
            CollectionPoint(point_id=10, node_id=2),
            CollectionPoint(point_id=20, node_id=3),
        ],
        vehicle_capacity=10.0,
        collection_service_duration=60.0,
        v_ref=10.0,
        sample_period=30.0,
        temperature_trajectories={10: (4.0,), 20: (4.5,)},
        volume_trajectories={10: (2.0,), 20: (1.0,)},
    )


def make_empty_instance() -> ProblemInstance:
    return ProblemInstance(
        instance_id=1,
        road_network=make_network(),
        depot=Depot(depot_id=1, node_id=1),
        collection_points=[],
        vehicle_capacity=10.0,
        collection_service_duration=60.0,
        v_ref=10.0,
        sample_period=30.0,
        temperature_trajectories={},
        volume_trajectories={},
    )


def test_route_stop_valid_for_depot() -> None:
    assert RouteStop(kind="depot", entity_id=1) == RouteStop("depot", 1)


def test_route_stop_valid_for_collection() -> None:
    assert RouteStop(kind="collection", entity_id=10) == RouteStop("collection", 10)


def test_route_stop_rejects_invalid_kind() -> None:
    with pytest.raises(ValueError, match="kind"):
        RouteStop(kind="warehouse", entity_id=1)  # type: ignore[arg-type]


@pytest.mark.parametrize("entity_id", [-1, 1.5, True, "1"])
def test_route_stop_rejects_invalid_entity_id(entity_id: object) -> None:
    with pytest.raises(ValueError, match="entity_id"):
        RouteStop(kind="collection", entity_id=entity_id)  # type: ignore[arg-type]


def test_route_normalizes_list_to_tuple() -> None:
    route = Route(
        stops=[RouteStop("depot", 1), RouteStop("depot", 1)],
    )

    assert isinstance(route.stops, tuple)


def test_route_is_valid() -> None:
    instance = make_instance()
    route = Route(
        stops=[
            RouteStop("depot", 1),
            RouteStop("collection", 10),
            RouteStop("collection", 20),
            RouteStop("depot", 1),
        ]
    )

    assert route.validate(instance) is None


def test_route_must_start_with_depot() -> None:
    route = Route(
        stops=[
            RouteStop("collection", 10),
            RouteStop("collection", 20),
            RouteStop("depot", 1),
        ]
    )

    with pytest.raises(ValueError, match="start"):
        route.validate(make_instance())


def test_route_must_end_with_depot() -> None:
    route = Route(
        stops=[
            RouteStop("depot", 1),
            RouteStop("collection", 10),
            RouteStop("collection", 20),
        ]
    )

    with pytest.raises(ValueError, match="end"):
        route.validate(make_instance())


def test_route_must_use_the_instance_depot() -> None:
    route = Route(
        stops=[
            RouteStop("depot", 999),
            RouteStop("collection", 10),
            RouteStop("collection", 20),
            RouteStop("depot", 999),
        ]
    )

    with pytest.raises(ValueError, match="instance depot"):
        route.validate(make_instance())


def test_route_rejects_wrong_final_depot_after_visiting_all_points() -> None:
    route = Route(
        stops=[
            RouteStop("depot", 1),
            RouteStop("collection", 10),
            RouteStop("collection", 20),
            RouteStop("depot", 999),
        ]
    )

    with pytest.raises(ValueError, match="instance depot"):
        route.validate(make_instance())


def test_route_rejects_depot_in_the_middle() -> None:
    route = Route(
        stops=[
            RouteStop("depot", 1),
            RouteStop("depot", 1),
            RouteStop("collection", 10),
            RouteStop("collection", 20),
            RouteStop("depot", 1),
        ]
    )

    with pytest.raises(ValueError, match="between"):
        route.validate(make_instance())


def test_route_rejects_unknown_collection_point() -> None:
    route = Route(
        stops=[
            RouteStop("depot", 1),
            RouteStop("collection", 999),
            RouteStop("depot", 1),
        ]
    )

    with pytest.raises(ValueError, match="unknown collection point"):
        route.validate(make_instance())


def test_route_rejects_repeated_collection_point() -> None:
    route = Route(
        stops=[
            RouteStop("depot", 1),
            RouteStop("collection", 10),
            RouteStop("collection", 10),
            RouteStop("collection", 20),
            RouteStop("depot", 1),
        ]
    )

    with pytest.raises(ValueError, match="more than once"):
        route.validate(make_instance())


def test_route_rejects_missing_collection_point() -> None:
    route = Route(
        stops=[
            RouteStop("depot", 1),
            RouteStop("collection", 10),
            RouteStop("depot", 1),
        ]
    )

    with pytest.raises(ValueError, match="missing"):
        route.validate(make_instance())


def test_depot_to_depot_is_valid_without_collection_points() -> None:
    instance = make_empty_instance()
    route = Route(stops=[RouteStop("depot", 1), RouteStop("depot", 1)])

    assert route.validate(instance) is None
