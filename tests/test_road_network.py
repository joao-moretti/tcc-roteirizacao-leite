from dataclasses import FrozenInstanceError

import pytest

from src.models.road_network import RoadArc, RoadNetwork, RoadNode


def test_road_node_valid_and_immutable() -> None:
    node = RoadNode(node_id=1, x=10.0, y=20.0)

    assert node == RoadNode(node_id=1, x=10.0, y=20.0)
    with pytest.raises(FrozenInstanceError):
        node.node_id = 2  # type: ignore[misc]


@pytest.mark.parametrize("node_id", [-1, 1.5, True, "1"])
def test_road_node_rejects_invalid_id(node_id: object) -> None:
    with pytest.raises(ValueError, match="node_id"):
        RoadNode(node_id=node_id, x=0.0, y=0.0)  # type: ignore[arg-type]


@pytest.mark.parametrize("coordinate", [float("nan"), float("inf"), float("-inf")])
def test_road_node_rejects_non_finite_coordinate(coordinate: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        RoadNode(node_id=1, x=coordinate, y=0.0)


def test_road_arc_valid_and_does_not_store_travel_time() -> None:
    arc = RoadArc(from_node_id=1, to_node_id=2, distance=12.5)

    assert arc.distance == 12.5
    assert not hasattr(arc, "travel_time")


def test_road_arc_rejects_self_loop() -> None:
    with pytest.raises(ValueError, match="different nodes"):
        RoadArc(from_node_id=1, to_node_id=1, distance=1.0)


@pytest.mark.parametrize("field_name", ["from_node_id", "to_node_id"])
@pytest.mark.parametrize("invalid_id", [-1, 1.5, True, "1"])
def test_road_arc_rejects_invalid_identifier(
    field_name: str, invalid_id: object
) -> None:
    with pytest.raises(ValueError, match=field_name):
        if field_name == "from_node_id":
            RoadArc(invalid_id, 2, 1.0)  # type: ignore[arg-type]
        else:
            RoadArc(1, invalid_id, 1.0)  # type: ignore[arg-type]


@pytest.mark.parametrize("distance", [0, -1, float("nan"), float("inf")])
def test_road_arc_rejects_invalid_distance(distance: float) -> None:
    with pytest.raises(ValueError, match="distance"):
        RoadArc(from_node_id=1, to_node_id=2, distance=distance)


def test_road_network_validates_and_normalizes_collections() -> None:
    network = RoadNetwork(
        network_id=7,
        nodes=[RoadNode(node_id=1, x=0.0, y=0.0), RoadNode(2, 1.0, 0.0)],
        arcs=[RoadArc(1, 2, 10.0)],
    )

    assert network.network_id == 7
    assert isinstance(network.nodes, tuple)
    assert isinstance(network.arcs, tuple)


def test_road_network_rejects_invalid_network_id() -> None:
    with pytest.raises(ValueError, match="network_id"):
        RoadNetwork(network_id=-1, nodes=(), arcs=())


def test_road_network_rejects_duplicate_node_id() -> None:
    nodes = [RoadNode(1, 0.0, 0.0), RoadNode(1, 1.0, 1.0)]

    with pytest.raises(ValueError, match="duplicate node_id"):
        RoadNetwork(network_id=1, nodes=nodes, arcs=[])


def test_road_network_rejects_arc_with_unknown_node() -> None:
    nodes = [RoadNode(1, 0.0, 0.0)]

    with pytest.raises(ValueError, match="unknown node"):
        RoadNetwork(network_id=1, nodes=nodes, arcs=[RoadArc(1, 2, 10.0)])


def test_road_network_rejects_duplicate_ordered_arc_pair() -> None:
    nodes = [RoadNode(1, 0.0, 0.0), RoadNode(2, 1.0, 0.0)]
    arcs = [RoadArc(1, 2, 10.0), RoadArc(1, 2, 20.0)]

    with pytest.raises(ValueError, match="duplicate directed arc"):
        RoadNetwork(network_id=1, nodes=nodes, arcs=arcs)


def test_road_network_accepts_both_arc_directions() -> None:
    nodes = [RoadNode(1, 0.0, 0.0), RoadNode(2, 1.0, 0.0)]
    arcs = [RoadArc(1, 2, 10.0), RoadArc(2, 1, 12.0)]

    network = RoadNetwork(network_id=1, nodes=nodes, arcs=arcs)

    assert network.arcs == tuple(arcs)
