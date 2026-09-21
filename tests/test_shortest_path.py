import pytest

from src.algorithms.shortest_path import shortest_path
from src.models.road_network import RoadArc, RoadNetwork, RoadNode


def make_network(arcs: list[RoadArc]) -> RoadNetwork:
    return RoadNetwork(
        network_id=1,
        nodes=[
            RoadNode(node_id=node_id, x=float(node_id), y=0.0)
            for node_id in range(1, 6)
        ],
        arcs=arcs,
    )


def test_shortest_path_direct() -> None:
    network = make_network([RoadArc(1, 2, 5.0)])

    distance, path = shortest_path(network, 1, 2)

    assert distance == 5.0
    assert path == (1, 2)


def test_shortest_path_prefers_shorter_indirect_path_and_sums_distances() -> None:
    network = make_network(
        [
            RoadArc(1, 4, 10.0),
            RoadArc(1, 2, 3.0),
            RoadArc(2, 4, 4.0),
        ]
    )

    distance, path = shortest_path(network, 1, 4)

    assert distance == 7.0
    assert path == (1, 2, 4)


def test_shortest_path_respects_arc_direction() -> None:
    network = make_network([RoadArc(1, 2, 5.0), RoadArc(2, 3, 4.0)])

    with pytest.raises(ValueError, match="no directed path"):
        shortest_path(network, 3, 1)


def test_shortest_path_origin_equal_to_destination() -> None:
    network = make_network([])

    distance, path = shortest_path(network, 3, 3)

    assert distance == 0.0
    assert path == (3,)


def test_shortest_path_rejects_missing_origin() -> None:
    network = make_network([])

    with pytest.raises(ValueError, match="start node"):
        shortest_path(network, 99, 1)


def test_shortest_path_rejects_missing_destination() -> None:
    network = make_network([])

    with pytest.raises(ValueError, match="end node"):
        shortest_path(network, 1, 99)


@pytest.mark.parametrize("invalid_id", [-1, 1.5, True, "1"])
def test_shortest_path_rejects_invalid_origin_id(invalid_id: object) -> None:
    network = make_network([])

    with pytest.raises(ValueError, match="start_node_id"):
        shortest_path(network, invalid_id, 1)  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid_id", [-1, 1.5, True, "1"])
def test_shortest_path_rejects_invalid_destination_id(invalid_id: object) -> None:
    network = make_network([])

    with pytest.raises(ValueError, match="end_node_id"):
        shortest_path(network, 1, invalid_id)  # type: ignore[arg-type]


def test_shortest_path_rejects_unreachable_destination() -> None:
    network = make_network([RoadArc(1, 2, 5.0)])

    with pytest.raises(ValueError, match="no directed path"):
        shortest_path(network, 1, 5)


def test_shortest_path_uses_node_id_for_deterministic_tie_breaking() -> None:
    network = make_network(
        [
            RoadArc(1, 3, 1.0),
            RoadArc(3, 4, 1.0),
            RoadArc(1, 2, 1.0),
            RoadArc(2, 4, 1.0),
        ]
    )

    distance, path = shortest_path(network, 1, 4)

    assert distance == 2.0
    assert path == (1, 2, 4)


def test_shortest_path_is_independent_of_arc_input_order() -> None:
    arcs = [
        RoadArc(1, 3, 1.0),
        RoadArc(3, 4, 1.0),
        RoadArc(1, 2, 1.0),
        RoadArc(2, 4, 1.0),
    ]
    network = make_network(arcs)
    reversed_network = make_network(list(reversed(arcs)))

    assert shortest_path(network, 1, 4) == shortest_path(reversed_network, 1, 4)
