import math
import random

import pytest

import src.experiments.instance_generation as generation
from src.algorithms.shortest_path import shortest_path
from src.experiments.instance_generation import (
    create_instance_rng,
    generate_grid_road_network,
)


def make_grid(
    rows: int = 2,
    columns: int = 3,
    *,
    spacing: float = 10.0,
    p_uni: float = 0.0,
    seed: int = 123,
):
    return generate_grid_road_network(
        network_id=1,
        rows=rows,
        columns=columns,
        spacing=spacing,
        p_uni=p_uni,
        rng=create_instance_rng(seed),
    )


def test_create_instance_rng_accepts_zero_and_positive_seed() -> None:
    assert isinstance(create_instance_rng(0), random.Random)
    assert isinstance(create_instance_rng(123), random.Random)


@pytest.mark.parametrize("invalid_seed", [-1, True, 1.5, "1"])
def test_create_instance_rng_rejects_invalid_seed(invalid_seed: object) -> None:
    with pytest.raises(ValueError, match="instance_seed"):
        create_instance_rng(invalid_seed)  # type: ignore[arg-type]


def test_grid_has_expected_node_count_and_row_major_ids() -> None:
    network = make_grid()

    assert len(network.nodes) == 6
    assert tuple(node.node_id for node in network.nodes) == (1, 2, 3, 4, 5, 6)


def test_grid_coordinates_follow_spacing() -> None:
    network = make_grid(spacing=2.5)

    coordinates = {(node.node_id): (node.x, node.y) for node in network.nodes}
    assert coordinates == {
        1: (0.0, 0.0),
        2: (2.5, 0.0),
        3: (5.0, 0.0),
        4: (0.0, 2.5),
        5: (2.5, 2.5),
        6: (5.0, 2.5),
    }


def test_grid_connects_only_orthogonal_neighbors() -> None:
    network = make_grid()

    for arc in network.arcs:
        first_row, first_column = divmod(arc.from_node_id - 1, 3)
        second_row, second_column = divmod(arc.to_node_id - 1, 3)
        assert abs(first_row - second_row) + abs(first_column - second_column) == 1


def test_grid_has_no_diagonal_or_self_loop_arcs() -> None:
    network = make_grid()

    for arc in network.arcs:
        first_row, first_column = divmod(arc.from_node_id - 1, 3)
        second_row, second_column = divmod(arc.to_node_id - 1, 3)
        assert arc.from_node_id != arc.to_node_id
        assert not (
            abs(first_row - second_row) == 1 and abs(first_column - second_column) == 1
        )


def test_all_grid_arcs_have_spacing_as_distance() -> None:
    network = make_grid(spacing=7.5)

    assert {arc.distance for arc in network.arcs} == {7.5}


def test_zero_unidirectional_probability_creates_both_directions() -> None:
    network = make_grid(rows=1, columns=3, p_uni=0.0)

    assert set((arc.from_node_id, arc.to_node_id) for arc in network.arcs) == {
        (1, 2),
        (2, 1),
        (2, 3),
        (3, 2),
    }


def test_partial_unidirectional_probability_has_unique_ordered_arcs() -> None:
    network = make_grid(p_uni=0.5)
    pairs = [(arc.from_node_id, arc.to_node_id) for arc in network.arcs]

    assert len(pairs) == len(set(pairs))


def test_generated_network_is_strongly_connected() -> None:
    network = make_grid(p_uni=0.5)
    node_ids = [node.node_id for node in network.nodes]

    for start_node_id in node_ids:
        for end_node_id in node_ids:
            distance, path = shortest_path(network, start_node_id, end_node_id)
            assert distance >= 0.0
            assert path[0] == start_node_id
            assert path[-1] == end_node_id


def test_same_seed_and_parameters_generate_equal_networks() -> None:
    first_network = make_grid(p_uni=0.5, seed=123)
    second_network = make_grid(p_uni=0.5, seed=123)

    assert first_network == second_network


def test_different_seeds_can_generate_different_orientations() -> None:
    networks = {
        make_grid(rows=2, columns=2, p_uni=1.0, seed=seed).arcs for seed in range(10)
    }

    assert len(networks) >= 2


@pytest.mark.parametrize("rows", [0, -1, 1.5, True, "2"])
def test_grid_rejects_invalid_rows(rows: object) -> None:
    with pytest.raises(ValueError, match="rows"):
        generate_grid_road_network(1, rows, 2, 1.0, 0.0, random.Random(1))  # type: ignore[arg-type]


@pytest.mark.parametrize("columns", [0, -1, 1.5, True, "2"])
def test_grid_rejects_invalid_columns(columns: object) -> None:
    with pytest.raises(ValueError, match="columns"):
        generate_grid_road_network(1, 2, columns, 1.0, 0.0, random.Random(1))  # type: ignore[arg-type]


def test_grid_rejects_single_node() -> None:
    with pytest.raises(ValueError, match="at least two nodes"):
        generate_grid_road_network(1, 1, 1, 1.0, 0.0, random.Random(1))


@pytest.mark.parametrize("spacing", [0.0, -1.0, math.nan, math.inf, True, "1"])
def test_grid_rejects_invalid_spacing(spacing: object) -> None:
    with pytest.raises(ValueError, match="spacing"):
        generate_grid_road_network(
            1,
            1,
            2,
            spacing,  # type: ignore[arg-type]
            0.0,
            random.Random(1),
        )


@pytest.mark.parametrize("p_uni", [-0.1, 1.1, math.nan, math.inf, True, "0.5"])
def test_grid_rejects_invalid_unidirectional_probability(p_uni: object) -> None:
    with pytest.raises(ValueError, match="p_uni"):
        generate_grid_road_network(
            1,
            1,
            2,
            1.0,
            p_uni,  # type: ignore[arg-type]
            random.Random(1),
        )


def test_grid_rejects_non_random_rng() -> None:
    with pytest.raises(ValueError, match="random.Random"):
        generate_grid_road_network(1, 1, 2, 1.0, 0.0, object())  # type: ignore[arg-type]


def test_grid_delegates_network_id_validation_to_road_network() -> None:
    with pytest.raises(ValueError, match="network_id"):
        generate_grid_road_network(-1, 1, 2, 1.0, 0.0, random.Random(1))


def test_grid_raises_after_attempt_limit_when_strong_connectivity_is_impossible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(generation, "GRID_NETWORK_MAX_ATTEMPTS", 2)

    with pytest.raises(ValueError, match="could not generate a strongly connected"):
        generate_grid_road_network(1, 1, 2, 1.0, 1.0, random.Random(1))
