import math
import random

import pytest

import src.experiments.instance_generation as generation
from src.algorithms.shortest_path import shortest_path
from src.experiments.instance_generation import (
    create_instance_rng,
    generate_grid_road_network,
    generate_synthetic_problem_instance,
    generate_temperature_trajectories,
    generate_volume_trajectories,
    required_vehicle_capacity,
    select_depot_and_collection_points,
)
from src.models.problem import CollectionPoint, ProblemInstance


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


def test_selects_central_depot_on_odd_grid() -> None:
    network = make_grid(rows=3, columns=3)

    depot, _ = select_depot_and_collection_points(
        network,
        point_count=1,
        min_separation=0.0,
        rng=random.Random(1),
    )

    assert depot.depot_id == 0
    assert depot.node_id == 5


def test_central_depot_tie_uses_smallest_node_id() -> None:
    network = make_grid(rows=2, columns=2)

    depot, _ = select_depot_and_collection_points(
        network,
        point_count=1,
        min_separation=0.0,
        rng=random.Random(1),
    )

    assert depot.node_id == 1


def test_depot_selection_is_independent_of_seed() -> None:
    network = make_grid(rows=3, columns=3)

    first_depot, _ = select_depot_and_collection_points(
        network, 1, 0.0, random.Random(1)
    )
    second_depot, _ = select_depot_and_collection_points(
        network, 1, 0.0, random.Random(999)
    )

    assert first_depot == second_depot


@pytest.mark.parametrize("point_count", [0, -1, True, 1.5, "1"])
def test_point_selection_rejects_invalid_point_count(point_count: object) -> None:
    with pytest.raises(ValueError, match="point_count"):
        select_depot_and_collection_points(
            make_grid(),
            point_count,
            0.0,
            random.Random(1),  # type: ignore[arg-type]
        )


def test_point_selection_rejects_too_few_network_nodes() -> None:
    with pytest.raises(ValueError, match=r"point_count \+ 1"):
        select_depot_and_collection_points(
            make_grid(rows=1, columns=2), 2, 0.0, random.Random(1)
        )


@pytest.mark.parametrize(
    "min_separation",
    [-1.0, math.nan, math.inf, True, "1"],
)
def test_point_selection_rejects_invalid_minimum_separation(
    min_separation: object,
) -> None:
    with pytest.raises(ValueError, match="min_separation"):
        select_depot_and_collection_points(
            make_grid(),
            1,
            min_separation,  # type: ignore[arg-type]
            random.Random(1),
        )


def test_point_selection_accepts_zero_minimum_separation() -> None:
    depot, points = select_depot_and_collection_points(
        make_grid(rows=1, columns=3), 2, 0, random.Random(1)
    )

    assert depot.node_id not in {point.node_id for point in points}
    assert len(points) == 2


def test_point_selection_rejects_non_random_rng() -> None:
    with pytest.raises(ValueError, match="random.Random"):
        select_depot_and_collection_points(
            make_grid(),
            1,
            0.0,
            object(),  # type: ignore[arg-type]
        )


def test_point_selection_excludes_depot_and_assigns_sorted_sequential_ids() -> None:
    network = make_grid(rows=3, columns=3)

    depot, points = select_depot_and_collection_points(
        network, 3, 0.0, random.Random(7)
    )

    assert all(point.node_id != depot.node_id for point in points)
    assert len({point.node_id for point in points}) == 3
    assert tuple(point.point_id for point in points) == (1, 2, 3)
    assert tuple(point.node_id for point in points) == tuple(
        sorted(point.node_id for point in points)
    )


def test_point_ids_are_sequential_and_independent_of_node_ids() -> None:
    network = make_grid(rows=1, columns=4)

    _, points = select_depot_and_collection_points(network, 2, 0.0, random.Random(4))

    assert tuple(point.point_id for point in points) == (1, 2)
    assert all(point.point_id <= 2 for point in points)


def test_selected_points_respect_depot_and_pairwise_separation() -> None:
    network = make_grid(rows=4, columns=4, spacing=10.0)
    depot, points = select_depot_and_collection_points(
        network, 4, 10.0, random.Random(4)
    )
    nodes_by_id = {node.node_id: node for node in network.nodes}
    selected_nodes = [nodes_by_id[point.node_id] for point in points]
    depot_node = nodes_by_id[depot.node_id]

    for point_node in selected_nodes:
        assert (
            math.dist((point_node.x, point_node.y), (depot_node.x, depot_node.y))
            >= 10.0
        )
    for index, first_node in enumerate(selected_nodes):
        for second_node in selected_nodes[index + 1 :]:
            assert (
                math.dist((first_node.x, first_node.y), (second_node.x, second_node.y))
                >= 10.0
            )


def test_same_seed_reproduces_depot_points_and_ids() -> None:
    network = make_grid(rows=3, columns=3)

    first = select_depot_and_collection_points(network, 3, 0.0, random.Random(123))
    second = select_depot_and_collection_points(network, 3, 0.0, random.Random(123))

    assert first == second


def test_different_seeds_can_select_different_points() -> None:
    network = make_grid(rows=2, columns=4)

    selections = {
        tuple(
            point.node_id
            for point in select_depot_and_collection_points(
                network, 2, 0.0, random.Random(seed)
            )[1]
        )
        for seed in range(5)
    }

    assert len(selections) > 1


def test_unused_nodes_remain_in_network() -> None:
    network = make_grid(rows=3, columns=3)
    original_nodes = network.nodes
    original_arcs = network.arcs

    depot, points = select_depot_and_collection_points(
        network, 2, 0.0, random.Random(5)
    )

    used_node_ids = {depot.node_id} | {point.node_id for point in points}
    assert len(used_node_ids) == 3
    assert network.nodes == original_nodes
    assert network.arcs == original_arcs
    assert len(set(node.node_id for node in network.nodes) - used_node_ids) == 6


def test_point_selection_raises_after_attempt_limit_when_impossible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(generation, "POINT_PLACEMENT_MAX_ATTEMPTS", 2)

    with pytest.raises(ValueError, match="could not place collection points"):
        select_depot_and_collection_points(
            make_grid(rows=2, columns=2, spacing=1.0),
            1,
            100.0,
            random.Random(1),
        )


def test_exactly_point_count_plus_one_nodes_can_be_used_with_zero_separation() -> None:
    network = make_grid(rows=1, columns=3)

    depot, points = select_depot_and_collection_points(
        network, 2, 0.0, random.Random(1)
    )

    assert len(points) == 2
    assert {depot.node_id} | {point.node_id for point in points} == {
        node.node_id for node in network.nodes
    }


class ControlledRandom(random.Random):
    """Small deterministic random.Random used for trajectory tests."""

    def __init__(self, draws: list[float]) -> None:
        super().__init__(0)
        self.draws = iter(draws)

    def random(self) -> float:
        return next(self.draws)


def make_points() -> tuple[CollectionPoint, ...]:
    return (
        CollectionPoint(point_id=20, node_id=2),
        CollectionPoint(point_id=10, node_id=1),
        CollectionPoint(point_id=30, node_id=3),
    )


def generate_temperatures(
    points: tuple[CollectionPoint, ...] | None = None,
    *,
    sample_count: int = 4,
    rng: random.Random | None = None,
) -> dict[int, tuple[float, ...]]:
    return generate_temperature_trajectories(
        make_points() if points is None else points,
        sample_count,
        (1.0, 2.0, 3.0),
        (0.2, 0.5, 0.3),
        0.5,
        (0.2, 0.5, 0.3),
        -2.0,
        5.0,
        random.Random(123) if rng is None else rng,
    )


def generate_volumes(
    points: tuple[CollectionPoint, ...] | None = None,
    *,
    sample_count: int = 4,
    rng: random.Random | None = None,
) -> dict[int, tuple[float, ...]]:
    return generate_volume_trajectories(
        make_points() if points is None else points,
        sample_count,
        (10.0, 20.0, 30.0),
        (0.2, 0.5, 0.3),
        0.1,
        (0.2, 0.5, 0.3),
        0.0,
        100.0,
        random.Random(123) if rng is None else rng,
    )


def test_temperature_empty_collection_returns_empty_dict() -> None:
    assert generate_temperatures(()) == {}


def test_temperature_trajectories_have_expected_shape_and_immutable_values() -> None:
    trajectories = generate_temperatures(sample_count=5)

    assert set(trajectories) == {10, 20, 30}
    assert all(len(values) == 5 for values in trajectories.values())
    assert all(isinstance(values, tuple) for values in trajectories.values())


def test_temperature_generation_is_reproducible_and_independent_of_point_order() -> (
    None
):
    first = generate_temperatures(rng=random.Random(123))
    second = generate_temperatures(
        tuple(reversed(make_points())), rng=random.Random(123)
    )

    assert first == second


def test_temperature_initial_value_is_one_of_the_initial_levels() -> None:
    trajectories = generate_temperatures(sample_count=1)

    assert all(values[0] in {1.0, 2.0, 3.0} for values in trajectories.values())


def test_temperature_unclipped_changes_are_additive() -> None:
    trajectories = generate_temperatures(
        points=(CollectionPoint(1, 1),),
        sample_count=4,
        rng=ControlledRandom([0.5, 0.0, 0.5, 0.99]),
    )
    values = trajectories[1]
    changes = [second - first for first, second in zip(values, values[1:])]

    assert changes == [-0.5, 0.0, 0.5]


def test_temperature_lower_and_upper_clipping_work() -> None:
    lower = generate_temperature_trajectories(
        (CollectionPoint(1, 1),),
        2,
        (0.5, 1.0, 2.0),
        (1.0, 0.0, 0.0),
        1.0,
        (0.2, 0.3, 0.5),
        0.0,
        3.0,
        ControlledRandom([0.0, 0.0]),
    )
    upper = generate_temperature_trajectories(
        (CollectionPoint(1, 1),),
        2,
        (1.0, 2.0, 3.0),
        (0.0, 0.0, 1.0),
        2.0,
        (0.2, 0.3, 0.5),
        0.0,
        4.0,
        ControlledRandom([0.0, 0.99]),
    )

    assert lower[1] == (0.5, 0.0)
    assert upper[1] == (3.0, 4.0)


def test_temperature_values_remain_within_simulation_limits() -> None:
    trajectories = generate_temperatures()

    assert all(
        -2.0 <= value <= 5.0 for values in trajectories.values() for value in values
    )


@pytest.mark.parametrize("sample_count", [0, -1, True, 1.5, "4"])
def test_trajectory_generation_rejects_invalid_sample_count(
    sample_count: object,
) -> None:
    with pytest.raises(ValueError, match="sample_count"):
        generate_temperatures(sample_count=sample_count)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "delta_temperature", [0.0, -1.0, math.nan, math.inf, True, "1"]
)
def test_temperature_generation_rejects_invalid_delta(
    delta_temperature: object,
) -> None:
    with pytest.raises(ValueError, match="delta_temperature"):
        generate_temperature_trajectories(
            make_points(),
            2,
            (1.0, 2.0, 3.0),
            (0.2, 0.5, 0.3),
            delta_temperature,  # type: ignore[arg-type]
            (0.2, 0.5, 0.3),
            -2.0,
            5.0,
            random.Random(1),
        )


@pytest.mark.parametrize(
    "levels",
    [(1.0, 1.0, 2.0), (2.0, 1.0, 3.0), (1.0, 2.0), (1.0, math.nan, 3.0)],
)
def test_temperature_generation_rejects_invalid_levels(levels: object) -> None:
    with pytest.raises(ValueError, match="initial_levels"):
        generate_temperature_trajectories(
            make_points(),
            2,
            levels,  # type: ignore[arg-type]
            (0.2, 0.5, 0.3),
            0.5,
            (0.2, 0.5, 0.3),
            -2.0,
            5.0,
            random.Random(1),
        )


def test_temperature_levels_must_be_within_limits() -> None:
    with pytest.raises(ValueError, match="simulation limits"):
        generate_temperature_trajectories(
            make_points(),
            2,
            (-3.0, 0.0, 3.0),
            (0.2, 0.5, 0.3),
            0.5,
            (0.2, 0.5, 0.3),
            -2.0,
            5.0,
            random.Random(1),
        )


@pytest.mark.parametrize(
    "probabilities",
    [(0.5, 0.5), (0.5, 0.5, 0.5), (-0.1, 0.6, 0.5), (True, 0.0, 0.0)],
)
def test_temperature_generation_rejects_invalid_probabilities(
    probabilities: object,
) -> None:
    with pytest.raises(ValueError, match="probabilities"):
        generate_temperature_trajectories(
            make_points(),
            2,
            (1.0, 2.0, 3.0),
            probabilities,  # type: ignore[arg-type]
            0.5,
            (0.2, 0.5, 0.3),
            -2.0,
            5.0,
            random.Random(1),
        )


@pytest.mark.parametrize("transition_probabilities", [(0.4, 0.4, 0.2), (0.3, 0.2, 0.5)])
def test_temperature_generation_rejects_invalid_transition_order(
    transition_probabilities: tuple[float, float, float],
) -> None:
    with pytest.raises(ValueError, match="transition_probabilities"):
        generate_temperature_trajectories(
            make_points(),
            2,
            (1.0, 2.0, 3.0),
            (0.2, 0.5, 0.3),
            0.5,
            transition_probabilities,
            -2.0,
            5.0,
            random.Random(1),
        )


def test_temperature_generation_rejects_invalid_limits_and_rng() -> None:
    with pytest.raises(ValueError, match="simulation_minimum"):
        generate_temperature_trajectories(
            make_points(),
            2,
            (1.0, 2.0, 3.0),
            (0.2, 0.5, 0.3),
            0.5,
            (0.2, 0.5, 0.3),
            5.0,
            -2.0,
            random.Random(1),
        )
    with pytest.raises(ValueError, match="random.Random"):
        generate_temperature_trajectories(
            make_points(),
            2,
            (1.0, 2.0, 3.0),
            (0.2, 0.5, 0.3),
            0.5,
            (0.2, 0.5, 0.3),
            -2.0,
            5.0,
            object(),  # type: ignore[arg-type]
        )


def test_volume_empty_collection_returns_empty_dict() -> None:
    assert generate_volumes(()) == {}


def test_volume_trajectories_have_expected_shape_and_type() -> None:
    trajectories = generate_volumes(sample_count=5)

    assert set(trajectories) == {10, 20, 30}
    assert all(len(values) == 5 for values in trajectories.values())
    assert all(isinstance(values, tuple) for values in trajectories.values())


def test_volume_generation_is_reproducible_and_independent_of_point_order() -> None:
    first = generate_volumes(rng=random.Random(123))
    second = generate_volumes(tuple(reversed(make_points())), rng=random.Random(123))

    assert first == second


def test_volume_initial_value_is_one_of_the_initial_levels() -> None:
    trajectories = generate_volumes(sample_count=1)

    assert all(values[0] in {10.0, 20.0, 30.0} for values in trajectories.values())


def test_volume_update_is_multiplicative_for_decrease_hold_and_increase() -> None:
    trajectories = generate_volume_trajectories(
        (CollectionPoint(1, 1),),
        4,
        (10.0, 20.0, 30.0),
        (0.0, 1.0, 0.0),
        0.1,
        (0.2, 0.5, 0.3),
        0.0,
        100.0,
        ControlledRandom([0.5, 0.0, 0.5, 0.99]),
    )

    assert trajectories[1] == (20.0, 18.0, 18.0, 19.8)


def test_volume_lower_and_upper_clipping_work() -> None:
    lower = generate_volume_trajectories(
        (CollectionPoint(1, 1),),
        2,
        (1.0, 2.0, 3.0),
        (1.0, 0.0, 0.0),
        2.0,
        (0.2, 0.3, 0.5),
        0.0,
        10.0,
        ControlledRandom([0.0, 0.0]),
    )
    upper = generate_volume_trajectories(
        (CollectionPoint(1, 1),),
        2,
        (10.0, 12.0, 14.0),
        (1.0, 0.0, 0.0),
        1.0,
        (0.2, 0.3, 0.5),
        0.0,
        15.0,
        ControlledRandom([0.0, 0.99]),
    )

    assert lower[1] == (1.0, 0.0)
    assert upper[1] == (10.0, 15.0)


def test_volume_values_are_non_negative_with_non_negative_minimum() -> None:
    trajectories = generate_volumes()

    assert all(value >= 0.0 for values in trajectories.values() for value in values)


@pytest.mark.parametrize("gamma_volume", [0.0, -1.0, math.nan, math.inf, True, "1"])
def test_volume_generation_rejects_invalid_gamma(gamma_volume: object) -> None:
    with pytest.raises(ValueError, match="gamma_volume"):
        generate_volume_trajectories(
            make_points(),
            2,
            (10.0, 20.0, 30.0),
            (0.2, 0.5, 0.3),
            gamma_volume,  # type: ignore[arg-type]
            (0.2, 0.5, 0.3),
            0.0,
            100.0,
            random.Random(1),
        )


@pytest.mark.parametrize(
    "levels",
    [(-1.0, 10.0, 20.0), (10.0, 10.0, 20.0), (10.0, 20.0, 15.0)],
)
def test_volume_generation_rejects_invalid_levels(levels: object) -> None:
    with pytest.raises(ValueError, match="initial_levels"):
        generate_volume_trajectories(
            make_points(),
            2,
            levels,  # type: ignore[arg-type]
            (0.2, 0.5, 0.3),
            0.1,
            (0.2, 0.5, 0.3),
            0.0,
            100.0,
            random.Random(1),
        )


def test_volume_levels_must_be_within_limits_and_limits_must_be_valid() -> None:
    with pytest.raises(ValueError, match="simulation limits"):
        generate_volume_trajectories(
            make_points(),
            2,
            (10.0, 20.0, 30.0),
            (0.2, 0.5, 0.3),
            0.1,
            (0.2, 0.5, 0.3),
            0.0,
            25.0,
            random.Random(1),
        )
    with pytest.raises(ValueError, match="simulation_minimum"):
        generate_volume_trajectories(
            make_points(),
            2,
            (10.0, 20.0, 30.0),
            (0.2, 0.5, 0.3),
            0.1,
            (0.2, 0.5, 0.3),
            10.0,
            5.0,
            random.Random(1),
        )


def test_required_vehicle_capacity_sums_individual_maxima_without_mutation() -> None:
    trajectories = {1: (1.0, 3.0, 2.0), 2: (4.0, 2.0)}
    original = trajectories.copy()

    assert required_vehicle_capacity(trajectories) == 7.0
    assert required_vehicle_capacity({}) == 0.0
    assert trajectories == original


def make_synthetic_instance(**overrides: object) -> ProblemInstance:
    parameters: dict[str, object] = {
        "instance_id": 7,
        "network_id": 9,
        "instance_seed": 123,
        "rows": 2,
        "columns": 3,
        "spacing": 10.0,
        "p_uni": 0.0,
        "point_count": 2,
        "min_separation": 0.0,
        "sample_count": 3,
        "temperature_initial_levels": (1.0, 2.0, 3.0),
        "temperature_initial_probabilities": (1.0, 0.0, 0.0),
        "delta_temperature": 0.5,
        "temperature_transition_probabilities": (0.2, 0.5, 0.3),
        "temperature_simulation_minimum": -2.0,
        "temperature_simulation_maximum": 5.0,
        "volume_initial_levels": (10.0, 20.0, 30.0),
        "volume_initial_probabilities": (1.0, 0.0, 0.0),
        "gamma_volume": 0.1,
        "volume_transition_probabilities": (0.2, 0.5, 0.3),
        "volume_simulation_minimum": 0.0,
        "volume_simulation_maximum": 100.0,
        "vehicle_capacity": 1000.0,
        "collection_service_duration": 60.0,
        "v_ref": 2.0,
    }
    parameters.update(overrides)
    return generate_synthetic_problem_instance(**parameters)  # type: ignore[arg-type]


def instance_signature(instance: ProblemInstance) -> tuple[object, ...]:
    return (
        instance.road_network,
        instance.depot,
        instance.collection_points,
        instance.temperature_trajectories,
        instance.volume_trajectories,
        instance.vehicle_capacity,
        instance.v_ref,
    )


def test_synthetic_generation_returns_a_complete_problem_instance() -> None:
    instance = make_synthetic_instance()

    assert isinstance(instance, ProblemInstance)
    assert instance.instance_id == 7
    assert instance.road_network.network_id == 9
    assert len(instance.collection_points) == 2
    point_node_ids = {point.node_id for point in instance.collection_points}
    assert len(point_node_ids) == 2
    assert instance.depot.node_id not in point_node_ids
    assert instance.sample_period == 30.0
    assert instance.vehicle_capacity == 1000.0
    assert instance.collection_service_duration == 60.0
    assert instance.v_ref == 2.0


def test_synthetic_generation_has_complete_trajectory_mappings() -> None:
    instance = make_synthetic_instance(sample_count=4)
    point_ids = {point.point_id for point in instance.collection_points}

    assert set(instance.temperature_trajectories) == point_ids
    assert set(instance.volume_trajectories) == point_ids
    assert all(
        len(values) == 4 for values in instance.temperature_trajectories.values()
    )
    assert all(len(values) == 4 for values in instance.volume_trajectories.values())
    assert required_vehicle_capacity(instance.volume_trajectories) <= (
        instance.vehicle_capacity
    )


def test_capacity_equal_to_required_capacity_is_accepted() -> None:
    instance = make_synthetic_instance(
        sample_count=1,
        volume_initial_probabilities=(1.0, 0.0, 0.0),
        vehicle_capacity=20.0,
    )

    assert required_vehicle_capacity(instance.volume_trajectories) == 20.0
    assert instance.vehicle_capacity == 20.0


def test_insufficient_capacity_raises_without_regenerating_volumes() -> None:
    with pytest.raises(
        ValueError,
        match="vehicle_capacity is insufficient for the generated volume trajectories",
    ):
        make_synthetic_instance(
            sample_count=1,
            volume_initial_probabilities=(1.0, 0.0, 0.0),
            vehicle_capacity=19.0,
        )


@pytest.mark.parametrize("vehicle_capacity", [0.0, -1.0, math.nan, math.inf, True, "1"])
def test_invalid_vehicle_capacity_is_rejected(vehicle_capacity: object) -> None:
    with pytest.raises(ValueError, match="vehicle_capacity"):
        make_synthetic_instance(vehicle_capacity=vehicle_capacity)


@pytest.mark.parametrize(
    "collection_service_duration",
    [0.0, -1.0, math.nan, math.inf, True, "1"],
)
def test_invalid_collection_service_duration_is_rejected(
    collection_service_duration: object,
) -> None:
    with pytest.raises(ValueError, match="collection_service_duration"):
        make_synthetic_instance(collection_service_duration=collection_service_duration)


@pytest.mark.parametrize("v_ref", [0.0, -1.0, math.nan, math.inf, True, "1"])
def test_invalid_v_ref_is_rejected(v_ref: object) -> None:
    with pytest.raises(ValueError, match="v_ref"):
        make_synthetic_instance(v_ref=v_ref)


def test_same_seed_and_parameters_reproduce_the_instance() -> None:
    first = make_synthetic_instance(instance_seed=321)
    second = make_synthetic_instance(instance_seed=321)

    assert instance_signature(first) == instance_signature(second)


def test_different_seeds_can_change_generated_instance_data() -> None:
    first = make_synthetic_instance(instance_seed=321)
    second = make_synthetic_instance(instance_seed=654)

    assert instance_signature(first) != instance_signature(second)


def test_generation_does_not_depend_on_global_random_state() -> None:
    random.seed(1)
    random.random()
    first = make_synthetic_instance(instance_seed=321)
    random.seed(999)
    random.random()
    second = make_synthetic_instance(instance_seed=321)

    assert instance_signature(first) == instance_signature(second)


@pytest.mark.parametrize("instance_seed", [-1, True, 1.5, "1"])
def test_invalid_instance_seed_is_rejected(instance_seed: object) -> None:
    with pytest.raises(ValueError, match="instance_seed"):
        make_synthetic_instance(instance_seed=instance_seed)


def test_network_generation_error_is_propagated() -> None:
    with pytest.raises(ValueError, match="at least two nodes"):
        make_synthetic_instance(rows=1, columns=1)


def test_point_selection_error_is_propagated() -> None:
    with pytest.raises(ValueError, match=r"point_count \+ 1"):
        make_synthetic_instance(point_count=6)


def test_temperature_generation_error_is_propagated() -> None:
    with pytest.raises(ValueError, match="delta_temperature"):
        make_synthetic_instance(delta_temperature=0.0)


def test_volume_generation_error_is_propagated() -> None:
    with pytest.raises(ValueError, match="gamma_volume"):
        make_synthetic_instance(gamma_volume=0.0)


def test_invalid_instance_id_is_delegated_to_problem_instance() -> None:
    with pytest.raises(ValueError, match="instance_id"):
        make_synthetic_instance(instance_id=-1)


def test_invalid_network_id_is_delegated_to_road_network() -> None:
    with pytest.raises(ValueError, match="network_id"):
        make_synthetic_instance(network_id=-1)
