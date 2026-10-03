import math
import random

import pytest

from src.algorithms.ant_colony import (
    ant_count,
    construct_ant_route,
    initialize_pheromones,
    update_pheromones,
)
from src.models.candidate_route import CandidateRoute
from src.models.problem import CollectionPoint, Depot, ProblemInstance
from src.models.road_network import RoadArc, RoadNetwork, RoadNode
from src.simulation.snapshot import OptimizationSnapshot


def make_instance(
    point_count: int = 3,
    *,
    node_count: int | None = None,
    arcs: list[RoadArc] | None = None,
    temperature_trajectories: dict[int, tuple[float, ...]] | None = None,
    volume_trajectories: dict[int, tuple[float, ...]] | None = None,
    coordinate_scale: float = 1.0,
) -> ProblemInstance:
    points = [
        CollectionPoint(point_id=10 * index, node_id=index)
        for index in range(2, point_count + 2)
    ]
    if node_count is None:
        node_count = point_count + 2

    return ProblemInstance(
        instance_id=1,
        road_network=RoadNetwork(
            network_id=1,
            nodes=[
                RoadNode(
                    node_id=node_id,
                    x=float(node_id) * coordinate_scale,
                    y=0.0,
                )
                for node_id in range(1, node_count + 1)
            ],
            arcs=[] if arcs is None else arcs,
        ),
        depot=Depot(depot_id=1, node_id=1),
        collection_points=points,
        vehicle_capacity=100.0,
        collection_service_duration=60.0,
        v_ref=2.0,
        sample_period=30.0,
        temperature_trajectories=(
            {point.point_id: (4.0,) for point in points}
            if temperature_trajectories is None
            else temperature_trajectories
        ),
        volume_trajectories=(
            {point.point_id: (1.0,) for point in points}
            if volume_trajectories is None
            else volume_trajectories
        ),
    )


def make_snapshot(
    point_ids: tuple[int, ...],
    *,
    current_node_id: int = 1,
    temperatures: dict[int, float] | None = None,
    volumes: dict[int, float] | None = None,
) -> OptimizationSnapshot:
    return OptimizationSnapshot(
        current_time=0.0,
        current_node_id=current_node_id,
        current_load=0.0,
        remaining_point_ids=point_ids,
        temperatures=(
            {point_id: 4.0 for point_id in point_ids}
            if temperatures is None
            else temperatures
        ),
        volumes=(
            {point_id: 1.0 for point_id in point_ids} if volumes is None else volumes
        ),
    )


def make_routing_instance(
    *,
    coordinate_scale: float = 1.0,
    arcs: list[RoadArc] | None = None,
) -> ProblemInstance:
    if arcs is None:
        arcs = [
            RoadArc(1, 2, 1.0),
            RoadArc(1, 3, 2.0),
            RoadArc(2, 1, 1.0),
            RoadArc(2, 3, 1.0),
            RoadArc(3, 1, 1.0),
            RoadArc(3, 2, 1.0),
        ]

    return make_instance(
        2,
        node_count=3,
        arcs=arcs,
        coordinate_scale=coordinate_scale,
    )


def construct(
    instance: ProblemInstance,
    snapshot: OptimizationSnapshot,
    *,
    pheromones: dict[tuple[int, int], float] | None = None,
    t_min: float = 2.0,
    t_max: float = 6.0,
    weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
    beta: float = 1.0,
    rng: random.Random | None = None,
) -> CandidateRoute:
    if pheromones is None:
        pheromones = initialize_pheromones(instance, snapshot, 1.0)
    if rng is None:
        rng = random.Random(123)

    return construct_ant_route(
        instance,
        snapshot,
        pheromones,
        t_min,
        t_max,
        weights,
        beta,
        rng,
    )


def test_ant_count_is_capped_at_twenty() -> None:
    cases = ((0, 0), (10, 10), (25, 20), (50, 20))

    for point_count, expected_count in cases:
        instance = make_instance(point_count)
        point_ids = tuple(point.point_id for point in instance.collection_points)

        assert ant_count(make_snapshot(point_ids)) == expected_count


def test_initialize_one_point_has_current_to_point_and_point_to_depot() -> None:
    instance = make_instance(1)
    snapshot = make_snapshot((20,))

    pheromones = initialize_pheromones(instance, snapshot, 4.0)

    assert set(pheromones) == {(1, 2), (2, 1)}


def test_initialize_multiple_points_has_exact_logical_movements() -> None:
    instance = make_instance(2)
    snapshot = make_snapshot((20, 30))

    pheromones = initialize_pheromones(instance, snapshot, 4.0)

    assert set(pheromones) == {
        (1, 2),
        (1, 3),
        (2, 3),
        (3, 2),
        (2, 1),
        (3, 1),
    }


def test_initialize_does_not_add_other_logical_movements() -> None:
    instance = make_instance(2)
    snapshot = make_snapshot((20, 30), current_node_id=4)

    pheromones = initialize_pheromones(instance, snapshot, 4.0)

    assert set(pheromones) == {
        (4, 2),
        (4, 3),
        (2, 3),
        (3, 2),
        (2, 1),
        (3, 1),
    }
    assert (1, 2) not in pheromones
    assert (1, 3) not in pheromones
    assert (4, 1) not in pheromones


def test_initialize_assigns_one_over_baseline_to_every_movement() -> None:
    instance = make_instance(2)
    snapshot = make_snapshot((20, 30))

    pheromones = initialize_pheromones(instance, snapshot, 4.0)

    assert set(pheromones.values()) == {0.25}


def test_initialize_terminal_snapshot_returns_empty_dict() -> None:
    instance = make_instance(0)
    snapshot = make_snapshot(())

    assert initialize_pheromones(instance, snapshot, 4.0) == {}


@pytest.mark.parametrize(
    "baseline_cost",
    [0.0, -1.0, math.nan, math.inf, True, "1"],
)
def test_initialize_rejects_invalid_baseline_cost(baseline_cost: object) -> None:
    instance = make_instance(1)
    snapshot = make_snapshot((20,))

    with pytest.raises(ValueError, match="baseline_cost"):
        initialize_pheromones(instance, snapshot, baseline_cost)  # type: ignore[arg-type]


def test_update_applies_evaporation_and_best_route_deposit() -> None:
    instance = make_instance(3)
    snapshot = make_snapshot((20, 30, 40))
    pheromones = initialize_pheromones(instance, snapshot, 10.0)
    best_route = CandidateRoute((20, 30, 40))

    updated = update_pheromones(
        instance,
        snapshot,
        pheromones,
        best_route,
        best_cost=5.0,
        evaporation_rate=0.3,
    )

    evaporated_value = 0.1 * 0.7
    deposited_value = evaporated_value + 0.2
    assert updated[(1, 2)] == pytest.approx(deposited_value)
    assert updated[(2, 3)] == pytest.approx(deposited_value)
    assert updated[(3, 4)] == pytest.approx(deposited_value)
    assert updated[(4, 1)] == pytest.approx(deposited_value)
    assert updated[(1, 3)] == pytest.approx(evaporated_value)


def test_update_deposit_is_exactly_one_over_best_cost() -> None:
    instance = make_instance(1)
    snapshot = make_snapshot((20,))
    pheromones = {(1, 2): 2.0, (2, 1): 3.0}

    updated = update_pheromones(
        instance,
        snapshot,
        pheromones,
        CandidateRoute((20,)),
        best_cost=4.0,
        evaporation_rate=0.5,
    )

    assert updated[(1, 2)] == pytest.approx(1.0 + 0.25)
    assert updated[(2, 1)] == pytest.approx(1.5 + 0.25)


def test_update_does_not_mutate_the_original_pheromone_dict() -> None:
    instance = make_instance(1)
    snapshot = make_snapshot((20,))
    pheromones = {(1, 2): 2.0, (2, 1): 3.0}
    original = pheromones.copy()

    updated = update_pheromones(
        instance,
        snapshot,
        pheromones,
        CandidateRoute((20,)),
        best_cost=4.0,
        evaporation_rate=0.5,
    )

    assert pheromones == original
    assert updated is not pheromones


def test_update_only_deposits_on_logical_route_movements() -> None:
    instance = make_instance(
        2,
        node_count=5,
        arcs=[RoadArc(1, 5, 1.0), RoadArc(5, 2, 1.0), RoadArc(2, 3, 1.0)],
    )
    snapshot = make_snapshot((20, 30))
    pheromones = initialize_pheromones(instance, snapshot, 10.0)

    updated = update_pheromones(
        instance,
        snapshot,
        pheromones,
        CandidateRoute((20, 30)),
        best_cost=5.0,
        evaporation_rate=0.5,
    )

    assert (1, 5) not in updated
    assert (5, 2) not in updated
    assert updated[(1, 2)] == pytest.approx(0.05 + 0.2)
    assert updated[(2, 3)] == pytest.approx(0.05 + 0.2)
    assert updated[(3, 1)] == pytest.approx(0.05 + 0.2)


def test_update_does_not_create_reverse_movement_automatically() -> None:
    instance = make_instance(1)
    snapshot = make_snapshot((20,), current_node_id=3)
    pheromones = initialize_pheromones(instance, snapshot, 10.0)

    assert set(pheromones) == {(3, 2), (2, 1)}
    assert (2, 3) not in pheromones
    assert (1, 2) not in pheromones


@pytest.mark.parametrize(
    "best_cost",
    [0.0, -1.0, math.nan, math.inf, True, "1"],
)
def test_update_rejects_invalid_best_cost(best_cost: object) -> None:
    instance = make_instance(1)
    snapshot = make_snapshot((20,))
    pheromones = initialize_pheromones(instance, snapshot, 10.0)

    with pytest.raises(ValueError, match="best_cost"):
        update_pheromones(
            instance,
            snapshot,
            pheromones,
            CandidateRoute((20,)),
            best_cost=best_cost,  # type: ignore[arg-type]
            evaporation_rate=0.3,
        )


@pytest.mark.parametrize(
    "evaporation_rate",
    [0.0, -0.1, 1.0, 1.1, math.nan, math.inf, True, "0.3"],
)
def test_update_rejects_invalid_evaporation_rate(
    evaporation_rate: object,
) -> None:
    instance = make_instance(1)
    snapshot = make_snapshot((20,))
    pheromones = initialize_pheromones(instance, snapshot, 10.0)

    with pytest.raises(ValueError, match="evaporation_rate"):
        update_pheromones(
            instance,
            snapshot,
            pheromones,
            CandidateRoute((20,)),
            best_cost=5.0,
            evaporation_rate=evaporation_rate,  # type: ignore[arg-type]
        )


def test_update_rejects_missing_pheromone_key() -> None:
    instance = make_instance(1)
    snapshot = make_snapshot((20,))
    pheromones = initialize_pheromones(instance, snapshot, 10.0)
    pheromones.pop((1, 2))

    with pytest.raises(ValueError, match="keys"):
        update_pheromones(
            instance,
            snapshot,
            pheromones,
            CandidateRoute((20,)),
            best_cost=5.0,
            evaporation_rate=0.3,
        )


def test_update_rejects_extra_pheromone_key() -> None:
    instance = make_instance(1)
    snapshot = make_snapshot((20,))
    pheromones = initialize_pheromones(instance, snapshot, 10.0)
    pheromones[(99, 100)] = 1.0

    with pytest.raises(ValueError, match="keys"):
        update_pheromones(
            instance,
            snapshot,
            pheromones,
            CandidateRoute((20,)),
            best_cost=5.0,
            evaporation_rate=0.3,
        )


@pytest.mark.parametrize(
    "pheromone_value",
    [0.0, -1.0, math.nan, math.inf, True, "1"],
)
def test_update_rejects_invalid_pheromone_value(pheromone_value: object) -> None:
    instance = make_instance(1)
    snapshot = make_snapshot((20,))
    pheromones = initialize_pheromones(instance, snapshot, 10.0)
    pheromones[(1, 2)] = pheromone_value  # type: ignore[assignment]

    with pytest.raises(ValueError, match="pheromone"):
        update_pheromones(
            instance,
            snapshot,
            pheromones,
            CandidateRoute((20,)),
            best_cost=5.0,
            evaporation_rate=0.3,
        )


def test_update_terminal_snapshot_with_empty_route_returns_empty_dict() -> None:
    instance = make_instance(0)
    snapshot = make_snapshot(())

    assert (
        update_pheromones(
            instance,
            snapshot,
            {},
            CandidateRoute(()),
            best_cost=5.0,
            evaporation_rate=0.3,
        )
        == {}
    )


def test_update_propagates_invalid_candidate_route() -> None:
    instance = make_instance(2)
    snapshot = make_snapshot((20, 30))
    pheromones = initialize_pheromones(instance, snapshot, 10.0)

    with pytest.raises(ValueError, match="missing"):
        update_pheromones(
            instance,
            snapshot,
            pheromones,
            CandidateRoute((20,)),
            best_cost=5.0,
            evaporation_rate=0.3,
        )


def test_construct_empty_snapshot_returns_empty_candidate_route() -> None:
    instance = make_instance(0)
    snapshot = make_snapshot(())

    route = construct(instance, snapshot, pheromones={})

    assert route == CandidateRoute(())


def test_construct_with_one_option_always_selects_that_point() -> None:
    instance = make_instance(
        1,
        arcs=[RoadArc(1, 2, 1.0), RoadArc(2, 1, 1.0)],
    )
    snapshot = make_snapshot((20,))

    route = construct(instance, snapshot, weights=(1.0, 0.0, 0.0))

    assert route == CandidateRoute((20,))


def test_construct_visits_all_points_once_and_returns_valid_route() -> None:
    instance = make_routing_instance()
    snapshot = make_snapshot((20, 30))

    route = construct(instance, snapshot)

    assert sorted(route.point_ids) == [20, 30]
    assert len(route.point_ids) == len(set(route.point_ids))
    assert route.validate(instance, snapshot) is None


def test_construct_does_not_modify_snapshot_remaining_order() -> None:
    instance = make_routing_instance()
    snapshot = make_snapshot((30, 20))
    original_point_ids = snapshot.remaining_point_ids

    construct(instance, snapshot)

    assert snapshot.remaining_point_ids == original_point_ids


def test_shorter_distance_increases_local_attractiveness() -> None:
    instance = make_routing_instance()
    snapshot = make_snapshot((20, 30))

    route = construct(
        instance,
        snapshot,
        weights=(1.0, 0.0, 0.0),
        rng=random.Random(1),
    )

    assert route.point_ids[0] == 20


def test_higher_thermal_risk_increases_local_attractiveness() -> None:
    instance = make_routing_instance()
    snapshot = make_snapshot(
        (20, 30),
        temperatures={20: 6.0, 30: 4.0},
    )

    route = construct(
        instance,
        snapshot,
        t_max=10.0,
        weights=(0.0, 1.0, 0.0),
        rng=random.Random(9),
    )

    assert route.point_ids[0] == 20


def test_higher_volume_increases_local_attractiveness() -> None:
    instance = make_routing_instance()
    high_first_volume = make_snapshot(
        (20, 30),
        volumes={20: 2.0, 30: 1.0},
    )
    high_second_volume = make_snapshot(
        (20, 30),
        volumes={20: 1.0, 30: 2.0},
    )

    first_route = construct(
        instance,
        high_first_volume,
        weights=(0.0, 0.0, 1.0),
        rng=random.Random(0),
    )
    second_route = construct(
        instance,
        high_second_volume,
        weights=(0.0, 0.0, 1.0),
        rng=random.Random(0),
    )

    assert first_route.point_ids[0] == 20
    assert second_route.point_ids[0] == 30


def test_zero_local_cost_is_handled_by_epsilon() -> None:
    instance = make_routing_instance()
    snapshot = make_snapshot(
        (20, 30),
        temperatures={20: 6.0, 30: 6.0},
    )

    route = construct(
        instance,
        snapshot,
        weights=(0.0, 1.0, 0.0),
        rng=random.Random(0),
    )

    assert sorted(route.point_ids) == [20, 30]


def test_larger_pheromone_increases_transition_probability() -> None:
    instance = make_routing_instance()
    snapshot = make_snapshot((20, 30))
    first_pheromones = initialize_pheromones(instance, snapshot, 1.0)
    second_pheromones = first_pheromones.copy()
    first_pheromones[(1, 2)] = 100.0
    first_pheromones[(1, 3)] = 1.0
    second_pheromones[(1, 2)] = 1.0
    second_pheromones[(1, 3)] = 100.0

    first_route = construct(
        instance,
        snapshot,
        pheromones=first_pheromones,
        weights=(1.0, 0.0, 0.0),
        rng=random.Random(0),
    )
    second_route = construct(
        instance,
        snapshot,
        pheromones=second_pheromones,
        weights=(1.0, 0.0, 0.0),
        rng=random.Random(0),
    )

    assert first_route.point_ids[0] == 20
    assert second_route.point_ids[0] == 30


def test_larger_beta_reinforces_heuristic_influence() -> None:
    instance = make_routing_instance()
    snapshot = make_snapshot((20, 30))
    pheromones = initialize_pheromones(instance, snapshot, 1.0)

    route_with_beta_one = construct(
        instance,
        snapshot,
        pheromones=pheromones,
        weights=(1.0, 0.0, 0.0),
        beta=1.0,
        rng=random.Random(6),
    )
    route_with_beta_two = construct(
        instance,
        snapshot,
        pheromones=pheromones,
        weights=(1.0, 0.0, 0.0),
        beta=2.0,
        rng=random.Random(6),
    )

    assert route_with_beta_one.point_ids[0] == 30
    assert route_with_beta_two.point_ids[0] == 20


@pytest.mark.parametrize("beta", [0.0, -1.0, math.nan, math.inf, True, "1"])
def test_construct_rejects_invalid_beta(beta: object) -> None:
    instance = make_routing_instance()
    snapshot = make_snapshot((20, 30))
    pheromones = initialize_pheromones(instance, snapshot, 1.0)

    with pytest.raises(ValueError, match="beta"):
        construct_ant_route(
            instance,
            snapshot,
            pheromones,
            2.0,
            6.0,
            (1.0, 0.0, 0.0),
            beta,  # type: ignore[arg-type]
            random.Random(1),
        )


@pytest.mark.parametrize(
    "weights",
    [
        (1.0, 0.0),
        (0.4, 0.4, 0.1),
        (-0.1, 0.5, 0.6),
        (math.nan, 0.0, 1.0),
        (True, 0.0, 0.0),
        ("1", 0.0, 0.0),
    ],
)
def test_construct_rejects_invalid_weights(weights: object) -> None:
    instance = make_routing_instance()
    snapshot = make_snapshot((20, 30))
    pheromones = initialize_pheromones(instance, snapshot, 1.0)

    with pytest.raises(ValueError, match="weights|w_"):
        construct_ant_route(
            instance,
            snapshot,
            pheromones,
            2.0,
            6.0,
            weights,  # type: ignore[arg-type]
            1.0,
            random.Random(1),
        )


def test_critical_precedence_excludes_non_critical_candidates() -> None:
    instance = make_routing_instance()
    snapshot = make_snapshot(
        (20, 30),
        temperatures={20: 6.0, 30: 2.0},
    )

    route = construct(instance, snapshot, rng=random.Random(0))

    assert route.point_ids[0] == 20


def test_two_critical_candidates_both_participate_in_roulette() -> None:
    instance = make_routing_instance()
    snapshot = make_snapshot(
        (20, 30),
        temperatures={20: 6.0, 30: 6.0},
    )

    first_route = construct(instance, snapshot, rng=random.Random(1))
    second_route = construct(instance, snapshot, rng=random.Random(2))

    assert first_route.point_ids[0] == 20
    assert second_route.point_ids[0] == 30


def test_non_critical_candidates_become_eligible_after_critical_points() -> None:
    arcs = [
        RoadArc(from_node, to_node, 1.0)
        for from_node in range(1, 5)
        for to_node in range(1, 5)
        if from_node != to_node
    ]
    instance = make_instance(3, arcs=arcs)
    snapshot = make_snapshot(
        (20, 30, 40),
        temperatures={20: 6.0, 30: 2.0, 40: 2.0},
    )

    route = construct(instance, snapshot, rng=random.Random(1))

    assert route.point_ids[0] == 20
    assert set(route.point_ids[1:]) == {30, 40}


def test_construct_uses_snapshot_temperature_not_future_trajectory() -> None:
    instance = make_instance(
        2,
        arcs=[
            RoadArc(1, 2, 1.0),
            RoadArc(1, 3, 2.0),
            RoadArc(2, 1, 1.0),
            RoadArc(2, 3, 1.0),
            RoadArc(3, 1, 1.0),
            RoadArc(3, 2, 1.0),
        ],
        temperature_trajectories={20: (100.0,), 30: (0.0,)},
    )
    snapshot = make_snapshot(
        (20, 30),
        temperatures={20: 2.0, 30: 6.0},
    )

    route = construct(
        instance,
        snapshot,
        t_max=10.0,
        weights=(0.0, 1.0, 0.0),
        rng=random.Random(0),
    )

    assert route.point_ids[0] == 30


def test_construct_uses_snapshot_volume_not_future_trajectory() -> None:
    instance_with_future_volumes = make_instance(
        2,
        arcs=[
            RoadArc(1, 2, 1.0),
            RoadArc(1, 3, 2.0),
            RoadArc(2, 1, 1.0),
            RoadArc(2, 3, 1.0),
            RoadArc(3, 1, 1.0),
            RoadArc(3, 2, 1.0),
        ],
        volume_trajectories={20: (2.0,), 30: (1.0,)},
    )
    snapshot = make_snapshot(
        (20, 30),
        volumes={20: 1.0, 30: 2.0},
    )

    route = construct(
        instance_with_future_volumes,
        snapshot,
        weights=(0.0, 0.0, 1.0),
        rng=random.Random(0),
    )

    assert route.point_ids[0] == 30


def test_construct_uses_directed_shortest_path_and_logical_pheromones() -> None:
    instance = make_instance(
        2,
        node_count=4,
        arcs=[
            RoadArc(1, 2, 10.0),
            RoadArc(1, 4, 1.0),
            RoadArc(4, 2, 1.0),
            RoadArc(1, 3, 5.0),
            RoadArc(2, 1, 1.0),
            RoadArc(2, 3, 1.0),
            RoadArc(3, 1, 1.0),
            RoadArc(3, 2, 1.0),
        ],
    )
    snapshot = make_snapshot((20, 30))
    pheromones = initialize_pheromones(instance, snapshot, 1.0)

    route = construct(
        instance,
        snapshot,
        pheromones=pheromones,
        weights=(1.0, 0.0, 0.0),
        rng=random.Random(1),
    )

    assert route.point_ids[0] == 20
    assert (1, 4) not in pheromones
    assert (4, 2) not in pheromones


def test_coordinates_do_not_influence_construction() -> None:
    first_instance = make_routing_instance(coordinate_scale=1.0)
    second_instance = make_routing_instance(coordinate_scale=1_000_000.0)
    first_snapshot = make_snapshot((20, 30))
    second_snapshot = make_snapshot((20, 30))

    first_route = construct(
        first_instance,
        first_snapshot,
        rng=random.Random(123),
    )
    second_route = construct(
        second_instance,
        second_snapshot,
        rng=random.Random(123),
    )

    assert first_route == second_route


def test_construct_rejects_missing_pheromone_movement() -> None:
    instance = make_instance(
        1,
        arcs=[RoadArc(1, 2, 1.0), RoadArc(2, 1, 1.0)],
    )
    snapshot = make_snapshot((20,))
    pheromones = initialize_pheromones(instance, snapshot, 1.0)
    pheromones.pop((1, 2))

    with pytest.raises(ValueError, match="missing logical movement"):
        construct(instance, snapshot, pheromones=pheromones)


@pytest.mark.parametrize("pheromone_value", [0.0, -1.0, math.nan, math.inf, True, "1"])
def test_construct_rejects_invalid_pheromone_value(
    pheromone_value: object,
) -> None:
    instance = make_instance(
        1,
        arcs=[RoadArc(1, 2, 1.0), RoadArc(2, 1, 1.0)],
    )
    snapshot = make_snapshot((20,))
    pheromones = initialize_pheromones(instance, snapshot, 1.0)
    pheromones[(1, 2)] = pheromone_value  # type: ignore[assignment]

    with pytest.raises(ValueError, match="pheromone"):
        construct(instance, snapshot, pheromones=pheromones)


def test_same_rng_and_inputs_produce_same_candidate_route() -> None:
    instance = make_routing_instance()
    snapshot = make_snapshot((20, 30))
    first_pheromones = initialize_pheromones(instance, snapshot, 1.0)
    second_pheromones = initialize_pheromones(instance, snapshot, 1.0)

    first_route = construct(
        instance,
        snapshot,
        pheromones=first_pheromones,
        rng=random.Random(123),
    )
    second_route = construct(
        instance,
        snapshot,
        pheromones=second_pheromones,
        rng=random.Random(123),
    )

    assert first_route == second_route


def test_remaining_point_order_does_not_influence_construction() -> None:
    instance = make_routing_instance()
    first_snapshot = make_snapshot((20, 30))
    second_snapshot = make_snapshot((30, 20))
    first_pheromones = initialize_pheromones(instance, first_snapshot, 1.0)
    second_pheromones = initialize_pheromones(instance, second_snapshot, 1.0)

    first_route = construct(
        instance,
        first_snapshot,
        pheromones=first_pheromones,
        rng=random.Random(123),
    )
    second_route = construct(
        instance,
        second_snapshot,
        pheromones=second_pheromones,
        rng=random.Random(123),
    )

    assert first_route == second_route


def test_different_roulette_draws_can_select_different_candidates() -> None:
    instance = make_routing_instance(
        arcs=[
            RoadArc(1, 2, 1.0),
            RoadArc(1, 3, 1.0),
            RoadArc(2, 1, 1.0),
            RoadArc(2, 3, 1.0),
            RoadArc(3, 1, 1.0),
            RoadArc(3, 2, 1.0),
        ]
    )
    snapshot = make_snapshot((20, 30))

    first_route = construct(
        instance,
        snapshot,
        weights=(1.0, 0.0, 0.0),
        rng=random.Random(1),
    )
    second_route = construct(
        instance,
        snapshot,
        weights=(1.0, 0.0, 0.0),
        rng=random.Random(2),
    )

    assert first_route.point_ids[0] == 20
    assert second_route.point_ids[0] == 30


def test_construct_does_not_modify_pheromone_dictionary() -> None:
    instance = make_routing_instance()
    snapshot = make_snapshot((20, 30))
    pheromones = initialize_pheromones(instance, snapshot, 1.0)
    original = pheromones.copy()

    construct(instance, snapshot, pheromones=pheromones)

    assert pheromones == original


def test_construct_propagates_missing_candidate_path() -> None:
    instance = make_instance(1, arcs=[RoadArc(2, 1, 1.0)])
    snapshot = make_snapshot((20,))
    pheromones = initialize_pheromones(instance, snapshot, 1.0)

    with pytest.raises(ValueError, match="no directed path"):
        construct(instance, snapshot, pheromones=pheromones)


def test_construct_propagates_missing_return_path() -> None:
    instance = make_instance(1, arcs=[RoadArc(1, 2, 1.0)])
    snapshot = make_snapshot((20,))
    pheromones = initialize_pheromones(instance, snapshot, 1.0)

    with pytest.raises(ValueError, match="no directed path"):
        construct(instance, snapshot, pheromones=pheromones)
