import math

import pytest

from src.algorithms.ant_colony import (
    ant_count,
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
                RoadNode(node_id=node_id, x=float(node_id), y=0.0)
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
        temperature_trajectories={point.point_id: (4.0,) for point in points},
        volume_trajectories={point.point_id: (1.0,) for point in points},
    )


def make_snapshot(
    point_ids: tuple[int, ...],
    *,
    current_node_id: int = 1,
) -> OptimizationSnapshot:
    return OptimizationSnapshot(
        current_time=0.0,
        current_node_id=current_node_id,
        current_load=0.0,
        remaining_point_ids=point_ids,
        temperatures={point_id: 4.0 for point_id in point_ids},
        volumes={point_id: 1.0 for point_id in point_ids},
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
