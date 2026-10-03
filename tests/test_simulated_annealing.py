import random

import pytest

from src.algorithms.simulated_annealing import (
    create_algorithm_rng,
    metropolis_accept,
    simulated_annealing,
    swap_neighbor,
)
from src.models.candidate_route import CandidateRoute
from src.models.problem import CollectionPoint, Depot, ProblemInstance
from src.models.road_network import RoadArc, RoadNetwork, RoadNode
from src.simulation.snapshot import OptimizationSnapshot


def make_instance(
    point_count: int = 3,
    arcs: list[RoadArc] | None = None,
) -> ProblemInstance:
    points = [
        CollectionPoint(point_id=10 * index, node_id=index)
        for index in range(2, point_count + 2)
    ]
    trajectory = (4.0,)
    volume = (1.0,)
    return ProblemInstance(
        instance_id=1,
        road_network=RoadNetwork(
            network_id=1,
            nodes=[
                RoadNode(node_id=node_id, x=float(node_id), y=0.0)
                for node_id in range(1, point_count + 2)
            ],
            arcs=[] if arcs is None else arcs,
        ),
        depot=Depot(depot_id=1, node_id=1),
        collection_points=points,
        vehicle_capacity=10.0,
        collection_service_duration=60.0,
        v_ref=2.0,
        sample_period=30.0,
        temperature_trajectories={point.point_id: trajectory for point in points},
        volume_trajectories={point.point_id: volume for point in points},
    )


def make_snapshot(
    point_ids: tuple[int, ...],
) -> OptimizationSnapshot:
    return OptimizationSnapshot(
        current_time=0.0,
        current_node_id=1,
        current_load=0.0,
        remaining_point_ids=point_ids,
        temperatures={point_id: 4.0 for point_id in point_ids},
        volumes={point_id: 1.0 for point_id in point_ids},
    )


def make_routing_instance() -> ProblemInstance:
    return make_instance(
        2,
        arcs=[
            RoadArc(1, 2, 2.0),
            RoadArc(1, 3, 4.0),
            RoadArc(2, 3, 2.0),
            RoadArc(3, 2, 2.0),
            RoadArc(2, 1, 6.0),
            RoadArc(3, 1, 6.0),
        ],
    )


def run_sa(
    instance: ProblemInstance,
    snapshot: OptimizationSnapshot,
    initial_route: CandidateRoute,
    *,
    logistic_reference_distance: float = 1.0,
    thermal_reference_value: float = 1.0,
    occupancy_reference_value: float = 1.0,
    t_min: float = 2.0,
    t_max: float = 6.0,
    weights: tuple[float, float, float] = (1.0, 0.0, 0.0),
    initial_temperature: float = 10.0,
    cooling_rate: float = 0.5,
    iterations: int = 0,
    algorithm_seed: int = 123,
) -> CandidateRoute:
    return simulated_annealing(
        instance,
        snapshot,
        initial_route,
        logistic_reference_distance,
        thermal_reference_value,
        occupancy_reference_value,
        t_min,
        t_max,
        weights,
        initial_temperature,
        cooling_rate,
        iterations,
        algorithm_seed,
    )


def test_create_algorithm_rng_accepts_non_negative_integer_seed() -> None:
    rng = create_algorithm_rng(123)

    assert isinstance(rng, random.Random)


def test_create_algorithm_rng_accepts_zero_seed() -> None:
    assert isinstance(create_algorithm_rng(0), random.Random)


@pytest.mark.parametrize("invalid_seed", [-1, True, 1.5, "1"])
def test_create_algorithm_rng_rejects_invalid_seed(invalid_seed: object) -> None:
    with pytest.raises(ValueError, match="algorithm_seed"):
        create_algorithm_rng(invalid_seed)  # type: ignore[arg-type]


def test_same_seed_produces_same_neighbor() -> None:
    instance = make_instance()
    snapshot = make_snapshot((20, 30, 40))
    candidate_route = CandidateRoute((20, 30, 40))

    neighbor_a = swap_neighbor(
        instance,
        snapshot,
        candidate_route,
        create_algorithm_rng(123),
    )
    neighbor_b = swap_neighbor(
        instance,
        snapshot,
        candidate_route,
        create_algorithm_rng(123),
    )

    assert neighbor_a == neighbor_b


def test_empty_route_produces_empty_neighbor() -> None:
    instance = make_instance(0)
    snapshot = make_snapshot(())

    neighbor = swap_neighbor(instance, snapshot, CandidateRoute(()), random.Random(1))

    assert neighbor == CandidateRoute(())


def test_single_point_route_remains_unchanged() -> None:
    instance = make_instance(1)
    snapshot = make_snapshot((20,))
    candidate_route = CandidateRoute((20,))

    neighbor = swap_neighbor(instance, snapshot, candidate_route, random.Random(1))

    assert neighbor == candidate_route


def test_two_point_route_swaps_both_positions() -> None:
    instance = make_instance(2)
    snapshot = make_snapshot((20, 30))

    neighbor = swap_neighbor(
        instance,
        snapshot,
        CandidateRoute((20, 30)),
        random.Random(1),
    )

    assert neighbor.point_ids == (30, 20)


def test_longer_route_changes_exactly_two_positions() -> None:
    instance = make_instance()
    snapshot = make_snapshot((20, 30, 40))
    original = (20, 30, 40)

    neighbor = swap_neighbor(
        instance,
        snapshot,
        CandidateRoute(original),
        random.Random(1),
    )

    changed_positions = sum(
        original_point_id != neighbor_point_id
        for original_point_id, neighbor_point_id in zip(original, neighbor.point_ids)
    )
    assert changed_positions == 2


def test_swap_preserves_all_point_ids_exactly_once() -> None:
    instance = make_instance()
    snapshot = make_snapshot((20, 30, 40))

    neighbor = swap_neighbor(
        instance,
        snapshot,
        CandidateRoute((20, 30, 40)),
        random.Random(1),
    )

    assert sorted(neighbor.point_ids) == [20, 30, 40]
    assert len(neighbor.point_ids) == len(set(neighbor.point_ids))


def test_original_route_remains_unchanged() -> None:
    instance = make_instance()
    snapshot = make_snapshot((20, 30, 40))
    candidate_route = CandidateRoute((20, 30, 40))

    swap_neighbor(instance, snapshot, candidate_route, random.Random(1))

    assert candidate_route.point_ids == (20, 30, 40)


def test_neighbor_is_a_validated_candidate_route() -> None:
    instance = make_instance()
    snapshot = make_snapshot((20, 30, 40))

    neighbor = swap_neighbor(
        instance,
        snapshot,
        CandidateRoute((20, 30, 40)),
        random.Random(1),
    )

    assert isinstance(neighbor, CandidateRoute)
    assert neighbor.validate(instance, snapshot) is None


def test_successive_calls_advance_the_same_rng_state() -> None:
    instance = make_instance()
    snapshot = make_snapshot((20, 30, 40))
    candidate_route = CandidateRoute((20, 30, 40))
    actual_rng = create_algorithm_rng(123)
    expected_rng = create_algorithm_rng(123)

    actual_first = swap_neighbor(instance, snapshot, candidate_route, actual_rng)
    actual_second = swap_neighbor(instance, snapshot, candidate_route, actual_rng)
    expected_first = swap_neighbor(instance, snapshot, candidate_route, expected_rng)
    expected_second = swap_neighbor(instance, snapshot, candidate_route, expected_rng)

    assert actual_first == expected_first
    assert actual_second == expected_second


def test_invalid_route_is_rejected_before_rng_is_used() -> None:
    instance = make_instance(2)
    snapshot = make_snapshot((20, 30))
    rng = create_algorithm_rng(123)
    state_before = rng.getstate()

    with pytest.raises(ValueError, match="missing"):
        swap_neighbor(instance, snapshot, CandidateRoute((20,)), rng)

    assert rng.getstate() == state_before


@pytest.mark.parametrize("delta_cost", [-1.0, 0.0])
def test_metropolis_accepts_non_worse_cost_without_consuming_rng(
    delta_cost: float,
) -> None:
    rng = random.Random(123)
    state_before = rng.getstate()

    assert metropolis_accept(delta_cost, 1.0, rng) is True
    assert rng.getstate() == state_before


def test_metropolis_can_accept_a_worse_solution() -> None:
    rng = random.Random(123)

    assert metropolis_accept(1.0, 1.0, rng) is True


def test_metropolis_can_reject_a_worse_solution() -> None:
    rng = random.Random(0)

    assert metropolis_accept(1.0, 1.0, rng) is False


def test_higher_temperature_has_higher_acceptance_probability() -> None:
    high_temperature_rng = random.Random(0)
    low_temperature_rng = random.Random(0)

    high_temperature_result = metropolis_accept(1.0, 10.0, high_temperature_rng)
    low_temperature_result = metropolis_accept(1.0, 1.0, low_temperature_rng)

    assert high_temperature_result is True
    assert low_temperature_result is False


@pytest.mark.parametrize(
    "delta_cost",
    [float("nan"), float("inf"), True, "1"],
)
def test_metropolis_rejects_invalid_delta_cost(delta_cost: object) -> None:
    with pytest.raises(ValueError, match="delta_cost"):
        metropolis_accept(delta_cost, 1.0, random.Random(1))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "temperature",
    [0.0, -1.0, float("nan"), float("inf"), True, "1"],
)
def test_metropolis_rejects_invalid_temperature(temperature: object) -> None:
    with pytest.raises(ValueError, match="temperature"):
        metropolis_accept(1.0, temperature, random.Random(1))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "initial_temperature",
    [0.0, -1.0, float("nan"), float("inf"), True, "1"],
)
def test_simulated_annealing_rejects_invalid_initial_temperature(
    initial_temperature: object,
) -> None:
    instance = make_routing_instance()
    snapshot = make_snapshot((20, 30))

    with pytest.raises(ValueError, match="initial_temperature"):
        run_sa(
            instance,
            snapshot,
            CandidateRoute((20, 30)),
            initial_temperature=initial_temperature,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("cooling_rate", [0.0, -0.1])
def test_simulated_annealing_rejects_non_positive_cooling_rate(
    cooling_rate: float,
) -> None:
    instance = make_routing_instance()
    snapshot = make_snapshot((20, 30))

    with pytest.raises(ValueError, match="cooling_rate"):
        run_sa(
            instance,
            snapshot,
            CandidateRoute((20, 30)),
            cooling_rate=cooling_rate,
        )


@pytest.mark.parametrize("cooling_rate", [1.0, 1.1])
def test_simulated_annealing_rejects_cooling_rate_at_or_above_one(
    cooling_rate: float,
) -> None:
    instance = make_routing_instance()
    snapshot = make_snapshot((20, 30))

    with pytest.raises(ValueError, match="cooling_rate"):
        run_sa(
            instance,
            snapshot,
            CandidateRoute((20, 30)),
            cooling_rate=cooling_rate,
        )


@pytest.mark.parametrize("iterations", [-1, 1.5, True])
def test_simulated_annealing_rejects_invalid_iterations(
    iterations: object,
) -> None:
    instance = make_routing_instance()
    snapshot = make_snapshot((20, 30))

    with pytest.raises(ValueError, match="iterations"):
        run_sa(
            instance,
            snapshot,
            CandidateRoute((20, 30)),
            iterations=iterations,  # type: ignore[arg-type]
        )


def test_zero_iterations_returns_initial_route() -> None:
    instance = make_routing_instance()
    snapshot = make_snapshot((20, 30))
    initial_route = CandidateRoute((30, 20))

    result = run_sa(instance, snapshot, initial_route, iterations=0)

    assert result == initial_route


def test_simulated_annealing_supports_empty_route() -> None:
    instance = make_instance(0)
    snapshot = make_snapshot(())

    result = run_sa(
        instance,
        snapshot,
        CandidateRoute(()),
        logistic_reference_distance=0.0,
        thermal_reference_value=0.0,
        occupancy_reference_value=0.0,
        iterations=1,
    )

    assert result == CandidateRoute(())


def test_simulated_annealing_supports_one_point_route() -> None:
    instance = make_instance(
        1,
        arcs=[RoadArc(1, 2, 4.0), RoadArc(2, 1, 6.0)],
    )
    snapshot = make_snapshot((20,))
    initial_route = CandidateRoute((20,))

    result = run_sa(instance, snapshot, initial_route, iterations=2)

    assert result == initial_route


def test_same_seed_and_inputs_produce_same_sa_result() -> None:
    instance = make_routing_instance()
    snapshot = make_snapshot((20, 30))
    initial_route = CandidateRoute((20, 30))

    first_result = run_sa(
        instance,
        snapshot,
        initial_route,
        iterations=5,
        algorithm_seed=123,
    )
    second_result = run_sa(
        instance,
        snapshot,
        initial_route,
        iterations=5,
        algorithm_seed=123,
    )

    assert first_result == second_result


def test_sa_preserves_point_ids_and_returns_valid_route() -> None:
    instance = make_routing_instance()
    snapshot = make_snapshot((20, 30))

    result = run_sa(
        instance,
        snapshot,
        CandidateRoute((20, 30)),
        iterations=5,
    )

    assert sorted(result.point_ids) == [20, 30]
    assert result.validate(instance, snapshot) is None


def test_sa_can_return_a_route_better_than_the_initial_route() -> None:
    instance = make_routing_instance()
    snapshot = make_snapshot((20, 30))
    initial_route = CandidateRoute((30, 20))

    result = run_sa(
        instance,
        snapshot,
        initial_route,
        iterations=1,
    )

    assert result == CandidateRoute((20, 30))


def test_accepted_worse_neighbor_does_not_replace_best_route() -> None:
    instance = make_routing_instance()
    snapshot = make_snapshot((20, 30))
    initial_route = CandidateRoute((20, 30))

    result = run_sa(
        instance,
        snapshot,
        initial_route,
        initial_temperature=1e308,
        iterations=1,
    )

    assert result == initial_route


def test_external_reference_values_are_used() -> None:
    instance = make_routing_instance()
    snapshot = make_snapshot((20, 30))

    with pytest.raises(ValueError, match="reference_distance"):
        run_sa(
            instance,
            snapshot,
            CandidateRoute((20, 30)),
            logistic_reference_distance=0.0,
            iterations=0,
        )


def test_invalid_reference_error_is_propagated() -> None:
    instance = make_routing_instance()
    snapshot = make_snapshot((20, 30))

    with pytest.raises(ValueError, match="reference"):
        run_sa(
            instance,
            snapshot,
            CandidateRoute((20, 30)),
            thermal_reference_value=-1.0,
            iterations=0,
        )


def test_invalid_initial_route_error_is_propagated() -> None:
    instance = make_routing_instance()
    snapshot = make_snapshot((20, 30))

    with pytest.raises(ValueError, match="missing"):
        run_sa(instance, snapshot, CandidateRoute((20,)), iterations=0)


def test_invalid_weights_are_propagated_by_objective() -> None:
    instance = make_routing_instance()
    snapshot = make_snapshot((20, 30))

    with pytest.raises(ValueError, match="weights"):
        run_sa(
            instance,
            snapshot,
            CandidateRoute((20, 30)),
            weights=(0.5, 0.2, 0.2),
            iterations=0,
        )
