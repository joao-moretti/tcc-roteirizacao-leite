import pytest

import src.experiments.planners as planners_module
from src.experiments.algorithm_presets import (
    ACO_BETA,
    ACO_EVAPORATION_RATE,
    SA_COOLING_RATE,
    SA_INITIAL_TEMPERATURE,
    SA_ITERATIONS,
)
from src.experiments.planners import (
    EXPERIMENTAL_ALGORITHM_SEEDS,
    make_aco_planner,
    make_classical_nn_planner,
    make_multicriteria_nn_planner,
    make_sa_planner,
)
from src.models.candidate_route import CandidateRoute
from src.models.problem import CollectionPoint, Depot, ProblemInstance
from src.models.road_network import RoadArc, RoadNetwork, RoadNode
from src.simulation.snapshot import OptimizationSnapshot


def make_instance() -> ProblemInstance:
    points = (
        CollectionPoint(point_id=10, node_id=2),
        CollectionPoint(point_id=20, node_id=3),
        CollectionPoint(point_id=30, node_id=4),
    )
    arcs = [
        RoadArc(1, 2, 2.0),
        RoadArc(1, 3, 4.0),
        RoadArc(1, 4, 6.0),
        RoadArc(2, 3, 1.0),
        RoadArc(2, 4, 3.0),
        RoadArc(3, 4, 1.0),
        RoadArc(2, 1, 5.0),
        RoadArc(3, 1, 5.0),
        RoadArc(4, 1, 5.0),
        RoadArc(3, 2, 2.0),
        RoadArc(4, 2, 2.0),
        RoadArc(4, 3, 2.0),
    ]
    trajectory = (3.0,)
    volumes = (1.0,)
    return ProblemInstance(
        instance_id=1,
        road_network=RoadNetwork(
            network_id=1,
            nodes=[
                RoadNode(node_id=node_id, x=float(node_id), y=0.0)
                for node_id in range(1, 5)
            ],
            arcs=arcs,
        ),
        depot=Depot(depot_id=1, node_id=1),
        collection_points=points,
        vehicle_capacity=10.0,
        collection_service_duration=10.0,
        v_ref=1.0,
        sample_period=30.0,
        temperature_trajectories={point.point_id: trajectory for point in points},
        volume_trajectories={point.point_id: volumes for point in points},
    )


def make_snapshot(
    instance: ProblemInstance,
    remaining_point_ids: tuple[int, ...] = (10, 20, 30),
    *,
    current_node_id: int = 1,
    temperatures: dict[int, float] | None = None,
    volumes: dict[int, float] | None = None,
) -> OptimizationSnapshot:
    if temperatures is None:
        temperatures = {point_id: 3.0 for point_id in remaining_point_ids}
    if volumes is None:
        volumes = {point_id: 1.0 for point_id in remaining_point_ids}
    snapshot = OptimizationSnapshot(
        current_time=0.0,
        current_node_id=current_node_id,
        current_load=0.0,
        remaining_point_ids=remaining_point_ids,
        temperatures=temperatures,
        volumes=volumes,
    )
    snapshot.validate(instance)
    return snapshot


def test_official_algorithm_seeds_are_thirty_unique_non_negative_ints() -> None:
    assert len(EXPERIMENTAL_ALGORITHM_SEEDS) == 30
    assert len(set(EXPERIMENTAL_ALGORITHM_SEEDS)) == 30
    assert all(
        isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0
        for seed in EXPERIMENTAL_ALGORITHM_SEEDS
    )


def test_classical_planner_freezes_order_and_filters_remaining_points() -> None:
    instance = make_instance()
    planner = make_classical_nn_planner(instance)
    initial_snapshot = make_snapshot(instance)

    initial_route = planner(instance, initial_snapshot)
    after_first_collection = make_snapshot(instance, (20, 30), current_node_id=2)
    remaining_route = planner(instance, after_first_collection)

    assert initial_route == CandidateRoute((10, 20, 30))
    assert remaining_route == CandidateRoute((20, 30))
    assert remaining_route.validate(instance, after_first_collection) is None


def test_classical_planner_does_not_recompute_after_factory_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = make_instance()
    planner = make_classical_nn_planner(instance)

    def fail_if_called(_instance: ProblemInstance) -> object:
        raise AssertionError("classical NN was recomputed")

    monkeypatch.setattr(
        planners_module,
        "classical_nearest_neighbor",
        fail_if_called,
    )
    snapshot = make_snapshot(instance, (20, 30), current_node_id=2)

    assert planner(instance, snapshot) == CandidateRoute((20, 30))


def test_classical_planner_ignores_snapshot_temperature_and_volume_changes() -> None:
    instance = make_instance()
    planner = make_classical_nn_planner(instance)
    cool_snapshot = make_snapshot(
        instance,
        temperatures={10: 3.0, 20: 3.0, 30: 3.0},
        volumes={10: 1.0, 20: 1.0, 30: 1.0},
    )
    changed_snapshot = make_snapshot(
        instance,
        temperatures={10: 9.0, 20: -2.0, 30: 8.0},
        volumes={10: 9.0, 20: 1.0, 30: 7.0},
    )

    assert planner(instance, cool_snapshot) == planner(instance, changed_snapshot)


def test_multicriteria_nn_planner_returns_valid_route() -> None:
    instance = make_instance()
    snapshot = make_snapshot(instance)

    route = make_multicriteria_nn_planner((1 / 3, 1 / 3, 1 / 3))(
        instance,
        snapshot,
    )

    assert isinstance(route, CandidateRoute)
    assert route.validate(instance, snapshot) is None


def test_multicriteria_nn_profiles_can_change_the_selected_point() -> None:
    instance = make_instance()
    snapshot = make_snapshot(
        instance,
        temperatures={10: 6.0, 20: 3.0, 30: 3.0},
        volumes={10: 1.0, 20: 5.0, 30: 1.0},
    )

    thermal_route = make_multicriteria_nn_planner((0.0, 1.0, 0.0))(
        instance,
        snapshot,
    )
    occupancy_route = make_multicriteria_nn_planner((0.0, 0.0, 1.0))(
        instance,
        snapshot,
    )

    assert thermal_route.point_ids[0] == 10
    assert occupancy_route.point_ids[0] == 20


@pytest.mark.parametrize("invalid_seed", [-1, True, 1.5, "1"])
def test_stochastic_planners_reject_invalid_run_seed(invalid_seed: object) -> None:
    with pytest.raises(ValueError, match="run_seed"):
        make_sa_planner((1 / 3, 1 / 3, 1 / 3), invalid_seed)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="run_seed"):
        make_aco_planner((1 / 3, 1 / 3, 1 / 3), invalid_seed)  # type: ignore[arg-type]


def test_sa_planner_uses_official_parameters_and_returns_valid_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = make_instance()
    snapshot = make_snapshot(instance)
    received: list[tuple[object, ...]] = []

    def fake_sa(*args: object) -> CandidateRoute:
        received.append(args)
        return args[2]  # type: ignore[return-value]

    monkeypatch.setattr(planners_module, "simulated_annealing", fake_sa)
    route = make_sa_planner((1 / 3, 1 / 3, 1 / 3), 7)(instance, snapshot)

    assert route.validate(instance, snapshot) is None
    assert received[0][9:12] == (
        SA_INITIAL_TEMPERATURE,
        SA_COOLING_RATE,
        SA_ITERATIONS,
    )


def test_aco_planner_uses_official_parameters_and_baseline_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = make_instance()
    snapshot = make_snapshot(instance)
    objective_calls: list[tuple[object, ...]] = []
    aco_calls: list[tuple[object, ...]] = []

    def fake_objective(*args: object) -> float:
        objective_calls.append(args)
        return 12.5

    def fake_aco(*args: object) -> CandidateRoute:
        aco_calls.append(args)
        return CandidateRoute(snapshot.remaining_point_ids)

    monkeypatch.setattr(planners_module, "weighted_objective_cost", fake_objective)
    monkeypatch.setattr(planners_module, "ant_colony_optimization", fake_aco)
    route = make_aco_planner((1 / 3, 1 / 3, 1 / 3), 7)(instance, snapshot)

    assert route.validate(instance, snapshot) is None
    assert len(objective_calls) == 1
    assert aco_calls[0][2] == 12.5
    assert aco_calls[0][9:11] == (ACO_BETA, ACO_EVAPORATION_RATE)


def test_stochastic_planners_use_different_child_seeds_on_consecutive_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = make_instance()
    snapshot = make_snapshot(instance)
    sa_seeds: list[int] = []
    aco_seeds: list[int] = []

    def fake_sa(*args: object) -> CandidateRoute:
        sa_seeds.append(args[-1])  # type: ignore[arg-type]
        return args[2]  # type: ignore[return-value]

    def fake_aco(*args: object) -> CandidateRoute:
        aco_seeds.append(args[-1])  # type: ignore[arg-type]
        return CandidateRoute(snapshot.remaining_point_ids)

    monkeypatch.setattr(planners_module, "simulated_annealing", fake_sa)
    monkeypatch.setattr(planners_module, "ant_colony_optimization", fake_aco)
    sa_planner = make_sa_planner((1 / 3, 1 / 3, 1 / 3), 9)
    aco_planner = make_aco_planner((1 / 3, 1 / 3, 1 / 3), 9)
    sa_planner(instance, snapshot)
    sa_planner(instance, snapshot)
    aco_planner(instance, snapshot)
    aco_planner(instance, snapshot)

    assert sa_seeds[0] != sa_seeds[1]
    assert aco_seeds[0] != aco_seeds[1]


def test_independent_stochastic_planners_derive_the_same_seed_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = make_instance()
    snapshot = make_snapshot(instance)
    sa_sequences: list[list[int]] = [[], []]
    aco_sequences: list[list[int]] = [[], []]

    def fake_sa_factory(sequence: list[int]):
        def fake_sa(*args: object) -> CandidateRoute:
            sequence.append(args[-1])  # type: ignore[arg-type]
            return args[2]  # type: ignore[return-value]

        return fake_sa

    def fake_aco_factory(sequence: list[int]):
        def fake_aco(*args: object) -> CandidateRoute:
            sequence.append(args[-1])  # type: ignore[arg-type]
            return CandidateRoute(snapshot.remaining_point_ids)

        return fake_aco

    monkeypatch.setattr(
        planners_module,
        "simulated_annealing",
        fake_sa_factory(sa_sequences[0]),
    )
    first_sa = make_sa_planner((1 / 3, 1 / 3, 1 / 3), 11)
    first_sa(instance, snapshot)
    first_sa(instance, snapshot)

    monkeypatch.setattr(
        planners_module,
        "simulated_annealing",
        fake_sa_factory(sa_sequences[1]),
    )
    second_sa = make_sa_planner((1 / 3, 1 / 3, 1 / 3), 11)
    second_sa(instance, snapshot)
    second_sa(instance, snapshot)

    monkeypatch.setattr(
        planners_module,
        "ant_colony_optimization",
        fake_aco_factory(aco_sequences[0]),
    )
    first_aco = make_aco_planner((1 / 3, 1 / 3, 1 / 3), 11)
    first_aco(instance, snapshot)
    first_aco(instance, snapshot)

    monkeypatch.setattr(
        planners_module,
        "ant_colony_optimization",
        fake_aco_factory(aco_sequences[1]),
    )
    second_aco = make_aco_planner((1 / 3, 1 / 3, 1 / 3), 11)
    second_aco(instance, snapshot)
    second_aco(instance, snapshot)

    assert sa_sequences[0] == sa_sequences[1]
    assert aco_sequences[0] == aco_sequences[1]


def test_dynamic_references_are_recalculated_for_each_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = make_instance()
    first_snapshot = make_snapshot(instance)
    second_snapshot = make_snapshot(instance, (20, 30), current_node_id=2)
    seen_snapshots: list[OptimizationSnapshot] = []

    def record_logistic(
        _instance: ProblemInstance,
        snapshot: OptimizationSnapshot,
    ) -> float:
        seen_snapshots.append(snapshot)
        return 1.0

    monkeypatch.setattr(
        planners_module,
        "dynamic_logistic_reference",
        record_logistic,
    )
    monkeypatch.setattr(
        planners_module,
        "thermal_reference",
        lambda *_args: 2.0,
    )
    monkeypatch.setattr(
        planners_module,
        "occupancy_reference",
        lambda *_args: 3.0,
    )
    monkeypatch.setattr(
        planners_module,
        "simulated_annealing",
        lambda *args: args[2],
    )
    planner = make_sa_planner((1 / 3, 1 / 3, 1 / 3), 13)

    planner(instance, first_snapshot)
    planner(instance, second_snapshot)

    assert seen_snapshots == [first_snapshot, second_snapshot]


def test_stochastic_planners_do_not_mutate_instance_or_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = make_instance()
    snapshot = make_snapshot(instance)
    monkeypatch.setattr(
        planners_module,
        "simulated_annealing",
        lambda *args: args[2],
    )
    monkeypatch.setattr(
        planners_module,
        "ant_colony_optimization",
        lambda *args: CandidateRoute(snapshot.remaining_point_ids),
    )
    sa_planner = make_sa_planner((1 / 3, 1 / 3, 1 / 3), 17)
    aco_planner = make_aco_planner((1 / 3, 1 / 3, 1 / 3), 17)

    assert sa_planner(instance, snapshot).validate(instance, snapshot) is None
    assert aco_planner(instance, snapshot).validate(instance, snapshot) is None
    assert instance.collection_points == make_instance().collection_points
    assert snapshot.remaining_point_ids == (10, 20, 30)
