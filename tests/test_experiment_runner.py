from types import SimpleNamespace

import pytest

import src.experiments.runner as runner_module
from src.experiments.planners import EXPERIMENTAL_ALGORITHM_SEEDS
from src.experiments.runner import (
    CollectionRecord,
    ExperimentalRunResult,
    run_experimental_case,
)
from src.models.problem import CollectionPoint, Depot, ProblemInstance
from src.models.road_network import RoadArc, RoadNetwork, RoadNode
from src.simulation.dynamic import DynamicSimulationResult
from src.simulation.events import RecalculationDecision
from src.simulation.executor import NextCollectionExecution
from src.simulation.snapshot import OptimizationSnapshot


def make_instance(instance_id: int = 1) -> ProblemInstance:
    points = (
        CollectionPoint(point_id=10, node_id=2),
        CollectionPoint(point_id=20, node_id=3),
    )
    return ProblemInstance(
        instance_id=instance_id,
        road_network=RoadNetwork(
            network_id=instance_id,
            nodes=[
                RoadNode(node_id=1, x=0.0, y=0.0),
                RoadNode(node_id=2, x=100.0, y=0.0),
                RoadNode(node_id=3, x=200.0, y=0.0),
            ],
            arcs=[
                RoadArc(1, 2, 4.0),
                RoadArc(2, 3, 6.0),
                RoadArc(1, 3, 20.0),
                RoadArc(3, 2, 2.0),
                RoadArc(2, 1, 3.0),
                RoadArc(3, 1, 5.0),
            ],
        ),
        depot=Depot(depot_id=1, node_id=1),
        collection_points=points,
        vehicle_capacity=1000.0,
        collection_service_duration=5.0,
        v_ref=1.0,
        sample_period=30.0,
        temperature_trajectories={
            10: (3.0, 4.0, 5.0, 6.0),
            20: (3.0, 5.0, 7.0, 8.0),
        },
        volume_trajectories={
            10: (2.0, 2.0, 2.0, 2.0),
            20: (3.0, 3.0, 3.0, 3.0),
        },
    )


def make_snapshot(
    instance: ProblemInstance,
    current_time: float,
    current_node_id: int,
    remaining_point_ids: tuple[int, ...],
) -> OptimizationSnapshot:
    return OptimizationSnapshot(
        current_time=current_time,
        current_node_id=current_node_id,
        current_load=0.0,
        remaining_point_ids=remaining_point_ids,
        temperatures={point_id: 3.0 for point_id in remaining_point_ids},
        volumes={point_id: 2.0 for point_id in remaining_point_ids},
    )


def make_simulation_result(instance: ProblemInstance) -> DynamicSimulationResult:
    interrupted = NextCollectionExecution(
        target_point_id=20,
        reached_target=False,
        collected_volume=None,
        distance_traveled=1.0,
        snapshot=make_snapshot(instance, 15.0, 1, (10, 20)),
        decision=RecalculationDecision(True, ("volume_band_changed",)),
    )
    first_collection = NextCollectionExecution(
        target_point_id=10,
        reached_target=True,
        collected_volume=2.0,
        distance_traveled=4.0,
        snapshot=make_snapshot(instance, 30.0, 2, (20,)),
        decision=RecalculationDecision(
            True,
            ("collection_completed", "thermal_band_changed"),
        ),
    )
    second_collection = NextCollectionExecution(
        target_point_id=20,
        reached_target=True,
        collected_volume=3.0,
        distance_traveled=6.0,
        snapshot=make_snapshot(instance, 60.0, 3, ()),
        decision=RecalculationDecision(
            True,
            ("collection_completed", "volume_band_changed"),
        ),
    )
    final_snapshot = OptimizationSnapshot(
        current_time=60.0,
        current_node_id=instance.depot.node_id,
        current_load=5.0,
        remaining_point_ids=(),
        temperatures={},
        volumes={},
    )
    return DynamicSimulationResult(
        final_snapshot=final_snapshot,
        total_distance=123.0,
        optimization_count=7,
        executions=(interrupted, first_collection, second_collection),
    )


def patch_fake_case(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[SimpleNamespace, list[dict[str, object]], ProblemInstance, ProblemInstance]:
    nominal_instance = make_instance(1)
    disturbance_instance = make_instance(2)
    pair = SimpleNamespace(
        point_count=2,
        replicate_index=1,
        instance_seed=501,
        nominal_instance=nominal_instance,
        disturbance_instance=disturbance_instance,
    )
    monkeypatch.setattr(
        runner_module,
        "generate_experimental_instance_pair",
        lambda _point_count, _replicate_index: pair,
    )
    simulation_calls: list[dict[str, object]] = []

    def fake_simulation(instance: ProblemInstance, planner: object, **kwargs: object):
        simulation_calls.append(
            {
                "instance": instance,
                "planner": planner,
                **kwargs,
            }
        )
        return make_simulation_result(instance)

    monkeypatch.setattr(runner_module, "run_dynamic_simulation", fake_simulation)
    return pair, simulation_calls, nominal_instance, disturbance_instance


def patch_fake_factories(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, tuple[object, ...]]]:
    factory_calls: list[tuple[str, tuple[object, ...]]] = []

    def fake_planner(*_args: object):
        return lambda _instance, _snapshot: None

    monkeypatch.setattr(
        runner_module,
        "make_classical_nn_planner",
        lambda *args: factory_calls.append(("classical_nn", args)) or fake_planner(),
    )
    monkeypatch.setattr(
        runner_module,
        "make_multicriteria_nn_planner",
        lambda *args: factory_calls.append(("mcnn", args)) or fake_planner(),
    )
    monkeypatch.setattr(
        runner_module,
        "make_sa_planner",
        lambda *args: factory_calls.append(("sa", args)) or fake_planner(),
    )
    monkeypatch.setattr(
        runner_module,
        "make_aco_planner",
        lambda *args: factory_calls.append(("aco", args)) or fake_planner(),
    )
    return factory_calls


def test_experimental_run_result_is_immutable_and_has_collection_record_type() -> None:
    assert ExperimentalRunResult.__dataclass_params__.frozen is True
    assert CollectionRecord.__dataclass_params__.frozen is True


def test_invalid_algorithm_and_scenario_are_rejected() -> None:
    with pytest.raises(ValueError, match="algorithm"):
        run_experimental_case(5, 1, "nominal", "invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="scenario"):
        run_experimental_case(5, 1, "invalid", "classical_nn")  # type: ignore[arg-type]


def test_weight_profile_rules_are_rejected_before_execution() -> None:
    with pytest.raises(ValueError, match="classical_nn"):
        run_experimental_case(
            5,
            1,
            "nominal",
            "classical_nn",
            weight_profile="balanced",
        )
    with pytest.raises(ValueError, match="weight_profile"):
        run_experimental_case(5, 1, "nominal", "mcnn")
    with pytest.raises(ValueError, match="weight_profile"):
        run_experimental_case(
            5,
            1,
            "nominal",
            "sa",
            run_seed=1,
        )
    with pytest.raises(ValueError, match="weight_profile"):
        run_experimental_case(
            5,
            1,
            "nominal",
            "aco",
            run_seed=1,
        )


def test_run_seed_rules_are_rejected() -> None:
    with pytest.raises(ValueError, match="run_seed"):
        run_experimental_case(5, 1, "nominal", "classical_nn", run_seed=1)
    with pytest.raises(ValueError, match="run_seed"):
        run_experimental_case(
            5, 1, "nominal", "mcnn", weight_profile="balanced", run_seed=1
        )
    with pytest.raises(ValueError, match="run_seed"):
        run_experimental_case(5, 1, "nominal", "sa", weight_profile="balanced")

    for invalid_seed in (0, 31, True, 1.5, "1"):
        with pytest.raises(ValueError, match="run_seed"):
            run_experimental_case(
                5,
                1,
                "nominal",
                "sa",
                weight_profile="balanced",
                run_seed=invalid_seed,  # type: ignore[arg-type]
            )
        with pytest.raises(ValueError, match="run_seed"):
            run_experimental_case(
                5,
                1,
                "nominal",
                "aco",
                weight_profile="balanced",
                run_seed=invalid_seed,  # type: ignore[arg-type]
            )


@pytest.mark.parametrize(
    ("algorithm", "weight_profile", "run_seed", "factory_name"),
    [
        ("classical_nn", None, None, "classical_nn"),
        ("mcnn", "balanced", None, "mcnn"),
        ("sa", "logistic", 7, "sa"),
        ("aco", "thermal", 7, "aco"),
    ],
)
def test_correct_planner_scenario_and_metadata_are_selected(
    monkeypatch: pytest.MonkeyPatch,
    algorithm: str,
    weight_profile: str | None,
    run_seed: int | None,
    factory_name: str,
) -> None:
    pair, simulation_calls, nominal_instance, disturbance_instance = patch_fake_case(
        monkeypatch
    )
    factory_calls = patch_fake_factories(monkeypatch)

    result = run_experimental_case(
        2,
        1,
        "disturbance",
        algorithm,  # type: ignore[arg-type]
        weight_profile=weight_profile,  # type: ignore[arg-type]
        run_seed=run_seed,
    )

    assert simulation_calls[0]["instance"] is disturbance_instance
    assert pair.disturbance_instance is disturbance_instance
    assert factory_calls[0][0] == factory_name
    assert result.scenario == "disturbance"
    assert result.algorithm == algorithm
    assert result.instance_seed == 501


def test_planner_creation_is_inside_computational_measurement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_fake_case(monkeypatch)
    events: list[str] = []
    clock_values = iter((10.0, 12.5))

    def fake_perf_counter() -> float:
        events.append("timer")
        return next(clock_values)

    def fake_planner_factory(_instance: ProblemInstance):
        events.append("planner")
        return lambda _instance, _snapshot: None

    def fake_simulation(
        instance: ProblemInstance,
        _planner: object,
        **_kwargs: object,
    ) -> DynamicSimulationResult:
        events.append("simulation")
        return make_simulation_result(instance)

    monkeypatch.setattr(runner_module.time, "perf_counter", fake_perf_counter)
    monkeypatch.setattr(
        runner_module,
        "make_classical_nn_planner",
        fake_planner_factory,
    )
    monkeypatch.setattr(runner_module, "run_dynamic_simulation", fake_simulation)

    result = run_experimental_case(2, 1, "nominal", "classical_nn")

    assert events == ["timer", "planner", "simulation", "timer"]
    assert result.computational_seconds == 2.5


def test_official_dynamic_parameters_are_forwarded() -> None:
    assert len(EXPERIMENTAL_ALGORITHM_SEEDS) == 30


def test_run_collects_metrics_records_reasons_and_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, simulation_calls, nominal_instance, _ = patch_fake_case(monkeypatch)
    patch_fake_factories(monkeypatch)

    result = run_experimental_case(
        2,
        1,
        "nominal",
        "classical_nn",
    )

    assert simulation_calls[0]["instance"] is nominal_instance
    assert result.total_distance == 123.0
    assert result.simulated_duration == 60.0
    assert result.computational_seconds >= 0.0
    assert result.final_load == 5.0
    assert result.planner_call_count == 7
    assert result.effective_optimization_count == 1
    assert result.interruption_count == 1
    assert result.collection_recalculation_count == 2
    assert result.thermal_recalculation_count == 1
    assert result.volume_recalculation_count == 2
    assert result.collection_order == (10, 20)
    assert result.collections == (
        CollectionRecord(10, 30.0, 2.0, 4.0),
        CollectionRecord(20, 60.0, 3.0, 7.0),
    )


def test_all_dynamic_algorithms_use_planner_call_count_as_effective_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for algorithm in ("mcnn", "sa", "aco"):
        patch_fake_case(monkeypatch)
        patch_fake_factories(monkeypatch)
        result = run_experimental_case(
            2,
            1,
            "nominal",
            algorithm,  # type: ignore[arg-type]
            weight_profile="balanced",
            run_seed=1 if algorithm in ("sa", "aco") else None,
        )
        assert result.effective_optimization_count == result.planner_call_count


def test_official_dynamic_parameters_are_sent_to_simulator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, simulation_calls, _, _ = patch_fake_case(monkeypatch)
    patch_fake_factories(monkeypatch)

    run_experimental_case(2, 1, "nominal", "classical_nn")

    assert {
        key: simulation_calls[0][key]
        for key in (
            "t_min",
            "t_max",
            "q_low",
            "q_high",
            "minimum_interval",
        )
    } == {
        "t_min": 4.0,
        "t_max": 7.0,
        "q_low": 300.0,
        "q_high": 500.0,
        "minimum_interval": 300.0,
    }


def test_runner_does_not_modify_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, nominal_instance, _ = patch_fake_case(monkeypatch)
    patch_fake_factories(monkeypatch)
    before = nominal_instance

    run_experimental_case(2, 1, "nominal", "classical_nn")

    assert nominal_instance == before
