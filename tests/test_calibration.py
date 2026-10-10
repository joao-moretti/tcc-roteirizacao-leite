import math

import pytest

from src.algorithms.nearest_neighbor import multicriteria_nearest_neighbor
from src.experiments.calibration import (
    ACO_BETAS,
    ACO_CALIBRATION_GRID,
    CALIBRATION_ALGORITHM_SEEDS,
    CALIBRATION_INSTANCE_SEEDS_BY_POINT_COUNT,
    CALIBRATION_POINT_COUNTS,
    SA_CALIBRATION_GRID,
    SA_COOLING_RATES,
    SA_INITIAL_TEMPERATURES,
    SA_ITERATIONS,
    ACOParameterSet,
    CalibrationRun,
    CalibrationSummary,
    SAParameterSet,
    _generate_pilot_instances,
    _prepare_case,
    _summarize_runs,
    _summary_sort_key,
    evaluate_aco_configuration,
    evaluate_sa_configuration,
    run_sa_calibration,
)
from src.experiments.experimental_instances import (
    EXPERIMENTAL_INSTANCE_SEEDS_BY_POINT_COUNT,
)
from src.experiments.experimental_presets import (
    EXPERIMENTAL_T_RISK_MAX,
    EXPERIMENTAL_T_RISK_MIN,
    ThermalDisturbance,
)
from src.models.candidate_route import CandidateRoute
from src.objective.logistics import dynamic_logistic_reference
from src.objective.occupancy import occupancy_reference
from src.objective.thermal import thermal_reference
from src.objective.weight_profiles import BALANCED_WEIGHTS
from src.simulation.snapshot import OptimizationSnapshot
from src.simulation.state import (
    build_optimization_snapshot,
    create_initial_simulation_state,
)


def test_calibration_has_exact_official_scales_and_pilot_seeds() -> None:
    assert CALIBRATION_POINT_COUNTS == (5, 15, 20)
    assert {
        point_count: len(seeds)
        for point_count, seeds in CALIBRATION_INSTANCE_SEEDS_BY_POINT_COUNT.items()
    } == {5: 3, 15: 3, 20: 3}
    for point_count, pilot_seeds in CALIBRATION_INSTANCE_SEEDS_BY_POINT_COUNT.items():
        assert set(pilot_seeds).isdisjoint(
            EXPERIMENTAL_INSTANCE_SEEDS_BY_POINT_COUNT[point_count]
        )


def test_calibration_has_five_algorithm_seeds() -> None:
    assert CALIBRATION_ALGORITHM_SEEDS == (9101, 9102, 9103, 9104, 9105)
    assert len(set(CALIBRATION_ALGORITHM_SEEDS)) == 5


def test_calibration_grids_have_expected_unique_configurations() -> None:
    assert len(SA_CALIBRATION_GRID) == 18
    assert len(set(SA_CALIBRATION_GRID)) == 18
    assert len(ACO_CALIBRATION_GRID) == 9
    assert len(set(ACO_CALIBRATION_GRID)) == 9


def test_calibration_grid_parameters_match_algorithm_domains() -> None:
    assert all(
        parameter.initial_temperature > 0.0
        and 0.0 < parameter.cooling_rate < 1.0
        and parameter.iterations >= 0
        for parameter in SA_CALIBRATION_GRID
    )
    assert all(
        parameter.beta > 0.0 and 0.0 < parameter.evaporation_rate < 1.0
        for parameter in ACO_CALIBRATION_GRID
    )
    assert set(SA_INITIAL_TEMPERATURES) == {
        parameter.initial_temperature for parameter in SA_CALIBRATION_GRID
    }
    assert set(SA_COOLING_RATES) == {
        parameter.cooling_rate for parameter in SA_CALIBRATION_GRID
    }
    assert set(SA_ITERATIONS) == {
        parameter.iterations for parameter in SA_CALIBRATION_GRID
    }
    assert set(ACO_BETAS) == {parameter.beta for parameter in ACO_CALIBRATION_GRID}


@pytest.mark.parametrize("point_count", CALIBRATION_POINT_COUNTS)
def test_pilot_generation_is_nominal_reproducible_and_has_expected_size(
    point_count: int,
) -> None:
    first = _generate_pilot_instances(point_count)
    second = _generate_pilot_instances(point_count)

    assert [seed for seed, _, _ in first] == list(
        CALIBRATION_INSTANCE_SEEDS_BY_POINT_COUNT[point_count]
    )
    assert all(disturbance is None for _, _, disturbance in first)
    assert all(
        not isinstance(disturbance, ThermalDisturbance) for _, _, disturbance in first
    )
    assert all(
        len(instance.collection_points) == point_count for _, instance, _ in first
    )
    assert [
        (
            instance.road_network,
            instance.collection_points,
            instance.temperature_trajectories,
        )
        for _, instance, _ in first
    ] == [
        (
            instance.road_network,
            instance.collection_points,
            instance.temperature_trajectories,
        )
        for _, instance, _ in second
    ]


@pytest.mark.parametrize("invalid_point_count", [0, 10, True, 1.5, "5"])
def test_calibration_rejects_invalid_point_count(invalid_point_count: object) -> None:
    with pytest.raises(ValueError, match="point_count"):
        run_sa_calibration(invalid_point_count)  # type: ignore[arg-type]


def test_initial_pilot_snapshot_references_and_baseline_are_valid() -> None:
    seed, instance, disturbance = _generate_pilot_instances(5)[0]
    assert disturbance is None
    state = create_initial_simulation_state(instance)
    snapshot = build_optimization_snapshot(instance, state)
    assert isinstance(snapshot, OptimizationSnapshot)

    logistic_reference = dynamic_logistic_reference(instance, snapshot)
    thermal_reference_value = thermal_reference(
        instance,
        snapshot,
        EXPERIMENTAL_T_RISK_MIN,
        EXPERIMENTAL_T_RISK_MAX,
    )
    occupancy_reference_value = occupancy_reference(instance, snapshot)
    assert logistic_reference >= 0.0
    assert thermal_reference_value >= 0.0
    assert occupancy_reference_value >= 0.0

    baseline = multicriteria_nearest_neighbor(
        instance,
        snapshot,
        EXPERIMENTAL_T_RISK_MIN,
        EXPERIMENTAL_T_RISK_MAX,
        BALANCED_WEIGHTS,
    )
    assert isinstance(baseline, CandidateRoute)
    assert baseline.validate(instance, snapshot) is None
    _, _, _, _, _, _, _, baseline_cost = _prepare_case(instance, snapshot, seed)
    assert math.isfinite(baseline_cost)
    assert baseline_cost >= 0.0


def test_sa_calibration_run_is_valid_reproducible_and_non_mutating() -> None:
    seed, instance, _ = _generate_pilot_instances(5)[0]
    state = create_initial_simulation_state(instance)
    snapshot = build_optimization_snapshot(instance, state)
    instance_before = instance
    snapshot_before = snapshot

    first = evaluate_sa_configuration(
        instance,
        snapshot,
        SA_CALIBRATION_GRID[0],
        CALIBRATION_ALGORITHM_SEEDS[0],
        point_count=5,
        instance_seed=seed,
    )
    second = evaluate_sa_configuration(
        instance,
        snapshot,
        SA_CALIBRATION_GRID[0],
        CALIBRATION_ALGORITHM_SEEDS[0],
        point_count=5,
        instance_seed=seed,
    )

    assert first.algorithm == "sa"
    assert first.parameters == SA_CALIBRATION_GRID[0]
    assert first.objective_cost == second.objective_cost
    assert first.objective_cost >= 0.0
    assert first.elapsed_seconds >= 0.0
    assert instance == instance_before
    assert snapshot == snapshot_before


def test_aco_calibration_run_is_valid_reproducible_and_non_mutating() -> None:
    seed, instance, _ = _generate_pilot_instances(5)[0]
    state = create_initial_simulation_state(instance)
    snapshot = build_optimization_snapshot(instance, state)
    instance_before = instance
    snapshot_before = snapshot

    first = evaluate_aco_configuration(
        instance,
        snapshot,
        ACO_CALIBRATION_GRID[0],
        CALIBRATION_ALGORITHM_SEEDS[0],
        point_count=5,
        instance_seed=seed,
    )
    second = evaluate_aco_configuration(
        instance,
        snapshot,
        ACO_CALIBRATION_GRID[0],
        CALIBRATION_ALGORITHM_SEEDS[0],
        point_count=5,
        instance_seed=seed,
    )

    assert first.algorithm == "aco"
    assert first.parameters == ACO_CALIBRATION_GRID[0]
    assert first.objective_cost == second.objective_cost
    assert first.objective_cost >= 0.0
    assert first.elapsed_seconds >= 0.0
    assert instance == instance_before
    assert snapshot == snapshot_before


def test_calibration_summary_aggregates_population_statistics() -> None:
    parameters = SAParameterSet(0.1, 0.95, 100)
    runs = tuple(
        CalibrationRun(
            algorithm="sa",
            point_count=5,
            instance_seed=index,
            algorithm_seed=index,
            parameters=parameters,
            objective_cost=cost,
            elapsed_seconds=elapsed,
        )
        for index, (cost, elapsed) in enumerate(
            ((1.0, 2.0), (2.0, 4.0), (3.0, 6.0)),
            start=1,
        )
    )

    summary = _summarize_runs(runs)

    assert summary.mean_objective_cost == 2.0
    assert summary.std_objective_cost == pytest.approx(math.sqrt(2.0 / 3.0))
    assert summary.mean_elapsed_seconds == 4.0
    assert summary.run_count == 3
    assert _summary_sort_key(summary) == (
        summary.mean_objective_cost,
        summary.std_objective_cost,
        summary.mean_elapsed_seconds,
    )


def test_calibration_summary_sorting_uses_cost_then_std_then_time() -> None:
    summaries = (
        CalibrationSummary("sa", 5, SAParameterSet(0.1, 0.95, 100), 1.0, 0.2, 3.0, 15),
        CalibrationSummary("sa", 5, SAParameterSet(0.1, 0.98, 100), 1.0, 0.1, 4.0, 15),
        CalibrationSummary("sa", 5, SAParameterSet(0.1, 0.995, 100), 1.0, 0.1, 2.0, 15),
        CalibrationSummary("sa", 5, SAParameterSet(0.5, 0.95, 100), 0.5, 0.9, 9.0, 15),
    )

    ordered = sorted(summaries, key=_summary_sort_key)

    assert ordered == [summaries[3], summaries[2], summaries[1], summaries[0]]


def test_calibration_summary_rejects_mixed_algorithms_or_parameters() -> None:
    sa_run = CalibrationRun(
        "sa",
        5,
        1,
        1,
        SAParameterSet(0.1, 0.95, 100),
        1.0,
        1.0,
    )
    aco_run = CalibrationRun(
        "aco",
        5,
        1,
        1,
        ACOParameterSet(1.0, 0.1),
        1.0,
        1.0,
    )
    with pytest.raises(ValueError, match="same algorithm"):
        _summarize_runs((sa_run, aco_run))

    different_parameters = CalibrationRun(
        "sa",
        5,
        1,
        1,
        SAParameterSet(0.5, 0.95, 100),
        1.0,
        1.0,
    )
    with pytest.raises(ValueError, match="same parameters"):
        _summarize_runs((sa_run, different_parameters))
