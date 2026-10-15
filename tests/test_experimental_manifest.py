import json
from pathlib import Path

import pytest

from src.algorithms.ant_colony import ACO_MAX_ITERATIONS, ACO_STAGNATION_LIMIT
from src.experiments.algorithm_presets import (
    ACO_BETA,
    ACO_EVAPORATION_RATE,
    SA_COOLING_RATE,
    SA_INITIAL_TEMPERATURE,
    SA_ITERATIONS,
)
from src.experiments.experimental_instances import (
    EXPERIMENTAL_INSTANCE_SEEDS_BY_POINT_COUNT,
    EXPERIMENTAL_INSTANCES_PER_SCALE,
    EXPERIMENTAL_POINT_COUNTS,
)
from src.experiments.experimental_presets import (
    EXPERIMENTAL_COLLECTION_SERVICE_DURATION,
    EXPERIMENTAL_GAMMA_VOLUME,
    EXPERIMENTAL_MINIMUM_RECALCULATION_INTERVAL,
    EXPERIMENTAL_P_UNI,
    EXPERIMENTAL_Q_HIGH,
    EXPERIMENTAL_Q_LOW,
    EXPERIMENTAL_SAMPLE_COUNT,
    EXPERIMENTAL_SAMPLE_PERIOD,
    EXPERIMENTAL_SPACING,
    EXPERIMENTAL_T_RISK_MAX,
    EXPERIMENTAL_T_RISK_MIN,
    EXPERIMENTAL_TEMPERATURE_INITIAL_LEVELS,
    EXPERIMENTAL_TEMPERATURE_MAXIMUM,
    EXPERIMENTAL_TEMPERATURE_MINIMUM,
    EXPERIMENTAL_VOLUME_INITIAL_LEVELS,
    EXPERIMENTAL_VOLUME_MAXIMUM,
    EXPERIMENTAL_VOLUME_MINIMUM,
)
from src.experiments.planners import EXPERIMENTAL_ALGORITHM_SEEDS
from src.objective.weight_profiles import (
    BALANCED_WEIGHTS,
    LOGISTIC_WEIGHTS,
    THERMAL_WEIGHTS,
)

MANIFEST_PATH = (
    Path(__file__).resolve().parents[1] / "results" / "experimental_manifest.json"
)


@pytest.fixture(scope="module")
def manifest() -> dict[str, object]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_manifest_has_core_problem_values_and_units(
    manifest: dict[str, object],
) -> None:
    units = manifest["units"]
    problem = manifest["problem"]

    assert units == {
        "distance": "m",
        "time": "s",
        "speed": "m/s",
        "temperature": "degC",
        "volume": "L",
    }
    assert problem["point_counts"] == list(EXPERIMENTAL_POINT_COUNTS)
    assert problem["vehicle_capacity"] == 12000.0
    assert problem["sample_period"] == EXPERIMENTAL_SAMPLE_PERIOD
    assert problem["sample_count"] == EXPERIMENTAL_SAMPLE_COUNT
    assert problem["collection_service_duration"] == (
        EXPERIMENTAL_COLLECTION_SERVICE_DURATION
    )


def test_manifest_has_official_weight_profiles(
    manifest: dict[str, object],
) -> None:
    profiles = manifest["objective"]["profiles"]
    expected_profiles = {
        "logistic": LOGISTIC_WEIGHTS,
        "balanced": BALANCED_WEIGHTS,
        "thermal": THERMAL_WEIGHTS,
    }

    assert set(profiles) == set(expected_profiles)
    for profile_name, expected_weights in expected_profiles.items():
        weights = tuple(profiles[profile_name].values())
        assert weights == pytest.approx(expected_weights)
        assert sum(weights) == pytest.approx(1.0)


def test_manifest_matches_thermal_and_volume_presets(
    manifest: dict[str, object],
) -> None:
    thermal = manifest["thermal_model"]
    volume = manifest["volume_model"]

    assert thermal["t_min"] == EXPERIMENTAL_T_RISK_MIN
    assert thermal["t_max"] == EXPERIMENTAL_T_RISK_MAX
    assert thermal["initial_levels"] == list(EXPERIMENTAL_TEMPERATURE_INITIAL_LEVELS)
    assert thermal["minimum"] == EXPERIMENTAL_TEMPERATURE_MINIMUM
    assert thermal["maximum"] == EXPERIMENTAL_TEMPERATURE_MAXIMUM
    assert volume["initial_levels"] == list(EXPERIMENTAL_VOLUME_INITIAL_LEVELS)
    assert volume["q_low"] == EXPERIMENTAL_Q_LOW
    assert volume["q_high"] == EXPERIMENTAL_Q_HIGH
    assert volume["gamma"] == EXPERIMENTAL_GAMMA_VOLUME
    assert volume["minimum"] == EXPERIMENTAL_VOLUME_MINIMUM
    assert volume["maximum"] == EXPERIMENTAL_VOLUME_MAXIMUM


def test_manifest_has_official_seeds_and_scale_grid_presets(
    manifest: dict[str, object],
) -> None:
    seeds = manifest["seeds"]
    road_network = manifest["road_network"]

    assert seeds["algorithm_seeds"] == list(EXPERIMENTAL_ALGORITHM_SEEDS)
    assert all(
        len(seed_values) == EXPERIMENTAL_INSTANCES_PER_SCALE
        for seed_values in seeds["instance_seeds_by_point_count"].values()
    )
    assert road_network["grid_by_point_count"] == {
        "5": [3, 4],
        "15": [5, 6],
        "20": [6, 7],
    }
    assert road_network["grid_spacing"] == EXPERIMENTAL_SPACING
    assert road_network["unidirectional_probability"] == EXPERIMENTAL_P_UNI
    assert set(seeds["instance_seeds_by_point_count"]) == {
        str(point_count) for point_count in EXPERIMENTAL_POINT_COUNTS
    }
    assert seeds["instance_seeds_by_point_count"] == {
        str(point_count): list(seed_values)
        for point_count, seed_values in (
            EXPERIMENTAL_INSTANCE_SEEDS_BY_POINT_COUNT.items()
        )
    }


def test_manifest_has_frozen_algorithm_parameters(
    manifest: dict[str, object],
) -> None:
    algorithms = manifest["algorithms"]

    assert algorithms["simulated_annealing"] == {
        "initial_temperature": SA_INITIAL_TEMPERATURE,
        "cooling_rate": SA_COOLING_RATE,
        "iterations": SA_ITERATIONS,
        "neighbor_operator": "swap two candidate-route positions",
        "acceptance": "Metropolis criterion",
        "seed_policy": "official run seed with deterministic child seeds per planner",
    }
    aco = algorithms["ant_colony_optimization"]
    assert aco["beta"] == ACO_BETA
    assert aco["evaporation_rate"] == ACO_EVAPORATION_RATE
    assert aco["max_iterations"] == ACO_MAX_ITERATIONS
    assert aco["stagnation_limit"] == ACO_STAGNATION_LIMIT
    assert aco["alpha"] == 1.0
    assert aco["alpha_parameterized"] is False


def test_manifest_has_official_matrix_counts(
    manifest: dict[str, object],
) -> None:
    matrix = manifest["experimental_matrix"]

    assert matrix["instances_per_scale"] == 10
    assert matrix["scenarios_per_instance"] == 2
    assert matrix["executions_per_scale"] == 3680
    assert matrix["total_executions"] == 11040


def test_manifest_records_the_event_policy(
    manifest: dict[str, object],
) -> None:
    policy = manifest["event_policy"]

    assert policy["minimum_recalculation_interval"] == (
        EXPERIMENTAL_MINIMUM_RECALCULATION_INTERVAL
    )
    assert policy["additional_hysteresis"] is False
    assert policy["collection_completion_bypasses_cooldown"] is True
