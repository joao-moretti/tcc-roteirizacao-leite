import random

import pytest

from src.experiments.experimental_presets import (
    DISTURBANCE_HEATING_RATE_C_PER_HOUR,
    DISTURBANCE_START_MAX_SERVICE_FRACTION,
    DISTURBANCE_START_MIN_SERVICE_FRACTION,
    EXPERIMENTAL_COLLECTION_SERVICE_DURATION,
    EXPERIMENTAL_MINIMUM_RECALCULATION_INTERVAL,
    EXPERIMENTAL_Q_HIGH,
    EXPERIMENTAL_Q_LOW,
    EXPERIMENTAL_SAMPLE_COUNT,
    EXPERIMENTAL_SAMPLE_PERIOD,
    EXPERIMENTAL_T_RISK_MAX,
    EXPERIMENTAL_T_RISK_MIN,
    EXPERIMENTAL_V_REF,
    EXPERIMENTAL_VEHICLE_CAPACITY,
    EXPERIMENTAL_VOLUME_INITIAL_LEVELS,
    ThermalDisturbance,
    _apply_thermal_disturbance,
    _disturbance_start_sample_bounds,
    generate_experimental_problem_instance,
)
from src.models.problem import ProblemInstance


def generate_preset(
    point_count: int = 5,
    instance_seed: int = 123,
    scenario: str = "nominal",
) -> tuple[ProblemInstance, ThermalDisturbance | None]:
    return generate_experimental_problem_instance(
        instance_id=1,
        network_id=2,
        instance_seed=instance_seed,
        point_count=point_count,
        scenario=scenario,  # type: ignore[arg-type]
    )


def instance_signature(instance: ProblemInstance) -> tuple[object, ...]:
    return (
        instance.road_network,
        instance.depot,
        instance.collection_points,
        instance.temperature_trajectories,
        instance.volume_trajectories,
        instance.vehicle_capacity,
        instance.collection_service_duration,
        instance.v_ref,
        instance.sample_period,
    )


@pytest.mark.parametrize(
    ("point_count", "expected_node_count"),
    [(5, 12), (15, 30), (20, 42)],
)
def test_official_scales_have_expected_grid_and_point_count(
    point_count: int,
    expected_node_count: int,
) -> None:
    instance, disturbance = generate_preset(point_count=point_count)

    assert len(instance.road_network.nodes) == expected_node_count
    assert len(instance.collection_points) == point_count
    assert disturbance is None


def test_preset_preserves_frozen_operational_constants() -> None:
    instance, _ = generate_preset()

    assert instance.road_network.nodes[1].x - instance.road_network.nodes[0].x == 1000.0
    assert instance.vehicle_capacity == EXPERIMENTAL_VEHICLE_CAPACITY
    assert instance.collection_service_duration == (
        EXPERIMENTAL_COLLECTION_SERVICE_DURATION
    )
    assert instance.v_ref == EXPERIMENTAL_V_REF
    assert instance.sample_period == 30.0


def test_preset_trajectories_have_frozen_horizon_and_valid_ranges() -> None:
    instance, _ = generate_preset(point_count=5)

    assert all(
        len(values) == EXPERIMENTAL_SAMPLE_COUNT
        for values in instance.temperature_trajectories.values()
    )
    assert all(
        len(values) == EXPERIMENTAL_SAMPLE_COUNT
        for values in instance.volume_trajectories.values()
    )
    assert all(
        0.0 <= value <= 10.0
        for values in instance.temperature_trajectories.values()
        for value in values
    )
    assert all(
        0.0 <= value <= 600.0
        for values in instance.volume_trajectories.values()
        for value in values
    )
    assert sum(max(values) for values in instance.volume_trajectories.values()) <= (
        EXPERIMENTAL_VEHICLE_CAPACITY
    )
    assert EXPERIMENTAL_T_RISK_MIN == 4.0
    assert EXPERIMENTAL_T_RISK_MAX == 7.0


def test_official_volume_band_limits_are_frozen() -> None:
    assert EXPERIMENTAL_Q_LOW == 300.0
    assert EXPERIMENTAL_Q_HIGH == 500.0
    assert EXPERIMENTAL_Q_LOW < EXPERIMENTAL_Q_HIGH

    low_level, medium_level, high_level = EXPERIMENTAL_VOLUME_INITIAL_LEVELS
    assert low_level < EXPERIMENTAL_Q_LOW
    assert EXPERIMENTAL_Q_LOW < medium_level < EXPERIMENTAL_Q_HIGH
    assert high_level >= EXPERIMENTAL_Q_HIGH


def test_official_recalculation_interval_matches_ten_samples() -> None:
    assert EXPERIMENTAL_MINIMUM_RECALCULATION_INTERVAL == 300.0
    assert (
        EXPERIMENTAL_MINIMUM_RECALCULATION_INTERVAL / EXPERIMENTAL_SAMPLE_PERIOD == 10.0
    )


@pytest.mark.parametrize(
    ("point_count", "expected_bounds"),
    [(5, (25, 75)), (15, (75, 225)), (20, (100, 300))],
)
def test_disturbance_start_sample_bounds_are_frozen_for_official_scales(
    point_count: int,
    expected_bounds: tuple[int, int],
) -> None:
    assert _disturbance_start_sample_bounds(point_count) == expected_bounds


@pytest.mark.parametrize("point_count", [5, 15, 20])
def test_disturbance_start_sample_bounds_follow_service_time_fractions(
    point_count: int,
) -> None:
    minimum_service_time = point_count * EXPERIMENTAL_COLLECTION_SERVICE_DURATION
    expected_bounds = (
        int(
            minimum_service_time
            * DISTURBANCE_START_MIN_SERVICE_FRACTION
            / EXPERIMENTAL_SAMPLE_PERIOD
        ),
        int(
            minimum_service_time
            * DISTURBANCE_START_MAX_SERVICE_FRACTION
            / EXPERIMENTAL_SAMPLE_PERIOD
        ),
    )

    assert _disturbance_start_sample_bounds(point_count) == expected_bounds


@pytest.mark.parametrize("point_count", [0, 10, 25, 50, True])
def test_non_official_point_counts_are_rejected(point_count: object) -> None:
    with pytest.raises(ValueError, match="point_count"):
        generate_preset(point_count=point_count)  # type: ignore[arg-type]


def test_unknown_scenario_is_rejected() -> None:
    with pytest.raises(ValueError, match="scenario"):
        generate_preset(scenario="unexpected")


def test_nominal_scenario_returns_instance_without_disturbance() -> None:
    instance, disturbance = generate_preset(scenario="nominal")

    assert isinstance(instance, ProblemInstance)
    assert disturbance is None


def test_same_seed_reproduces_nominal_scenario() -> None:
    first, first_disturbance = generate_preset(instance_seed=321)
    second, second_disturbance = generate_preset(instance_seed=321)

    assert instance_signature(first) == instance_signature(second)
    assert first_disturbance == second_disturbance is None


def test_global_random_state_does_not_affect_nominal_scenario() -> None:
    random.seed(1)
    random.random()
    first, _ = generate_preset(instance_seed=321)
    random.seed(999)
    random.random()
    second, _ = generate_preset(instance_seed=321)

    assert instance_signature(first) == instance_signature(second)


@pytest.mark.parametrize("instance_seed", [-1, True, 1.5, "1"])
def test_invalid_instance_seed_is_rejected(instance_seed: object) -> None:
    with pytest.raises(ValueError, match="instance_seed"):
        generate_preset(instance_seed=instance_seed)  # type: ignore[arg-type]


def test_disturbance_returns_metadata_for_one_existing_point() -> None:
    instance, disturbance = generate_preset(scenario="disturbance")

    assert isinstance(instance, ProblemInstance)
    assert isinstance(disturbance, ThermalDisturbance)
    assert disturbance.point_id in {
        point.point_id for point in instance.collection_points
    }
    minimum_sample, maximum_sample = _disturbance_start_sample_bounds(5)
    assert minimum_sample <= disturbance.start_sample <= maximum_sample
    assert (
        disturbance.start_time == disturbance.start_sample * EXPERIMENTAL_SAMPLE_PERIOD
    )
    assert disturbance.heating_rate_c_per_hour == (DISTURBANCE_HEATING_RATE_C_PER_HOUR)


@pytest.mark.parametrize("point_count", [5, 15, 20])
def test_disturbance_start_is_within_scale_service_window(point_count: int) -> None:
    _, disturbance = generate_preset(
        point_count=point_count,
        scenario="disturbance",
    )

    assert disturbance is not None
    minimum_sample, maximum_sample = _disturbance_start_sample_bounds(point_count)
    assert minimum_sample <= disturbance.start_sample <= maximum_sample
    assert (
        disturbance.start_time == disturbance.start_sample * EXPERIMENTAL_SAMPLE_PERIOD
    )
    assert disturbance.start_time <= (
        DISTURBANCE_START_MAX_SERVICE_FRACTION
        * point_count
        * EXPERIMENTAL_COLLECTION_SERVICE_DURATION
    )
    assert disturbance.start_time < (
        point_count * EXPERIMENTAL_COLLECTION_SERVICE_DURATION
    )


def test_same_seed_reproduces_disturbance_metadata_and_instance() -> None:
    first, first_disturbance = generate_preset(
        instance_seed=321,
        scenario="disturbance",
    )
    second, second_disturbance = generate_preset(
        instance_seed=321,
        scenario="disturbance",
    )

    assert first_disturbance == second_disturbance
    assert instance_signature(first) == instance_signature(second)


def test_different_seeds_can_change_disturbance_metadata() -> None:
    metadata = {
        (
            generate_preset(instance_seed=seed, scenario="disturbance")[1].point_id,
            generate_preset(instance_seed=seed, scenario="disturbance")[1].start_sample,
        )
        for seed in range(5)
    }

    assert len(metadata) > 1


def test_nominal_and_disturbance_are_paired_on_static_data() -> None:
    nominal, _ = generate_preset(instance_seed=321, scenario="nominal")
    disturbance_instance, disturbance = generate_preset(
        instance_seed=321,
        scenario="disturbance",
    )

    assert disturbance is not None
    assert nominal.road_network == disturbance_instance.road_network
    assert nominal.depot == disturbance_instance.depot
    assert nominal.collection_points == disturbance_instance.collection_points
    assert nominal.volume_trajectories == disturbance_instance.volume_trajectories
    assert nominal.vehicle_capacity == disturbance_instance.vehicle_capacity
    assert nominal.collection_service_duration == (
        disturbance_instance.collection_service_duration
    )
    assert nominal.v_ref == disturbance_instance.v_ref
    assert nominal.sample_period == disturbance_instance.sample_period


def test_disturbance_preserves_nominal_temperatures_until_failure() -> None:
    nominal, _ = generate_preset(instance_seed=321, scenario="nominal")
    disturbed, disturbance = generate_preset(
        instance_seed=321,
        scenario="disturbance",
    )

    assert disturbance is not None
    for point_id, nominal_values in nominal.temperature_trajectories.items():
        disturbed_values = disturbed.temperature_trajectories[point_id]
        if point_id == disturbance.point_id:
            assert disturbed_values[: disturbance.start_sample + 1] == (
                nominal_values[: disturbance.start_sample + 1]
            )
        else:
            assert disturbed_values == nominal_values


def test_disturbance_overlay_matches_incremental_heating() -> None:
    nominal, _ = generate_preset(instance_seed=321, scenario="nominal")
    disturbed, disturbance = generate_preset(
        instance_seed=321,
        scenario="disturbance",
    )

    assert disturbance is not None
    nominal_values = nominal.temperature_trajectories[disturbance.point_id]
    disturbed_values = disturbed.temperature_trajectories[disturbance.point_id]
    for elapsed_samples in (1, 10, 100):
        sample_index = disturbance.start_sample + elapsed_samples
        expected = min(10.0, nominal_values[sample_index] + elapsed_samples * 0.005)
        assert disturbed_values[sample_index] == expected


def test_thermal_disturbance_clips_at_temperature_maximum_without_mutation() -> None:
    nominal = {1: (9.9, 9.99, 10.0), 2: (3.0, 3.0, 3.0)}
    disturbance = ThermalDisturbance(1, 0, 0.0, 0.6)

    disturbed = _apply_thermal_disturbance(nominal, disturbance)

    assert disturbed[1] == pytest.approx((9.9, 9.995, 10.0))
    assert disturbed[2] == nominal[2]
    assert nominal == {1: (9.9, 9.99, 10.0), 2: (3.0, 3.0, 3.0)}


def test_disturbance_does_not_mutate_base_instance() -> None:
    nominal, _ = generate_preset(instance_seed=321, scenario="nominal")
    nominal_signature = instance_signature(nominal)

    generate_preset(instance_seed=321, scenario="disturbance")

    assert instance_signature(nominal) == nominal_signature
