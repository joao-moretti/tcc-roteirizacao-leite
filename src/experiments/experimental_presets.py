"""Frozen experimental presets and paired thermal-disturbance scenarios."""

from dataclasses import dataclass
from typing import Literal

from src.experiments.instance_generation import (
    create_instance_rng,
    generate_synthetic_problem_instance,
)
from src.models.problem import ProblemInstance

ExperimentalScenario = Literal["nominal", "disturbance"]

EXPERIMENTAL_GRID_BY_POINT_COUNT = {
    5: (3, 4),
    15: (5, 6),
    20: (6, 7),
}

EXPERIMENTAL_SPACING = 1000.0
EXPERIMENTAL_P_UNI = 0.20
EXPERIMENTAL_MIN_SEPARATION = 1000.0

EXPERIMENTAL_V_REF = 11.11
EXPERIMENTAL_COLLECTION_SERVICE_DURATION = 600.0
EXPERIMENTAL_VEHICLE_CAPACITY = 12000.0
EXPERIMENTAL_SAMPLE_PERIOD = 30.0
EXPERIMENTAL_SAMPLE_COUNT = 1921

EXPERIMENTAL_TEMPERATURE_INITIAL_LEVELS = (3.0, 5.0, 7.0)
EXPERIMENTAL_TEMPERATURE_INITIAL_PROBABILITIES = (
    1 / 3,
    1 / 3,
    1 / 3,
)
EXPERIMENTAL_DELTA_TEMPERATURE = 0.2
EXPERIMENTAL_TEMPERATURE_TRANSITION_PROBABILITIES = (
    0.05,
    0.89,
    0.06,
)
EXPERIMENTAL_TEMPERATURE_MINIMUM = 0.0
EXPERIMENTAL_TEMPERATURE_MAXIMUM = 10.0
EXPERIMENTAL_T_RISK_MIN = 4.0
EXPERIMENTAL_T_RISK_MAX = 7.0

EXPERIMENTAL_VOLUME_INITIAL_LEVELS = (
    200.0,
    400.0,
    600.0,
)
EXPERIMENTAL_VOLUME_INITIAL_PROBABILITIES = (
    1 / 3,
    1 / 3,
    1 / 3,
)
EXPERIMENTAL_GAMMA_VOLUME = 0.01
EXPERIMENTAL_VOLUME_TRANSITION_PROBABILITIES = (
    0.05,
    0.89,
    0.06,
)
EXPERIMENTAL_VOLUME_MINIMUM = 0.0
EXPERIMENTAL_VOLUME_MAXIMUM = 600.0
EXPERIMENTAL_Q_LOW = 300.0
EXPERIMENTAL_Q_HIGH = 500.0
EXPERIMENTAL_MINIMUM_RECALCULATION_INTERVAL = 300.0

DISTURBANCE_START_MIN_HOURS = 2.0
DISTURBANCE_START_MAX_HOURS = 6.0
DISTURBANCE_START_MIN_SAMPLE = 240
DISTURBANCE_START_MAX_SAMPLE = 720
DISTURBANCE_HEATING_RATE_C_PER_HOUR = 0.6
DISTURBANCE_SEED_OFFSET = 1_000_003


@dataclass(frozen=True)
class ThermalDisturbance:
    """Metadata describing one frozen thermal refrigeration failure."""

    point_id: int
    start_sample: int
    start_time: float
    heating_rate_c_per_hour: float


def _validate_experimental_point_count(point_count: object) -> int:
    """Validate that the point count belongs to the official scales."""
    if isinstance(point_count, bool) or not isinstance(point_count, int):
        raise ValueError("point_count must be one of the official experimental scales.")
    if point_count not in EXPERIMENTAL_GRID_BY_POINT_COUNT:
        raise ValueError("point_count must be one of the official experimental scales.")
    return point_count


def _validate_scenario(scenario: object) -> ExperimentalScenario:
    """Validate and return one of the two supported experimental scenarios."""
    if scenario not in ("nominal", "disturbance"):
        raise ValueError("scenario must be 'nominal' or 'disturbance'.")
    return scenario


def _apply_thermal_disturbance(
    temperature_trajectories: dict[int, tuple[float, ...]],
    disturbance: ThermalDisturbance,
) -> dict[int, tuple[float, ...]]:
    """Overlay post-failure heating without mutating nominal trajectories."""
    additional_temperature_per_sample = (
        disturbance.heating_rate_c_per_hour * EXPERIMENTAL_SAMPLE_PERIOD / 3600.0
    )
    disturbed_trajectories = dict(temperature_trajectories)
    nominal_trajectory = temperature_trajectories[disturbance.point_id]
    disturbed_values: list[float] = []

    for sample_index, nominal_temperature in enumerate(nominal_trajectory):
        if sample_index <= disturbance.start_sample:
            disturbed_values.append(nominal_temperature)
            continue

        elapsed_samples = sample_index - disturbance.start_sample
        disturbed_values.append(
            min(
                EXPERIMENTAL_TEMPERATURE_MAXIMUM,
                nominal_temperature
                + elapsed_samples * additional_temperature_per_sample,
            )
        )

    disturbed_trajectories[disturbance.point_id] = tuple(disturbed_values)
    return disturbed_trajectories


def generate_experimental_problem_instance(
    instance_id: int,
    network_id: int,
    instance_seed: int,
    point_count: int,
    scenario: ExperimentalScenario,
) -> tuple[ProblemInstance, ThermalDisturbance | None]:
    """Generate one frozen experimental instance and optional disturbance metadata."""
    point_count = _validate_experimental_point_count(point_count)
    scenario = _validate_scenario(scenario)
    rows, columns = EXPERIMENTAL_GRID_BY_POINT_COUNT[point_count]

    base_instance = generate_synthetic_problem_instance(
        instance_id=instance_id,
        network_id=network_id,
        instance_seed=instance_seed,
        rows=rows,
        columns=columns,
        spacing=EXPERIMENTAL_SPACING,
        p_uni=EXPERIMENTAL_P_UNI,
        point_count=point_count,
        min_separation=EXPERIMENTAL_MIN_SEPARATION,
        sample_count=EXPERIMENTAL_SAMPLE_COUNT,
        temperature_initial_levels=EXPERIMENTAL_TEMPERATURE_INITIAL_LEVELS,
        temperature_initial_probabilities=(
            EXPERIMENTAL_TEMPERATURE_INITIAL_PROBABILITIES
        ),
        delta_temperature=EXPERIMENTAL_DELTA_TEMPERATURE,
        temperature_transition_probabilities=(
            EXPERIMENTAL_TEMPERATURE_TRANSITION_PROBABILITIES
        ),
        temperature_simulation_minimum=EXPERIMENTAL_TEMPERATURE_MINIMUM,
        temperature_simulation_maximum=EXPERIMENTAL_TEMPERATURE_MAXIMUM,
        volume_initial_levels=EXPERIMENTAL_VOLUME_INITIAL_LEVELS,
        volume_initial_probabilities=EXPERIMENTAL_VOLUME_INITIAL_PROBABILITIES,
        gamma_volume=EXPERIMENTAL_GAMMA_VOLUME,
        volume_transition_probabilities=EXPERIMENTAL_VOLUME_TRANSITION_PROBABILITIES,
        volume_simulation_minimum=EXPERIMENTAL_VOLUME_MINIMUM,
        volume_simulation_maximum=EXPERIMENTAL_VOLUME_MAXIMUM,
        vehicle_capacity=EXPERIMENTAL_VEHICLE_CAPACITY,
        collection_service_duration=EXPERIMENTAL_COLLECTION_SERVICE_DURATION,
        v_ref=EXPERIMENTAL_V_REF,
    )

    if scenario == "nominal":
        return base_instance, None

    disturbance_rng = create_instance_rng(instance_seed + DISTURBANCE_SEED_OFFSET)
    ordered_points = sorted(
        base_instance.collection_points,
        key=lambda collection_point: collection_point.point_id,
    )
    affected_point = ordered_points[disturbance_rng.randrange(len(ordered_points))]
    start_sample = disturbance_rng.randrange(
        DISTURBANCE_START_MIN_SAMPLE,
        DISTURBANCE_START_MAX_SAMPLE + 1,
    )
    disturbance = ThermalDisturbance(
        point_id=affected_point.point_id,
        start_sample=start_sample,
        start_time=start_sample * EXPERIMENTAL_SAMPLE_PERIOD,
        heating_rate_c_per_hour=DISTURBANCE_HEATING_RATE_C_PER_HOUR,
    )
    disturbed_temperature_trajectories = _apply_thermal_disturbance(
        dict(base_instance.temperature_trajectories),
        disturbance,
    )
    disturbed_instance = ProblemInstance(
        instance_id=base_instance.instance_id,
        road_network=base_instance.road_network,
        depot=base_instance.depot,
        collection_points=base_instance.collection_points,
        vehicle_capacity=base_instance.vehicle_capacity,
        collection_service_duration=base_instance.collection_service_duration,
        v_ref=base_instance.v_ref,
        sample_period=base_instance.sample_period,
        temperature_trajectories=disturbed_temperature_trajectories,
        volume_trajectories=base_instance.volume_trajectories,
    )
    return disturbed_instance, disturbance
