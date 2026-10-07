"""Frozen collections of paired experimental instances."""

from dataclasses import dataclass

from src.experiments.experimental_presets import (
    ThermalDisturbance,
    generate_experimental_problem_instance,
)
from src.models.problem import ProblemInstance

EXPERIMENTAL_POINT_COUNTS = (5, 15, 20)
EXPERIMENTAL_INSTANCES_PER_SCALE = 10

EXPERIMENTAL_INSTANCE_SEEDS_BY_POINT_COUNT = {
    5: (
        5001,
        5002,
        5003,
        5004,
        5005,
        5006,
        5007,
        5008,
        5009,
        5010,
    ),
    15: (
        15001,
        15002,
        15003,
        15004,
        15005,
        15006,
        15007,
        15008,
        15009,
        15010,
    ),
    20: (
        20001,
        20002,
        20003,
        20004,
        20005,
        20006,
        20007,
        20008,
        20009,
        20010,
    ),
}

EXPERIMENTAL_INSTANCE_ID_OFFSET_BY_POINT_COUNT = {
    5: 0,
    15: 10,
    20: 20,
}


@dataclass(frozen=True)
class ExperimentalInstancePair:
    """Nominal and disturbance instances from one base realization."""

    point_count: int
    replicate_index: int
    instance_seed: int
    nominal_instance: ProblemInstance
    disturbance_instance: ProblemInstance
    disturbance: ThermalDisturbance


def _validate_point_count(point_count: object) -> int:
    """Validate one of the three official experimental scales."""
    if isinstance(point_count, bool) or not isinstance(point_count, int):
        raise ValueError("point_count must be one of the official experimental scales.")
    if point_count not in EXPERIMENTAL_POINT_COUNTS:
        raise ValueError("point_count must be one of the official experimental scales.")
    return point_count


def _validate_replicate_index(replicate_index: object) -> int:
    """Validate one replicate index within an experimental scale."""
    if (
        isinstance(replicate_index, bool)
        or not isinstance(replicate_index, int)
        or not 1 <= replicate_index <= EXPERIMENTAL_INSTANCES_PER_SCALE
    ):
        raise ValueError("replicate_index must be an integer between one and ten.")
    return replicate_index


def generate_experimental_instance_pair(
    point_count: int,
    replicate_index: int,
) -> ExperimentalInstancePair:
    """Generate one nominal/disturbance pair from the frozen plan."""
    point_count = _validate_point_count(point_count)
    replicate_index = _validate_replicate_index(replicate_index)
    instance_seed = EXPERIMENTAL_INSTANCE_SEEDS_BY_POINT_COUNT[point_count][
        replicate_index - 1
    ]
    instance_id = (
        EXPERIMENTAL_INSTANCE_ID_OFFSET_BY_POINT_COUNT[point_count] + replicate_index
    )
    network_id = instance_id

    nominal_instance, nominal_disturbance = generate_experimental_problem_instance(
        instance_id=instance_id,
        network_id=network_id,
        instance_seed=instance_seed,
        point_count=point_count,
        scenario="nominal",
    )
    if nominal_disturbance is not None:
        raise ValueError("nominal scenario unexpectedly returned disturbance metadata.")

    disturbance_instance, disturbance = generate_experimental_problem_instance(
        instance_id=instance_id,
        network_id=network_id,
        instance_seed=instance_seed,
        point_count=point_count,
        scenario="disturbance",
    )
    if disturbance is None:
        raise ValueError("disturbance scenario did not return disturbance metadata.")

    return ExperimentalInstancePair(
        point_count=point_count,
        replicate_index=replicate_index,
        instance_seed=instance_seed,
        nominal_instance=nominal_instance,
        disturbance_instance=disturbance_instance,
        disturbance=disturbance,
    )


def generate_experimental_scale_instances(
    point_count: int,
) -> tuple[ExperimentalInstancePair, ...]:
    """Generate the ten frozen pairs for one official scale."""
    point_count = _validate_point_count(point_count)
    return tuple(
        generate_experimental_instance_pair(point_count, replicate_index)
        for replicate_index in range(1, EXPERIMENTAL_INSTANCES_PER_SCALE + 1)
    )


def generate_full_experimental_instance_set() -> tuple[ExperimentalInstancePair, ...]:
    """Generate all thirty frozen experimental pairs in official order."""
    return tuple(
        pair
        for point_count in EXPERIMENTAL_POINT_COUNTS
        for pair in generate_experimental_scale_instances(point_count)
    )
