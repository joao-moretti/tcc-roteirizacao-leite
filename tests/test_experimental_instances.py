import random

import pytest

from src.experiments.experimental_instances import (
    EXPERIMENTAL_INSTANCE_ID_OFFSET_BY_POINT_COUNT,
    EXPERIMENTAL_INSTANCE_SEEDS_BY_POINT_COUNT,
    EXPERIMENTAL_INSTANCES_PER_SCALE,
    EXPERIMENTAL_POINT_COUNTS,
    ExperimentalInstancePair,
    generate_experimental_instance_pair,
    generate_experimental_scale_instances,
    generate_full_experimental_instance_set,
)
from src.experiments.experimental_presets import ThermalDisturbance


@pytest.fixture(scope="module")
def full_instance_set() -> tuple[ExperimentalInstancePair, ...]:
    return generate_full_experimental_instance_set()


def pair_signature(pair: ExperimentalInstancePair) -> tuple[object, ...]:
    return (
        pair.point_count,
        pair.replicate_index,
        pair.instance_seed,
        pair.nominal_instance,
        pair.disturbance_instance,
        pair.disturbance,
    )


def test_constants_define_three_scales_and_ten_seeds_each() -> None:
    assert EXPERIMENTAL_POINT_COUNTS == (5, 15, 20)
    assert EXPERIMENTAL_INSTANCES_PER_SCALE == 10
    assert set(EXPERIMENTAL_INSTANCE_SEEDS_BY_POINT_COUNT) == {5, 15, 20}
    assert all(
        len(seeds) == EXPERIMENTAL_INSTANCES_PER_SCALE
        for seeds in EXPERIMENTAL_INSTANCE_SEEDS_BY_POINT_COUNT.values()
    )
    all_seeds = tuple(
        seed
        for seeds in EXPERIMENTAL_INSTANCE_SEEDS_BY_POINT_COUNT.values()
        for seed in seeds
    )
    assert len(all_seeds) == 30
    assert len(set(all_seeds)) == 30


def test_first_pair_uses_frozen_seed_and_ids() -> None:
    pair = generate_experimental_instance_pair(5, 1)

    assert isinstance(pair, ExperimentalInstancePair)
    assert pair.instance_seed == 5001
    assert pair.nominal_instance.instance_id == 1
    assert pair.disturbance_instance.instance_id == 1
    assert pair.nominal_instance.road_network.network_id == 1
    assert pair.disturbance_instance.road_network.network_id == 1


def test_frozen_seed_and_ids_are_correct_at_scale_boundaries() -> None:
    cases = (
        (5, 10, 5010, 10),
        (15, 1, 15001, 11),
        (15, 10, 15010, 20),
        (20, 1, 20001, 21),
        (20, 10, 20010, 30),
    )

    for point_count, replicate_index, seed, instance_id in cases:
        pair = generate_experimental_instance_pair(point_count, replicate_index)
        assert pair.instance_seed == seed
        assert pair.nominal_instance.instance_id == instance_id
        assert pair.disturbance_instance.instance_id == instance_id
        assert pair.nominal_instance.road_network.network_id == instance_id


def test_pair_preserves_nominal_disturbance_pairing() -> None:
    pair = generate_experimental_instance_pair(5, 1)

    assert pair.nominal_instance.road_network == pair.disturbance_instance.road_network
    assert pair.nominal_instance.depot == pair.disturbance_instance.depot
    assert pair.nominal_instance.collection_points == (
        pair.disturbance_instance.collection_points
    )
    assert pair.nominal_instance.volume_trajectories == (
        pair.disturbance_instance.volume_trajectories
    )
    assert pair.nominal_instance.vehicle_capacity == (
        pair.disturbance_instance.vehicle_capacity
    )
    assert isinstance(pair.disturbance, ThermalDisturbance)
    assert pair.disturbance.point_id in {
        point.point_id for point in pair.nominal_instance.collection_points
    }


def test_scale_generation_returns_ten_pairs_in_replicate_order() -> None:
    pairs = generate_experimental_scale_instances(5)

    assert isinstance(pairs, tuple)
    assert len(pairs) == 10
    assert tuple(pair.replicate_index for pair in pairs) == tuple(range(1, 11))
    assert tuple(pair.instance_seed for pair in pairs) == (
        EXPERIMENTAL_INSTANCE_SEEDS_BY_POINT_COUNT[5]
    )
    assert tuple(pair.nominal_instance.instance_id for pair in pairs) == tuple(
        range(1, 11)
    )


def test_scale_id_offsets_match_the_official_plan() -> None:
    for point_count, offset in EXPERIMENTAL_INSTANCE_ID_OFFSET_BY_POINT_COUNT.items():
        pairs = generate_experimental_scale_instances(point_count)
        assert tuple(pair.nominal_instance.instance_id for pair in pairs) == tuple(
            range(offset + 1, offset + 11)
        )
        assert all(
            pair.nominal_instance.road_network.network_id
            == pair.nominal_instance.instance_id
            for pair in pairs
        )


def test_full_set_has_official_order_and_unique_global_ids(
    full_instance_set: tuple[ExperimentalInstancePair, ...],
) -> None:
    assert len(full_instance_set) == 30
    assert (
        tuple(pair.point_count for pair in full_instance_set)
        == (5,) * 10 + (15,) * 10 + (20,) * 10
    )
    assert (
        tuple(pair.replicate_index for pair in full_instance_set)
        == tuple(range(1, 11)) * 3
    )
    assert tuple(
        pair.nominal_instance.instance_id for pair in full_instance_set
    ) == tuple(range(1, 31))
    assert (
        len(
            {
                pair.nominal_instance.road_network.network_id
                for pair in full_instance_set
            }
        )
        == 30
    )
    assert len({pair.instance_seed for pair in full_instance_set}) == 30


def test_full_set_contains_exactly_ten_pairs_per_scale(
    full_instance_set: tuple[ExperimentalInstancePair, ...],
) -> None:
    for point_count in EXPERIMENTAL_POINT_COUNTS:
        assert sum(pair.point_count == point_count for pair in full_instance_set) == 10


@pytest.mark.parametrize("point_count", [0, 10, 25, 50, True, 5.0, "5"])
def test_invalid_point_count_is_rejected(point_count: object) -> None:
    with pytest.raises(ValueError, match="point_count"):
        generate_experimental_instance_pair(point_count, 1)  # type: ignore[arg-type]


@pytest.mark.parametrize("replicate_index", [0, 11, True, 1.5, "1"])
def test_invalid_replicate_index_is_rejected(replicate_index: object) -> None:
    with pytest.raises(ValueError, match="replicate_index"):
        generate_experimental_instance_pair(5, replicate_index)  # type: ignore[arg-type]


def test_pair_generation_is_reproducible() -> None:
    first = generate_experimental_instance_pair(5, 1)
    second = generate_experimental_instance_pair(5, 1)

    assert pair_signature(first) == pair_signature(second)


def test_global_random_state_does_not_affect_pair_generation() -> None:
    random.seed(1)
    random.random()
    first = generate_experimental_instance_pair(5, 1)
    random.seed(999)
    random.random()
    second = generate_experimental_instance_pair(5, 1)

    assert pair_signature(first) == pair_signature(second)


def test_disturbance_metadata_is_valid_for_every_pair(
    full_instance_set: tuple[ExperimentalInstancePair, ...],
) -> None:
    for pair in full_instance_set:
        assert isinstance(pair.disturbance, ThermalDisturbance)
        assert pair.disturbance.point_id in {
            point.point_id for point in pair.nominal_instance.collection_points
        }
        assert 240 <= pair.disturbance.start_sample <= 720
        assert pair.disturbance.start_time == pair.disturbance.start_sample * 30.0
        assert pair.disturbance.heating_rate_c_per_hour == 0.6
