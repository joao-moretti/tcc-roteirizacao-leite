from collections import Counter

import pytest

from src.experiments.experiment_matrix import (
    ExperimentalRunSpec,
    generate_official_run_specs,
    make_run_id,
)
from src.experiments.planners import EXPERIMENTAL_ALGORITHM_SEEDS


@pytest.mark.parametrize(
    ("algorithm", "weight_profile", "run_seed"),
    [
        ("classical_nn", None, None),
        ("mcnn", "balanced", None),
        ("sa", "thermal", 17),
        ("aco", "logistic", 3),
    ],
)
def test_experimental_run_spec_accepts_each_algorithm(
    algorithm: str,
    weight_profile: str | None,
    run_seed: int | None,
) -> None:
    spec = ExperimentalRunSpec(
        point_count=15,
        replicate_index=4,
        scenario="disturbance",
        algorithm=algorithm,  # type: ignore[arg-type]
        weight_profile=weight_profile,  # type: ignore[arg-type]
        run_seed=run_seed,
    )

    assert spec.algorithm == algorithm
    assert spec.weight_profile == weight_profile
    assert spec.run_seed == run_seed


@pytest.mark.parametrize(
    ("algorithm", "weight_profile", "run_seed"),
    [
        ("classical_nn", "logistic", None),
        ("classical_nn", None, 1),
        ("mcnn", None, None),
        ("mcnn", "thermal", 1),
        ("sa", None, 1),
        ("sa", "logistic", None),
        ("aco", "balanced", 999),
    ],
)
def test_experimental_run_spec_rejects_invalid_algorithm_combinations(
    algorithm: str,
    weight_profile: str | None,
    run_seed: int | None,
) -> None:
    with pytest.raises(ValueError):
        ExperimentalRunSpec(
            point_count=5,
            replicate_index=1,
            scenario="nominal",
            algorithm=algorithm,  # type: ignore[arg-type]
            weight_profile=weight_profile,  # type: ignore[arg-type]
            run_seed=run_seed,
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("point_count", 10),
        ("replicate_index", 0),
        ("replicate_index", 11),
        ("scenario", "invalid"),
    ],
)
def test_experimental_run_spec_rejects_invalid_matrix_dimensions(
    field_name: str,
    value: object,
) -> None:
    values: dict[str, object] = {
        "point_count": 5,
        "replicate_index": 1,
        "scenario": "nominal",
    }
    values[field_name] = value

    with pytest.raises(ValueError):
        ExperimentalRunSpec(
            point_count=values["point_count"],  # type: ignore[arg-type]
            replicate_index=values["replicate_index"],  # type: ignore[arg-type]
            scenario=values["scenario"],  # type: ignore[arg-type]
            algorithm="classical_nn",
            weight_profile=None,
            run_seed=None,
        )


@pytest.mark.parametrize(
    ("spec", "expected_run_id"),
    [
        (
            ExperimentalRunSpec(5, 1, "nominal", "classical_nn", None, None),
            "n05_r01_nominal_classical_nn",
        ),
        (
            ExperimentalRunSpec(15, 4, "disturbance", "mcnn", "balanced", None),
            "n15_r04_disturbance_mcnn_balanced",
        ),
        (
            ExperimentalRunSpec(15, 4, "disturbance", "sa", "thermal", 17),
            "n15_r04_disturbance_sa_thermal_s17",
        ),
        (
            ExperimentalRunSpec(20, 10, "nominal", "aco", "logistic", 3),
            "n20_r10_nominal_aco_logistic_s03",
        ),
    ],
)
def test_make_run_id_uses_the_official_readable_format(
    spec: ExperimentalRunSpec,
    expected_run_id: str,
) -> None:
    assert make_run_id(spec) == expected_run_id


def test_make_run_id_is_deterministic() -> None:
    spec = ExperimentalRunSpec(15, 4, "disturbance", "sa", "thermal", 17)

    assert make_run_id(spec) == make_run_id(spec)


def test_official_matrix_has_expected_size_and_algorithm_counts() -> None:
    specs = generate_official_run_specs()

    assert len(specs) == 11_040
    assert Counter(spec.algorithm for spec in specs) == Counter(
        {
            "classical_nn": 60,
            "mcnn": 180,
            "sa": 5_400,
            "aco": 5_400,
        }
    )


def test_official_matrix_run_ids_are_unique_and_generation_is_deterministic() -> None:
    first_specs = generate_official_run_specs()
    second_specs = generate_official_run_specs()

    assert first_specs == second_specs
    run_ids = tuple(make_run_id(spec) for spec in first_specs)
    assert len(run_ids) == len(set(run_ids))


def test_official_matrix_contains_all_scales_and_scenarios() -> None:
    specs = generate_official_run_specs()

    assert {spec.point_count for spec in specs} == {5, 15, 20}
    assert {spec.scenario for spec in specs} == {"nominal", "disturbance"}


def test_stochastic_algorithms_use_all_official_seeds() -> None:
    specs = generate_official_run_specs()

    for algorithm in ("sa", "aco"):
        seeds = {spec.run_seed for spec in specs if spec.algorithm == algorithm}
        assert seeds == set(EXPERIMENTAL_ALGORITHM_SEEDS)


def test_deterministic_algorithms_have_no_seed_or_profile_when_required() -> None:
    specs = generate_official_run_specs()

    classical_specs = [spec for spec in specs if spec.algorithm == "classical_nn"]
    mcnn_specs = [spec for spec in specs if spec.algorithm == "mcnn"]

    assert all(spec.weight_profile is None for spec in classical_specs)
    assert all(spec.run_seed is None for spec in classical_specs)
    assert all(spec.run_seed is None for spec in mcnn_specs)


def test_official_matrix_starts_in_the_expected_order() -> None:
    specs = generate_official_run_specs()

    assert specs[:4] == (
        ExperimentalRunSpec(5, 1, "nominal", "classical_nn", None, None),
        ExperimentalRunSpec(5, 1, "nominal", "mcnn", "logistic", None),
        ExperimentalRunSpec(5, 1, "nominal", "mcnn", "balanced", None),
        ExperimentalRunSpec(5, 1, "nominal", "mcnn", "thermal", None),
    )


def test_experimental_run_spec_rejects_invalid_algorithm() -> None:
    with pytest.raises(ValueError):
        ExperimentalRunSpec(
            point_count=5,
            replicate_index=1,
            scenario="nominal",
            algorithm="invalid",  # type: ignore[arg-type]
            weight_profile=None,
            run_seed=None,
        )
