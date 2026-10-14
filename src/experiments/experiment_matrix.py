"""Deterministic specification of the official experimental run matrix."""

from dataclasses import dataclass
from typing import cast

from src.experiments.experimental_instances import (
    EXPERIMENTAL_INSTANCES_PER_SCALE,
    EXPERIMENTAL_POINT_COUNTS,
)
from src.experiments.experimental_presets import ExperimentalScenario
from src.experiments.planners import EXPERIMENTAL_ALGORITHM_SEEDS
from src.experiments.runner import (
    EXPERIMENTAL_WEIGHT_PROFILES,
    ExperimentalAlgorithm,
    WeightProfileName,
)

_OFFICIAL_SCENARIOS: tuple[ExperimentalScenario, ...] = (
    "nominal",
    "disturbance",
)
_OFFICIAL_ALGORITHMS: tuple[ExperimentalAlgorithm, ...] = (
    "classical_nn",
    "mcnn",
    "sa",
    "aco",
)
_OFFICIAL_WEIGHT_PROFILES: tuple[WeightProfileName, ...] = tuple(
    EXPERIMENTAL_WEIGHT_PROFILES
)


@dataclass(frozen=True)
class ExperimentalRunSpec:
    """Identify one valid configuration in the official experiment matrix."""

    point_count: int
    replicate_index: int
    scenario: ExperimentalScenario
    algorithm: ExperimentalAlgorithm
    weight_profile: WeightProfileName | None = None
    run_seed: int | None = None

    def __post_init__(self) -> None:
        """Validate the dimensions and algorithm-specific fields."""
        _validate_point_count(self.point_count)
        _validate_replicate_index(self.replicate_index)
        _validate_scenario(self.scenario)
        algorithm = _validate_algorithm(self.algorithm)
        _validate_weight_profile(algorithm, self.weight_profile)
        _validate_run_seed(algorithm, self.run_seed)


def _validate_point_count(point_count: object) -> int:
    """Validate one of the official experimental scales."""
    if (
        isinstance(point_count, bool)
        or not isinstance(point_count, int)
        or point_count not in EXPERIMENTAL_POINT_COUNTS
    ):
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


def _validate_scenario(scenario: object) -> ExperimentalScenario:
    """Validate one official experimental scenario."""
    if scenario not in _OFFICIAL_SCENARIOS:
        raise ValueError("scenario must be 'nominal' or 'disturbance'.")
    return cast(ExperimentalScenario, scenario)


def _validate_algorithm(algorithm: object) -> ExperimentalAlgorithm:
    """Validate one official experimental algorithm."""
    if algorithm not in _OFFICIAL_ALGORITHMS:
        raise ValueError("algorithm must be one of: classical_nn, mcnn, sa, or aco.")
    return cast(ExperimentalAlgorithm, algorithm)


def _validate_weight_profile(
    algorithm: ExperimentalAlgorithm,
    weight_profile: object,
) -> WeightProfileName | None:
    """Validate the weight-profile requirement for one algorithm."""
    if algorithm == "classical_nn":
        if weight_profile is not None:
            raise ValueError("classical_nn requires weight_profile=None.")
        return None

    if weight_profile not in _OFFICIAL_WEIGHT_PROFILES:
        raise ValueError(
            "mcnn, sa, and aco require weight_profile to be logistic, balanced, "
            "or thermal."
        )
    return cast(WeightProfileName, weight_profile)


def _validate_run_seed(
    algorithm: ExperimentalAlgorithm,
    run_seed: object,
) -> int | None:
    """Validate the run-seed requirement for one algorithm."""
    if algorithm in ("classical_nn", "mcnn"):
        if run_seed is not None:
            raise ValueError(f"{algorithm} requires run_seed=None.")
        return None

    if (
        isinstance(run_seed, bool)
        or not isinstance(run_seed, int)
        or run_seed not in EXPERIMENTAL_ALGORITHM_SEEDS
    ):
        raise ValueError(
            "sa and aco require run_seed to be one of the 30 official seeds."
        )
    return run_seed


def make_run_id(spec: ExperimentalRunSpec) -> str:
    """Return a deterministic, readable identifier for one run specification."""
    run_id = (
        f"n{spec.point_count:02d}_r{spec.replicate_index:02d}_"
        f"{spec.scenario}_{spec.algorithm}"
    )
    if spec.weight_profile is not None:
        run_id += f"_{spec.weight_profile}"
    if spec.run_seed is not None:
        run_id += f"_s{spec.run_seed:02d}"
    return run_id


def _algorithm_profiles(
    algorithm: ExperimentalAlgorithm,
) -> tuple[WeightProfileName | None, ...]:
    """Return the profiles required by one official algorithm."""
    if algorithm == "classical_nn":
        return (None,)
    return _OFFICIAL_WEIGHT_PROFILES


def _algorithm_seeds(algorithm: ExperimentalAlgorithm) -> tuple[int | None, ...]:
    """Return the seeds required by one official algorithm."""
    if algorithm in ("classical_nn", "mcnn"):
        return (None,)
    return EXPERIMENTAL_ALGORITHM_SEEDS


def generate_official_run_specs() -> tuple[ExperimentalRunSpec, ...]:
    """Generate the complete official experiment matrix in deterministic order."""
    return tuple(
        ExperimentalRunSpec(
            point_count=point_count,
            replicate_index=replicate_index,
            scenario=scenario,
            algorithm=algorithm,
            weight_profile=weight_profile,
            run_seed=run_seed,
        )
        for point_count in EXPERIMENTAL_POINT_COUNTS
        for replicate_index in range(1, EXPERIMENTAL_INSTANCES_PER_SCALE + 1)
        for scenario in _OFFICIAL_SCENARIOS
        for algorithm in _OFFICIAL_ALGORITHMS
        for weight_profile in _algorithm_profiles(algorithm)
        for run_seed in _algorithm_seeds(algorithm)
    )
