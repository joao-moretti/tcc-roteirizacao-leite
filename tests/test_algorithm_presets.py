from src.experiments.algorithm_presets import (
    ACO_BETA,
    ACO_EVAPORATION_RATE,
    SA_COOLING_RATE,
    SA_INITIAL_TEMPERATURE,
    SA_ITERATIONS,
)


def test_official_simulated_annealing_presets() -> None:
    assert SA_INITIAL_TEMPERATURE == 0.5
    assert SA_COOLING_RATE == 0.95
    assert SA_ITERATIONS == 500


def test_official_ant_colony_presets() -> None:
    assert ACO_BETA == 3.0
    assert ACO_EVAPORATION_RATE == 0.1


def test_official_presets_satisfy_algorithm_domains() -> None:
    assert SA_INITIAL_TEMPERATURE > 0.0
    assert 0.0 < SA_COOLING_RATE < 1.0
    assert SA_ITERATIONS > 0
    assert ACO_BETA > 0.0
    assert 0.0 < ACO_EVAPORATION_RATE < 1.0
