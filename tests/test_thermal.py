import pytest

from src.objective.thermal import thermal_risk


@pytest.mark.parametrize("temperature", [1.0, 2.0])
def test_thermal_risk_is_zero_at_or_below_lower_threshold(
    temperature: float,
) -> None:
    assert thermal_risk(temperature, 2.0, 6.0) == 0.0


@pytest.mark.parametrize("temperature, expected", [(4.0, 0.5), (3.0, 0.25)])
def test_thermal_risk_interpolates_inside_thresholds(
    temperature: float,
    expected: float,
) -> None:
    assert thermal_risk(temperature, 2.0, 6.0) == expected


@pytest.mark.parametrize("temperature", [6.0, 7.0])
def test_thermal_risk_is_one_at_or_above_upper_threshold(
    temperature: float,
) -> None:
    assert thermal_risk(temperature, 2.0, 6.0) == 1.0


@pytest.mark.parametrize("t_min, t_max", [(2.0, 2.0), (6.0, 2.0)])
def test_thermal_risk_rejects_invalid_threshold_order(
    t_min: float,
    t_max: float,
) -> None:
    with pytest.raises(ValueError, match="t_min must be less than t_max"):
        thermal_risk(4.0, t_min, t_max)


@pytest.mark.parametrize(
    "temperature, t_min, t_max",
    [
        (float("nan"), 2.0, 6.0),
        (4.0, float("inf"), 6.0),
        (4.0, 2.0, float("-inf")),
    ],
)
def test_thermal_risk_rejects_non_finite_values(
    temperature: float,
    t_min: float,
    t_max: float,
) -> None:
    with pytest.raises(ValueError, match="finite number"):
        thermal_risk(temperature, t_min, t_max)


@pytest.mark.parametrize(
    "temperature, t_min, t_max",
    [
        (True, 2.0, 6.0),
        (4.0, False, 6.0),
        (4.0, 2.0, True),
    ],
)
def test_thermal_risk_rejects_bool_values(
    temperature: object,
    t_min: object,
    t_max: object,
) -> None:
    with pytest.raises(ValueError, match="finite number"):
        thermal_risk(temperature, t_min, t_max)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "temperature, t_min, t_max",
    [
        ("4", 2.0, 6.0),
        (4.0, "2", 6.0),
        (4.0, 2.0, "6"),
    ],
)
def test_thermal_risk_rejects_non_numeric_values(
    temperature: object,
    t_min: object,
    t_max: object,
) -> None:
    with pytest.raises(ValueError, match="finite number"):
        thermal_risk(temperature, t_min, t_max)  # type: ignore[arg-type]
