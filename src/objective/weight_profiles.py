"""Official weight profiles for Formulation V1."""

from typing import Final

# Values are ordered as (w_logistic, w_thermal, w_occupancy).
LOGISTIC_WEIGHTS: Final[tuple[float, float, float]] = (0.60, 0.20, 0.20)
BALANCED_WEIGHTS: Final[tuple[float, float, float]] = (1 / 3, 1 / 3, 1 / 3)
THERMAL_WEIGHTS: Final[tuple[float, float, float]] = (0.20, 0.60, 0.20)
