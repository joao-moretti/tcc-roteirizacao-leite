"""Official algorithm hyperparameters selected for the final experiments."""

from typing import Final

# These values were selected through independent pilot calibration at the 5,
# 15, and 20 producer scales.
# One global parameterization per algorithm avoids scale-specific final tuning.
SA_INITIAL_TEMPERATURE: Final[float] = 0.5
SA_COOLING_RATE: Final[float] = 0.95
SA_ITERATIONS: Final[int] = 500

ACO_BETA: Final[float] = 3.0
ACO_EVAPORATION_RATE: Final[float] = 0.1
