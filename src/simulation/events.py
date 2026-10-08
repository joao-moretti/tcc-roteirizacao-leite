"""Generic recalculation-event detection for the dynamic simulation."""

from dataclasses import dataclass
from math import isfinite
from typing import Literal

from src.models.problem import ProblemInstance
from src.simulation.snapshot import OptimizationSnapshot

ThermalBand = Literal[
    "safe",
    "attention",
    "critical",
]

VolumeBand = Literal[
    "low",
    "medium",
    "high",
]

RecalculationReason = Literal[
    "collection_completed",
    "thermal_band_changed",
    "volume_band_changed",
]


@dataclass(frozen=True)
class RecalculationDecision:
    """Result of evaluating detected recalculation reasons."""

    should_recalculate: bool
    reasons: tuple[RecalculationReason, ...]


def _validate_finite_number(value: object, field_name: str) -> float:
    """Validate and normalize a finite numeric value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite numeric value.")

    normalized = float(value)
    if not isfinite(normalized):
        raise ValueError(f"{field_name} must be a finite numeric value.")

    return normalized


def _validate_non_negative_number(value: object, field_name: str) -> float:
    """Validate and normalize a finite non-negative number."""
    normalized = _validate_finite_number(value, field_name)
    if normalized < 0.0:
        raise ValueError(f"{field_name} must be greater than or equal to zero.")

    return normalized


def _validate_thermal_limits(t_min: object, t_max: object) -> tuple[float, float]:
    """Validate and normalize the thermal-band limits."""
    normalized_t_min = _validate_finite_number(t_min, "t_min")
    normalized_t_max = _validate_finite_number(t_max, "t_max")
    if normalized_t_min >= normalized_t_max:
        raise ValueError("t_min must be less than t_max.")

    return normalized_t_min, normalized_t_max


def _validate_volume_limits(q_low: object, q_high: object) -> tuple[float, float]:
    """Validate and normalize the volume-band limits."""
    normalized_q_low = _validate_non_negative_number(q_low, "q_low")
    normalized_q_high = _validate_non_negative_number(q_high, "q_high")
    if normalized_q_low >= normalized_q_high:
        raise ValueError("q_low must be less than q_high.")

    return normalized_q_low, normalized_q_high


def thermal_band(
    temperature: float,
    t_min: float,
    t_max: float,
) -> ThermalBand:
    """Classify a temperature as safe, attention, or critical."""
    normalized_temperature = _validate_finite_number(temperature, "temperature")
    normalized_t_min, normalized_t_max = _validate_thermal_limits(t_min, t_max)

    if normalized_temperature <= normalized_t_min:
        return "safe"
    if normalized_temperature < normalized_t_max:
        return "attention"
    return "critical"


def volume_band(
    volume: float,
    q_low: float,
    q_high: float,
) -> VolumeBand:
    """Classify a volume as low, medium, or high."""
    normalized_volume = _validate_non_negative_number(volume, "volume")
    normalized_q_low, normalized_q_high = _validate_volume_limits(q_low, q_high)

    if normalized_volume <= normalized_q_low:
        return "low"
    if normalized_volume < normalized_q_high:
        return "medium"
    return "high"


def detect_recalculation_reasons(
    instance: ProblemInstance,
    previous_snapshot: OptimizationSnapshot,
    current_snapshot: OptimizationSnapshot,
    *,
    t_min: float,
    t_max: float,
    q_low: float,
    q_high: float,
) -> tuple[RecalculationReason, ...]:
    """Detect collection and band-change reasons between two snapshots."""
    previous_snapshot.validate(instance)
    current_snapshot.validate(instance)
    normalized_t_min, normalized_t_max = _validate_thermal_limits(t_min, t_max)
    normalized_q_low, normalized_q_high = _validate_volume_limits(q_low, q_high)

    if current_snapshot.current_time < previous_snapshot.current_time:
        raise ValueError(
            "current_snapshot.current_time cannot be less than "
            "previous_snapshot.current_time."
        )

    previous_ids = set(previous_snapshot.remaining_point_ids)
    current_ids = set(current_snapshot.remaining_point_ids)
    reappeared_ids = current_ids - previous_ids
    if reappeared_ids:
        raise ValueError(
            "remaining_point_ids cannot increase or reappear: "
            f"{sorted(reappeared_ids)}."
        )

    reasons: list[RecalculationReason] = []
    if previous_ids - current_ids:
        reasons.append("collection_completed")

    common_ids = previous_ids & current_ids
    thermal_changed = any(
        thermal_band(
            previous_snapshot.temperatures[point_id],
            normalized_t_min,
            normalized_t_max,
        )
        != thermal_band(
            current_snapshot.temperatures[point_id],
            normalized_t_min,
            normalized_t_max,
        )
        for point_id in common_ids
    )
    if thermal_changed:
        reasons.append("thermal_band_changed")

    volume_changed = any(
        volume_band(
            previous_snapshot.volumes[point_id],
            normalized_q_low,
            normalized_q_high,
        )
        != volume_band(
            current_snapshot.volumes[point_id],
            normalized_q_low,
            normalized_q_high,
        )
        for point_id in common_ids
    )
    if volume_changed:
        reasons.append("volume_band_changed")

    return tuple(reasons)


def minimum_recalculation_interval_elapsed(
    current_time: float,
    last_recalculation_time: float | None,
    minimum_interval: float,
) -> bool:
    """Return whether the inclusive minimum recalculation interval elapsed."""
    normalized_current_time = _validate_non_negative_number(
        current_time,
        "current_time",
    )
    normalized_minimum_interval = _validate_non_negative_number(
        minimum_interval,
        "minimum_interval",
    )
    if last_recalculation_time is None:
        return True

    normalized_last_time = _validate_non_negative_number(
        last_recalculation_time,
        "last_recalculation_time",
    )
    if normalized_current_time < normalized_last_time:
        raise ValueError("current_time cannot be less than last_recalculation_time.")

    return normalized_current_time - normalized_last_time >= normalized_minimum_interval


def evaluate_recalculation(
    instance: ProblemInstance,
    previous_snapshot: OptimizationSnapshot,
    current_snapshot: OptimizationSnapshot,
    *,
    t_min: float,
    t_max: float,
    q_low: float,
    q_high: float,
    last_recalculation_time: float | None,
    minimum_interval: float,
) -> RecalculationDecision:
    """Evaluate detected reasons and apply the V1 recalculation cooldown."""
    reasons = detect_recalculation_reasons(
        instance,
        previous_snapshot,
        current_snapshot,
        t_min=t_min,
        t_max=t_max,
        q_low=q_low,
        q_high=q_high,
    )

    if not reasons:
        return RecalculationDecision(False, ())
    if "collection_completed" in reasons:
        return RecalculationDecision(True, reasons)

    should_recalculate = minimum_recalculation_interval_elapsed(
        current_snapshot.current_time,
        last_recalculation_time,
        minimum_interval,
    )
    return RecalculationDecision(should_recalculate, reasons)
