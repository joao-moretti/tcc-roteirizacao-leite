import pytest

from src.models.problem import CollectionPoint, Depot, ProblemInstance
from src.models.road_network import RoadNetwork, RoadNode
from src.simulation.events import (
    RecalculationDecision,
    detect_recalculation_reasons,
    evaluate_recalculation,
    minimum_recalculation_interval_elapsed,
    thermal_band,
    volume_band,
)
from src.simulation.snapshot import OptimizationSnapshot

T_MIN = 4.0
T_MAX = 7.0
Q_LOW = 2.0
Q_HIGH = 5.0


def make_instance() -> ProblemInstance:
    points = [
        CollectionPoint(point_id=10, node_id=2),
        CollectionPoint(point_id=20, node_id=3),
        CollectionPoint(point_id=30, node_id=4),
    ]
    return ProblemInstance(
        instance_id=1,
        road_network=RoadNetwork(
            network_id=1,
            nodes=[
                RoadNode(node_id=node_id, x=float(node_id), y=0.0)
                for node_id in range(1, 5)
            ],
            arcs=[],
        ),
        depot=Depot(depot_id=1, node_id=1),
        collection_points=points,
        vehicle_capacity=20.0,
        collection_service_duration=60.0,
        v_ref=2.0,
        sample_period=30.0,
        temperature_trajectories={point.point_id: (1.0, 2.0, 3.0) for point in points},
        volume_trajectories={point.point_id: (1.0, 2.0, 3.0) for point in points},
    )


def make_snapshot(
    *,
    current_time: float = 0.0,
    remaining_point_ids: tuple[int, ...] = (10, 20, 30),
    temperatures: dict[int, float] | None = None,
    volumes: dict[int, float] | None = None,
) -> OptimizationSnapshot:
    if temperatures is None:
        temperatures = {point_id: 5.0 for point_id in remaining_point_ids}
    if volumes is None:
        volumes = {point_id: 3.0 for point_id in remaining_point_ids}

    return OptimizationSnapshot(
        current_time=current_time,
        current_node_id=1,
        current_load=0.0,
        remaining_point_ids=remaining_point_ids,
        temperatures=temperatures,
        volumes=volumes,
    )


def detect(
    previous_snapshot: OptimizationSnapshot,
    current_snapshot: OptimizationSnapshot,
) -> tuple[str, ...]:
    return detect_recalculation_reasons(
        make_instance(),
        previous_snapshot,
        current_snapshot,
        t_min=T_MIN,
        t_max=T_MAX,
        q_low=Q_LOW,
        q_high=Q_HIGH,
    )


@pytest.mark.parametrize(
    ("temperature", "expected_band"),
    [
        (3.0, "safe"),
        (4.0, "safe"),
        (5.0, "attention"),
        (7.0, "critical"),
        (8.0, "critical"),
    ],
)
def test_thermal_band_classifies_boundaries(
    temperature: float,
    expected_band: str,
) -> None:
    assert thermal_band(temperature, T_MIN, T_MAX) == expected_band


@pytest.mark.parametrize("limits", [(4.0, 4.0), (7.0, 4.0)])
def test_thermal_band_rejects_invalid_limits(limits: tuple[float, float]) -> None:
    with pytest.raises(ValueError, match="t_min"):
        thermal_band(5.0, *limits)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "5"])
def test_thermal_band_rejects_invalid_temperature(value: object) -> None:
    with pytest.raises(ValueError, match="temperature"):
        thermal_band(value, T_MIN, T_MAX)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("volume", "expected_band"),
    [
        (1.0, "low"),
        (2.0, "low"),
        (3.0, "medium"),
        (5.0, "high"),
        (6.0, "high"),
    ],
)
def test_volume_band_classifies_boundaries(volume: float, expected_band: str) -> None:
    assert volume_band(volume, Q_LOW, Q_HIGH) == expected_band


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), True, "3"])
def test_volume_band_rejects_invalid_volume(value: object) -> None:
    with pytest.raises(ValueError, match="volume"):
        volume_band(value, Q_LOW, Q_HIGH)  # type: ignore[arg-type]


@pytest.mark.parametrize("limits", [(2.0, 2.0), (5.0, 2.0), (-1.0, 2.0)])
def test_volume_band_rejects_invalid_limits(limits: tuple[float, float]) -> None:
    with pytest.raises(ValueError, match="q_"):
        volume_band(3.0, *limits)


def test_identical_snapshots_have_no_recalculation_reason() -> None:
    snapshot = make_snapshot()

    assert detect(snapshot, snapshot) == ()


@pytest.mark.parametrize(
    ("previous_temperature", "current_temperature"),
    [(3.0, 4.0), (4.5, 5.5), (7.0, 8.0)],
)
def test_temperature_change_within_same_band_has_no_reason(
    previous_temperature: float,
    current_temperature: float,
) -> None:
    previous = make_snapshot(temperatures={10: previous_temperature, 20: 5.0, 30: 5.0})
    current = make_snapshot(temperatures={10: current_temperature, 20: 5.0, 30: 5.0})

    assert detect(previous, current) == ()


@pytest.mark.parametrize(
    ("previous_temperature", "current_temperature"),
    [(3.0, 5.0), (5.0, 8.0), (8.0, 5.0)],
)
def test_temperature_band_change_generates_one_reason(
    previous_temperature: float,
    current_temperature: float,
) -> None:
    previous = make_snapshot(temperatures={10: previous_temperature, 20: 5.0, 30: 5.0})
    current = make_snapshot(temperatures={10: current_temperature, 20: 5.0, 30: 5.0})

    assert detect(previous, current) == ("thermal_band_changed",)


def test_volume_change_within_same_band_has_no_reason() -> None:
    previous = make_snapshot(volumes={10: 2.1, 20: 3.0, 30: 3.0})
    current = make_snapshot(volumes={10: 4.9, 20: 3.0, 30: 3.0})

    assert detect(previous, current) == ()


@pytest.mark.parametrize(
    ("previous_volume", "current_volume"),
    [(1.0, 3.0), (3.0, 6.0)],
)
def test_volume_band_change_generates_one_reason(
    previous_volume: float,
    current_volume: float,
) -> None:
    previous = make_snapshot(volumes={10: previous_volume, 20: 3.0, 30: 3.0})
    current = make_snapshot(volumes={10: current_volume, 20: 3.0, 30: 3.0})

    assert detect(previous, current) == ("volume_band_changed",)


def test_removed_point_generates_collection_only() -> None:
    previous = make_snapshot()
    current = make_snapshot(remaining_point_ids=(20, 30))

    assert detect(previous, current) == ("collection_completed",)


def test_removed_point_does_not_generate_band_reasons() -> None:
    previous = make_snapshot(
        temperatures={10: 3.0, 20: 5.0, 30: 3.0},
        volumes={10: 1.0, 20: 3.0, 30: 1.0},
    )
    current = make_snapshot(
        remaining_point_ids=(20, 30),
        temperatures={20: 5.0, 30: 3.0},
        volumes={20: 3.0, 30: 1.0},
    )

    assert detect(previous, current) == ("collection_completed",)


def test_all_reasons_are_unique_and_returned_in_official_order() -> None:
    previous = make_snapshot(
        temperatures={10: 3.0, 20: 5.0, 30: 5.0},
        volumes={10: 1.0, 20: 3.0, 30: 3.0},
    )
    current = make_snapshot(
        remaining_point_ids=(20, 30),
        temperatures={20: 8.0, 30: 5.0},
        volumes={20: 6.0, 30: 3.0},
    )

    assert detect(previous, current) == (
        "collection_completed",
        "thermal_band_changed",
        "volume_band_changed",
    )


def test_reappearing_point_is_rejected() -> None:
    previous = make_snapshot(remaining_point_ids=(20,))
    current = make_snapshot(remaining_point_ids=(10, 20))

    with pytest.raises(ValueError, match="cannot increase or reappear"):
        detect(previous, current)


def test_time_moving_backwards_is_rejected() -> None:
    previous = make_snapshot(current_time=30.0)
    current = make_snapshot(current_time=0.0)

    with pytest.raises(ValueError, match="cannot be less"):
        detect(previous, current)


@pytest.mark.parametrize(
    ("current_time", "last_time", "minimum_interval", "expected"),
    [
        (10.0, None, 300.0, True),
        (599.999, 300.0, 300.0, False),
        (600.0, 300.0, 300.0, True),
        (600.001, 300.0, 300.0, True),
    ],
)
def test_minimum_recalculation_interval(
    current_time: float,
    last_time: float | None,
    minimum_interval: float,
    expected: bool,
) -> None:
    assert (
        minimum_recalculation_interval_elapsed(
            current_time,
            last_time,
            minimum_interval,
        )
        is expected
    )


@pytest.mark.parametrize(
    ("current_time", "last_time", "minimum_interval"),
    [
        (-1.0, None, 1.0),
        (1.0, None, -1.0),
        (1.0, float("nan"), 1.0),
        (True, None, 1.0),
    ],
)
def test_minimum_recalculation_interval_rejects_invalid_values(
    current_time: object,
    last_time: object,
    minimum_interval: object,
) -> None:
    with pytest.raises(ValueError):
        minimum_recalculation_interval_elapsed(
            current_time,  # type: ignore[arg-type]
            last_time,  # type: ignore[arg-type]
            minimum_interval,  # type: ignore[arg-type]
        )


def test_minimum_recalculation_interval_rejects_time_before_last_recalculation() -> (
    None
):
    with pytest.raises(ValueError, match="cannot be less"):
        minimum_recalculation_interval_elapsed(10.0, 20.0, 1.0)


def evaluate(
    previous_snapshot: OptimizationSnapshot,
    current_snapshot: OptimizationSnapshot,
    *,
    last_recalculation_time: float | None = 0.0,
    minimum_interval: float = 300.0,
) -> RecalculationDecision:
    return evaluate_recalculation(
        make_instance(),
        previous_snapshot,
        current_snapshot,
        t_min=T_MIN,
        t_max=T_MAX,
        q_low=Q_LOW,
        q_high=Q_HIGH,
        last_recalculation_time=last_recalculation_time,
        minimum_interval=minimum_interval,
    )


def test_evaluate_returns_false_without_events() -> None:
    snapshot = make_snapshot()

    assert evaluate(snapshot, snapshot) == RecalculationDecision(False, ())


def test_evaluate_collection_ignores_cooldown() -> None:
    previous = make_snapshot()
    current = make_snapshot(remaining_point_ids=(20, 30))

    assert evaluate(previous, current, minimum_interval=300.0) == RecalculationDecision(
        True,
        ("collection_completed",),
    )


def test_evaluate_thermal_event_respects_cooldown() -> None:
    previous = make_snapshot(temperatures={10: 3.0, 20: 5.0, 30: 5.0})
    current = make_snapshot(
        current_time=30.0,
        temperatures={10: 5.0, 20: 5.0, 30: 5.0},
    )

    assert evaluate(
        previous, current, last_recalculation_time=0.0
    ) == RecalculationDecision(
        False,
        ("thermal_band_changed",),
    )
    assert evaluate(
        previous, current, last_recalculation_time=None
    ) == RecalculationDecision(
        True,
        ("thermal_band_changed",),
    )


def test_evaluate_volume_event_respects_cooldown() -> None:
    previous = make_snapshot(volumes={10: 1.0, 20: 3.0, 30: 3.0})
    current = make_snapshot(
        current_time=30.0,
        volumes={10: 3.0, 20: 3.0, 30: 3.0},
    )

    assert evaluate(
        previous, current, last_recalculation_time=0.0
    ) == RecalculationDecision(
        False,
        ("volume_band_changed",),
    )
    assert evaluate(
        previous, current, last_recalculation_time=None
    ) == RecalculationDecision(
        True,
        ("volume_band_changed",),
    )


def test_evaluate_thermal_and_volume_events_keep_both_reasons_when_blocked() -> None:
    previous = make_snapshot(
        current_time=0.0,
        temperatures={10: 3.0, 20: 5.0, 30: 5.0},
        volumes={10: 1.0, 20: 3.0, 30: 3.0},
    )
    current = make_snapshot(
        current_time=30.0,
        temperatures={10: 5.0, 20: 5.0, 30: 5.0},
        volumes={10: 3.0, 20: 3.0, 30: 3.0},
    )

    decision = evaluate(previous, current, last_recalculation_time=0.0)

    assert decision == RecalculationDecision(
        False,
        ("thermal_band_changed", "volume_band_changed"),
    )


def test_evaluate_collection_and_thermal_always_recalculates() -> None:
    previous = make_snapshot(temperatures={10: 3.0, 20: 5.0, 30: 5.0})
    current = make_snapshot(
        remaining_point_ids=(20, 30),
        temperatures={20: 5.0, 30: 5.0},
    )
    current = OptimizationSnapshot(
        current_time=current.current_time,
        current_node_id=current.current_node_id,
        current_load=current.current_load,
        remaining_point_ids=current.remaining_point_ids,
        temperatures={20: 8.0, 30: 5.0},
        volumes=dict(current.volumes),
    )

    assert (
        evaluate(previous, current, minimum_interval=300.0).should_recalculate is True
    )
