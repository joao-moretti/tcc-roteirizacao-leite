import pytest

from src.models.candidate_route import CandidateRoute
from src.models.problem import CollectionPoint, Depot, ProblemInstance
from src.models.road_network import RoadArc, RoadNetwork, RoadNode
from src.objective.thermal import thermal_cost, thermal_risk
from src.simulation.snapshot import OptimizationSnapshot


def make_instance(
    arcs: list[RoadArc],
    *,
    trajectory_temperature: float = 4.0,
    collection_service_duration: float = 60.0,
    collection_points: list[CollectionPoint] | None = None,
) -> ProblemInstance:
    points = collection_points
    if points is None:
        points = [
            CollectionPoint(point_id=10, node_id=2),
            CollectionPoint(point_id=20, node_id=3),
            CollectionPoint(point_id=30, node_id=4),
        ]

    temperature_trajectory = tuple(trajectory_temperature for _ in range(20))
    volume_trajectory = tuple(1.0 for _ in range(20))

    return ProblemInstance(
        instance_id=1,
        road_network=RoadNetwork(
            network_id=1,
            nodes=[
                RoadNode(node_id=node_id, x=float(node_id), y=0.0)
                for node_id in range(1, 6)
            ],
            arcs=arcs,
        ),
        depot=Depot(depot_id=1, node_id=1),
        collection_points=points,
        vehicle_capacity=10.0,
        collection_service_duration=collection_service_duration,
        v_ref=2.0,
        sample_period=30.0,
        temperature_trajectories={
            point.point_id: temperature_trajectory for point in points
        },
        volume_trajectories={point.point_id: volume_trajectory for point in points},
    )


def make_snapshot(
    remaining_point_ids: tuple[int, ...],
    temperatures: dict[int, float],
    *,
    current_node_id: int = 1,
    current_time: float = 300.0,
) -> OptimizationSnapshot:
    return OptimizationSnapshot(
        current_time=current_time,
        current_node_id=current_node_id,
        current_load=0.0,
        remaining_point_ids=remaining_point_ids,
        temperatures=temperatures,
        volumes={point_id: 1.0 for point_id in remaining_point_ids},
    )


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


def test_thermal_cost_for_one_point_with_zero_risk() -> None:
    instance = make_instance([RoadArc(1, 2, 4.0)])
    snapshot = make_snapshot((10,), {10: 1.0})

    assert thermal_cost(instance, snapshot, CandidateRoute((10,)), 2.0, 6.0) == 0.0


def test_thermal_cost_for_one_point_with_intermediate_risk() -> None:
    instance = make_instance([RoadArc(1, 2, 4.0)])
    snapshot = make_snapshot((10,), {10: 4.0})

    assert thermal_cost(instance, snapshot, CandidateRoute((10,)), 2.0, 6.0) == 1.0


def test_thermal_cost_for_one_point_with_maximum_risk() -> None:
    instance = make_instance([RoadArc(1, 2, 4.0)])
    snapshot = make_snapshot((10,), {10: 6.0})

    assert thermal_cost(instance, snapshot, CandidateRoute((10,)), 2.0, 6.0) == 2.0


def test_thermal_cost_sums_risk_times_accumulated_arrival() -> None:
    instance = make_instance([RoadArc(1, 2, 4.0), RoadArc(2, 3, 6.0)])
    snapshot = make_snapshot((10, 20), {10: 4.0, 20: 6.0})

    assert thermal_cost(instance, snapshot, CandidateRoute((10, 20)), 2.0, 6.0) == 6.0


def test_candidate_route_order_changes_thermal_cost_when_risks_differ() -> None:
    instance = make_instance(
        [
            RoadArc(1, 2, 2.0),
            RoadArc(2, 3, 4.0),
            RoadArc(1, 3, 10.0),
            RoadArc(3, 2, 1.0),
        ]
    )
    snapshot = make_snapshot((10, 20), {10: 6.0, 20: 3.0})

    first_order_cost = thermal_cost(
        instance, snapshot, CandidateRoute((10, 20)), 2.0, 6.0
    )
    second_order_cost = thermal_cost(
        instance, snapshot, CandidateRoute((20, 10)), 2.0, 6.0
    )

    assert first_order_cost == 1.75
    assert second_order_cost == 4.25


def test_all_zero_risks_produce_zero_thermal_cost() -> None:
    instance = make_instance([RoadArc(1, 2, 4.0), RoadArc(2, 3, 6.0)])
    snapshot = make_snapshot((10, 20), {10: 1.0, 20: 2.0})

    assert thermal_cost(instance, snapshot, CandidateRoute((10, 20)), 2.0, 6.0) == 0.0


def test_empty_candidate_route_produces_zero_thermal_cost() -> None:
    instance = make_instance([], collection_points=[])
    snapshot = make_snapshot((), {})

    assert thermal_cost(instance, snapshot, CandidateRoute(()), 2.0, 6.0) == 0.0


def test_thermal_cost_uses_temperature_from_snapshot() -> None:
    instance = make_instance(
        [RoadArc(1, 2, 4.0)],
        trajectory_temperature=1.0,
    )
    snapshot = make_snapshot((10,), {10: 6.0})

    assert thermal_cost(instance, snapshot, CandidateRoute((10,)), 2.0, 6.0) == 2.0


def test_instance_temperature_trajectories_do_not_change_thermal_cost() -> None:
    arcs = [RoadArc(1, 2, 4.0)]
    instance_with_low_trajectory = make_instance(arcs, trajectory_temperature=1.0)
    instance_with_high_trajectory = make_instance(arcs, trajectory_temperature=6.0)
    snapshot = make_snapshot((10,), {10: 4.0})
    candidate_route = CandidateRoute((10,))

    low_trajectory_cost = thermal_cost(
        instance_with_low_trajectory,
        snapshot,
        candidate_route,
        2.0,
        6.0,
    )
    high_trajectory_cost = thermal_cost(
        instance_with_high_trajectory,
        snapshot,
        candidate_route,
        2.0,
        6.0,
    )

    assert low_trajectory_cost == 1.0
    assert high_trajectory_cost == 1.0


def test_thermal_cost_propagates_arrival_times_error() -> None:
    instance = make_instance([RoadArc(2, 1, 4.0)])
    snapshot = make_snapshot((10,), {10: 4.0})

    with pytest.raises(ValueError, match="no directed path"):
        thermal_cost(instance, snapshot, CandidateRoute((10,)), 2.0, 6.0)


def test_thermal_cost_propagates_thermal_risk_error() -> None:
    instance = make_instance([RoadArc(1, 2, 4.0)])
    snapshot = make_snapshot((10,), {10: 4.0})

    with pytest.raises(ValueError, match="t_min must be less than t_max"):
        thermal_cost(instance, snapshot, CandidateRoute((10,)), 6.0, 2.0)
