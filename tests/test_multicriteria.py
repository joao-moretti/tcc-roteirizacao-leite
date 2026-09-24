import pytest

from src.models.candidate_route import CandidateRoute
from src.models.problem import CollectionPoint, Depot, ProblemInstance
from src.models.road_network import RoadArc, RoadNetwork, RoadNode
from src.objective.multicriteria import weighted_objective_cost
from src.simulation.snapshot import OptimizationSnapshot


def make_instance(
    arcs: list[RoadArc],
    *,
    collection_points: list[CollectionPoint] | None = None,
) -> ProblemInstance:
    points = collection_points
    if points is None:
        points = [CollectionPoint(point_id=10, node_id=2)]

    trajectory = tuple(4.0 for _ in range(20))
    volumes = tuple(1.0 for _ in range(20))
    return ProblemInstance(
        instance_id=1,
        road_network=RoadNetwork(
            network_id=1,
            nodes=[
                RoadNode(node_id=node_id, x=float(node_id), y=0.0)
                for node_id in range(1, 4)
            ],
            arcs=arcs,
        ),
        depot=Depot(depot_id=1, node_id=1),
        collection_points=points,
        vehicle_capacity=10.0,
        collection_service_duration=60.0,
        v_ref=2.0,
        sample_period=30.0,
        temperature_trajectories={point.point_id: trajectory for point in points},
        volume_trajectories={point.point_id: volumes for point in points},
    )


def make_snapshot(
    remaining_point_ids: tuple[int, ...],
    *,
    temperatures: dict[int, float] | None = None,
    volumes: dict[int, float] | None = None,
) -> OptimizationSnapshot:
    return OptimizationSnapshot(
        current_time=0.0,
        current_node_id=1,
        current_load=0.0,
        remaining_point_ids=remaining_point_ids,
        temperatures=temperatures
        or {point_id: 4.0 for point_id in remaining_point_ids},
        volumes=volumes or {point_id: 2.0 for point_id in remaining_point_ids},
    )


def one_point_case() -> tuple[ProblemInstance, OptimizationSnapshot, CandidateRoute]:
    instance = make_instance([RoadArc(1, 2, 4.0), RoadArc(2, 1, 6.0)])
    snapshot = make_snapshot((10,))
    return instance, snapshot, CandidateRoute((10,))


def test_weighted_objective_combines_three_components() -> None:
    instance, snapshot, candidate_route = one_point_case()

    result = weighted_objective_cost(
        instance,
        snapshot,
        candidate_route,
        logistic_reference_distance=5.0,
        thermal_reference_value=2.0,
        occupancy_reference_value=2.0,
        t_min=2.0,
        t_max=6.0,
        w_logistic=0.1,
        w_thermal=0.2,
        w_occupancy=0.7,
    )

    assert result == pytest.approx(1.7)


def test_weights_are_assigned_to_their_respective_components() -> None:
    instance, snapshot, candidate_route = one_point_case()

    result = weighted_objective_cost(
        instance,
        snapshot,
        candidate_route,
        logistic_reference_distance=5.0,
        thermal_reference_value=1.0,
        occupancy_reference_value=4.0,
        t_min=2.0,
        t_max=6.0,
        w_logistic=0.6,
        w_thermal=0.2,
        w_occupancy=0.2,
    )

    assert result == pytest.approx(1.6)


def test_logistic_weight_profile_is_accepted() -> None:
    instance, snapshot, candidate_route = one_point_case()

    result = weighted_objective_cost(
        instance,
        snapshot,
        candidate_route,
        logistic_reference_distance=10.0,
        thermal_reference_value=1.0,
        occupancy_reference_value=4.0,
        t_min=2.0,
        t_max=6.0,
        w_logistic=0.60,
        w_thermal=0.20,
        w_occupancy=0.20,
    )

    assert result == pytest.approx(1.0)


def test_fractional_equal_weights_are_accepted() -> None:
    instance, snapshot, candidate_route = one_point_case()

    result = weighted_objective_cost(
        instance,
        snapshot,
        candidate_route,
        logistic_reference_distance=10.0,
        thermal_reference_value=1.0,
        occupancy_reference_value=4.0,
        t_min=2.0,
        t_max=6.0,
        w_logistic=1 / 3,
        w_thermal=1 / 3,
        w_occupancy=1 / 3,
    )

    assert result == pytest.approx(1.0)


def test_objective_result_may_be_greater_than_one() -> None:
    instance, snapshot, candidate_route = one_point_case()

    result = weighted_objective_cost(
        instance,
        snapshot,
        candidate_route,
        logistic_reference_distance=5.0,
        thermal_reference_value=1.0,
        occupancy_reference_value=4.0,
        t_min=2.0,
        t_max=6.0,
        w_logistic=0.6,
        w_thermal=0.2,
        w_occupancy=0.2,
    )

    assert result == pytest.approx(1.6)
    assert result > 1.0


def test_terminal_snapshot_produces_zero() -> None:
    instance = make_instance([], collection_points=[])
    snapshot = make_snapshot(())

    assert (
        weighted_objective_cost(
            instance,
            snapshot,
            CandidateRoute(()),
            logistic_reference_distance=0.0,
            thermal_reference_value=0.0,
            occupancy_reference_value=0.0,
            t_min=2.0,
            t_max=6.0,
            w_logistic=0.6,
            w_thermal=0.2,
            w_occupancy=0.2,
        )
        == 0.0
    )


@pytest.mark.parametrize(
    "weight_name,bad_value",
    [
        ("w_logistic", -0.1),
        ("w_thermal", float("nan")),
        ("w_occupancy", float("inf")),
        ("w_logistic", True),
    ],
)
def test_invalid_weight_is_rejected(weight_name: str, bad_value: object) -> None:
    instance, snapshot, candidate_route = one_point_case()
    weights: dict[str, object] = {
        "w_logistic": 0.5,
        "w_thermal": 0.3,
        "w_occupancy": 0.2,
    }
    weights[weight_name] = bad_value

    with pytest.raises(ValueError, match=weight_name):
        weighted_objective_cost(  # type: ignore[arg-type]
            instance,
            snapshot,
            candidate_route,
            10.0,
            1.0,
            4.0,
            2.0,
            6.0,
            weights["w_logistic"],
            weights["w_thermal"],
            weights["w_occupancy"],
        )


@pytest.mark.parametrize("weights", [(0.5, 0.2, 0.2), (0.6, 0.6, 0.2)])
def test_weights_must_sum_to_one(weights: tuple[float, float, float]) -> None:
    instance, snapshot, candidate_route = one_point_case()

    with pytest.raises(ValueError, match="sum to 1.0"):
        weighted_objective_cost(
            instance,
            snapshot,
            candidate_route,
            10.0,
            1.0,
            4.0,
            2.0,
            6.0,
            *weights,
        )


def test_logistic_reference_error_is_propagated() -> None:
    instance, snapshot, candidate_route = one_point_case()

    with pytest.raises(ValueError, match="reference_distance"):
        weighted_objective_cost(
            instance,
            snapshot,
            candidate_route,
            -1.0,
            1.0,
            4.0,
            2.0,
            6.0,
            0.6,
            0.2,
            0.2,
        )


def test_thermal_reference_error_is_propagated() -> None:
    instance, snapshot, candidate_route = one_point_case()

    with pytest.raises(ValueError, match="reference"):
        weighted_objective_cost(
            instance,
            snapshot,
            candidate_route,
            10.0,
            -1.0,
            4.0,
            2.0,
            6.0,
            0.6,
            0.2,
            0.2,
        )


def test_occupancy_reference_error_is_propagated() -> None:
    instance, snapshot, candidate_route = one_point_case()

    with pytest.raises(ValueError, match="reference"):
        weighted_objective_cost(
            instance,
            snapshot,
            candidate_route,
            10.0,
            1.0,
            -1.0,
            2.0,
            6.0,
            0.6,
            0.2,
            0.2,
        )


def test_invalid_thermal_limits_are_propagated() -> None:
    instance, snapshot, candidate_route = one_point_case()

    with pytest.raises(ValueError, match="t_min must be less than t_max"):
        weighted_objective_cost(
            instance,
            snapshot,
            candidate_route,
            10.0,
            1.0,
            4.0,
            6.0,
            2.0,
            0.6,
            0.2,
            0.2,
        )


def test_invalid_candidate_route_is_propagated() -> None:
    instance, snapshot, _ = one_point_case()

    with pytest.raises(ValueError, match="missing"):
        weighted_objective_cost(
            instance,
            snapshot,
            CandidateRoute(()),
            10.0,
            1.0,
            4.0,
            2.0,
            6.0,
            0.6,
            0.2,
            0.2,
        )


def test_route_order_can_change_weighted_objective() -> None:
    instance = make_instance(
        [
            RoadArc(1, 2, 2.0),
            RoadArc(2, 3, 4.0),
            RoadArc(3, 1, 6.0),
            RoadArc(1, 3, 10.0),
            RoadArc(3, 2, 1.0),
            RoadArc(2, 1, 7.0),
        ],
        collection_points=[
            CollectionPoint(point_id=10, node_id=2),
            CollectionPoint(point_id=20, node_id=3),
        ],
    )
    snapshot = OptimizationSnapshot(
        current_time=0.0,
        current_node_id=1,
        current_load=0.0,
        remaining_point_ids=(10, 20),
        temperatures={10: 6.0, 20: 3.0},
        volumes={10: 1.0, 20: 2.0},
    )

    first_result = weighted_objective_cost(
        instance,
        snapshot,
        CandidateRoute((10, 20)),
        12.0,
        7.0,
        14.0,
        2.0,
        6.0,
        1 / 3,
        1 / 3,
        1 / 3,
    )
    second_result = weighted_objective_cost(
        instance,
        snapshot,
        CandidateRoute((20, 10)),
        12.0,
        7.0,
        14.0,
        2.0,
        6.0,
        1 / 3,
        1 / 3,
        1 / 3,
    )

    assert first_result != second_result
