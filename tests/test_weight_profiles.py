import pytest

from src.models.candidate_route import CandidateRoute
from src.models.problem import CollectionPoint, Depot, ProblemInstance
from src.models.road_network import RoadArc, RoadNetwork, RoadNode
from src.objective.multicriteria import weighted_objective_cost
from src.objective.weight_profiles import (
    BALANCED_WEIGHTS,
    LOGISTIC_WEIGHTS,
    THERMAL_WEIGHTS,
)
from src.simulation.snapshot import OptimizationSnapshot


def make_case() -> tuple[ProblemInstance, OptimizationSnapshot, CandidateRoute]:
    point = CollectionPoint(point_id=10, node_id=2)
    instance = ProblemInstance(
        instance_id=1,
        road_network=RoadNetwork(
            network_id=1,
            nodes=[
                RoadNode(node_id=1, x=0.0, y=0.0),
                RoadNode(node_id=2, x=1.0, y=0.0),
            ],
            arcs=[RoadArc(1, 2, 4.0), RoadArc(2, 1, 6.0)],
        ),
        depot=Depot(depot_id=1, node_id=1),
        collection_points=[point],
        vehicle_capacity=10.0,
        collection_service_duration=60.0,
        v_ref=2.0,
        sample_period=30.0,
        temperature_trajectories={point.point_id: (4.0,)},
        volume_trajectories={point.point_id: (2.0,)},
    )
    snapshot = OptimizationSnapshot(
        current_time=0.0,
        current_node_id=1,
        current_load=0.0,
        remaining_point_ids=(10,),
        temperatures={10: 4.0},
        volumes={10: 2.0},
    )
    return instance, snapshot, CandidateRoute((10,))


def test_logistic_weights_are_exact() -> None:
    assert LOGISTIC_WEIGHTS == (0.60, 0.20, 0.20)


def test_balanced_weights_are_equal_thirds() -> None:
    assert BALANCED_WEIGHTS == (1 / 3, 1 / 3, 1 / 3)


def test_thermal_weights_are_exact() -> None:
    assert THERMAL_WEIGHTS == (0.20, 0.60, 0.20)


@pytest.mark.parametrize(
    "weights",
    [LOGISTIC_WEIGHTS, BALANCED_WEIGHTS, THERMAL_WEIGHTS],
)
def test_each_profile_sums_to_one(weights: tuple[float, float, float]) -> None:
    assert sum(weights) == pytest.approx(1.0)


def test_profile_order_is_logistic_thermal_occupancy() -> None:
    w_logistic, w_thermal, w_occupancy = LOGISTIC_WEIGHTS

    assert w_logistic == 0.60
    assert w_thermal == 0.20
    assert w_occupancy == 0.20


@pytest.mark.parametrize(
    "profile, expected",
    [
        (LOGISTIC_WEIGHTS, 1.7),
        (BALANCED_WEIGHTS, 1.5),
        (THERMAL_WEIGHTS, 1.1),
    ],
)
def test_profiles_can_be_expanded_into_weighted_objective(
    profile: tuple[float, float, float],
    expected: float,
) -> None:
    instance, snapshot, candidate_route = make_case()

    result = weighted_objective_cost(
        instance,
        snapshot,
        candidate_route,
        5.0,
        2.0,
        2.0,
        2.0,
        6.0,
        *profile,
    )

    assert result == pytest.approx(expected)
