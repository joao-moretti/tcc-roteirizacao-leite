import pytest

from src.models.candidate_route import CandidateRoute
from src.models.problem import CollectionPoint, Depot, ProblemInstance
from src.models.road_network import RoadArc, RoadNetwork, RoadNode
from src.simulation.dynamic import (
    DynamicSimulationResult,
    run_dynamic_simulation,
)
from src.simulation.snapshot import OptimizationSnapshot


def make_instance(
    *,
    first_arc_distance: float = 4.0,
    second_arc_distance: float = 6.0,
    direct_arc_distance: float = 20.0,
    service_duration: float = 5.0,
    temperature_10: tuple[float, ...] = (3.0, 3.0, 3.0),
    volume_10: tuple[float, ...] = (1.0, 1.0, 1.0),
    collection_points: list[CollectionPoint] | None = None,
    arcs: list[RoadArc] | None = None,
) -> ProblemInstance:
    points = collection_points
    if points is None:
        points = [
            CollectionPoint(point_id=10, node_id=2),
            CollectionPoint(point_id=20, node_id=3),
        ]
    if arcs is None:
        arcs = [
            RoadArc(1, 2, first_arc_distance),
            RoadArc(2, 3, second_arc_distance),
            RoadArc(1, 3, direct_arc_distance),
            RoadArc(3, 2, 2.0),
            RoadArc(2, 1, 3.0),
        ]

    trajectory_length = len(temperature_10)
    trajectory_length = len(temperature_10)
    temperature_trajectories = {
        point.point_id: (
            temperature_10 if point.point_id == 10 else (3.0,) * trajectory_length
        )
        for point in points
    }
    volume_trajectories = {
        point.point_id: (
            volume_10 if point.point_id == 10 else (2.0,) * trajectory_length
        )
        for point in points
    }

    return ProblemInstance(
        instance_id=1,
        road_network=RoadNetwork(
            network_id=1,
            nodes=[
                RoadNode(node_id=1, x=0.0, y=0.0),
                RoadNode(node_id=2, x=100.0, y=0.0),
                RoadNode(node_id=3, x=200.0, y=0.0),
            ],
            arcs=arcs,
        ),
        depot=Depot(depot_id=1, node_id=1),
        collection_points=points,
        vehicle_capacity=20.0,
        collection_service_duration=service_duration,
        v_ref=1.0,
        sample_period=30.0,
        temperature_trajectories=temperature_trajectories,
        volume_trajectories=volume_trajectories,
    )


def ordered_planner(
    instance: ProblemInstance,
    snapshot: OptimizationSnapshot,
) -> CandidateRoute:
    """Return the remaining points in the snapshot order."""
    return CandidateRoute(snapshot.remaining_point_ids)


def test_run_dynamic_simulation_completes_trip_and_returns_to_depot() -> None:
    instance = make_instance()

    result = run_dynamic_simulation(
        instance,
        ordered_planner,
        t_min=4.0,
        t_max=7.0,
        q_low=2.0,
        q_high=5.0,
        minimum_interval=0.0,
    )

    assert isinstance(result, DynamicSimulationResult)
    assert result.final_snapshot.current_node_id == instance.depot.node_id
    assert result.final_snapshot.remaining_point_ids == ()
    assert dict(result.final_snapshot.temperatures) == {}
    assert dict(result.final_snapshot.volumes) == {}
    assert result.final_snapshot.current_load == 3.0
    assert result.total_distance == 15.0
    assert result.optimization_count == 2
    assert len(result.executions) == 2
    assert result.final_snapshot.current_time == 25.0


def test_route_planner_receives_snapshots_for_receding_horizon_steps() -> None:
    instance = make_instance()
    received: list[tuple[int, tuple[int, ...]]] = []

    def planner(
        _instance: ProblemInstance,
        snapshot: OptimizationSnapshot,
    ) -> CandidateRoute:
        received.append((snapshot.current_node_id, snapshot.remaining_point_ids))
        return CandidateRoute(snapshot.remaining_point_ids)

    result = run_dynamic_simulation(
        instance,
        planner,
        t_min=4.0,
        t_max=7.0,
        q_low=2.0,
        q_high=5.0,
        minimum_interval=0.0,
    )

    assert received == [(1, (10, 20)), (2, (20,))]
    assert result.optimization_count == 2
    assert len(result.executions) == 2


def test_intermediate_event_reoptimizes_from_new_node() -> None:
    instance = make_instance(
        first_arc_distance=30.0,
        second_arc_distance=5.0,
        direct_arc_distance=50.0,
        temperature_10=(3.0, 8.0, 8.0),
    )
    received: list[tuple[int, tuple[int, ...]]] = []

    def planner(
        _instance: ProblemInstance,
        snapshot: OptimizationSnapshot,
    ) -> CandidateRoute:
        received.append((snapshot.current_node_id, snapshot.remaining_point_ids))
        if len(received) == 1:
            return CandidateRoute((20, 10))
        return CandidateRoute(snapshot.remaining_point_ids)

    result = run_dynamic_simulation(
        instance,
        planner,
        t_min=4.0,
        t_max=7.0,
        q_low=2.0,
        q_high=5.0,
        minimum_interval=0.0,
    )

    assert received[0] == (1, (10, 20))
    assert received[1] == (2, (10, 20))
    assert result.executions[0].reached_target is False
    assert result.executions[0].distance_traveled == 30.0
    assert result.optimization_count == 3
    assert len(result.executions) == 3
    assert result.total_distance == 40.0
    assert result.final_snapshot.current_node_id == 1
    assert result.final_snapshot.remaining_point_ids == ()


def test_cooldown_blocked_event_does_not_add_planner_call_before_collection() -> None:
    instance = make_instance(
        first_arc_distance=30.0,
        second_arc_distance=5.0,
        direct_arc_distance=50.0,
        temperature_10=(3.0, 8.0, 8.0),
    )
    received: list[tuple[int, ...]] = []

    def planner(
        _instance: ProblemInstance,
        snapshot: OptimizationSnapshot,
    ) -> CandidateRoute:
        received.append(snapshot.remaining_point_ids)
        return CandidateRoute(snapshot.remaining_point_ids)

    result = run_dynamic_simulation(
        instance,
        planner,
        t_min=4.0,
        t_max=7.0,
        q_low=2.0,
        q_high=5.0,
        minimum_interval=300.0,
    )

    assert received == [(10, 20), (20,)]
    assert result.optimization_count == 2
    assert result.executions[0].reached_target is True
    assert result.executions[0].decision.reasons == ("collection_completed",)


def test_return_to_depot_uses_directed_shortest_path_without_optimization() -> None:
    instance = make_instance()
    planner_calls = 0

    def planner(
        _instance: ProblemInstance,
        snapshot: OptimizationSnapshot,
    ) -> CandidateRoute:
        nonlocal planner_calls
        planner_calls += 1
        return CandidateRoute(snapshot.remaining_point_ids)

    result = run_dynamic_simulation(
        instance,
        planner,
        t_min=4.0,
        t_max=7.0,
        q_low=2.0,
        q_high=5.0,
        minimum_interval=0.0,
    )

    assert result.total_distance == 15.0
    assert planner_calls == result.optimization_count == 2
    assert result.final_snapshot.current_load == 3.0


def test_empty_instance_finishes_without_calling_planner() -> None:
    instance = make_instance(collection_points=[], arcs=[])
    planner_calls = 0

    def planner(
        _instance: ProblemInstance,
        _snapshot: OptimizationSnapshot,
    ) -> CandidateRoute:
        nonlocal planner_calls
        planner_calls += 1
        return CandidateRoute(())

    result = run_dynamic_simulation(
        instance,
        planner,
        t_min=4.0,
        t_max=7.0,
        q_low=2.0,
        q_high=5.0,
        minimum_interval=0.0,
    )

    assert planner_calls == 0
    assert result.optimization_count == 0
    assert result.executions == ()
    assert result.total_distance == 0.0
    assert result.final_snapshot.remaining_point_ids == ()


def test_invalid_planner_result_is_rejected_before_counting_optimization() -> None:
    instance = make_instance()
    planner_calls = 0

    def planner(
        _instance: ProblemInstance,
        _snapshot: OptimizationSnapshot,
    ) -> object:
        nonlocal planner_calls
        planner_calls += 1
        return object()

    with pytest.raises(ValueError, match="route_planner"):
        run_dynamic_simulation(
            instance,
            planner,  # type: ignore[arg-type]
            t_min=4.0,
            t_max=7.0,
            q_low=2.0,
            q_high=5.0,
            minimum_interval=0.0,
        )

    assert planner_calls == 1


def test_invalid_candidate_route_is_propagated_before_counting_optimization() -> None:
    instance = make_instance()
    planner_calls = 0

    def planner(
        _instance: ProblemInstance,
        _snapshot: OptimizationSnapshot,
    ) -> CandidateRoute:
        nonlocal planner_calls
        planner_calls += 1
        return CandidateRoute((10,))

    with pytest.raises(ValueError, match="candidate route"):
        run_dynamic_simulation(
            instance,
            planner,
            t_min=4.0,
            t_max=7.0,
            q_low=2.0,
            q_high=5.0,
            minimum_interval=0.0,
        )

    assert planner_calls == 1
