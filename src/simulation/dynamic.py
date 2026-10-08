"""Complete receding-horizon simulation loop for the V1 formulation."""

from collections.abc import Callable
from dataclasses import dataclass

from src.algorithms.shortest_path import shortest_path
from src.models.candidate_route import CandidateRoute
from src.models.problem import ProblemInstance
from src.simulation.executor import (
    NextCollectionExecution,
    execute_candidate_route_step,
)
from src.simulation.operations import advance_to_adjacent_node
from src.simulation.snapshot import OptimizationSnapshot
from src.simulation.state import (
    SimulationState,
    build_optimization_snapshot,
    create_initial_simulation_state,
)

RoutePlanner = Callable[
    [ProblemInstance, OptimizationSnapshot],
    CandidateRoute,
]


@dataclass(frozen=True)
class DynamicSimulationResult:
    """Result of one complete dynamic simulation."""

    final_snapshot: OptimizationSnapshot
    total_distance: float
    optimization_count: int
    executions: tuple[NextCollectionExecution, ...]


def run_dynamic_simulation(
    instance: ProblemInstance,
    route_planner: RoutePlanner,
    *,
    t_min: float,
    t_max: float,
    q_low: float,
    q_high: float,
    minimum_interval: float,
) -> DynamicSimulationResult:
    """Run the receding-horizon simulation until all points are collected."""
    state: SimulationState = create_initial_simulation_state(instance)
    snapshot = build_optimization_snapshot(instance, state)
    total_distance = 0.0
    optimization_count = 0
    executions: list[NextCollectionExecution] = []
    last_recalculation_time: float | None = None

    while state.remaining_point_ids:
        candidate_route = route_planner(instance, snapshot)
        if not isinstance(candidate_route, CandidateRoute):
            raise ValueError("route_planner must return a CandidateRoute.")

        candidate_route.validate(instance, snapshot)
        last_recalculation_time = snapshot.current_time
        optimization_count += 1

        execution = execute_candidate_route_step(
            instance,
            state,
            candidate_route,
            snapshot,
            t_min=t_min,
            t_max=t_max,
            q_low=q_low,
            q_high=q_high,
            last_recalculation_time=last_recalculation_time,
            minimum_interval=minimum_interval,
        )
        if not execution.decision.should_recalculate:
            raise ValueError(
                "candidate route step ended without a recalculation decision."
            )

        total_distance += execution.distance_traveled
        executions.append(execution)
        snapshot = execution.snapshot

    _, return_path = shortest_path(
        instance.road_network,
        state.current_node_id,
        instance.depot.node_id,
    )
    for next_node_id in return_path[1:]:
        total_distance += advance_to_adjacent_node(
            instance,
            state,
            next_node_id,
        )

    final_snapshot = build_optimization_snapshot(instance, state)
    return DynamicSimulationResult(
        final_snapshot=final_snapshot,
        total_distance=total_distance,
        optimization_count=optimization_count,
        executions=tuple(executions),
    )
