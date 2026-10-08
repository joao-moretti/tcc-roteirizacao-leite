import pytest

from src.models.problem import CollectionPoint, Depot, ProblemInstance
from src.models.road_network import RoadArc, RoadNetwork, RoadNode
from src.simulation.operations import advance_to_adjacent_node, complete_collection
from src.simulation.state import SimulationState


def make_instance(*, capacity: float = 20.0) -> ProblemInstance:
    points = [
        CollectionPoint(point_id=10, node_id=2),
        CollectionPoint(point_id=20, node_id=3),
    ]
    return ProblemInstance(
        instance_id=1,
        road_network=RoadNetwork(
            network_id=1,
            nodes=[
                RoadNode(node_id=1, x=0.0, y=0.0),
                RoadNode(node_id=2, x=100.0, y=0.0),
                RoadNode(node_id=3, x=200.0, y=0.0),
                RoadNode(node_id=4, x=300.0, y=0.0),
            ],
            arcs=[
                RoadArc(from_node_id=1, to_node_id=2, distance=6.0),
                RoadArc(from_node_id=2, to_node_id=3, distance=4.0),
                RoadArc(from_node_id=3, to_node_id=4, distance=5.0),
            ],
        ),
        depot=Depot(depot_id=1, node_id=1),
        collection_points=points,
        vehicle_capacity=capacity,
        collection_service_duration=20.0,
        v_ref=2.0,
        sample_period=30.0,
        temperature_trajectories={
            10: (3.0, 4.0, 5.0, 6.0, 7.0),
            20: (8.0, 9.0, 10.0, 11.0, 12.0),
        },
        volume_trajectories={
            10: (1.0, 4.0, 6.0, 2.0, 3.0),
            20: (2.0, 3.0, 4.0, 5.0, 1.0),
        },
    )


def state_values(state: SimulationState) -> tuple[float, int, float, tuple[int, ...]]:
    """Return the mutable state fields for atomicity assertions."""
    return (
        state.current_time,
        state.current_node_id,
        state.current_load,
        state.remaining_point_ids,
    )


def test_advance_to_adjacent_node_updates_time_and_node() -> None:
    state = SimulationState(0.0, 1, 3.0, (10, 20))

    distance = advance_to_adjacent_node(make_instance(), state, 2)

    assert distance == 6.0
    assert state.current_node_id == 2
    assert state.current_time == 3.0
    assert state.current_load == 3.0
    assert state.remaining_point_ids == (10, 20)


def test_advance_uses_arc_distance_instead_of_coordinates() -> None:
    state = SimulationState(0.0, 1, 0.0, (10, 20))

    assert advance_to_adjacent_node(make_instance(), state, 2) == 6.0
    assert state.current_time == 3.0


def test_advance_respects_arc_direction() -> None:
    state = SimulationState(0.0, 2, 0.0, (10, 20))

    with pytest.raises(ValueError, match="directed arc"):
        advance_to_adjacent_node(make_instance(), state, 1)


@pytest.mark.parametrize("next_node_id", [99, True, 1.5, "2"])
def test_advance_rejects_invalid_or_unknown_next_node(next_node_id: object) -> None:
    state = SimulationState(0.0, 1, 0.0, (10, 20))

    with pytest.raises(ValueError, match="next_node_id"):
        advance_to_adjacent_node(
            make_instance(),
            state,
            next_node_id,  # type: ignore[arg-type]
        )


def test_advance_rejects_movement_to_current_node() -> None:
    state = SimulationState(0.0, 1, 0.0, (10, 20))

    with pytest.raises(ValueError, match="next_node_id"):
        advance_to_adjacent_node(make_instance(), state, 1)


def test_advance_rejects_missing_directed_arc_without_mutation() -> None:
    instance = make_instance()
    state = SimulationState(0.0, 2, 0.0, (10, 20))
    before = state_values(state)

    with pytest.raises(ValueError, match="directed arc"):
        advance_to_adjacent_node(instance, state, 1)

    assert state_values(state) == before


def test_advance_accepts_arrival_exactly_at_horizon() -> None:
    state = SimulationState(117.0, 1, 0.0, (10, 20))

    advance_to_adjacent_node(make_instance(), state, 2)

    assert state.current_time == 120.0


def test_advance_rejects_arrival_after_horizon_without_mutation() -> None:
    state = SimulationState(118.0, 1, 0.0, (10, 20))
    before = state_values(state)

    with pytest.raises(ValueError, match="trajectory horizon"):
        advance_to_adjacent_node(make_instance(), state, 2)

    assert state_values(state) == before


def test_complete_collection_updates_time_load_and_remaining_order() -> None:
    instance = make_instance()
    state = SimulationState(0.0, 2, 1.0, (20, 10))

    collected_volume = complete_collection(instance, state, 10)

    assert collected_volume == 1.0
    assert state.current_time == 20.0
    assert state.current_node_id == 2
    assert state.current_load == 2.0
    assert state.remaining_point_ids == (20,)


def test_complete_collection_uses_volume_at_service_completion() -> None:
    instance = make_instance()
    state = SimulationState(20.0, 2, 0.0, (10, 20))

    assert complete_collection(instance, state, 10) == 4.0
    assert state.current_time == 40.0


def test_complete_collection_rejects_invalid_or_unknown_point_id() -> None:
    state = SimulationState(0.0, 2, 0.0, (10, 20))

    for point_id in (99, True, 1.5, "10"):
        with pytest.raises(ValueError, match="point_id"):
            complete_collection(
                make_instance(),
                state,
                point_id,  # type: ignore[arg-type]
            )


def test_complete_collection_rejects_point_that_is_not_remaining() -> None:
    state = SimulationState(0.0, 2, 0.0, (20,))

    with pytest.raises(ValueError, match="remaining"):
        complete_collection(make_instance(), state, 10)


def test_complete_collection_rejects_wrong_current_node_without_mutation() -> None:
    state = SimulationState(0.0, 1, 0.0, (10, 20))
    before = state_values(state)

    with pytest.raises(ValueError, match="current_node_id"):
        complete_collection(make_instance(), state, 10)

    assert state_values(state) == before


def test_complete_collection_rejects_capacity_excess_without_mutation() -> None:
    instance = make_instance(capacity=20.0)
    state = SimulationState(0.0, 2, 20.0, (10, 20))
    before = state_values(state)

    with pytest.raises(ValueError, match="vehicle_capacity"):
        complete_collection(instance, state, 10)

    assert state_values(state) == before


def test_complete_collection_rejects_service_after_horizon_without_mutation() -> None:
    state = SimulationState(101.0, 2, 0.0, (10, 20))
    before = state_values(state)

    with pytest.raises(ValueError, match="trajectory horizon"):
        complete_collection(make_instance(), state, 10)

    assert state_values(state) == before


def test_complete_collection_accepts_completion_exactly_at_horizon() -> None:
    state = SimulationState(100.0, 2, 0.0, (10, 20))

    collected_volume = complete_collection(make_instance(), state, 10)

    assert collected_volume == 3.0
    assert state.current_time == 120.0
    assert state.remaining_point_ids == (20,)


def test_complete_collection_rejects_repeated_collection() -> None:
    instance = make_instance()
    state = SimulationState(0.0, 2, 0.0, (10, 20))
    complete_collection(instance, state, 10)

    with pytest.raises(ValueError, match="remaining"):
        complete_collection(instance, state, 10)


def test_complete_collection_does_not_modify_instance() -> None:
    instance = make_instance()
    before = (
        dict(instance.temperature_trajectories),
        dict(instance.volume_trajectories),
    )

    complete_collection(instance, SimulationState(0.0, 2, 0.0, (10, 20)), 10)

    assert dict(instance.temperature_trajectories) == before[0]
    assert dict(instance.volume_trajectories) == before[1]


def test_atomic_operations_support_manual_depot_to_collection_sequence() -> None:
    instance = make_instance()
    state = SimulationState(0.0, 1, 0.0, (10, 20))

    advance_to_adjacent_node(instance, state, 2)
    advance_to_adjacent_node(instance, state, 3)
    complete_collection(instance, state, 20)

    assert state.current_time == 25.0
    assert state.current_node_id == 3
    assert state.current_load == 2.0
    assert state.remaining_point_ids == (10,)
