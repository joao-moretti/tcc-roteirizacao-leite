import pytest

from src.models.problem import CollectionPoint, Depot, ProblemInstance
from src.models.road_network import RoadNetwork, RoadNode


def make_network() -> RoadNetwork:
    return RoadNetwork(
        network_id=1,
        nodes=[
            RoadNode(node_id=1, x=0.0, y=0.0),
            RoadNode(node_id=2, x=1.0, y=0.0),
            RoadNode(node_id=3, x=2.0, y=0.0),
        ],
        arcs=[],
    )


def make_problem(**overrides: object) -> ProblemInstance:
    values: dict[str, object] = {
        "instance_id": 1,
        "road_network": make_network(),
        "depot": Depot(depot_id=1, node_id=1),
        "collection_points": [
            CollectionPoint(point_id=1, node_id=2),
            CollectionPoint(point_id=2, node_id=3),
        ],
        "vehicle_capacity": 10.0,
        "collection_service_duration": 60.0,
        "v_ref": 10.0,
        "sample_period": 30.0,
        "temperature_trajectories": {
            1: (4.0, 4.5),
            2: (4.2, 4.7),
        },
        "volume_trajectories": {
            1: (2.0, 3.0),
            2: (1.0, 2.0),
        },
    }
    values.update(overrides)
    return ProblemInstance(**values)  # type: ignore[arg-type]


def test_collection_point_valid() -> None:
    assert CollectionPoint(point_id=1, node_id=2) == CollectionPoint(1, 2)


@pytest.mark.parametrize("field_name", ["point_id", "node_id"])
@pytest.mark.parametrize("invalid_id", [-1, 1.5, True, "1"])
def test_collection_point_rejects_invalid_id(
    field_name: str, invalid_id: object
) -> None:
    with pytest.raises(ValueError, match=field_name):
        if field_name == "point_id":
            CollectionPoint(invalid_id, 2)  # type: ignore[arg-type]
        else:
            CollectionPoint(1, invalid_id)  # type: ignore[arg-type]


def test_depot_valid() -> None:
    assert Depot(depot_id=1, node_id=2) == Depot(1, 2)


def test_depot_rejects_invalid_identifier() -> None:
    with pytest.raises(ValueError, match="depot_id"):
        Depot(depot_id=-1, node_id=2)


def test_problem_instance_valid_and_structurally_protected() -> None:
    problem = make_problem()

    assert isinstance(problem.collection_points, tuple)
    assert problem.temperature_trajectories[1] == (4.0, 4.5)
    assert problem.volume_trajectories[2] == (1.0, 2.0)

    with pytest.raises(TypeError):
        problem.temperature_trajectories[1] = (5.0, 5.5)  # type: ignore[index]
    with pytest.raises(TypeError):
        problem.volume_trajectories[1][0] = 5.0  # type: ignore[index]


def test_problem_instance_rejects_invalid_instance_id() -> None:
    with pytest.raises(ValueError, match="instance_id"):
        make_problem(instance_id=-1)


def test_problem_instance_rejects_invalid_capacity() -> None:
    with pytest.raises(ValueError, match="vehicle_capacity"):
        make_problem(vehicle_capacity=0.0)


@pytest.mark.parametrize("v_ref", [0.0, -1.0, float("nan")])
def test_problem_instance_rejects_invalid_v_ref(v_ref: float) -> None:
    with pytest.raises(ValueError, match="v_ref"):
        make_problem(v_ref=v_ref)


@pytest.mark.parametrize("sample_period", [0.0, 29.0, 30.1])
def test_problem_instance_rejects_invalid_sample_period(sample_period: float) -> None:
    with pytest.raises(ValueError, match="sample_period"):
        make_problem(sample_period=sample_period)


def test_problem_instance_rejects_duplicate_point_id() -> None:
    points = [CollectionPoint(1, 2), CollectionPoint(1, 3)]

    with pytest.raises(ValueError, match="duplicate point_id"):
        make_problem(collection_points=points)


def test_problem_instance_rejects_point_with_unknown_node() -> None:
    points = [CollectionPoint(1, 99)]

    with pytest.raises(ValueError, match="collection point node_id"):
        make_problem(
            collection_points=points,
            temperature_trajectories={1: (4.0,)},
            volume_trajectories={1: (1.0,)},
        )


def test_problem_instance_rejects_depot_with_unknown_node() -> None:
    with pytest.raises(ValueError, match="depot node_id"):
        make_problem(depot=Depot(depot_id=1, node_id=99))


def test_problem_instance_rejects_points_sharing_node() -> None:
    points = [CollectionPoint(1, 2), CollectionPoint(2, 2)]

    with pytest.raises(ValueError, match="share node_id"):
        make_problem(collection_points=points)


def test_problem_instance_rejects_depot_sharing_node_with_point() -> None:
    with pytest.raises(ValueError, match="depot node_id"):
        make_problem(depot=Depot(depot_id=1, node_id=2))


def test_problem_instance_rejects_missing_temperature_trajectory() -> None:
    with pytest.raises(ValueError, match="temperature_trajectories"):
        make_problem(temperature_trajectories={1: (4.0, 4.5)})


def test_problem_instance_rejects_missing_volume_trajectory() -> None:
    with pytest.raises(ValueError, match="volume_trajectories"):
        make_problem(volume_trajectories={1: (2.0, 3.0)})


def test_problem_instance_rejects_trajectory_for_unknown_point() -> None:
    with pytest.raises(ValueError, match="unknown point_id"):
        make_problem(
            temperature_trajectories={1: (4.0,), 2: (4.0,), 99: (4.0,)},
            volume_trajectories={1: (1.0,), 2: (1.0,)},
        )


def test_problem_instance_rejects_different_trajectory_lengths() -> None:
    with pytest.raises(ValueError, match="same number of samples"):
        make_problem(
            temperature_trajectories={1: (4.0,), 2: (4.0, 4.5)},
            volume_trajectories={1: (1.0,), 2: (1.0,)},
        )


def test_problem_instance_rejects_empty_trajectory() -> None:
    with pytest.raises(ValueError, match="at least one sample"):
        make_problem(
            temperature_trajectories={1: (), 2: (4.0,)},
            volume_trajectories={1: (1.0,), 2: (1.0,)},
        )


def test_problem_instance_rejects_non_finite_temperature() -> None:
    with pytest.raises(ValueError, match="temperature_trajectories"):
        make_problem(temperature_trajectories={1: (float("nan"), 4.0), 2: (4.0, 4.0)})


def test_problem_instance_rejects_negative_volume() -> None:
    with pytest.raises(ValueError, match="volume_trajectories"):
        make_problem(volume_trajectories={1: (2.0, -1.0), 2: (1.0, 2.0)})


def test_problem_instance_rejects_non_finite_volume() -> None:
    with pytest.raises(ValueError, match="volume_trajectories"):
        make_problem(volume_trajectories={1: (float("inf"), 2.0), 2: (1.0, 2.0)})


def test_problem_instance_rejects_volume_exceeding_capacity() -> None:
    with pytest.raises(ValueError, match="vehicle_capacity"):
        make_problem(vehicle_capacity=4.0)
