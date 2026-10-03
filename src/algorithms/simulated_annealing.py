"""Deterministic neighborhood infrastructure for simulated annealing."""

import random

from src.models.candidate_route import CandidateRoute
from src.models.problem import ProblemInstance
from src.simulation.snapshot import OptimizationSnapshot


def create_algorithm_rng(algorithm_seed: int) -> random.Random:
    """Create a reproducible random generator from a non-negative seed."""
    if (
        isinstance(algorithm_seed, bool)
        or not isinstance(algorithm_seed, int)
        or algorithm_seed < 0
    ):
        raise ValueError("algorithm_seed must be a non-negative integer.")

    return random.Random(algorithm_seed)


def swap_neighbor(
    instance: ProblemInstance,
    snapshot: OptimizationSnapshot,
    candidate_route: CandidateRoute,
    rng: random.Random,
) -> CandidateRoute:
    """Return a neighbor formed by swapping two route positions."""
    candidate_route.validate(instance, snapshot)

    modified_point_ids = list(candidate_route.point_ids)
    if len(modified_point_ids) < 2:
        neighbor = CandidateRoute(tuple(modified_point_ids))
        neighbor.validate(instance, snapshot)
        return neighbor

    first_index, second_index = rng.sample(range(len(modified_point_ids)), 2)
    modified_point_ids[first_index], modified_point_ids[second_index] = (
        modified_point_ids[second_index],
        modified_point_ids[first_index],
    )

    neighbor = CandidateRoute(tuple(modified_point_ids))
    neighbor.validate(instance, snapshot)
    return neighbor
