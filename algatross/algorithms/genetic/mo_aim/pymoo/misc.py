"""Miscellaneous PyMOO operators."""

import numpy as np

from pymoo.core.algorithm import Algorithm
from pymoo.core.population import Population
from pymoo.util.dominator import Dominator

from algatross.algorithms.genetic.mo_aim.pymoo.tournament import compare


def binary_tournament(
    pop: Population,
    parents: np.ndarray,
    algorithm: Algorithm,
    generator: np.random.Generator | None = None,
    **kwargs,
) -> np.ndarray:
    """
    Perform a binary tournament.

    Identical to PyMOO's ``binary_tournament`` except it respects NumPys new randomness.

    Parameters
    ----------
    pop : pymoo.core.population.Population
        Population undergoing tournament selection
    parents : np.ndarray
        The problem the population is solving
    algorithm : pymoo.core.algorithm.Algorithm
        The algorithm for evolving the population
    generator : np.random.Generator | None, optional
        Randomness generator, :data:`python:None`
    `**kwargs`
        Additional keyword arguments.

    Returns
    -------
    np.ndarray
        The solutions selected by the binary tournament

    Raises
    ------
    ValueError
        If the tournament ``parents`` doesn't have two parents
    ValueError
        If the algorithm has an invalid tournament type.
    """
    n_tournaments, n_parents = parents.shape

    if n_parents != 2:  # noqa: PLR2004
        msg = "Only implemented for binary tournament!"
        raise ValueError(msg)

    tournament_type = algorithm.tournament_type
    s = np.full(n_tournaments, np.nan)

    for i in range(n_tournaments):
        a, b = parents[i, 0], parents[i, 1]
        a_cv, a_f, b_cv, b_f = pop[a].CV[0], pop[a].F, pop[b].CV[0], pop[b].F
        rank_a, cd_a = pop[a].get("rank", "crowding")
        rank_b, cd_b = pop[b].get("rank", "crowding")

        # if at least one solution is infeasible
        if a_cv > 0.0 or b_cv > 0.0:
            s[i] = compare(a, a_cv, b, b_cv, method="smaller_is_better", return_random_if_equal=True, generator=generator)

        # both solutions are feasible
        else:
            if tournament_type == "comp_by_dom_and_crowding":
                rel = Dominator.get_relation(a_f, b_f)
                if rel == 1:
                    s[i] = a
                elif rel == -1:
                    s[i] = b

            elif tournament_type == "comp_by_rank_and_crowding":
                s[i] = compare(a, rank_a, b, rank_b, method="smaller_is_better")

            else:
                msg = f"Unknown tournament type: {tournament_type}"
                raise ValueError(msg)

            # if rank or domination relation didn't make a decision compare by crowding
            if np.isnan(s[i]):
                s[i] = compare(a, cd_a, b, cd_b, method="larger_is_better", return_random_if_equal=True, generator=generator)

    return s[:, None].astype(int, copy=False)
