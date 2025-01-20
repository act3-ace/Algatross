"""A module containing the implementation of the Efficient Dominance Degree Approach to NDS."""

import numpy as np

from algatross.algorithms.genetic.util.nds.dda_ns import construct_domination_matrix


def dda_ens(f_scores: np.ndarray, **kwargs) -> list[list[int]]:
    """Run the main DDA-ENS loop.

    Parameters
    ----------
    f_scores : np.ndarray
        The matrics of fitness scores for each solution
    `**kwargs`
        Additional keyword arguments.

    Returns
    -------
    list[list[int]]
        The domination matrix
    """
    d_mx = construct_domination_matrix(f_scores)

    fronts: list[list[int]] = []
    for s in np.lexsort(f_scores.T):
        isinserted = False
        for fk in fronts:
            if not (d_mx[fk, s] == f_scores.shape[1]).any():
                fk.append(s)
                isinserted = True
                break
        if not isinserted:
            fronts.append([s])
    return fronts
