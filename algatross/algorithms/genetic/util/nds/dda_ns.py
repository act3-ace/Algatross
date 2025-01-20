"""Module which implements Dominance Degree Approach for Non-dominated Sorting.

For the original work see: https://ieeexplore.ieee.org/document/7469397

Adapted from https://github.com/rsenwar/Non-Dominated-Sorting-Algorithms/tree/master
"""

import numpy as np


def construct_comp_matrix(vec: np.ndarray, sorted_idx: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Construct the comparison matrix from a row-vector w.

    Parameters
    ----------
    vec : np.ndarray
        The vector of objectives
    sorted_idx : np.ndarray
        The indices which sort ``vec``
    c : np.ndarray
        The comparison matrix

    Returns
    -------
    np.ndarray
        The updated comparison matrix
    """
    n = c.shape[0]
    c.fill(0)

    # the elements of the b(0)-th row in C are all set to 1
    c[sorted_idx[0], :] = 1

    for i in range(1, n):
        if vec[sorted_idx[i]] == vec[sorted_idx[i - 1]]:
            # the rows in C corresponding to the same elements in w are identical
            c[sorted_idx[i]] = c[sorted_idx[i - 1]]
        else:
            c[sorted_idx[i], sorted_idx[i:]] = 1

    return c


def construct_domination_matrix(f_scores: np.ndarray, **kwargs) -> np.ndarray:
    """
    Construct_domination_matrix calculates the dominance degree matrix for a set of vectors.

    The dominance degree indicate the degree of dominance of a solution, which is the number of
    objectives for which it is the dominating solution.

    Parameters
    ----------
    f_scores : np.ndarray
        An N x M matrix of N (population size) objective function values for M objectives
    `**kwargs`
        Additional keyword arguments.

    Returns
    -------
    np.ndarray
        The domination matrix
    """
    d = np.zeros((f_scores.shape[0], f_scores.shape[0]), dtype=np.int32)
    c = np.empty((f_scores.shape[0], f_scores.shape[0]), dtype=np.int32)
    b = np.apply_over_axes(np.argsort, f_scores, axes=0)
    for vec, srt in zip(f_scores.T, b.T, strict=True):
        d += construct_comp_matrix(vec, srt, c)
    return np.where(np.logical_and(d == f_scores.shape[-1], f_scores.shape[-1] == d.T), 0, d)


def dda_ns(f_scores: np.ndarray, **kwargs) -> list[list[int]]:
    """Run the DDA-NS algorithm.

    Parameters
    ----------
    f_scores : np.ndarray
        An N x M matrix of N (population size) objective function values for M objectives
    `**kwargs`
        Additional keyword arguments.

    Returns
    -------
    list[list[int]]
        A list of members of each Pareto front. The index in the outer most list corresponds to the level in the Pareto front
        while the value in the inner-most list is the id of the member of the population belonging to that front.
    """
    d_mx = construct_domination_matrix(f_scores)
    max_d = np.empty((f_scores.shape[0],), dtype=np.int32)

    fronts = []
    count = 0
    while count < f_scores.shape[0]:
        # Max(D) is the row vector containing the maximum elements from each column of D
        np.max(d_mx, out=max_d, axis=0)
        front = [i for i, m_d in enumerate(max_d) if 0 <= m_d < f_scores.shape[-1]]
        count += len(front)
        d_mx[front] = -1
        d_mx[:, front] = -1
        fronts.append(front)

    return fronts
