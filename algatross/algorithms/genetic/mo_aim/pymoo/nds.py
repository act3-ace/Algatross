"""Module containing modified pymoo classes for non-dominating sorts."""

from collections.abc import Callable

import numpy as np

from pymoo.util.nds.non_dominated_sorting import (
    NonDominatedSorting as _NonDominatedSorting,
    rank_from_fronts,
)


class NonDominatedSorting(_NonDominatedSorting):
    """
    A modified NDS class from PyMOO which allows passing any callable.

    Bypasses PyMOOs normal ``load_function`` routine so any valid callable can be used. This allows
    providing an NDS algorithm which is not implemented natively in PyMOO

    Parameters
    ----------
    epsilon : float | None, optional
        A small positive number for numerical stability, default is :data:`python:None`
    method : Callable | None, optional
        The non-dominating sort method to use, default is :data:`python:None`.
    """

    def __init__(self, epsilon: float | None = None, method: Callable | None = None) -> None:
        super().__init__(epsilon, method)

    def do(
        self,
        f: np.ndarray,
        return_rank: bool = False,
        only_non_dominated_front: bool = False,
        n_stop_if_ranked: int | None = None,
        **kwargs,
    ) -> list[list[np.ndarray]] | list[np.ndarray]:
        """
        Do do non-dominated sorting.

        Parameters
        ----------
        f : np.ndarray
            The solutions to be sorted
        return_rank : bool, optional
            Whether to return the non-dominating rank, :data:`python:False`
        only_non_dominated_front : bool, optional
            Whether to return only the non-dominating front, :data:`python:False`. This setting
            takes precedence over ``return_rank`` so if both are True, only the non-dominated
            front (no ranks) will be returned
        n_stop_if_ranked : int | None, optional
            Whether or not we stop sorting once with many solutions have been ranked, :data:`python:None`
            which sorts the entire population of solutions.
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        list[list[int]] | list[int]
            The list of sorted fronts whether the values in the front are the indices of the solutions
            belonging to the front. If ``only_non_dominated_front`` is specifed then only the non-dominated
            (first) front is returned
        """
        f = f.astype(float)

        # if not set just set it to a very large values because the cython algorithms do not take None
        if n_stop_if_ranked is None:
            n_stop_if_ranked = int(1e8)
        func = self.method

        # set the epsilon if it should be set
        if self.epsilon is not None:
            kwargs["epsilon"] = float(self.epsilon)

        fronts = func(f, **kwargs)

        # convert to numpy array for each front and filter by n_stop_if_ranked if desired
        fronts_ = []
        n_ranked = 0
        for front in fronts:
            fronts_.append(np.array(front, dtype=int))

            # increment the n_ranked solution counter
            n_ranked += len(front)

            # stop if more than this solutions are n_ranked
            if n_ranked >= n_stop_if_ranked:
                break

        fronts = fronts_

        if only_non_dominated_front:
            return fronts[0]

        if return_rank:
            rank = rank_from_fronts(fronts, f.shape[0])
            return fronts, rank  # type: ignore[return-value]

        return fronts
