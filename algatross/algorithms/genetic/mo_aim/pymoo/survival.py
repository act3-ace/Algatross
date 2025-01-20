"""Module containing methods for controlling Populations on the islands & mainlands in MO-AIM."""

from collections.abc import Callable
from typing import Literal

import numpy as np

from pymoo.core.population import Population
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding as _RankAndCrowding
from pymoo.operators.survival.rank_and_crowding.metrics import CrowdingDiversity

from algatross.algorithms.genetic.mo_aim.pymoo.argsort import RandomizedArgsorter
from algatross.algorithms.genetic.mo_aim.pymoo.nds import NonDominatedSorting
from algatross.utils.random import resolve_seed
from algatross.utils.types import NumpyRandomSeed

MOAIM_ISLAND_ALGORITHM_CONFIG_DEFAULTS = {"conspecific_data_keys": ["actions"]}


class RankAndCrowding(_RankAndCrowding):
    """
    RankAndCrowding a customized PyMOO rank-and-crowding class.

    This modifies the call to ``do`` so that the NDS method is not called. Instead the
    already-sorted front are provided. This is necessary because MO-AIM needs access to
    the non-dominating fronts directly and PyMOO RankAndCrowding does not return them.

    Parameters
    ----------
    nds : NonDominatedSorting, optional
        The non-dominated sorter to call for determing fronts, :data:`python:None`
    crowding_func : Literal["cd", "pcd", "ce", "mnn", "2nn"] | Callable | CrowdingDiversity, optional
        The crowding function to use, by default "cd". Valid values correspond to the following:

        - :python:`"cd"`: crowding distance
        - :python:`"pcd"` or :python:`"pruning-cd"`: pruning crowding distance
        - :python:`"ce"`: crowding entropy
        - :python:`"mnn"`: m-nearest-neighbors
        - :python:`"2nn"` 2-nearest-neighbors

        You may also pass a callable or a subclass of :class:`~pymoo.operators.survival.rank_and_crowding.metrics.CrowdingDiversity`
        See pymoo documentation for more information.
    seed : NumpyRandomSeed, optional
        The seed for the random state, :data:`python:None`
    `*args`
        Additional positional arguments.
    `**kwargs`
        Additional keyword arguments.
    """

    def __init__(
        self,
        nds: NonDominatedSorting = None,
        crowding_func: Literal["cd", "pcd", "pruning-cd", "ce", "mnn", "2nn"] | Callable | CrowdingDiversity = "cd",
        generator: np.random.Generator | None = None,
        seed: NumpyRandomSeed = None,
        *args,
        **kwargs,
    ):
        super().__init__(nds, crowding_func)
        self._numpy_generator = generator or resolve_seed(seed)  # type: ignore[arg-type]
        self._argsorter = RandomizedArgsorter(self._numpy_generator)

    def _do(self, pop: Population, fronts: list[np.ndarray], *args, n_survive: int | None = None, **kwargs) -> Population:
        """
        _do do rank-and-crowding.

        Parameters
        ----------
        pop : Population
            The pymoo population to R&C
        fronts : list[np.ndarray]
            The sorted Pareto fronts
        n_survive : int, optional
            The number of survivors, :data:`python:None`, in which all of the population survives.
        `*args`
            Additional positional arguments.
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        Population
            The surviving population
        """
        # get the objective space values and objects
        f = pop.get("F").astype(float, copy=False)

        # the final indices of surviving individuals
        survivors: list[np.ndarray] = []

        for k, front in enumerate(fronts):
            indices = np.arange(len(front))

            # current front sorted by crowding distance if splitting
            if len(survivors) + len(indices) > n_survive:
                # Define how many will be removed
                n_remove = len(survivors) + len(front) - n_survive

                # re-calculate the crowding distance of the front
                crowding_of_front = self.crowding_func.do(f[front, :], n_remove=n_remove)

                indices = self._argsorter(crowding_of_front, method="numpy", order="descending")
                indices = indices[:-n_remove]

            # otherwise take the whole front unsorted
            else:
                # calculate the crowding distance of the front
                crowding_of_front = self.crowding_func.do(f[front, :], n_remove=0)

            # save rank and crowding in the individual class
            for j, i in enumerate(front):
                pop[i].set("rank", k)
                pop[i].set("crowding", crowding_of_front[j])

            # extend the survivors by all or selected individuals
            survivors.extend(front[indices])

        return pop[survivors]
