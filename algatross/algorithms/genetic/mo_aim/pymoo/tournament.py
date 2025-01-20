"""Tournament selection operators adapted from PyMOO."""

import math

from collections.abc import Callable
from typing import Literal

import numpy as np

from pymoo.operators.selection.tournament import TournamentSelection as _TournamentSelection

from algatross.utils.random import resolve_seed
from algatross.utils.types import NumpyRandomSeed


class TournamentSelection(_TournamentSelection):
    """
    TournamentSelection conducts tournament selection.

    Identical to PyMOOs ``TournamentSelection`` except it respects NumPys new standard for random seeding.

    Parameters
    ----------
    func_comp : Callable | None, optional
        The function to use for comparison, default is :data:`python:None`
    pressure : int, optional
        The selection pressure to apply, default is 2 for binary tournament
    generator : np.random.Generator | None, optional
        The random number generator to share, default is :data:`python:None`
    seed : NumpyRandomSeed | None, optional
        The seed for randomness, default is :data:`python:None`
    `**kwargs`
        Additional keyword arguments.
    """

    def __init__(
        self,
        func_comp: Callable | None = None,
        pressure: int = 2,
        generator: np.random.Generator | None = None,
        seed: NumpyRandomSeed | None = None,
        **kwargs,
    ):
        super().__init__(func_comp, pressure, **kwargs)
        self._numpy_generator = generator or resolve_seed(seed=seed)  # type: ignore[arg-type]

    def _do(self, _, pop, n_select, n_parents=1, **kwargs):
        # number of random individuals needed
        n_random = n_select * n_parents * self.pressure

        # number of permutations needed
        n_perms = math.ceil(n_random / len(pop))

        # get random permutations and reshape them
        p = self.random_permuations(n_perms, len(pop))[:n_random]
        p = np.reshape(p, (n_select * n_parents, self.pressure))

        # compare using tournament function
        s = self.func_comp(pop, p, generator=self._numpy_generator, **kwargs)

        return np.reshape(s, (n_select, n_parents))

    def random_permuations(
        self,
        n_permutations: int,
        permutation_length: int,
        concat: bool = True,
        generator: np.random.Generator | None = None,
    ) -> list | np.ndarray:
        """
        Generate ``n_permutations`` random permutations of ``permutation_length`` items.

        Identical to PyMOO's ``random_permutations`` except it respects new rules for numpy randomness.

        Parameters
        ----------
        n_permutations : int
            Number of random permutations to generate
        permutation_length : int
            Number of items to be permuted
        concat : bool, optional
            Whether to concatenate the permutations, :data:`python:True`
        generator : np.random.Generator | None, optional
            The random number generator, :data:`python:None`

        Returns
        -------
        list | np.ndarray
            The list or concatenated array of permutations
        """
        p = [(self._numpy_generator if generator is None else generator).permutation(permutation_length) for _ in range(n_permutations)]
        if concat:
            p = np.concatenate(p)
        return p


def compare(
    a: np.ndarray,
    a_val: float,
    b: np.ndarray,
    b_val: float,
    method: Literal["larger_is_better", "smaller_is_better"],
    return_random_if_equal: bool = False,
    generator: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Compare two solutions and return the preferred one.

    Identical to the PyMOO implementation except it respects random seeds.

    Parameters
    ----------
    a : np.ndarray
        Solution a
    a_val : float
        Value to compare for solution a
    b : np.ndarray
        Solution b
    b_val : float
        Value to compare for solution b
    method : Literal["larger_is_better", "smaller_is_better"]
        Whether to prefer a larger or smaller value
    return_random_if_equal : bool, optional
        Whether to return a random solution in case of a tie, :data:`python:False`
    generator : np.random.Generator | None, optional
        Random bit generator to use, :data:`python:None`

    Returns
    -------
    np.ndarray
        The better solution

    Raises
    ------
    ValueError
        If an invalid ``method`` is given
    """
    better = None
    if method == "larger_is_better":
        if a_val > b_val:
            better = a
        if a_val < b_val:
            better = b
        if return_random_if_equal:
            better = (np.random.choice if generator is None else generator.choice)([a, b])
        return better
    if method == "smaller_is_better":
        if a_val < b_val:
            better = a
        if a_val > b_val:
            better = b
        if return_random_if_equal:
            better = (np.random.choice if generator is None else generator.choice)([a, b])
        return better
    msg = f"Unknown `compare` method: {method}"
    raise ValueError(msg)
