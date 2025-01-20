"""Module containing a customized argsorted based on pymoos randomized argsort."""

from typing import Literal

import numpy as np

from numpy.typing import ArrayLike

from pymoo.util.misc import swap

from algatross.utils.random import resolve_seed
from algatross.utils.types import NumpyRandomSeed


class RandomizedArgsorter:
    """
    A class for doing randomized arg sorts using the new Numpy generator API.

    This allows for clearer syncronicity between functions which use randomness. Rather
    than using a global random state one may provide a seed, which itself may be a generator
    and thus the user can explicitly link the random state of this random arg-sorter to another
    random state.

    An instance of this class can be called directly on data or the :meth:randomized_argsort:
    method can be invoked.

    Parameters
    ----------
    generator : np.random.Generator | None, optional
        The random number generator to share with this object, default is :data:`python:None`
    seed : NumpyRandomSeed | None, optional
        The seed for randomness, default is :data:`python:None`
    `*args`
        Additional positional arguments.
    `**kwargs`
        Additional keyword arguments.
    """

    def __init__(self, generator: np.random.Generator | None = None, seed: NumpyRandomSeed | None = None, *args, **kwargs):
        self._numpy_generator = generator or resolve_seed(seed)  # type: ignore[arg-type]

    def __call__(
        self,
        arr: ArrayLike,
        method: Literal["numpy", "quicksort"] = "numpy",
        order: Literal["ascending", "descending"] = "ascending",
    ) -> np.ndarray:
        """
        Sort the input using the method and ordering specified.

        Parameters
        ----------
        arr : ArrayLike
            The array to be sorted.
        method : Literal["numpy", "quicksort"], optional
            The argsort method to use, by default "numpy"
        order : Literal["ascending", "descending"], optional
            The sort order, by default "ascending"

        Returns
        -------
        np.ndarray
            The arguments which would sort the input array.
        """
        return self.randomized_argsort(arr, method=method, order=order)

    def randomized_argsort(
        self,
        arr: ArrayLike,
        method: Literal["numpy", "quicksort"] = "numpy",
        order: Literal["ascending", "descending"] = "ascending",
    ) -> np.ndarray:
        """Conduct a randomized argsort on the input array.

        Parameters
        ----------
        arr : ArrayLike
            The array to be sorted.
        method : Literal["numpy", "quicksort"], optional
            The argsort method to use, by default "numpy"
        order : Literal["ascending", "descending"], optional
            The sort order, by default "ascending"

        Returns
        -------
        np.ndarray
            The arguments which would sort the input array.

        Raises
        ------
        ValueError
            If an invalid sort method is given, or an invalid sort order is given.
        """
        if method == "numpy":
            perm = self._numpy_generator.permutation(len(arr))  # type: ignore[arg-type]
            indices = np.argsort(arr[perm], kind="quicksort")  # type: ignore[index]
            indices = perm[indices]

        elif method == "quicksort":
            indices = self.quicksort(arr)

        else:
            msg = "Randomized sort method not known."
            raise ValueError(msg)

        if order == "ascending":
            return indices
        if order == "descending":
            return np.flip(indices, axis=0)
        msg = "Unknown sorting order: ascending or descending."
        raise ValueError(msg)

    def quicksort(self, arr: ArrayLike) -> np.ndarray:
        """
        Quicksort performs an arg-quicksort on the input.

        Parameters
        ----------
        arr : ArrayLike
            The array to be quicksorted

        Returns
        -------
        np.ndarray
            The arguments which would sort the input array.
        """
        indices = np.arange(len(arr))  # type: ignore[arg-type]
        self._quicksort(arr, indices, 0, len(arr) - 1)  # type: ignore[arg-type]
        return indices

    def _quicksort(self, arr: ArrayLike, indices: np.ndarray, left: int, right: int):
        if left < right:
            index = self._numpy_generator.integers(left, right + 1)
            swap(indices, right, index)

            pivot = arr[indices[right]]  # type: ignore[index]

            i = left - 1

            for j in range(left, right):
                if arr[indices[j]] <= pivot:  # type: ignore[index, operator]
                    i += 1
                    swap(indices, i, j)

            index = i + 1
            swap(indices, right, index)

            self._quicksort(arr, indices, left, index - 1)
            self._quicksort(arr, indices, index + 1, right)
