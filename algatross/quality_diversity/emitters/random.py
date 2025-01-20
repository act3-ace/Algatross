"""A module with a custom emitter class."""

import numpy as np

from ribs._utils import check_batch_shape, check_shape  # noqa: PLC2701
from ribs.emitters import EmitterBase


class RandomEmitter(EmitterBase):
    """Emits solutions, unchanged, randomly from the archive solutions.

    If the archive is empty and there are no initial solutions then solutions will be
    generated using a random normal distribution.

    Parameters
    ----------
    archive : ribs.archives.ArchiveBase
        An archive to use when creating and inserting solutions. For instance, this can be
        :class:`~ribs.archives._grid_archive.GridArchive`.
    sigma : float | Sequence
        Standard deviation of the Gaussian distribution. Note we assume the Gaussian is
        diagonal, so if this argument is an array, it must be 1D.
    x0 : array-like | None
        Center of the Gaussian distribution from which to sample solutions when the archive
        is empty. Must be 1-dimensional. This argument is ignored if ``initial_solutions``
        is set.
    initial_solutions : array-like | None
        An (n, solution_dim) array of solutions to be used when the archive is empty. If
        this argument is None, then solutions will be ``x0``.
    bounds : array-like | None
        Bounds of the solution space. Solutions are clipped to these bounds. Pass None to
        indicate there are no bounds. Alternatively, pass an array-like to specify the
        bounds for each dim. Each element in this array-like can be None to indicate no
        bound, or a tuple of ``(lower_bound, upper_bound)``, where ``lower_bound`` or
        ``upper_bound`` may be None to indicate no bound.
    batch_size : int
        Number of solutions to return in :meth:`~algatross.quality_diversity.emitters.random.RandomEmitter.ask`.
    seed : int | None
        Value to seed the random number generator. Set to None to avoid a fixed seed.

    Raises
    ------
    ValueError

        - There is an error in x0 or initial_solutions.
        - There is an error in the bounds configuration.
    """

    def __init__(self, archive, *, sigma=1.0, x0=None, initial_solutions=None, bounds=None, batch_size=64, seed=None):
        self._rng = np.random.default_rng(seed)
        self._batch_size = batch_size

        self._sigma = np.array(sigma, dtype=archive.dtypes["solution"])

        self._x0 = None
        self._initial_solutions = None

        if x0 is None and initial_solutions is None:
            msg = "Either x0 or initial_solutions must be provided."
            raise ValueError(msg)
        if x0 is not None and initial_solutions is not None:
            msg = "x0 and initial_solutions cannot both be provided."
            raise ValueError(msg)

        if x0 is not None:
            self._x0 = np.array(x0, dtype=archive.dtypes["solution"])
            check_shape(self._x0, "x0", archive.solution_dim, "archive.solution_dim")
        elif initial_solutions is not None:
            self._initial_solutions = np.asarray(initial_solutions, dtype=archive.dtypes["solution"])
            check_batch_shape(self._initial_solutions, "initial_solutions", archive.solution_dim, "archive.solution_dim")

        EmitterBase.__init__(self, archive, solution_dim=archive.solution_dim, bounds=bounds)

    @property
    def x0(self) -> np.ndarray:
        """Center of the Gaussian distribution from which to sample solutions when the archive is empty.

        If initial_solutions is not set then this parameter is ignored.

        Returns
        -------
        np.ndarray
            The center of the Gaussian distribution from which to sample solutions when the archive is empty.
        """
        return self._x0

    @property
    def sigma(self) -> np.ndarray:
        """
        Standard deviation of the (diagonal) Gaussian distribution when the archive is empty.

        Returns
        -------
        np.ndarray
            The standard deviation of the (diagonal) Gaussian distribution when the archive is empty.
        """
        return self._sigma

    @property
    def initial_solutions(self) -> np.ndarray:
        """The initial solutions which are returned when the archive is empty.

        Returns
        -------
        np.ndarray
            The initial solutions which are returned when the archive is empty.
        """
        return self._initial_solutions

    @property
    def batch_size(self) -> int:
        """
        Number of solutions to return in :meth:`~algatross.quality_diversity.emitters.random.RandomEmitter.ask`.

        Returns
        -------
        int
            Number of solutions to return in :meth:`~algatross.quality_diversity.emitters.random.RandomEmitter.ask`.
        """
        return self._batch_size

    def ask(self) -> np.ndarray:
        """Create solutions randomly sampling elites in the archive.

        If the archive is empty and ``self._initial_solutions`` is set, we
        return ``self._initial_solutions``. If ``self._initial_solutions`` is
        not set, we draw from Gaussian distribution centered at ``self.x0``
        with standard deviation ``self.sigma``. Otherwise, each solution is
        drawn from a distribution centered at a randomly chosen elite with
        standard deviation ``self.sigma``.

        Returns
        -------
        np.ndarray
            If the archive is not empty, :python:`(batch_size, solution_dim)` array
            -- contains ``batch_size`` new solutions to evaluate. If the
            archive is empty, we return ``self._initial_solutions``, which
            might not have ``batch_size`` solutions.
        """
        if self.archive.empty:
            if self._initial_solutions is not None:
                return np.clip(self._initial_solutions, self.lower_bounds, self.upper_bounds)

            # only add noise if the archive is empty and no initial solutions
            noise = self._rng.normal(scale=self._sigma, size=(self._batch_size, self.solution_dim)).astype(self.archive.dtype)
            parents = np.expand_dims(self.x0, axis=0) + noise
        else:
            # sample from the archive
            parents = self.archive.sample_elites(self._batch_size)["solution"]

        # return samples unchanged
        return np.clip(parents, self.lower_bounds, self.upper_bounds)
