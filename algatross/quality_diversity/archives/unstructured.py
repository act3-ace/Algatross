"""Contains the UnstructuredArchive class."""

from collections.abc import Sequence
from functools import cached_property

import numpy as np

from pymoo.util.function_loader import load_function

from ribs._utils import check_batch_shape, check_finite, validate_batch, validate_single  # noqa: PLC2701
from ribs.archives import ArchiveBase, CQDScoreResult
from ribs.archives._transforms import batch_entries_with_threshold, compute_best_index, compute_objective_sum  # noqa: PLC2701
from scipy.spatial import KDTree

_UNSTRUCTURED_ARCHIVE_FIELDS = {"novelty_score", "objective"}
_NON_DOMINATING_SORT_METHODS = {
    "efficient_non_dominated_sort": "efficient_non_dominated_sort",
    "efficient": "efficient_non_dominated_sort",
    "efficient_ns": "efficient_non_dominated_sort",
    "efficient_nds": "efficient_non_dominated_sort",
    "ens": "efficient_non_dominated_sort",
    "ends": "efficient_non_dominated_sort",
    "fast_non_dominated_sort": "fast_non_dominated_sort",
    "fast_ns": "fast_non_dominated_sort",
    "fast_nds": "fast_non_dominated_sort",
    "fns": "fast_non_dominated_sort",
    "fnds": "fast_non_dominated_sort",
    "tree_based_non_dominated_sort": "tree_based_non_dominated_sort",
    "tree_ns": "tree_based_non_dominated_sort",
    "tree_nds": "tree_based_non_dominated_sort",
    "tns": "tree_based_non_dominated_sort",
    "tnds": "tree_based_non_dominated_sort",
}


def parse_dtype(dtype: str | type | dict[str, str | type]) -> type | dict[str, type]:
    """Parse the dtype for the archive.

    Parameters
    ----------
    dtype : str | type | dict[str, str  |  type]
        The type or types for fields in the archive

    Returns
    -------
    type | dict[str, type]
        The parsed dtypes of the archive, mapping fields to types

    Raises
    ------
    ValueError

        - ``dtype`` was a dict but missing :python:`"solution"`, :python:`"objective"`, or :python:`"measures"`
        - Unsupported dtype.
    """
    # First convert str dtype's to np.dtype.
    if isinstance(dtype, str):
        dtype = np.dtype(dtype)  # type: ignore[assignment]

    # np.dtype is not np.float32 or np.float64, but it compares equal.
    if dtype in {np.float32, np.float64}:
        return {
            "solution": dtype,  # type: ignore[dict-item]
            "objective": dtype,  # type: ignore[dict-item]
            "measures": dtype,  # type: ignore[dict-item]
        }
    if isinstance(dtype, dict):
        if "solution" not in dtype or "objective" not in dtype or "measures" not in dtype:
            msg = "If dtype is a dict, it must contain 'solution','objective', and 'measures' keys."
            raise ValueError(msg)
        return {k: parse_dtype(dt) for k, dt in dtype.items()}  # type: ignore[misc]
    msg = (
        "Unsupported dtype. Must be np.float32 or np.float64, or dict of the form "
        '{"solution": <dtype>, "objective": <dtype>, "measures": <dtype>}'
    )
    raise ValueError(msg)


class UnstructuredArchive(ArchiveBase):
    r"""An archive that adds new solutions based on their novelty.

    This archive is described in `Lehman 2011
    <https://www.cs.swarthmore.edu/~meeden/DevelopmentalRobotics/lehman_gecco11.pdf>`_.
    If a solution is in a sparse area of metric space it is added
    unconditionally. When a solution is in an overdense region it is added to
    the archive only if its objective improves upon the nearest existing
    solution. The archive uses the mean distance of the k-nearest neighbors to
    determine the novelty of metric space

    Parameters
    ----------
    solution_dim : int
        Dimension of the solution space.
    measure_dim : int
        The dimension of the measure space.
    k_neighbors : int
        The number of nearest neighbors to use for
        determining sparseness.
    novelty_threshold : float
        The level of novelty required to add a
        solution to the archive unconditionally
    local_competition : bool
        Whether to add solutions based on the objectives of the k-nearest neighbors
        or based purely on novelty, default is :data:`python:True`.
    initial_capacity : int
        The initial capacity for the archive, default is 128.
    qd_score_offset : float
        Archives often contain negative objective
        values, and if the QD score were to be computed with these negative
        objectives, the algorithm would be penalized for adding new cells
        with negative objectives. Thus, a standard practice is to normalize
        all the objectives so that they are non-negative by introducing an
        offset. This QD score offset will be *subtracted* from all
        objectives in the archive, e.g., if your objectives go as low as
        -300, pass in -300 so that each objective will be transformed as
        :python:`objective - (-300)`.
    target_size : int | None
        The target size for the archive, above which upward pressure will
        be put on the novelty threshold, default is :data:`python:None`.
    max_size : int | None
        The absolut maximum size for the archive, dfeault is None.
    seed : int
        Value to seed the random number generator. Set to None to
        avoid a fixed seed.
    dtype : str | numpy.dtype | dict[str, str | numpy.dtype]
        Data type of the solutions, objectives, and measures. We only support
        ``"f"`` / :class:`~np.float32` and ``"d"`` / :class:`~np.float64`.
        Alternatively, this can be a dict specifying separate dtypes, of the form
        :python:`{"solution": <dtype>, "objective": <dtype>, "measures": <dtype>}`.
    extra_fields : dict
        Description of extra fields of data that is stored
        next to elite data like solutions and objectives. The description is
        a dict mapping from a field name (:class:`str`) to a tuple of :python:`(shape, dtype)` .
        For instance, :python:`{"foo": ((), np.float32), "bar": ((10,), np.float32)}`
        will create a ``"foo"`` field that contains scalar values and a ``"bar"``
        field that contains 10D values. Note that field names must be valid
        Python identifiers, and names already used in the archive are not
        allowed.
    ckdtree_kwargs : dict | None
        The keyword args to pass to SciPy's :class:`~scipy.spatial.KDTree` class constructor. Default is None.
    ckdtree_query_kwargs : dict | None
        The keyword args to pass to SciPy's :meth:`~scipy.spatial.KDTree.query` method. Default is None
    nds_method : str, optional
        The non-dominating sort method to use for multi-objective local competition, default is "efficient_nds".
    nds_kwargs : dict | None, optional
        The keyword args to pass to the ``nds_method``
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        solution_dim,
        measure_dim,
        k_neighbors,
        novelty_threshold,
        local_competition=True,
        initial_capacity=128,
        qd_score_offset=0.0,
        target_size=None,
        max_size=None,
        seed=None,
        dtype=np.float64,
        extra_fields=None,
        extra_objective_fields=None,
        ckdtree_kwargs=None,
        ckdtree_query_kwargs=None,
        nds_method="efficient_nds",
        nds_kwargs=None,
    ):
        if initial_capacity < 1:
            msg = "initial_capacity must be at least 1."
            raise ValueError(msg)

        extra_fields = extra_fields or {}
        if _UNSTRUCTURED_ARCHIVE_FIELDS & extra_fields.keys():
            msg = f"The following names are not allowed in extra_fields: {_UNSTRUCTURED_ARCHIVE_FIELDS}"
            raise ValueError(msg)

        extra_objective_fields = extra_objective_fields or {}
        if _UNSTRUCTURED_ARCHIVE_FIELDS & extra_objective_fields.keys():
            msg = f"The following names are not allowed in extra_objective_fields: {_UNSTRUCTURED_ARCHIVE_FIELDS}"
            raise ValueError(msg)

        measure_dtype = parse_dtype(dtype)["measures"]
        extra_fields.update(extra_objective_fields)
        extra_objective_fields["objective"] = ((), dtype)

        ArchiveBase.__init__(
            self,
            solution_dim=solution_dim,
            cells=initial_capacity,
            measure_dim=measure_dim,
            qd_score_offset=qd_score_offset,
            seed=seed,
            dtype=dtype,
            extra_fields=extra_fields,
        )

        self._k_neighbors = int(k_neighbors)
        self._novelty_threshold = measure_dtype(novelty_threshold)
        self._max_size = int(max_size) if max_size else None
        self._target_size = int(target_size) if target_size else None
        self.local_competition = bool(local_competition)

        self._objective_fields = list(extra_objective_fields)
        self._single_objective = len(self._objective_fields) == 1

        # No default values to override.
        self._ckdtree_kwargs = {} if ckdtree_kwargs is None else ckdtree_kwargs.copy()

        # Apply default args for kd-tree query.
        self._ckdtree_query_kwargs = {} if ckdtree_query_kwargs is None else ckdtree_query_kwargs.copy()

        # k-D tree with current measures in the archive. Updated on add().
        self._cur_kd_tree = KDTree(self._store.data("measures"), **self._ckdtree_kwargs)

        self._nds_kwargs = nds_kwargs or {}
        self._nds_method = load_function(_NON_DOMINATING_SORT_METHODS[nds_method])

    @property
    def k_neighbors(self) -> int:
        """
        The number of nearest neighbors to use for determining novelty.

        Returns
        -------
        int
            The number of nearest neighbors to use for determining novelty.
        """
        return self._k_neighbors

    @property
    def novelty_threshold(self) -> float:
        """
        The degree of sparseness in metric space required for a solution to be added unconditionally.

        Returns
        -------
        float
            The degree of sparseness in metric space required for a solution to be added unconditionally.
        """
        return self._novelty_threshold

    @property
    def objective_fields(self) -> list:
        """The list of data fields used as objectives in determining whether a solution can be added to the archive.

        Solutions will be added if the value of these fields are at least as
        high as those in their k-neighborhoods. Ties are broken using the
        novelty objective.

        This behavior is the same as
        [NSGA-II](https://ieeexplore.ieee.org/document/996017) with the
        crowding function replaced with the novelty score as described in
        [Lehman, 2011](https://dl.acm.org/doi/10.1145/2001576.2001606) extended
        from fitness + genotypic diversity to an arbitrary number of objectives.

        Returns
        -------
        list
            The list of data fields used as objectives in determining whether a solution can be added to the archive.
        """
        return self._objective_fields

    @property
    def cells(self) -> int:
        """
        Total number of cells in the archive.

        Since this archive is unstructured and grows over time, the number of cells is equal to the
        number of solutions currently in the archive.

        Returns
        -------
        int
            Total number of cells in the archive.
        """
        return len(self)

    @property
    def capacity(self) -> int:
        """
        The number of solutions that can currently be stored in this archive.

        The capacity doubles every time the archive fills up, similar to a C++ vector.

        Returns
        -------
        int
            The number of solutions that can currently be stored in this archive.
        """
        return self._store.capacity

    @cached_property
    def dtypes(self):
        """
        Data types of fields in the store.

        Returns
        -------
        dict
            Data types of fields in the store.

        Examples
        --------
        >>> store.dtypes == {
            "objective": np.float32,
            "measures": np.float32,
        }
        """
        return {name: arr.dtype for name, arr in self._store._fields.items()}

    def index_of(self, measures: np.ndarray) -> np.ndarray:
        """Return the index of the closest solution to the given measures.

        Unlike the structured archives like :class:`~ribs.archives._grid_archive.GridArchive`,
        this archive does not have indexed cells where each measure "belongs."
        Thus, this method instead returns the index of the solution with the
        closest measure to each solution passed in.

        This means that ``retrieve`` will return the solution with the
        closest measure to each measure passed into that method.

        Parameters
        ----------
        measures : np.ndarray
            :python:`(batch_size, measure_dim)` array of coordinates in measure space.

        Returns
        -------
        np.ndarray
            :python:`(batch_size,)` array of integer indices representing the location of the solution in the archive.

        Raises
        ------
        RuntimeError
            If the archive is empty
        """
        measures = np.asarray(measures)
        check_batch_shape(measures, "measures", self.measure_dim, "measure_dim")
        check_finite(measures, "measures")

        if self.empty:
            msg = (
                "There were no solutions in the archive. "
                f"`{self.__class__.__name__}.index_of` computes the nearest "
                "neighbor to the input measures, so there must be at least one "
                "solution present in the archive."
            )
            raise RuntimeError(msg)

        _, indices = self._cur_kd_tree.query(measures, **self._ckdtree_query_kwargs)
        return indices.astype(np.int32)

    def compute_novelty(self, measures, local_competition=None):
        """Compute the novelty and local competition of the given measures.

        Parameters
        ----------
        measures : array-like
            :python:`(batch_size, measure_dim)` array of
            coordinates in measure space.
        local_competition : None | array-like
            This can be None to indicate not to compute local competition.
            Otherwise, it can be a (batch_size,) array of objective values
            to use as references for computing objective values.

        Returns
        -------
        np.ndarray | tuple[np.ndarray, np.ndarray]
            Either one value or a tuple of two values:

            - :class:`~np.ndarray`
                :python:`(batch_size,)` array holding the novelty score of
                each measure. If the archive is empty, the novelty is set to the
                :attr:`novelty_threshold`.
            - :class:`python:tuple` [:class:`~numpy.ndarray`, :class:`~numpy.ndarray` ]
                If ``local_competition`` is passed in, a :python:`(batch_size,)` array
                holding the local competition of each solution will also be returned.
                If the archive is empty, the local competition will be set to 0.
        """
        measures = np.asarray(measures)
        batch_size = len(measures)

        if local_competition is not None:
            objectives = np.asarray(local_competition)
            local_competition = True

        if self.empty:
            # Set default values for novelty and local competition when archive
            # is empty.
            novelty = np.full(batch_size, self.novelty_threshold, dtype=self.dtypes["measures"])

            if local_competition:
                local_competition_scores = np.zeros(len(novelty), dtype=np.int32)
        else:
            # Compute nearest neighbors.
            k_neighbors = min(len(self), self.k_neighbors)
            dists, indices = self._cur_kd_tree.query(measures, k=k_neighbors, **self._ckdtree_query_kwargs)

            # Expand since query() automatically squeezes the last dim when k=1.
            dists = dists[:, None] if k_neighbors == 1 else dists

            novelty = np.mean(dists, axis=1)

            if local_competition:
                indices = indices[:, None] if k_neighbors == 1 else indices

                # The first item returned by `retrieve` is `occupied` -- all
                # these indices are occupied since they are indices of solutions
                # in the archive.
                neighbor_objectives = self._store.retrieve(indices.ravel(), "objective")[1]
                neighbor_objectives = neighbor_objectives.reshape(indices.shape)

                # Local competition is the number of neighbors who have a lower
                # objective.
                local_competition_scores = np.sum(neighbor_objectives < objectives[:, None], axis=1, dtype=np.int32)

        if local_competition:
            return novelty, local_competition_scores
        return novelty

    def kdtree(self, measures: np.ndarray | None = None) -> tuple[KDTree, np.ndarray]:
        """Construct a KDTree using the archive plus the ``measures``.

        The ``measures`` are concatenated onto the end of the archive so any
        indices returned by the KDTree methods which are greater than the
        current archive size indicate that they come from the ``measures``.

        The construction of the tree can be controlled using the
        ``ckdtree_kwargs`` dictionary passed to this class's ``__init__``

        Parameters
        ----------
        measures : np.ndarray | None, optional
            Any measures to include in the KDTree, :data:`python:None`.

        Returns
        -------
        kd_tree : KDTree
            The KDTree constructed from the archive and measures.
        all_measures : np.ndarray
            The array passed as the ``data`` to the
            :class:`~scipy.spatial.KDTree` constructor method.
        """
        all_measures = self._store.data("measures") if measures is None else np.concatenate([self._store.data("measures"), measures])
        return KDTree(all_measures, **self._ckdtree_kwargs), all_measures

    def add(self, solution, objective, measures, **fields):
        r"""Insert a batch of solutions into the archive.

        Solutions are inserted if they have a high enough novelty score as
        discussed in the documentation for this class. The novelty is determined
        by comparing to solutions currently in the archive.

        If :attr:`local_competition` is turned on, solutions can also replace
        existing solutions in the archive. Namely, if the solution was not novel
        enough to be added, it will be compared to its nearest neighbor, and if
        it exceeds the objective value of its nearest neighbor, it will replace
        the nearest neighbor. If there are conflicts where multiple solutions
        may replace a single solution, the highest-performing is chosen.

        .. note:: The indices of all arguments should "correspond" to each
            other, i.e. ``solution[i]``, ``objective[i]``,
            ``measures[i]``, and should be the solution parameters,
            objective, and measures for solution ``i``.

        Parameters
        ----------
        solution : array-like
            :python:`(batch_size, solution_dim)` array of solution parameters.
        objective :  None | array-like
            A value of None will cause the
            objective values to default to 0. However, if the user wishes to
            associate an objective with each solution, this can be a
            :python:`(batch_size,)` array with objective function evaluations of the
            solutions. If :attr:`local_competition` is turned on, this
            argument must be provided.
        measures : array-like
            :python:`(batch_size, measure_dim)` array with
            measure space coordinates of all the solutions.
        `**fields` : dict
            Additional data for each solution. Each argument should be an array with
            batch_size as the first dimension.

        Returns
        -------
        dict
            Information describing the result of the add operation. The dict contains the following keys:

            - ``"status"`` (:class:`~numpy.ndarray` of :class:`int` )
                An array of integers that represent the "status" obtained when
                attempting to insert each solution in the batch. Each item has
                the following possible values:

                - ``0``: The solution was not added to the archive.
                - ``1``: The solution replaced an existing solution in the
                    archive due to having a higher objective (only applies if
                    :attr:`local_competition` is turned on).
                - ``2``: The solution was added to the archive due to being
                    sufficiently novel.

                To convert statuses to a more semantic format, cast all statuses
                to :class:`~ribs.archives.AddStatus` e.g. with :python:`[AddStatus(s) for s in add_info["status"]]`.

            - ``"novelty"`` (:class:`~numpy.ndarray` of :attr:`dtypes` :python:`["measures"]`)
                The computed novelty of the solutions passed in. If
                there were no solutions to compute novelty with respect to (i.e.,
                The archive was empty), the novelty is set to the
                :attr:`novelty_threshold`.

            - ``"local_competition"`` ( :class:`~numpy.ndarray` of :class:`int` )
                Only available if :attr:`local_competition` is turned on.
                Indicates, for each solution, how many of the nearest neighbors
                had lower objective values. Maximum value is :attr:`k_neighbors`.
                If there were no solutions to compute novelty with respect to,
                (i.e., the archive was empty), the local competition is set to 0.

            - ``"value"`` ( :class:`~numpy.ndarray` )
                :attr:`dtypes` ["objective"]): Only available if
                :attr:`local_competition` is turned on. The meaning of each value
                depends on the corresponding ``status`` and is inspired by the
                values in CMA-ME (`Fontaine 2020
                <https://arxiv.org/abs/1912.02400>`_):

                - ``0`` (not added): The value is the "negative improvement", i.e.
                    The objective of the solution passed in minus the objective of
                    The nearest neighbor (this value is negative because the
                    solution did not have a high enough objective to be added to the
                    archive).
                - ``1`` (replace/improve existing solution): The value is the
                    "improvement," i.e. the objective of the solution passed in
                    minus the objective of the elite that was replaced.
                - ``2`` (new solution): The value is just the objective of the
                    solution.

        Raises
        ------
        ValueError

            - The array arguments do not match their specified shapes.
            - ``objective`` or ``measures`` has non-finite values (inf or NaN).
            - :attr:`local_competition` is turned on but objective was not passed in.
        """
        if objective is None:
            if self.local_competition:
                msg = "If local competition is turned on, objective must be passed in to add()."
                raise ValueError(msg)
            objective = np.zeros(len(solution), dtype=self.dtypes["objective"])

        data = validate_batch(self, {"solution": solution, "objective": objective, "measures": measures, **fields})

        if self.local_competition:
            novelty, local_competition = self.compute_novelty(measures=data["measures"], local_competition=data["objective"])
        else:
            novelty = self.compute_novelty(measures=data["measures"])

        novel_enough = novelty >= self.novelty_threshold
        n_novel_enough = np.sum(novel_enough)
        new_size = len(self) + n_novel_enough

        if self.local_competition:
            # In the case of local competition, we consider all solutions for
            # addition.
            add_indices = np.empty(len(novelty), dtype=np.int32)

            # New solutions are assigned the new indices.
            add_indices[novel_enough] = np.arange(len(self), new_size)

            # Solutions that were not novel enough have the potential to replace
            # their nearest neighbors in the archive.
            not_novel_enough = ~novel_enough
            n_not_novel_enough = len(novelty) - n_novel_enough
            if n_not_novel_enough > 0:
                add_indices[not_novel_enough] = self.index_of(data["measures"][not_novel_enough])

            add_data = data
        else:
            # Without local competition, the only solutions that can be added
            # are the ones that were novel enough.
            add_indices = np.arange(len(self), new_size)
            add_data = {key: val[novel_enough] for key, val in data.items()}

        if new_size > self.capacity:
            # Resize the store by doubling its capacity. We may need to double
            # the capacity multiple times. The log2 below indicates how many
            # times we would need to double the capacity. We obtain the final
            # multiplier by raising to a power of 2.
            multiplier = 2 ** int(np.ceil(np.log2(new_size / self.capacity)))
            self._store.resize(multiplier * self.capacity)

        add_info = self._store.add(
            add_indices,
            add_data,
            {
                "dtype": self.dtypes["threshold"],
                "learning_rate": self._learning_rate,
                # Note that when only novelty is considered, objectives default
                # to 0, so all solutions specified will be added because the
                # threshold_min is -np.inf.
                "threshold_min": self._threshold_min,
                "objective_sum": self._objective_sum,
            },
            [batch_entries_with_threshold, compute_objective_sum, compute_best_index],
        )

        objective_sum = add_info.pop("objective_sum")
        best_index = add_info.pop("best_index")

        # Add novelty to the data.
        add_info["novelty"] = novelty

        if self.local_competition:
            # add_info contains results for all solutions. We also want to
            # return local_competition info.
            add_info["local_competition"] = local_competition
        else:
            # add_info only contains results for the solutions that were novel
            # enough. Here we create an add_info that contains results for all
            # solutions.
            all_status = np.zeros(len(data["measures"]), dtype=np.int32)
            all_status[novel_enough] = add_info["status"]
            add_info["status"] = all_status

            # We ignore objective/threshold when only novelty is considered.
            del add_info["value"]

        if not np.all(add_info["status"] == 0):
            self._stats_update(objective_sum, best_index)

            # Make a new tree with the updated solutions.
            self._cur_kd_tree = KDTree(self._store.data("measures"), **self._ckdtree_kwargs)

            # Clear the cached properties since they have now changed.
            if "upper_bounds" in self.__dict__:
                del self.__dict__["upper_bounds"]
            if "lower_bounds" in self.__dict__:
                del self.__dict__["lower_bounds"]

        return add_info

    def add_single(self, solution, objective, measures, **fields):
        """Insert a single solution into the archive.

        Parameters
        ----------
        solution : array-like
            Parameters of the solution.
        objective : None | float
            Set to None to get the default value of
            0; otherwise, a valid objective value is also acceptable.
        measures : array-like
            Coordinates in measure space of the solution.
        `**fields` : dict
            Additional data for the solution.

        Returns
        -------
        dict
            Information describing the result of the add operation. The dict contains ``status`` and ``novelty`` keys;
            refer to :meth:`add` for the meaning of status and novelty.

        Raises
        ------
        ValueError

            - The array arguments do not match their specified shapes.
            - ``objective`` is non-finite (inf or NaN) or ``measures`` has non-finite values.
            - :attr:`local_competition` is turned on but objective was not passed in.
        """
        if objective is None:
            if self.local_competition:
                msg = "If local competition is turned on, objective must be passed in to add_single()."
                raise ValueError(msg)
            objective = 0.0

        data = validate_single(self, {"solution": solution, "objective": objective, "measures": measures, **fields})

        return self.add(**{key: [val] for key, val in data.items()})

    def cqd_score(
        self,
        iterations: int,
        target_points: int | Sequence,
        penalties: int | Sequence,
        obj_min: float,
        obj_max: float,
        dist_max: float | None = None,
        dist_ord: int | str | None = None,
    ) -> CQDScoreResult:
        r"""Compute the CQD score of the archive.

        Refer to the documentation in :meth:`~ribs.archives.ArchiveBase.cqd_score` for more
        info. The key difference from the base implementation is that the
        implementation in :class:`~ribs.archives.ArchiveBase` assumes the archive has a pre-defined
        measure space with lower and upper bounds. However, by nature of being
        unstructured, this archive has lower and upper bounds that change over
        time. Thus, it is required to directly pass in ``target_points`` and
        ``dist_max``.

        Parameters
        ----------
        iterations : int
            Number of times to compute the CQD score. We return the mean CQD score across these iterations.
        target_points : int | Sequence
            Number of target points to generate, or an (iterations, n, measure_dim) array which lists n target
            points to list on each iteration. When an int is passed, the points are sampled uniformly within the
            bounds of the measure space.
        penalties : int | Sequence
            Number of penalty values over which to compute the score (the values are distributed evenly over the
            range [0,1]). Alternatively, this may be a 1D array which explicitly lists the penalty values.
            Known as :math:`\\theta` in Kent 2022.
        obj_min : float
            Minimum objective value, used when normalizing the objectives.
        obj_max : float
            Maximum objective value, used when normalizing the objectives.
        dist_max : float | None
            Maximum distance between points in measure space. Defaults to the distance between the extremes of
            The measure space bounds (the type of distance is computed with the order specified by ``dist_ord``).
            Known as :math:`\\delta_{max}` in Kent 2022.
        dist_ord : int | str | None, optional
            Order of the norm to use for calculating measure space distance; this is passed to
            :func:`numpy.linalg.norm` as the ``ord`` argument. See :func:`numpy.linalg.norm` for possible values.
            The default is to use Euclidean distance (L2 norm).

        Returns
        -------
        CQDScoreResult
            The CQD score.

        Raises
        ------
        ValueError
            The dist_max and target_points were not passed in.
        """
        if dist_max is None or np.isscalar(target_points):
            msg = (
                f"In {self.__class__.__name__}, dist_max must be passed in, "
                "and target_points must be passed in as a custom array of points."
            )
            raise ValueError(msg)

        return super().cqd_score(
            iterations=iterations,
            target_points=target_points,
            penalties=penalties,
            obj_min=obj_min,
            obj_max=obj_max,
            dist_max=dist_max,
            dist_ord=dist_ord,
        )

    @property
    def _cells(self) -> int:
        """
        Make sure the value returned by ``_cells`` always matches the store capacity.

        Returns
        -------
        int
            The number of cells in the archive.
        """
        if hasattr(self, "_store"):
            return self._store._props["capacity"]  # noqa: SLF001
        return self.__cells

    @_cells.setter
    def _cells(self, new_size):
        if hasattr(self, "_store"):
            self._store.resize(new_size)
        self.__cells = new_size

    @cached_property
    def upper_bounds(self) -> np.ndarray:
        """
        The upper bounds of the measures in the archive.

        Since the archive can grow arbitrarily this is calculated based on the
        maximum measure values of the solutions in the archive.

        The user should take care when the archive only has a single solution
        since the upper bound would equal the lower bound and my cause
        problems.
        problems.

        Returns
        -------
        np.ndarray
            The upper bounds of the measures in the archive.
        """
        return np.max(self._store.data("measures"), axis=0)

    @cached_property
    def lower_bounds(self) -> np.ndarray:
        """
        The lower bounds of the measures in the archive.

        Since the archive can grow arbitrarily this is calculated based on the
        minimum measure values of the solutions in the archive.

        The user should take care when the archive only has a single solution
        since the upper bound would equal the lower bound and my cause
        problems.

        Returns
        -------
        np.ndarray
            The lower bounds of the measures in the archive.
        """
        return np.min(self._store.data("measures"), axis=0)

    def _stats_update(self, new_objective_sum, new_best_index):
        super()._stats_update(new_objective_sum, new_best_index)

        # clear the cached properties since they've now changed
        if "upper_bounds" in self.__dict__:
            del self.__dict__["upper_bounds"]
        if "lower_bounds" in self.__dict__:
            del self.__dict__["lower_bounds"]

        measures = self._store.data("measures")
        self._cur_kd_tree = KDTree(measures, **self._ckdtree_kwargs)
