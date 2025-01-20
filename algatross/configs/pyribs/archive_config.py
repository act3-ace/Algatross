"""A module of configuration dataclasses for PyRibs Archives."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from algatross.quality_diversity.visualization import nearest_neighbor_archive_heatmap


@dataclass
class ArchiveConfig:
    """Base configuration for :class:`~ribs.archives.ArchiveBase`."""

    solution_dim: int | None = None
    """Dimensionality of the solutions in the archive."""
    learning_rate: float = 1.0
    """The learning rate for CQD, default is 1.0."""
    threshold_min: float = -np.inf
    """Minimum threshold for adding solutions to the archive, default is -:python:`np.inf`."""
    qd_score_offset: float = 0.0
    """Offset to subtract from QD scores, default is 0.0."""
    seed: int | None = None
    """Random number generator seed, default is :data:`python:None`."""
    dtype = np.float64
    """Datatype for the archive, default is :class:`np.float64`."""
    visualize: bool = False
    """Whether or not the archive should be visualized, default is :data:`python:False`."""


@dataclass
class GridArchiveConfig(ArchiveConfig):
    """A dataclass for configuration of grid archives."""

    dims: tuple[int, ...] | None = None
    """The number of chunks each dimension should be broken into."""
    ranges: Sequence[tuple[float, float]] | None = None
    """The bounds for each dimension."""
    epsilon: float = 1e-6
    """The value for numerical stability."""


@dataclass
class CVTArchiveConfig(ArchiveConfig):
    """A dataclass for configuration of CVT archives."""

    cells: int | None = None
    """The number of cells for the CVT, default is :data:`python:None`."""
    ranges: Sequence[tuple[float, float]] | None = None
    """The ranges for each dimension of the CVT."""
    samples: int = 100_000
    """The number of samples for constructing the CVT centroids."""
    custom_centroids: list[np.ndarray] | None = None
    """Custom centroids to use for the CVT."""
    k_means_kwargs: dict[str, Any] | None = None
    """Keyword arguments to pass to the k-means algorithm."""
    use_kd_tree: bool = True
    """Whether to use the kdTree."""
    ckdtree_kwargs: dict[str, Any] | None = None
    """Keyword arguments to pass to the cKDTree constructor."""


@dataclass
class SlidingBoundsArchiveConfig(ArchiveConfig):
    """A dataclass for configuration of sliding bounds archives."""

    dims: tuple[int, ...] | None = None
    """The dimensions for the sliding bounds, default is :data:`python:None`."""
    ranges: Sequence[tuple[float, float]] | None = None
    """The ranges for each dimension of the sliding bounds, default is :data:`python:None`."""
    remap_frequency: int = 100
    """The frequency at which the sliding bounds should be recomputed, default is 100."""
    buffer_capacity: int = 1000
    """The capacity of the sliding bounds buffer, default is 1000."""


@dataclass
class UnstructuredArchiveConfig:
    """Configuration class for :class:`~algatross.quality_diversity.archives.unstructured.UnstructuredArchive`."""

    solution_dim: int | None = None
    """Dimensionality of the solutions in the archive."""
    k_neighbors: int = 5
    """Number of neighbors to use for kNN comparisons."""
    novelty_threshold: float = 1
    """The novelty threshold solutions must exceed in order to be added to the archive, default is 1."""
    qd_score_offset: float = 0.0
    """The offset for the QD scores."""
    seed: int | None = None
    """The seed for the random number generator."""
    dtype = np.float64
    """The datatype for the archive."""
    visualize: bool = False
    """Whether to visualize the archive or not, default is :data:`python:False`."""
    visualizer: Callable = nearest_neighbor_archive_heatmap
    """Default is :func:`~algatross.quality_diversity.visualization.nearest_neighbor_archive_heatmap`."""
