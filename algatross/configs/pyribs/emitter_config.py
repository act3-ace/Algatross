"""A module of configuration dataclasses for PyRibs Emitters."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


@dataclass
class EmitterConfig:
    """EmitterConfig base dataclass for configuration of all emitters."""

    bounds: Sequence[tuple[float, float]] | np.ndarray | None = None
    """
    Bounds of the solution space. Pass None to indicate there are no bounds.

    Alternatively, pass an array-like to specify the bounds for each dim. Each element
    in this array-like can be None to indicate no bound, or a tuple of
    :python:`(lower_bound, upper_bound)`, where :python:`lower_bound` or
    :python:`upper_bound` may be :data:`python:None` to indicate no bound.
    """

    batch_size: int | None = None
    """
    Number of solutions to return in :meth:`~ribs.emitters.EmitterBase.ask`.

    If not passed in, a batch size will be automatically calculated using the default CMA-ES rules.
    """

    seed: int | None = None
    """
    Seed for the random number generator.

    If none is specified it will default to the algorithm configs seed value.
    """


@dataclass
class EvolutionStrategyEmitterConfig(EmitterConfig):
    """EvolutionStrategyEmitterConfig  dataclass for configuration of evolution strategy emitters."""

    x0: np.ndarray | None = None
    """
    Center of the Gaussian distribution from which to sample solutions when the archive is empty.

    Must be 1-dimensional. This argument is ignored if ``initial_solutions`` is set.
    sigma0 : float | Sequence[float] | np.ndarray
    """

    sigma0: float | Sequence[float] | np.ndarray = 0
    """
    Standard deviation of the Gaussian distribution.

    Note we assume the Gaussian is diagonal, so if this argument is an array, it must be 1D.
    """

    ranker: Callable | str = "2imp"
    """
    A :class:`~ribs.emitters.rankers.RankerBase` object that orders the solutions after they have been evaluated in the environment.

    This parameter may be a callable (e.g. a class or a lambda function) that takes in no parameters and
    returns an instance of :class:`~ribs.emitters.rankers.RankerBase`, or it may be a full or abbreviated
    ranker name as described in :func:`~ribs.emitters.rankers._get_ranker`.
    """

    es: Callable | str = "cma_es"
    """
    An :class:`~ribs.emitters.opt.EvolutionStrategyBase` object that is used to adapt the distribution from which new solutions are
    sampled.

    This parameter may be a callable (e.g. a class or a lambda function) that takes in the parameters of
    :class:`~ribs.emitters.opt.EvolutionStrategyBase` along with kwargs provided by the ``es_kwargs``
    argument, or it may be a full or abbreviated optimizer name as described in :mod:`ribs.emitters.opt`.
    """

    es_kwargs: dict[str, Any] | None = None
    """
    Additional arguments to pass to the evolution strategy optimizer.

    See the evolution-strategy-based optimizers in :mod:`ribs.emitters.opt` for the arguments allowed
    by each optimizer.
    """

    selection_rule: Literal["mu", "filter"] = "filter"
    """
    Method for selecting parents for the evolution strategy.

    With "mu" selection, the first half of the solutions will be selected as parents, while in "filter",
    any solutions that were added to the archive will be selected.
    """

    restart_rule: int | Literal["no_improvement", "basic"] = "no_improvement"
    """
    Method to use when checking for restarts.

    If given an integer, then the emitter will restart after this many iterations, where each iteration
    is a call to :meth:`~ribs.emitters.EmitterBase.tell`. With "basic", only the default CMA-ES convergence
    rules will be used, while with "no_improvement", the emitter will restart when none of the proposed
    solutions were added to the archive.
    """


@dataclass
class GaussianEmitterConfig(EmitterConfig):
    """GaussianEmitterConfig dataclass for configuration of gaussian emitters."""

    x0: np.ndarray | None = None
    """
    Center of the Gaussian distribution from which to sample solutions when the archive is empty.

    Must be 1-dimensional. This argument is ignored if ``initial_solutions`` is set.
    """

    sigma: float | Sequence[float] | np.ndarray = 0
    """
    Standard deviation of the Gaussian distribution.

    Note we assume the Gaussian is diagonal, so if this argument is an array, it must be 1D.
    """

    initial_solutions: Sequence[tuple[float, ...]] | np.ndarray | None = None
    """
    An (n, solution_dim) array of solutions to be used when the archive is empty.

    If this argument is None, then solutions will be sampled from a Gaussian distribution centered at
    ``x0`` with standard deviation ``sigma``.
    """


@dataclass
class RandomEmitterConfig(GaussianEmitterConfig):
    """RandomEmitterConfig dataclass for configuration of random emitters."""


@dataclass
class GradientArborescenceEmitterConfig(EvolutionStrategyEmitterConfig):
    """GradientArborescenceEmitterConfig dataclass for configuration of gradient arborescence emitters."""

    lr: float = 1e-3
    """Learning rate for the gradient optimizer, default is 3e-3."""

    grad_opt: Callable | str = "adam"
    """
    Gradient optimizer to use for the gradient ascent step of the algorithm.

    The optimizer is a :class:`~ribs.emitters.opt.GradientOptBase` object. This parameter may
    be a callable (e.g. a class or a lambda function) which takes in the ``theta0`` and ``lr``
    arguments, or it may be a full or abbreviated name as described in :mod:`ribs.emitters.opt`.
    """

    grad_opt_kwargs: dict[str, Any] | None = None
    """
    Additional arguments to pass to the gradient optimizer.

    See the gradient-based optimizers in :mod:`~ribs.emitters.opt` for the arguments allowed by
    each optimizer. Note that we already pass in ``theta0`` and ``lr``., default is :data:`python:None`.
    """

    normalize_grad: bool = True
    """
    If true (default), then gradient infomation will be normalized.

    Otherwise, it will not be normalized.
    """
    epsilon: float = 1e-8
    """
    For numerical stability, we add a small epsilon when normalizing gradients in :meth:`~ribs.emitters.EmitterBase.tell_dqd`.

    Refer to the implementation `here <../_modules/ribs/emitters/_gradient_arborescence_emitter.html#GradientArborescenceEmitter.tell_dqd>`
    . Pass this parameter to configure that epsilon.
    """


@dataclass
class IsoLineEmitterConfig(GaussianEmitterConfig):
    """IsoLineEmitterConfig dataclass for configuration of isoline emitters."""

    iso_sigma: float = 0.01
    """Scale factor for the isotropic distribution used to generate solutions. Default is 0.01."""
    line_sigma: float = 0.2
    """Scale factor for the line distribution used when generating solutions. Default is 0.2."""
