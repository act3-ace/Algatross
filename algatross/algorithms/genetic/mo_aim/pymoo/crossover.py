"""Module containing modified crossover pymoo objects."""

import numpy as np

from pymoo.core.problem import Problem
from pymoo.core.variable import Real, get
from pymoo.operators.crossover.sbx import SBX, cross_sbx

from algatross.utils.random import resolve_seed
from algatross.utils.types import NumpyRandomSeed


class SimulatedBinaryCrossover(SBX):
    r"""
    A modified PyMOO simulated binary crossover class.

    The only difference is that it is passed a numpy Generator or random seed so that
    randomness can be kept in-sync with other algorithms.

    Parameters
    ----------
    prob_var : float, optional
        Probability of undergoing crossover, default is 0.5.
    eta : float, optional
        The :math:`\eta` parameter for crossover, default is 15.
    prob_exch : float, optional
        The probability of exchange, default is 1.0.
    prob_bin : float, optional
        The binary probability, default is 0.5.
    n_offsprings : int, optional
        The number of offsprings from the crossover, default is 2.
    generator : np.random.Generator | None, optional
        The generator to share, default is :data:`python:None`.
    seed : NumpyRandomSeed | None, optional
        The seed for randomness, default is :data:`python:None`
    `**kwargs`
        Additional keyword arguments
    """

    def __init__(
        self,
        prob_var: float = 0.5,
        eta: float = 15,
        prob_exch: float = 1.0,
        prob_bin: float = 0.5,
        n_offsprings: int = 2,
        generator: np.random.Generator | None = None,
        seed: NumpyRandomSeed = None,
        **kwargs,
    ):
        super().__init__(2, n_offsprings, **kwargs)

        self.prob_var = Real(prob_var, bounds=(0.1, 0.9))
        self.eta = Real(eta, bounds=(3.0, 30.0), strict=(1.0, None))
        self.prob_exch = Real(prob_exch, bounds=(0.0, 1.0), strict=(0.0, 1.0))
        self.prob_bin = Real(prob_bin, bounds=(0.0, 1.0), strict=(0.0, 1.0))
        self._numpy_generator = generator or resolve_seed(seed=seed)  # type: ignore[arg-type]

    def _do(self, problem: Problem, genomes: np.ndarray, **kwargs) -> np.ndarray:
        _, n_matings, _ = genomes.shape

        # get the parameters required by SBX
        eta, prob_var, prob_exch, prob_bin = get(self.eta, self.prob_var, self.prob_exch, self.prob_bin, size=(n_matings, 1))

        # set the binomial probability to zero if no exchange between individuals shall happen
        rand = self._numpy_generator.random((len(prob_bin), 1))
        prob_bin[rand > prob_exch] = 0.0

        offspring = cross_sbx(genomes.astype(float), problem.xl, problem.xu, eta, prob_var, prob_bin)

        if self.n_offsprings == 1:
            rand = self._numpy_generator.random(size=n_matings) < 0.5  # type: ignore[assignment] # noqa: PLR2004
            offspring[0, rand] = offspring[1, rand]
            offspring = offspring[[0]]

        return offspring
