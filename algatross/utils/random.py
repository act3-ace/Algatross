"""Utilities for working with randomness."""

import operator
import os

from collections.abc import Callable

import numpy as np

from ray.rllib.policy.sample_batch import SampleBatch

import torch

import tree  # pip install dm_tree

from algatross.utils.types import NumpyRandomSeed


def seed_global(seed: int | NumpyRandomSeed):
    """Seed global state for environments that need it.

    Parameters
    ----------
    seed : int | NumpyRandomSeed
        The global seed to set, int at most 32-bit.
    """
    # 32-bit max
    # np.random.seed(seed)  # noqa: ERA001
    # random.seed(seed)  # noqa: ERA001
    # os.environ["PYTHONHASHSEED"] = str(seed)  # noqa: ERA001
    # See https://github.com/pytorch/pytorch/issues/47672.
    torch.cuda.manual_seed(seed)  # type: ignore[arg-type]
    torch.cuda.manual_seed_all(seed)  # type: ignore[arg-type]
    torch.backends.cudnn.deterministic = True
    cuda_version = torch.version.cuda
    if cuda_version is not None and float(torch.version.cuda) >= 10.2:  # noqa: PLR2004
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = "4096:8"
    else:
        # Not all Operations support this.
        torch.use_deterministic_algorithms(True)
        # This is only for Convolution no problem.


class SerializableTorchGenerator(torch.Generator):  # noqa: D101
    def __getstate__(self):  # noqa: D105
        return self.get_state()

    def __setstate__(self, state):  # noqa: D105
        self.set_state(state)


def get_torch_generator_from_numpy(numpy_generator: np.random.Generator) -> tuple[torch.Generator, int]:
    """Get a torch generator from the state of a numpy random number generator.

    Parameters
    ----------
    numpy_generator : np.random.Generator
        The :class:`~np.random.Generator` to use for seeding the :class:`~torch.Generator`.

    Returns
    -------
    torch.Generator
        The torch generator spawned from the same seed as ``numpy_generator``.
    int
        The value used to seed the generator
    """
    # make torch and numpy share a seed
    torch_seed = hash(int(numpy_generator.bit_generator._seed_seq.generate_state(1)[0].item()))  # type: ignore[attr-defined] # noqa: SLF001
    torch_seed &= 0xFFFF_FFFF_FFFF_FFFF

    generator = SerializableTorchGenerator()
    # generator = torch.Generator()  # noqa: ERA001
    generator.manual_seed(torch_seed)

    return generator, torch_seed


def get_generator_entropy(generator: np.random.Generator) -> int:
    """Get entropy of an rng.

    Parameters
    ----------
    generator : np.random.Generator
        Generator from process.

    Returns
    -------
    int
        The generator's entropy.
    """
    # TODO: fix this for earlier versions of numpy
    return generator.bit_generator.seed_seq.entropy  # type: ignore[attr-defined]


def get_generator_integer(generator: np.random.Generator) -> np.ndarray:
    """Query the generator for a random integer.

    Parameters
    ----------
    generator : np.random.Generator
        Generator from process.

    Returns
    -------
    np.ndarray
        Output integer.
    """
    return generator.integers(0, 2e8, dtype=int)  # type: ignore[call-overload]


def resolve_seed(seed: int | list | np.random.Generator, logfn: Callable | None = None) -> np.random.Generator:
    """Central seed resolution interface.

    Give a new generator (when type(seed) is int) otherwise return the same generator.
    This is the default behavior of np.random.default_rng.
    The bulk of this function is logging and this docstring :).

    Parameters
    ----------
        seed : int | list | np.random.Generator
            The random seed to use, int or list[int] each at most 128-bit, or spawned generator.
        logfn : Callable, optional
            Logger function. Defaults to None.

    Returns
    -------
        np.random.Generator
            Output RNG.
    """
    if isinstance(seed, np.random.Generator):
        if logfn is not None:
            logfn(f"pid={os.getpid()}: I was given a numpy generator with entropy={get_generator_entropy(seed)}")
    elif isinstance(seed, int) and logfn is not None:
        logfn(f"pid={os.getpid()}: I was given an integer: {seed}")

    gen = np.random.default_rng(seed)

    # # forward compat with older numpy
    # if not hasattr(gen, "spawn"):
    #     def spawner(num_spawns: int):
    #         return gen.bit_generator._seed_seq.spawn(num_spawns)  # noqa: ERA001

    #     gen.spawn = spawner.__get__(gen)  # noqa: ERA001

    return gen  # noqa: RET504


def egocentric_shuffle(batch: SampleBatch, agent_random_state: np.random.Generator) -> SampleBatch:
    """Shuffle the rows of a batch.

    SampleBatch.shuffle but use agent's local np generator instead of a global runtime generator.

    Parameters
    ----------
    batch : SampleBatch
        The batch to shuffle.
    agent_random_state : np.random.Generator
        The random state of the agent.

    Returns
    -------
    SampleBatch
        The shuffled batch.

    Raises
    ------
    ValueError
        If the batch cannot be shuffled due to the agent being stateful (rnn).
    """
    # Shuffling the data when we have `seq_lens` defined is probably
    # a bad idea!
    if batch.get(SampleBatch.SEQ_LENS) is not None:
        msg = "SampleBatch.shuffle not possible when your data has `seq_lens` defined!"
        raise ValueError(msg)

    # Get a permutation over the single items once and use the same
    # permutation for all the data (otherwise, data would become
    # meaningless).
    permutation = agent_random_state.permutation(batch.count)

    batch_as_dict = dict(batch.items())
    shuffled = tree.map_structure(operator.itemgetter(permutation), batch_as_dict)
    batch.update(shuffled)
    # Flush cache such that intercepted values are recalculated after the
    # shuffling.
    batch.intercepted_values = {}
    return batch


def seed_action_spaces(env, seed, log_fn=print):
    """
    Seed the action space.

    Parameters
    ----------
    env : Any
        The environment to seed.
    seed : int | None
        The random seed to set.
    log_fn : Callable
        The function to call to log the results, default is :func:`print`.
    """
    if hasattr(env, "agents"):
        log_fn("Seeding the action spaces")
        for i, agent in enumerate(env.agents):
            env.action_space(agent).seed(seed + i)


def seed_observation_spaces(env, seed, log_fn=print):
    """
    Seed the observation space.

    Parameters
    ----------
    env : Any
        The environment to seed.
    seed : int | None
        The random seed to set.
    log_fn : Callable
        The function to call to log the results, default is :func:`print`.
    """
    if hasattr(env, "agents"):
        log_fn("Seeding the observation spaces")
        for i, agent in enumerate(env.agents):
            env.observation_space(agent).seed(seed + i)
