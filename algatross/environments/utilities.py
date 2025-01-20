"""A module containing clean-rl style PPO agents, rollout, and training operations."""

import random

from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from hashlib import blake2b
from typing import Any

import numpy as np

from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch

import torch

from gymnasium.spaces import Box
from scipy.signal import lfilter

from algatross.utils.types import AgentID

try:
    from ray.rllib.core import Columns  # type: ignore[attr-defined, unused-ignore]
except ImportError:
    from algatross.utils.compatibility import Columns  # type: ignore[assignment, unused-ignore]


def calc_rewards(trajectory: SampleBatch, reward_metrics: Sequence[str], reward_metric_gains: Sequence[float]) -> np.ndarray:
    r"""
    Calculate the rewards to use as learning targets given a sample batch.

    Calculate the sum of the ``reward_metrics`` scaled by ``reward_metric_gains`` for each sample in ``trajectory``:

    .. math::

        \sum_{\tau}\sum_{r_{\tau}}\alpha_{r}\cdot r_{\tau}

    Parameters
    ----------
    trajectory : SampleBatch
        The sample batch containing the trajectory
    reward_metrics : Sequence[str]
        The reward metrics to find in the ``trajectory``
    reward_metric_gains : Sequence[float]
        The gain by which each metric should be scaled

    Returns
    -------
    np.ndarray
        The learning targets
    """
    return np.stack([trajectory[rew] * gain for rew, gain in zip(reward_metrics, reward_metric_gains, strict=True)]).sum(axis=0)


def episode_hash(episode_num: int) -> int:
    """Generate a new hash for the episode.

    Parameters
    ----------
    episode_num : int
        The episode number

    Returns
    -------
    int
        The hash of the episode based on the episode number and current datetime
    """
    base = f"{random.randrange(int(1e18))}{episode_num}".encode()  # noqa: S311
    salt = datetime.now(timezone.utc).strftime("%m%d%H%M%S%f").encode()
    digest = blake2b(base, digest_size=8, salt=salt, usedforsecurity=False).hexdigest()
    return int(digest, base=16)


def compute_advantage(
    rewards: np.ndarray,
    value_predictions: np.ndarray,
    last_r: np.ndarray,
    gae_lambda: float = 1.0,
    gamma: float = 0.99,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Compute the advantages for samples in a batch.

    Parameters
    ----------
    rewards : np.ndarray
        The episode rewards
    value_predictions : np.ndarray
        The critic network value predictions
    last_r : np.ndarray
        The final reward of the episodes
    gae_lambda : float, optional
        Value to use for :math:`\lambda` in GAE, by default 1.0
    gamma : float, optional
        Discount factor, by default 0.99

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The advantage and value estimates from GAE
    """
    v_pred_t = np.concatenate([value_predictions, last_r])
    delta_t = (rewards + gamma * v_pred_t[1:] - v_pred_t[:-1]).astype(np.float64)
    adv = discount_cumsum(delta_t, gae_lambda * gamma)
    val = (value_predictions + adv).astype(np.float32)
    return adv.astype(np.float32), val


def discount_cumsum(x: np.ndarray, gamma: float) -> np.ndarray:
    """Compute the discounted cumulative sum of rewards.

    Parameters
    ----------
    x : np.ndarray
        The input array to accumulate
    gamma : float
        The discout factor

    Returns
    -------
    np.ndarray
        The cumulative sum for each value in x discounted by ``gamma``
    """
    return lfilter([1], [1, float(-gamma)], x[::-1], axis=0)[::-1]


def get_team_fitness(team: Iterable[AgentID], sample_batch: MultiAgentBatch, fitness_metric: str = Columns.REWARDS) -> np.ndarray:
    """Get the fitness value for the team.

    Parameters
    ----------
    team : Iterable[AgentID]
        The sequence of agent ids to evaluate
    sample_batch : MultiAgentBatch
        The multi-agent sample batch
    fitness_metric : str, optional
        The fitness metric to collect, by default :attr:`~ray.rllib.core.Columns.REWARDS`

    Returns
    -------
    np.ndarray
        The fitness of the team of agents
    """
    # array of [B, T, M]
    fitness = np.stack([sample_batch[str(tm)][fitness_metric] for tm in team], axis=1)
    return fitness.sum(axis=0).mean(axis=0).flatten()


def explained_var(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Explained variance.

    Parameters
    ----------
    y_pred : torch.Tensor
        The predicted output variable
    y_true : torch.Tensor
        The true output variable

    Returns
    -------
    torch.Tensor
        The explained variance of the predictions
    """
    var_y = y_true.var()
    return torch.tensor(torch.nan, dtype=y_true.dtype, device=y_true.device) if var_y == 0 else 1 - (y_true - y_pred).var() / var_y


def is_continuous_env(env: Any) -> bool:  # noqa: ANN401
    """
    Return whether or not the environment is continuous.

    Parameters
    ----------
    env : Any
        The environment to check

    Returns
    -------
    bool
        Whether or not the environment is continuous
    """
    return bool(all(isinstance(env.action_space(agent), Box) for agent in getattr(env, "possible_agents", [])))
