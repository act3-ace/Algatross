"""On policy HARL runners."""

import copy

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import timedelta
from itertools import cycle
from operator import itemgetter
from time import time
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

import torch

from harl.envs.env_wrappers import ShareDummyVecEnv, ShareSubprocVecEnv
from harl.utils.envs_tools import set_seed
from harl.utils.models_tools import init_device
from harl.utils.trans_tools import _t2n  # noqa: PLC2701
from tree import map_structure, map_structure_up_to

from algatross.agents.torch_base import TorchBaseMARLAgent
from algatross.configs.harl.runners import HARLRunnerConfig
from algatross.environments.harl.env_tools import make_env
from algatross.environments.runners import BaseRunner
from algatross.utils.merge_dicts import merge_dicts
from algatross.utils.types import AgentID, ConstructorData, NumpyRandomSeed, PlatformID

if TYPE_CHECKING:
    from ray.rllib import SampleBatch

try:
    from ray.rllib.core import Columns  # type: ignore[attr-defined, unused-ignore]
except ImportError:
    from algatross.utils.compatibility import Columns  # type: ignore[assignment, unused-ignore]


class OnPolicyBaseRunner(BaseRunner):
    """
    Runner for a MARL/HARL On-Policy algorithm.

    Parameters
    ----------
    agent_constructors : Mapping[AgentID, ConstructorData]
        Mapping from agent IDs to their constructors.
    platform_map : Mapping[PlatformID, Sequence[AgentID]]
        Mapping from platform IDs to the agents which manage them
    env_name : str
        The name of the environment in the global registry.
    seed : NumpyRandomSeed | None, optional
        The seed to use for randomness, default is :data:`python:None`.
    n_envs : int, optional
        The number of environments to clone on this runner, default is 1.
    runner_config : HARLRunnerConfig | dict | None, optional
        The configuration to pass to the runner, default is :data:`python:None`
    train_config : dict[str, Any] | None, optional
        The base configuration for training, default is :data:`python:None`.
    rollout_config : dict[str, Any] | None = None
        The base configuration for rollouts, default is :data:`python:None`.
    `**kwargs`
        Additional keyword arguments.
    """

    env: ShareDummyVecEnv | ShareSubprocVecEnv
    """The environment in this runner."""
    platform_map: dict[AgentID, str]
    """The map from agent IDs to platforms they control."""
    default_config: type[HARLRunnerConfig] = HARLRunnerConfig
    """The default configuration, if one is not provided."""
    num_agents: int
    """The number of agents in this runners environment."""
    agents: dict[AgentID, TorchBaseMARLAgent]
    """The agents in this runners environment."""
    agent_slice_map: dict[AgentID, list[int]]
    """The mapping from agents to the index in the observation space where this agents observations begin."""
    agent_map: dict[PlatformID, dict[Literal["agent_id", "agent"], AgentID | TorchBaseMARLAgent]]
    """The mapping from platform ID to agents and their ids."""
    seed: int | None = None
    """The seed used for randomness."""
    device: torch.device
    """The device used by this runner."""

    def __init__(
        self,
        agent_constructors: Mapping[AgentID, ConstructorData],
        platform_map: Mapping[PlatformID, Sequence[AgentID]],
        env_name: str,
        seed: NumpyRandomSeed | None = None,
        n_envs: int = 1,
        runner_config: HARLRunnerConfig | dict | None = None,
        train_config: dict[str, Any] | None = None,
        rollout_config: dict[str, Any] | None = None,
        **kwargs,
    ):
        train_config = train_config or {}
        rollout_config = rollout_config or {}
        runner_config = copy.deepcopy(runner_config or {})
        runner_config = runner_config if isinstance(runner_config, dict) else asdict(runner_config)
        runner_config = merge_dicts(asdict(self.default_config()), runner_config)
        runner_config = merge_dicts(runner_config, kwargs)
        runner_config = merge_dicts(runner_config, train_config)
        runner_config = merge_dicts(runner_config, rollout_config)
        runner_config.setdefault("n_rollout_threads", n_envs)
        runner_config["seed"] = seed

        self.config = runner_config
        self._train_config = train_config
        self._rollout_config = rollout_config
        self._n_envs = runner_config["n_rollout_threads"]

        self.seed = seed if isinstance(seed, int) else int(self._numpy_generator.bit_generator.seed_seq.generate_state(1).item())

        self.device = init_device({
            "cuda": ("cuda" in self.config.get("device", "cpu")) or self.config["cuda"],
            "torch_threads": self.config["torch_threads"],
            "cuda_deterministic": self.config["cuda_deterministic"],
        })
        set_seed({"seed_specify": seed is not None, "seed": seed})

        self.env, self.platforms = make_env(env_name, seed, self.n_envs, self.config)  # type: ignore[arg-type]
        self.num_agents = self.env.n_agents

        # agent: algorithm that controls a platform
        # platform: a controllable actor in an environment
        # platform_map: a map from agent to the list of platforms it controls
        self.agents: dict[AgentID, TorchBaseMARLAgent] = {}
        self.platform_map: dict[AgentID, Sequence[PlatformID]] = {}
        self.agent_slice_map: dict[AgentID, list[int]] = {}
        self.agent_map: dict[PlatformID, dict[Literal["agent_id", "agent"], AgentID | TorchBaseMARLAgent]] = {}
        platform_map = platform_map or defaultdict(list)
        obs_spaces, act_spaces, share_obs_spaces = {}, {}, {}
        platform_indices = {}
        for platform_idx, platform in enumerate(self.platforms):
            obs_spaces[platform] = self.env.observation_space[platform_idx]
            act_spaces[platform] = self.env.action_space[platform_idx]
            share_obs_spaces[platform] = self.env.share_observation_space[platform_idx]
            platform_indices[platform] = platform_idx

        for agent_id, constructor in agent_constructors.items():
            self.platform_map[agent_id] = [platform for platform in platform_map[agent_id] if platform in self.platforms]  # type: ignore[assignment]
            self.agents[agent_id] = constructor.construct(
                platforms=self.platform_map[agent_id],
                obs_spaces=obs_spaces,
                act_spaces=act_spaces,
                shared_obs_space=share_obs_spaces[self.platform_map[agent_id][0]],
                runner_config=self.config,
            )
            # for agent_id, platforms in platform_map or {}:
            for platform in self.platform_map[agent_id]:
                self.agent_map[platform] = {"agent_id": agent_id, "agent": self.agents[agent_id]}

            self.agent_slice_map[agent_id] = [platform_indices[platform] for platform in self.platform_map[agent_id]]

    @property
    def n_envs(self) -> int:
        """
        The number of environments in this runner.

        Returns
        -------
        int
            The number of environments in this runner.
        """
        return self._n_envs

    def run(  # noqa: PLR0914, PLR0915
        self,
        trainable_agents: Sequence[PlatformID],
        max_episodes: int | None = None,
        batch_size: int = 3000,
        rollout_length: int = 300,
        reward_metrics: dict[AgentID, Sequence[str]] | None = None,
        reward_metric_gains: dict[AgentID, Sequence[float]] | None = None,
        training: bool = False,
        **kwargs,
    ) -> dict:
        """
        Run the training (or rendering) pipeline.

        Parameters
        ----------
        trainable_agents : Sequence[PlatformID]
            Sequence of trainable platforms
        max_episodes : int | None, optional
            Maximum number of episode to gather, :data:`python:None`
        batch_size : int, optional
            The batch size for training, by default 3000
        rollout_length : int, optional
            The length of each rollout, by default 300
        reward_metrics : dict[AgentID, Sequence[str]] | None, optional
            The reward metrics for each agent, :data:`python:None`
        reward_metric_gains : dict[AgentID, Sequence[float]] | None, optional
            The weight for each reward for each agent, :data:`python:None`
        training : bool, optional
            Whether the run is training or just evaluating, :data:`python:False`
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        dict
            Rollout or training info
        """
        self.clear_buffer()
        reward_metrics = {agent_id: [Columns.REWARDS] for agent_id in trainable_agents} if reward_metrics is None else reward_metrics
        reward_metrics = {agent_id: [Columns.REWARDS] if rew is None else rew for agent_id, rew in reward_metrics.items()}
        reward_metric_gains = (
            {agent_id: [1.0] * len(metric) for agent_id, metric in reward_metrics.items()}
            if reward_metric_gains is None
            else reward_metric_gains
        )
        reward_metric_gains = {
            agent_id: [1.0] * len(metric) if reward_metric_gains[agent_id] is None else reward_metric_gains[agent_id]
            for agent_id, metric in reward_metrics.items()
        }

        self.warmup()

        episodes = int(np.ceil(batch_size / (rollout_length * self._n_envs)))
        if max_episodes:
            episodes = min(episodes, max_episodes)

        episode_data: dict[PlatformID, SampleBatch] = {}
        training_data: dict[PlatformID, dict] = {}
        shallow: dict[PlatformID, dict] = {}

        t0 = time()
        times = [0.0]
        for episode in range(1, episodes + 1):
            # linear decay of learning rate
            for agent in self.agents.values():
                agent.lr_decay(episode, episodes)

            self.prep_rollout()  # change to eval mode
            for step in range(self.config["episode_length"]):
                # Sample actions from actors and values from critics
                (all_actions, agent_values, agent_actions, agent_action_log_probs, agent_rnn_states, agent_rnn_states_critic) = (
                    self.collect(step=step, trainable_agents=trainable_agents, deterministic=not training)
                )
                # actions: (n_threads, n_agents, action_dim)  # noqa: ERA001
                (obs, share_obs, rewards, dones, infos, available_actions) = self.env.step(
                    zip(all_actions, cycle([trainable_agents]), cycle([reward_metrics]), cycle([reward_metric_gains])),
                )
                # obs: (n_threads, n_agents, obs_dim)  # noqa: ERA001
                # share_obs: (n_threads, n_agents, share_obs_dim)  # noqa: ERA001
                # rewards: (n_threads, n_agents, 1)  # noqa: ERA001
                # dones: (n_threads, n_agents)  # noqa: ERA001
                # infos: (n_threads)  # noqa: ERA001
                # available_actions: (n_threads, ) of None or (n_threads, n_agents, action_number)
                agent_obs = {}
                agent_share_obs = {}
                agent_rewards = {agent_id: rewards.copy() for agent_id in self.agents}
                agent_dones = {}
                agent_infos = {}
                agent_available_actions = {}
                for platform_id in trainable_agents:
                    agent_id = self.agent_map[platform_id]["agent_id"]
                    indices = self.agent_slice_map[agent_id]  # type: ignore[index]
                    agent_obs[agent_id] = obs[:, indices]
                    agent_share_obs[agent_id] = share_obs[:, indices]
                    agent_dones[agent_id] = dones[:, indices]
                    agent_infos[agent_id] = np.array([itemgetter(*indices)(info) for info in infos])
                    agent_available_actions[agent_id] = available_actions if available_actions.ndim == 1 else available_actions[:, indices]

                data = (
                    agent_obs,
                    agent_share_obs,
                    agent_rewards,
                    agent_dones,
                    agent_infos,
                    agent_available_actions,
                    agent_values,
                    agent_actions,
                    agent_action_log_probs,
                    agent_rnn_states,
                    agent_rnn_states_critic,
                )

                # insert data into buffer
                self.insert(data, trainable_agents=trainable_agents, **kwargs)

            # compute return and update network
            self.compute()

            # keep the last "episode" of data
            for agent in self.agents.values():
                if hasattr(agent, "buffers_to_sample_batch"):
                    episode_data.update(agent.buffers_to_sample_batch())

            if training:
                self.prep_training()  # change to train mode
                for platform_id, platform_data in self.train().items():
                    if training_data.get(platform_id) is None:
                        training_data[platform_id] = map_structure(
                            lambda x: (
                                {"max": x.detach().max().item(), "min": x.detach().min().item(), "mean": x.detach().mean().item()}
                                if isinstance(x, torch.Tensor)
                                else (
                                    {"max": x.max().item(), "min": x.min().item(), "mean": x.mean().item()}
                                    if isinstance(x, np.ndarray)
                                    else {"max": x, "min": x, "mean": x}
                                )
                            ),
                            platform_data,
                        )
                        shallow[platform_id] = platform_data
                        continue
                    training_data[platform_id] = map_structure_up_to(
                        shallow[platform_id],
                        lambda x, y: (
                            {
                                "max": max([x["max"], y.detach().max().item()]),
                                "min": min([x["min"], y.detach().min().item()]),
                                "mean": sum([x["mean"], y.detach().mean().item()]),
                            }
                            if isinstance(y, torch.Tensor)
                            else (
                                {
                                    "max": max([x["max"], y.max().item()]),
                                    "min": min([x["min"], y.min().item()]),
                                    "mean": x["mean"] + y.mean().item(),
                                }
                                if isinstance(y, np.ndarray)
                                else {"max": max([x["max"], y]), "min": min([x["min"], y]), "mean": x["mean"] + y}
                            )
                        ),
                        training_data[platform_id],
                        platform_data,
                    )
            self.after_update()
            times.append(time() - t0)
            diff = np.diff(times).mean()
            total_time = episodes * diff
            remaining_time = (episodes - episode) * diff
            msg = f"{'Training' if training else 'Evaluation'} iteration {episode} of {episodes} ({100 * episode / episodes:.1f}%)\n"
            msg += f"\tElapsed time:\t{timedelta(seconds=times[-1])}\n"
            msg += f"\tAverage time:\t{timedelta(seconds=diff)}\n"
            msg += f"\tRemaining: \t{timedelta(seconds=remaining_time)} ({100 * times[-1] / total_time:.1f}%)\n"
            print(msg)
        diff = np.diff(times).mean()
        msg = "=" * 50 + "\n"
        msg += f"\tTotal time: {timedelta(seconds=time() - t0)}\n"
        msg += f"\tAverage: {timedelta(seconds=diff)}\n"
        print(msg)
        if training:
            for platform_id, platform_data in training_data.items():
                training_data[platform_id] = map_structure_up_to(
                    shallow[platform_id],
                    lambda x: {"max": x["max"], "min": x["min"], "mean": x["mean"] / episodes},
                    platform_data,
                )
        return {
            platform_id: {
                "training_stats": training_data.get(platform_id, {}),
                "rollout_stats": {},
                "extra_info": {"rollout_buffer": platform_rollout},
            }
            for platform_id, platform_rollout in episode_data.items()
        }

    def warmup(self):
        """Warm up the replay buffer."""
        # reset env
        obs, share_obs, infos, available_actions = self.env.reset()
        # replay buffer
        for agent_id, indices in self.agent_slice_map.items():
            if hasattr(self.agents[agent_id], "warmup"):
                self.agents[agent_id].warmup(
                    obs=obs[:, indices],
                    infos=np.array([itemgetter(*indices)(info) for info in infos]),
                    share_obs=share_obs[:, indices],
                    available_actions=available_actions if available_actions.ndim == 1 else available_actions[:, indices],
                )

    @torch.no_grad()
    def collect(
        self,
        step: int,
        trainable_agents: Sequence[AgentID],
        **kwargs,
    ) -> tuple[
        np.ndarray,
        dict[AgentID, np.ndarray],
        dict[AgentID, np.ndarray],
        dict[AgentID, np.ndarray],
        dict[AgentID, np.ndarray],
        dict[AgentID, np.ndarray],
    ]:
        """Collect actions and values from actors and critics.

        Parameters
        ----------
        step : int
            The step in the episode.
        trainable_agents : Sequence[AgentID]
            The trainable agents in the environment.
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
            all_actions: np.ndarray
                Actions from all platforms concatenated in the correct order
            values: dict[AgentID, np.ndarray]
                Dictionary of value predictions of each platform controlled by the agent
            actions: dict[AgentID, np.ndarray]
                Dictionary of actions of each platform controlled by the agent
            action_log_probs: dict[AgentID, np.ndarray]
                Dictionary of action log-probabilities of each platform controlled by the agent
            rnn_states: dict[AgentID, np.ndarray]
                Dictionary of actor RNN states of each platform controlled by the agent
            rnn_states_critic: dict[AgentID, np.ndarray]
                Dictionary of critic RNN states of each platform controlled by the agent
        """
        # collect actions, action_log_probs, rnn_states from n actors
        all_actions_list = []
        action_collector = defaultdict(list)
        action_log_prob_collector = defaultdict(list)
        values: dict[AgentID, np.ndarray] = {}
        rnn_state_collector = defaultdict(list)
        rnn_states_critic: dict[AgentID, np.ndarray] = {}
        agent_iters = {agent_id: agent.get_actions_and_values(step=step, **kwargs) for agent_id, agent in self.agents.items()}
        critic_logged = dict.fromkeys(self.agents, False)
        for platform_id in self.platforms:
            agent_id = self.agent_map[platform_id]["agent_id"]
            value, action, action_log_prob, rnn_state, rnn_state_critic = next(agent_iters[agent_id])  # type: ignore[index]
            all_actions_list.append(action)
            if platform_id in trainable_agents:
                action_collector[agent_id].append(_t2n(action))
                action_log_prob_collector[agent_id].append(_t2n(action_log_prob))
                rnn_state_collector[agent_id].append(_t2n(rnn_state))
                if not critic_logged[agent_id]:  # type: ignore[index]
                    values[agent_id] = value  # type: ignore[index]
                    rnn_states_critic[agent_id] = rnn_state_critic  # type: ignore[index]
        all_actions = np.stack(all_actions_list, axis=1)
        actions: dict[AgentID, np.ndarray] = {agent_id: np.stack(act, axis=1) for agent_id, act in action_collector.items()}  # type: ignore[misc]
        action_log_probs: dict[AgentID, np.ndarray] = {
            agent_id: np.stack(alp, axis=1)  # type: ignore[misc]
            for agent_id, alp in action_log_prob_collector.items()
        }
        rnn_states: dict[AgentID, np.ndarray] = {agent_id: np.stack(state, axis=1) for agent_id, state in rnn_state_collector.items()}  # type: ignore[misc]

        return all_actions, values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data: Sequence[dict], trainable_agents: Sequence[PlatformID], **kwargs):
        """
        Insert data into buffer.

        Parameters
        ----------
        data : Sequence[dict]
            The data to insert into the buffer.
        trainable_agents : Sequence[PlatformID]
            The trainable agents in the environment
        `**kwargs`
            Additional keyword arguments
        """
        agents = {self.agent_map[platform_id]["agent_id"] for platform_id in trainable_agents}
        for agent_id in agents:
            agent = self.agents[agent_id]  # type: ignore[index]
            if hasattr(agent, "insert"):
                agent.insert(tuple(d[agent_id] for d in data))

    @torch.no_grad()
    def compute(self):
        """Compute returns and advantages.

        Compute critic evaluation of the last state,
        and then let buffer compute returns, which will be used during training.
        """
        for agent in self.agents.values():
            if hasattr(agent, "compute"):
                agent.compute()

    def train(self):
        """Train the model."""
        raise NotImplementedError

    def after_update(self):
        """Do the necessary data operations after an update.

        After an update, copy the data at the last step to the first position of the buffer.
        This will be used for then generating new actions.
        """
        for agent in self.agents.values():
            if hasattr(agent, "after_update"):
                agent.after_update()

    def clear_buffer(self):
        """Clear data from previous runs from the buffer."""
        for agent in self.agents.values():
            if hasattr(agent, "clear_buffer"):
                agent.clear_buffer()

    def prep_rollout(self):
        """Prepare for rollout."""
        """Prepare for training."""
        for agent in self.agents.values():
            agent.prep_rollout()

    def prep_training(self):
        """Prepare for training."""
        for agent in self.agents.values():
            agent.prep_training()

    def close(self):
        """Close environment, writter, and logger."""
        self.env.close()

    def train_island(  # type: ignore[override]
        self,
        trainable_agents: Sequence[AgentID],
        reward_metrics: dict[AgentID, Sequence[str]] | None = None,
        reward_metric_gains: dict[AgentID, Sequence[float]] | None = None,
        rollout_config: dict | None = None,
        train_config: dict | None = None,
        **kwargs,
    ) -> dict[AgentID, dict[str, Any]]:
        """
        Gather a rollout from the island with training.

        Parameters
        ----------
        trainable_agents : Sequence[AgentID]
            Sequence of trainable agent ids
        reward_metrics : dict[AgentID, Sequence[str]] | None, optional
            Reward metrics for each agent, :data:`python:None`
        reward_metric_gains : dict[AgentID, Sequence[float]] | None, optional
            Weights for each reward metrics for each agent, :data:`python:None`
        rollout_config : dict | None, optional
            The configuration to use when gathering rollouts, :data:`python:None`
        train_config : dict | None, optional
            The configuration to use when training, :data:`python:None`
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        dict[AgentID, dict[str, Any]]
            The training results for the agents
        """
        rollout_config = rollout_config or {}
        train_config = train_config or {}
        run_kwargs = {**kwargs}
        run_kwargs |= rollout_config
        run_kwargs |= train_config
        run_kwargs["training"] = True

        return self.run(
            trainable_agents=trainable_agents,
            reward_metrics=reward_metrics,
            reward_metric_gains=reward_metric_gains,
            **run_kwargs,
        )

    def rollout_island(  # type: ignore[override]
        self,
        trainable_agents: Sequence[AgentID],
        reward_metrics: dict[AgentID, Sequence[str]] | None = None,
        reward_metric_gains: dict[AgentID, Sequence[float]] | None = None,
        rollout_config: dict | None = None,
        train_config: dict | None = None,
        **kwargs,
    ) -> dict[AgentID, dict[str, Any]]:
        """Gather a rollout from the island with training.

        Parameters
        ----------
        trainable_agents : Sequence[AgentID]
            Sequence of trainable agent ids
        reward_metrics : dict[AgentID, Sequence[str]] | None, optional
            Reward metrics for each agent, :data:`python:None`
        reward_metric_gains : dict[AgentID, Sequence[float]] | None, optional
            Weights for each reward metrics for each agent, :data:`python:None`
        rollout_config : dict | None, optional
            The configuration to use when gathering rollouts, :data:`python:None`
        train_config : dict | None, optional
            The configuration to use when training, :data:`python:None`
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        dict[AgentID, dict[str, Any]]
            The rollout results
        """
        rollout_config = merge_dicts(self._rollout_config, rollout_config or {})
        train_config = merge_dicts(self._train_config, train_config or {})
        run_kwargs = {**kwargs}
        run_kwargs |= rollout_config
        run_kwargs |= train_config
        run_kwargs["training"] = False

        with torch.no_grad():
            return self.run(
                trainable_agents=trainable_agents,
                reward_metrics=reward_metrics,
                reward_metric_gains=reward_metric_gains,
                **run_kwargs,
            )
