"""Classes of environment runners."""

import functools
import logging

from collections import defaultdict
from collections.abc import Callable, Sequence
from itertools import chain
from typing import Any, Literal

import numpy as np

import ray
import ray.remote_function

from ray.rllib import MultiAgentEnv, SampleBatch
from ray.rllib.env.multi_agent_env import MultiAgentEnvWrapper
from ray.rllib.utils.torch_utils import convert_to_torch_tensor

import torch

import PIL
import PIL.Image

from gymnasium.spaces import Dict
from pettingzoo.utils.env import AECEnv, ParallelEnv
from rich.console import Console
from rich.table import Table

from algatross.agents.base import BaseAgent
from algatross.environments.ma_inspection.inspection.rendered_env import RenderedParallelPettingZooInspection
from algatross.environments.utilities import calc_rewards, episode_hash
from algatross.utils.debugging import log_agent_params
from algatross.utils.io import total_size
from algatross.utils.random import (
    egocentric_shuffle,
    get_torch_generator_from_numpy,
    resolve_seed,
    seed_action_spaces,
    seed_observation_spaces,
)
from algatross.utils.sample_batch_builder import SampleBatchBuilder, concat_agent_buffers
from algatross.utils.types import AgentID, NumpyRandomSeed

try:
    from ray.rllib.core import Columns  # type: ignore[attr-defined, unused-ignore]
except ImportError:
    from algatross.utils.compatibility import Columns  # type: ignore[assignment, unused-ignore]

try:
    import jax
except ImportError:
    jax = None


class ManagedEnvironmentsContext:
    """
    A context manager which handles automatic setup and teardown of environments.

    Parameters
    ----------
    env_list : list[Callable | AECEnv | ParallelEnv]
        The list of environments to manage in this context.
    """

    was_callable: bool = False
    """Whether or not the environments were :class:`Callable`."""
    logger: Any = logging.getLogger("ray")
    """The logger for this context manager."""

    def __init__(self, env_list: list[Callable | AECEnv | ParallelEnv]):
        self.raw_envs = env_list

    @property
    def mec_envs(self) -> list:
        """
        The environments managed by this context.

        Returns
        -------
        list
            The environments managed by this context.
        """
        return self._mec_envs

    @mec_envs.setter
    def mec_envs(self, envs):
        self._mec_envs = envs

    @mec_envs.deleter
    def mec_envs(self):
        for env in self._mec_envs:
            env: AECEnv | ParallelEnv
            env.close()

        del self._mec_envs

    def __enter__(self):  # noqa: D105
        if jax is not None:
            jax.config.update("jax_enable_compilation_cache", False)
        mec_envs = []
        for env in self.raw_envs:
            if isinstance(env, Callable):
                mec_envs.append(env())
                self.was_callable = True
            else:
                mec_envs.append(env)

        self.mec_envs = mec_envs
        return self.mec_envs

    def __exit__(self, exc_type, exc_val, exc_tb):  # noqa: D105
        if self.was_callable:
            self.logger.debug(f"Flag {len(self.mec_envs)} environments for gc")
            del self.mec_envs

        if jax is not None:
            self.logger.debug("Clearing jax cache and backends!")
            jax.clear_caches()
            jax.clear_backends()


class BaseRunner:
    """
    A runner class for gathering rollouts and training agents.

    Parameters
    ----------
    n_envs : int, optional
        The number of environments to run in parallel on this runner, default is 1.
    train_config : dict[str, Any] | None, optional
        The base configuration to pass to the training function, default is :data:`python:None`.
    rollout_config : dict[str, Any] | None, optional
        The base configuration to pass to the training function, default is :data:`python:None`.
    seed : np.random.Generator | NumpyRandomSeed | None, optional
        The seed for randomness.
    `**kwargs`
        Additional keyword arguments
    """

    rollout_function: Callable
    """The function to use to gather rollout data."""
    seed: int | None = None
    """The seed for randomness."""
    tape_index: int
    """The tape index for randomess repeatability."""
    logger: logging.Logger
    """The logger for the runner."""

    def __init__(
        self,
        n_envs: int = 1,
        train_config: dict[str, Any] | None = None,
        rollout_config: dict[str, Any] | None = None,
        seed: np.random.Generator | NumpyRandomSeed | None = None,
        **kwargs,
    ):
        self.logger = logging.getLogger("ray")
        if seed is None:
            msg = f"No seed given to {self.__class__.__name__}"
            raise RuntimeError(msg)
            # seed = np.random.randint(int(2e16))  # noqa: ERA001

        self.logger.debug("Runner seed debug:")
        self._numpy_generator = resolve_seed(seed)  # type: ignore[arg-type]
        _, self.seed = get_torch_generator_from_numpy(self._numpy_generator)

        # self.seed = (
        #     seed if isinstance(seed, int)
        #     else int(self._numpy_generator.bit_generator.seed_seq.generate_state(1).item())
        # )  # noqa: RUF100, ERA001
        self.logger.debug(f"Runner got integer seed: {self.seed}")
        self._n_envs = n_envs
        self._train_config = train_config or {}
        self._rollout_config = rollout_config or {}
        self.tape_index = 0

    def set_state(self, state: dict):
        """
        Set the runner state.

        Parameters
        ----------
        state : dict
            The runner state to set.
        """
        self.tape_index = state["tape_index"]

    def get_state(self) -> dict:
        """Get the runner state.

        Returns
        -------
        dict
            The runner state
        """
        # This will depend on the environment implement
        # some environments implement local seed generator, while others implement global seed
        # For now, support MPE.
        return {"tape_index": self.tape_index}

    def finalize_batch(  # noqa: PLR6301
        self,
        agent_map: dict[AgentID, BaseAgent],
        sample_builder: dict[AgentID, defaultdict[str, list]],
        rollout_builder: dict[AgentID, SampleBatchBuilder],
        trainable_agents: Sequence[AgentID],  # noqa: ARG002
        opponent_agents: Sequence[AgentID] | None = None,  # noqa: ARG002
        reward_metrics: dict[AgentID, Sequence[str] | None] | None = None,
        reward_metric_gains: dict[AgentID, Sequence[float] | None] | None = None,
        **kwargs,
    ):
        """
        Finalize the batch of episode data in-place for PPO so it can be trained upon.

        Parameters
        ----------
        agent_map : dict[AgentID, BaseAgent]
            The map of agent IDs to agents.
        sample_builder : dict[AgentID, defaultdict[str, list]]
            The builder for sample batches.
        rollout_builder : dict[AgentID, SampleBatchBuilder]
            The builder for rollouts.
        trainable_agents : Sequence[AgentID]
            The trainable agents in the environment
        opponent_agents : Sequence[AgentID] | None, optional
            The opponent agents in the environment, default is :data:`python:None`.
        reward_metrics : dict[AgentID, Sequence[str] | None] | None, optional
            The reward metrics for each agent, default is :data:`python:None`.
        reward_metric_gains : dict[AgentID, Sequence[float] | None] | None, optional
            The reward metric weights for each reward and each agent, default is :data:`python:None`.
        `**kwargs`
            Additional keyword arguments.
        """
        eb = {agent_id: SampleBatch(**sb) for agent_id, sb in sample_builder.items()}
        for agent_id, agent in agent_map.items():
            if agent_id in eb:
                rewards = calc_rewards(
                    eb[agent_id],
                    reward_metrics=reward_metrics[agent_id],
                    reward_metric_gains=reward_metric_gains[agent_id],
                )
                eb[agent_id][f"learning_{Columns.REWARDS}"] = rewards
                eb[agent_id] = agent.process_episode(eb[agent_id], rewards=rewards, **kwargs)
                rollout_builder[agent_id].add_batch(eb[agent_id])

    def gather_batch_data(  # noqa: PLR6301
        self,
        env: ParallelEnv | AECEnv,  # noqa: ARG002
        sample_batch: dict[AgentID, defaultdict[str, list]],  # noqa: ARG002
        agent_map: dict[AgentID, BaseAgent],  # noqa: ARG002
        trainable_agents: Sequence[AgentID],  # noqa: ARG002
        opponent_agents: Sequence[AgentID] | None = None,  # noqa: ARG002
        reportable_agent: AgentID | None = None,  # noqa: ARG002
        **kwargs,
    ):
        """
        Gather additional data from into the sample batch.

        Parameters
        ----------
        env : ParallelEnv | AECEnv
            The environment to gather data from
        sample_batch : dict[AgentID, defaultdict[str, list]]
            Batches of agent data for the episode
        agent_map : dict[AgentID, BaseAgent]
            Mapping to agent modules
        trainable_agents : Iterable[AgentID]
            Iterable of trainable agent IDs
        opponent_agents : Sequence[AgentID] | None, optional
            Opponent agent names in the environment, default is :data:`python:None`
        reportable_agent : AgentID | None = None
            The agent ID to gather info for in the case of AEC api. Defaults to None causing data for all agents to be added to the batch
            (parallel env API).
        `**kwargs`
            Additional keyword arguments.
        """
        return

    def visualize_step(  # noqa: PLR6301
        self,
        env: MultiAgentEnv | ParallelEnv | AECEnv,
        sample_batch: dict[AgentID, defaultdict[str, list]],  # noqa: ARG002
        agent_map: dict[AgentID, BaseAgent],  # noqa: ARG002
        trainable_agents: Sequence[AgentID],  # noqa: ARG002
        opponent_agents: Sequence[AgentID] | None = None,  # noqa: ARG002
        reportable_agent: AgentID | None = None,  # noqa: ARG002
        **kwargs,
    ) -> list[np.ndarray]:
        """
        Visualize the current environment to np array.

        Parameters
        ----------
        env : MultiAgentEnv | ParallelEnv | AECEnv
            The environment to visualize
        sample_batch : dict[AgentID, defaultdict[str, list]]
            Batches of agent data for the episode
        agent_map : dict[AgentID, BaseAgent]
            Mapping to agent modules
        trainable_agents : Iterable[AgentID]
            Iterable of trainable agent IDs
        opponent_agents : Sequence[AgentID] | None, optional
            Opponent agent names in the environment, default is :data:`python:None`
        reportable_agent : AgentID | None = None
            The agent ID to gather info for in the case of AEC api. Defaults to None causing data for all agents to be added to the batch
            (parallel env API).
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        list[np.ndarray]
            The image representation of the step.
        """
        if isinstance(env, RenderedParallelPettingZooInspection):
            # PIL image returned
            frame = env.render()
        elif isinstance(env, MultiAgentEnvWrapper):
            # unwrap
            frame = env.get_sub_environments()[0].render()
            PIL.Image.fromarray(frame)
        else:
            # probably ParallelEnv or something
            frame = env.render()
            PIL.Image.fromarray(frame)

        return frame

    def postprocess_batch(  # noqa: PLR6301
        self,
        agent_map: dict[AgentID, BaseAgent],
        sample_batches: dict[AgentID, SampleBatch],
        trainable_agents: Sequence[AgentID],
        opponent_agents: Sequence[AgentID] | None = None,
        rendered_episodes: list[list[np.ndarray]] | None = None,
    ) -> dict[AgentID, dict[str, Any]]:
        """Run postprocessing on the rollout.

        Parameters
        ----------
        agent_map : dict[AgentID, BaseAgent]
            A mapping from agent ID to a nn.Module instance
        sample_batches : dict[AgentID, SampleBatch]
            A mapping from agent id to their sample batch
        trainable_agents : Sequence[AgentID]
            Iterable of trainable agent IDs
        opponent_agents : Sequence[AgentID] | None, optional
            Opponent agent names in the environment, default is :data:`python:None`
        rendered_episodes : list[list[np.ndarray]] | None, optional
            The sequence of rendered episodes, defaults to :data:`python:None`

        Returns
        -------
        dict[AgentID, dict[str, Any]]
            The postprocessed batch
        """
        rendered_episodes = rendered_episodes or []
        postprocessed_batch = {}

        for agent_id in chain(trainable_agents, opponent_agents or []):
            agent_batch = sample_batches[str(agent_id)]
            agent_map[agent_id].postprocess_batch(agent_batch)
            episode_batches = agent_batch.split_by_episode()
            episode_returns = np.stack([episode_batch["returns"][-1] for episode_batch in episode_batches])
            extra_info = {}
            extra_info["episode_returns"] = episode_returns
            extra_info["episode_returns_mean"] = episode_returns.mean()
            extra_info["episode_returns_max"] = episode_returns.max()
            extra_info["episode_returns_min"] = episode_returns.min()
            extra_info["rollout_buffer"] = sample_batches[str(agent_id)].copy()
            postprocessed_batch[agent_id] = {"training_batch": agent_batch, "extra_info": extra_info}
        return postprocessed_batch

    @torch.no_grad()
    def gather_parallel_rollouts(
        self,
        envs: Sequence[ParallelEnv | AECEnv],
        agent_map: dict[AgentID, BaseAgent],
        trainable_agents: Sequence[AgentID],
        opponent_agents: Sequence[AgentID] | None = None,
        batch_size: int = 3000,
        reward_metrics: dict[AgentID, Sequence[str]] | None = None,
        reward_metric_gains: dict[AgentID, Sequence[float]] | None = None,
        visualize: bool = False,
        **kwargs,
    ) -> tuple[dict[AgentID, dict[str, Any]], list]:
        """Gather a batch of rollouts from the environment.

        Parameters
        ----------
        envs : Sequence[ParallelEnv | AECEnv]
            The list of environments to use for experience gathering
        agent_map : dict[AgentID, BaseAgent]
            A mapping from agent ID to the agents which will output actions from the observations.
        trainable_agents : Iterable[AgentID]
            The ids of the trainable agents in the environment.
        opponent_agents : Sequence[AgentID] | None, optional
            Opponent agent names in the environment, default is :data:`python:None`
        batch_size : int
            The size of the batch of experiences
        reward_metrics : dict[AgentID, Sequence[str]], optional
            Which metrics to use as learning targets for each agent. Defaults to None, in which case the environment rewards are used.
        reward_metric_gains : dict[AgentID, Sequence[float]], optional
            How to scale the metrics each agent uses to learn. Defaults to None, in which case all metrics are unscaled.
        visualize : bool
            Whether to visualize the trajectories
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        dict[AgentID, dict[str, Any]
            The mapping from agentID to the extra info and samples collected during rollout
        """
        logger = logging.getLogger("ray")
        # rollout_length = np.ceil(batch_size / self._n_envs)  # noqa: ERA001
        rollout_length = batch_size

        reward_metrics = {agent_id: [Columns.REWARDS] for agent_id in agent_map} if reward_metrics is None else reward_metrics
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

        self.rollout_function = self.rollout_parallel_env if isinstance(envs[0], MultiAgentEnv) else self.rollout_aec_env
        base_task = dict(
            function=self.rollout_function,
            gather_batch_data=self.gather_batch_data,
            finalize_batch=self.finalize_batch,
            agent_map=agent_map,
            trainable_agents=trainable_agents,
            opponent_agents=opponent_agents,
            rollout_length=rollout_length,
            reward_metrics=reward_metrics,
            reward_metric_gains=reward_metric_gains,
            visualize=visualize,
            visualizer_fn=self.visualize_step if visualize else None,
            **kwargs,
        )

        # TODO: reimplement parallel rollouts
        s_envs = [envs[0]]
        for env_rank, env in enumerate(s_envs):
            logger.debug(f"\treset env with params seed={self.seed}, tape_index={self.tape_index}, env_rank={env_rank}")
            # TODO: seed_seq([...]) -> entropy -> env.seed(entropy)
            seed = self.seed + self.tape_index + env_rank

            _ = env.reset(seed=seed)
            seed_observation_spaces(env, seed)
            seed_action_spaces(env, seed)

        rbs = []
        rendered_episodes = []
        for rank, env in enumerate(s_envs):
            rb_, rendered_episode_batch = self.rollout_function(
                **(base_task | {"env": env, "env_rank": rank, "tape_index": self.tape_index, "seed": self.seed}),
            )
            rbs.append(rb_)
            rendered_episodes.extend(rendered_episode_batch)

        rb = concat_agent_buffers(rbs)
        batches = {agent_id: ag_rb.build_and_reset() for agent_id, ag_rb in rb.items()}

        sample_batch = self.postprocess_batch(
            agent_map=agent_map,
            sample_batches=batches,
            trainable_agents=trainable_agents,
            opponent_agents=opponent_agents,
        )

        return sample_batch, rendered_episodes

    @staticmethod
    @torch.no_grad()
    def rollout_parallel_env(  # noqa: PLR0915, PLR0914, PLR0913, PLR0917
        env: ParallelEnv,
        agent_map: dict[AgentID, BaseAgent],
        trainable_agents: Sequence[AgentID],
        gather_batch_data: Callable,
        finalize_batch: Callable,
        opponent_agents: Sequence[AgentID] | None = None,
        rollout_length: int = 300,
        max_episodes: int | None = None,
        batch_mode: Literal["truncate_episodes", "complete_episodes"] = "complete_episodes",
        reward_metrics: dict[AgentID, Sequence[str]] | None = None,
        reward_metric_gains: dict[AgentID, Sequence[float]] | None = None,
        tape_index: int = 0,  # noqa: ARG004
        env_rank: int = 0,  # noqa: ARG004
        seed: int = 0,  # noqa: ARG004
        visualize: bool = False,
        visualizer_fn: Callable | None = None,
        **kwargs,
    ) -> tuple[dict[AgentID, SampleBatchBuilder], list]:
        """Conduct a rollout using the parallel environment API.

        Parameters
        ----------
        env : ParallelEnv
            The environment to use for gathering experiences
        agent_map : dict[AgentID, BaseAgent]
            A mapping from agent ID to the agents which will output actions from the observations.
        trainable_agents : Sequence[AgentID]
            The ids of the agents in the environment which are trainable.
        gather_batch_data : Callable
            A callable that returns extra data from the environment
        finalize_batch : Callable
            A callable that finalizes a batch of episode data
        opponent_agents : Sequence[AgentID] | None, optional
            Opponent agent names in the environment, default is :data:`python:None`
        rollout_length : int
            The length of rollout fragments to collect
        max_episodes : int | None
            The maximum number of episodes to collect
        batch_mode : Literal["truncate_episodes", "complete_episodes"]
            How to handle trajectories when the rollout length differs from the episode length
        reward_metrics : dict[AgentID, Sequence[str]], optional
            Which metrics to use as learning targets for each agent. Defaults to None, in which case the environment rewards are used.
        reward_metric_gains : dict[AgentID, Sequence[float]], optional
            How to scale the metrics each agent uses to learn. Defaults to None, in which case all metrics are unscaled.
        tape_index : int, optional
            A tape index for the random generator state
        env_rank : int, optional
            The rank of the environment for seeding
        seed : int, optional
            The random seed to use with the environment
        visualize : bool, optional
            Whether to visualize the trajectory
        visualizer_fn : Callable | None, optional
            The function to call in order to visualize the trajectory
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        dict[AgentID, SampleBatchBuilder]
            A mapping from agent IDs to their corresponding sample batch builder
        list
            The visualized episode

        Raises
        ------
        ValueError
            If the visualization failed
        """
        episode_done = False

        sb: dict[AgentID, defaultdict[str, list]] = {agent_id: defaultdict(list) for agent_id in agent_map}
        rb: dict[AgentID, SampleBatchBuilder] = {agent_id: SampleBatchBuilder() for agent_id in agent_map}
        # T x W x H x C
        rendered_frames: list[np.ndarray] = []
        # B x T x W x H x C
        rendered_episodes: list[list[np.ndarray]] = []

        logger = logging.getLogger("ray")
        # logger.debug(f"\treset env with params seed={seed}, tape_index={tape_index}, env_rank={env_rank}")  # noqa: ERA001

        # torch.manual_seed(_seed)  # noqa: ERA001
        next_obs, _info = env.reset()
        # seed_observation_spaces(env, _seed)  # noqa: ERA001
        # seed_action_spaces(env, _seed)  # noqa: ERA001

        episode_num = 0
        step = 0
        n_steps = 0
        episode_id = episode_hash(episode_num)
        actions: dict[AgentID, torch.Tensor | np.ndarray] = {}
        logits: dict[AgentID, torch.Tensor | np.ndarray] = {}
        values: dict[AgentID, torch.Tensor | np.ndarray] = {}
        logp: dict[AgentID, torch.Tensor | np.ndarray] = {}

        logger.debug("inside rollout_parallel_env:")
        logger.debug(f"trainable: {trainable_agents}, opp: {opponent_agents}")
        while n_steps < rollout_length and (not episode_done if batch_mode == "complete_episodes" else True):
            finalized = False
            obs = next_obs
            gather_batch_data(
                env=env,
                sample_batch=sb,
                agent_map=agent_map,
                trainable_agents=trainable_agents,
                opponent_agents=opponent_agents,
                **kwargs,
            )
            if visualize:
                if visualizer_fn is None:
                    msg = "Tried to visualize but visualizer_fn is None! This should be defined by the runner."
                    raise ValueError(msg)

                frame = visualizer_fn(
                    env=env,
                    sample_batch=sb,
                    agent_map=agent_map,
                    trainable_agents=trainable_agents,
                    opponent_agents=opponent_agents,
                )
                if n_steps == 0 and np.any(frame == None):  # only check first step  # noqa: E711
                    msg = "Found NoneType in rendered frame, this is not what you want! Check env render mode."
                    raise ValueError(msg)

                rendered_frames.append(frame)

            for agent_id, agent in agent_map.items():
                agent_obs = obs[agent_id]
                if isinstance(agent_obs, dict | Dict) and "obs" in agent_obs:
                    torch_obs = torch.from_numpy(agent_obs["obs"])[None]
                    action_mask: np.ndarray = agent_obs.get("action_mask", None)  # type: ignore[assignment]
                else:
                    torch_obs = torch.from_numpy(agent_obs)[None]
                    action_mask = None

                torch_obs = torch_obs.to(agent.device, dtype=agent.dtype)
                ag_logits = agent.actor(torch_obs)  # type: ignore[attr-defined]
                ag_actions, ag_values, dist = agent.get_actions_and_values(torch_obs, logits=ag_logits)
                ag_logp = dist.log_prob(ag_actions)
                logits[agent_id] = ag_logits.numpy()
                values[agent_id] = ag_values.numpy()
                logp[agent_id] = ag_logp.numpy()
                if action_mask is None:
                    actions[agent_id] = ag_actions.numpy()[0]
                else:
                    actions[agent_id] = np.logical_and(ag_actions.numpy()[0], action_mask)

            next_obs, rewards, terms, truncs, infos = env.step(actions)
            episode_done = any(terms.values()) or any(truncs.values())

            for agent_id in next_obs:
                # only store the rollouts for trainable agents
                sb[agent_id][Columns.T].append(step)
                sb[agent_id][Columns.OBS].append(obs[agent_id])
                sb[agent_id][Columns.ACTIONS].append([actions[agent_id]])
                sb[agent_id][Columns.REWARDS].append([rewards[agent_id]])
                sb[agent_id][Columns.NEXT_OBS].append(next_obs[agent_id])
                sb[agent_id][Columns.VF_PREDS].append(values[agent_id][0])
                sb[agent_id][Columns.ACTION_LOGP].append(logp[agent_id][0])
                sb[agent_id][Columns.ACTION_DIST_INPUTS].append(logits[agent_id][0])
                sb[agent_id][Columns.TERMINATEDS].append(terms[agent_id])
                sb[agent_id][Columns.TRUNCATEDS].append(truncs[agent_id])
                sb[agent_id][Columns.INFOS].append(infos[agent_id])
                sb[agent_id][Columns.EPS_ID].append(episode_id)
                sb[agent_id]["returns"].append([(sb[agent_id]["returns"][-1][0] if step > 0 else 0) + rewards[agent_id]])

            if episode_done:
                # compute advantages and add the episode data to the rollout buffer
                finalize_batch(
                    agent_map=agent_map,
                    sample_builder=sb,
                    rollout_builder=rb,
                    trainable_agents=trainable_agents,
                    opponent_agents=opponent_agents,
                    reward_metrics=reward_metrics,
                    reward_metric_gains=reward_metric_gains,
                    **kwargs,
                )
                finalized = True
                if visualize:
                    rendered_episodes.append(rendered_frames)
                    logger.info(f"Length of rendered: {len(rendered_episodes)}")

                if episode_num + 1 == max_episodes:
                    break

                # reset the episode data
                step = 0
                episode_num += 1
                episode_done = False
                next_obs, _info = env.reset()
                episode_id = episode_hash(episode_num)
                sb = {agent_id: defaultdict(list) for agent_id in agent_map}
                rendered_frames = []

            else:
                step += 1

            # reset the step data
            n_steps += 1
            actions.clear()
            logits.clear()
            values.clear()
            logp.clear()

        if not finalized:
            finalize_batch(
                agent_map=agent_map,
                sample_builder=sb,
                rollout_builder=rb,
                trainable_agents=trainable_agents,
                opponent_agents=opponent_agents,
                reward_metrics=reward_metrics,
                reward_metric_gains=reward_metric_gains,
                **kwargs,
            )
            # save frames for episode that didn't end up truncating or finishing from env side
            if visualize and episode_num < max_episodes:
                logger.warning(
                    f"Requested render for {max_episodes} episode(s) but ended up with {episode_num} after {n_steps} steps."
                    "Check agent behavior.",
                )
                rendered_episodes.append(rendered_frames)
                logger.info(f"Length of rendered: {len(rendered_episodes)}")
                rendered_frames = []

        env.close()
        if visualize:
            logger.info(f"After collection, got {len(rendered_episodes)} episodes with total size={total_size(rendered_episodes)} bytes")

        return rb, rendered_episodes

    @staticmethod
    @torch.no_grad()
    def rollout_aec_env(
        env: AECEnv,
        agent_map: dict[AgentID, BaseAgent],
        trainable_agents: Sequence[AgentID],
        gather_batch_data: Callable,
        finalize_batch: Callable,
        opponent_agents: Sequence[AgentID] | None = None,
        rollout_length: int = 300,
        max_episodes: int | None = None,
        batch_mode: Literal["truncate_episodes", "complete_episodes"] = "complete_episodes",
        reward_metrics: dict[AgentID, Sequence[str]] | None = None,
        reward_metric_gains: dict[AgentID, Sequence[float]] | None = None,
        tape_index: int = 0,  # noqa: ARG004
        env_rank: int = 0,  # noqa: ARG004
        seed: int = 0,  # noqa: ARG004
        **kwargs,
    ) -> tuple[dict[AgentID, SampleBatchBuilder], list]:
        """Conduct a rollout using the AEC environment API.

        Parameters
        ----------
        agent_map : dict[AgentID, BaseAgent]
            The mapping from agent ids to agent classes
        trainable_agents : Iterable[AgentID]
            The ids of the agents in the environment which are trainable.
        reward_metrics : dict[AgentID, Sequence[str]], optional
            Which metrics to use as learning targets for each agent. Defaults to None, in which case the environment rewards are used.
        reward_metric_gains : dict[AgentID, Sequence[float]], optional
            How to scale the metrics each agent uses to learn. Defaults to None, in which case all metrics are unscaled.
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        dict[AgentID, SampleBatchBuilder]
            A mapping from agent IDs to their corresponding sample batch builder
        """
        step = 0
        n_steps = 0
        episode_num = 0

        rb = {agent_id: SampleBatchBuilder() for agent_id in agent_map}
        # logger.debug(f"\treset env with params seed={seed}, tape_index={tape_index}, env_rank={env_rank}")  # noqa: ERA001

        # keep rolling out new episodes until the stop conditions are met
        while n_steps < rollout_length:
            n_steps += step

            # new episodic variables
            episode_id = episode_hash(episode_num)
            agent_steps = dict.fromkeys(agent_map, 0)
            sb: dict[AgentID, defaultdict[str, list]] = {agent_id: defaultdict(list) for agent_id in agent_map}
            end_agents: list[AgentID] | None = None

            # env.reset(seed=seed + tape_index + env_rank + n_steps)  # noqa: ERA001
            env.reset()
            # torch.manual_seed(seed + tape_index + env_rank + n_steps)  # noqa: ERA001

            # main episode loop
            for agent_id in env.agent_iter():
                gather_batch_data(
                    env=env,
                    sample_batch=sb,
                    agent_map=agent_map,
                    trainable_agents=trainable_agents,
                    opponent_agents=opponent_agents,
                    reportable_agent=agent_id,
                    **kwargs,
                )
                obs, reward, term, trunc, info = env.last()
                logits = agent_map[agent_id].actor(obs)  # type: ignore[attr-defined]

                if term or trunc:
                    actions = None
                else:
                    actions, values, dist = agent_map[agent_id].get_actions_and_values(obs, logits=logits)
                    logp = dist.log_prob(actions)

                env.step(actions)

                if actions and agent_id in chain(trainable_agents, opponent_agents or []):
                    sb[agent_id][Columns.T].append(agent_steps[agent_id])
                    sb[agent_id][Columns.OBS].append(obs)
                    sb[agent_id][Columns.ACTIONS].append(actions)
                    sb[agent_id][Columns.REWARDS].append([reward])
                    sb[agent_id][Columns.NEXT_OBS].append(env.observe(agent_id))
                    sb[agent_id][Columns.VF_PREDS].append(values)
                    sb[agent_id][Columns.ACTION_LOGP].append(logp)
                    sb[agent_id][Columns.ACTION_DIST_INPUTS].append(logits)
                    sb[agent_id][Columns.TERMINATEDS].append(term)
                    sb[agent_id][Columns.TRUNCATEDS].append(trunc)
                    sb[agent_id][Columns.INFOS].append(info)
                    sb[agent_id][Columns.EPS_ID].append(episode_id)
                    sb[agent_id]["returns"].append([(sb[agent_id]["returns"][-1][0] if step > 0 else 0) + reward])
                    agent_steps[agent_id] += 1
                if end_agents and agent_id in end_agents:
                    # we've hit the rollout length stop condition and we've come back around to the same agent
                    break
                if batch_mode != "complete_episodes" and n_steps + step >= rollout_length:
                    # set the "end_agent" so the next time this agent takes a turn we know to end the rollout
                    end_agents = [agent_id] if end_agents is None else [*end_agents, agent_id]

            step = max(agent_steps.values())

            finalize_batch(
                agent_map=agent_map,
                sample_builder=sb,
                rollout_builder=rb,
                trainable_agents=trainable_agents,
                opponent_agents=opponent_agents,
                reward_metrics=reward_metrics,
                reward_metric_gains=reward_metric_gains,
                **kwargs,
            )

            episode_num += 1
            if episode_num == max_episodes:
                break
        return rb, []

    def island_training_step(  # noqa: PLR0915
        self,
        agent_map: dict[AgentID, BaseAgent],
        sample_batch: dict[AgentID, dict[str, SampleBatch | Any]],
        trainable_agents: Sequence[AgentID],
        opponent_agents: Sequence[AgentID] | None = None,
        device: str | torch.device = "cpu",
        **kwargs,
    ) -> tuple[dict[AgentID, BaseAgent], dict[AgentID, dict[str, Any]]]:
        """Conduct a single rollout on the island, with training.

        Parameters
        ----------
        agent_map : dict[AgentID, BaseAgent]
            A mapping from agent ID to a nn.Module instance
        sample_batch : dict[AgentID, dict[str, SampleBatch | Any]]
            A mapping from agent id to their sample batch
        trainable_agents : Sequence[AgentID]
            Iterable of trainable agent IDs
        opponent_agents : Sequence[AgentID] | None, optional
            Opponent agent names in the environment, default is :data:`python:None`
        device : str | torch.device
            The device to use for training
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        dict[AgentID, BaseAgent]
            The trained agents
        dict[AgentID, dict[str, Any]]
            A mapping from agent id to their training results.
        """
        torch.manual_seed(self.seed + self.tape_index)

        def render_debug_panel(d: dict):
            table = Table(title="Runner remote_train_agent() stats")

            table.add_column("Metric", justify="left", style="cyan", no_wrap=False)
            table.add_column("value", justify="center", style="green")

            for k, v in d.items():
                if isinstance(v, torch.Tensor):
                    v = f"{v.detach().cpu().item():.3f}"  # noqa: PLW2901
                elif isinstance(v, np.ndarray):
                    v = f"{v:.3f}"  # noqa: PLW2901
                else:
                    v = f"{v}"  # noqa: PLW2901

                table.add_row(k, v)

            console = Console()
            console.print(table)

        @ray.remote
        def remote_train_agent(
            batch: SampleBatch,
            agent: BaseAgent,
            agent_id: AgentID,
            **kwargs,
        ) -> tuple[dict, BaseAgent, AgentID]:
            logger = logging.getLogger("ray")

            batch.set_training(True)
            tb_builder = SampleBatchBuilder()
            training_infos: defaultdict[str, list[np.ndarray]] = defaultdict(list)
            training_stats = {}
            batch = egocentric_shuffle(batch, agent.np_random)
            index_start = 0
            batch_size = kwargs.get("sgd_minibatch_size", batch.count)
            if batch_size > batch.count:
                msg = f"Minibatch size ({batch_size}) must be less than the total batch size ({batch.count})"
                raise RuntimeError(msg)

            for _sgd_iter in range(kwargs.get("num_sgd_iter", 1)):
                if _sgd_iter % 5 == 0:
                    logger.debug(f"\t ========= sgd iter {_sgd_iter}")

                # perform minibatch SGD
                index_end = index_start + batch_size
                if index_end > batch.count:
                    batch = egocentric_shuffle(batch, agent.np_random)
                    index_start = 0
                    index_end = index_start + batch_size

                train_batch = batch.slice(index_start, index_end)
                tb_builder.add_batch(train_batch)
                train_batch = tb_builder.build_and_reset()
                train_batch.set_get_interceptor(functools.partial(convert_to_torch_tensor, device=device))

                _, infos = agent.learn(train_batch=train_batch, **kwargs)
                for key, val in infos.items():
                    training_infos[key].append(val)

                if _sgd_iter % 25 == 0:
                    render_debug_panel(infos)

            training_stats = {key: np.stack(sval, axis=-1) for key, sval in training_infos.items()}
            training_stats |= agent.post_training_hook(training_stats=training_stats)

            return training_stats, agent, agent_id

        train_config = {key: value for key, value in kwargs.items() if key not in {"env", "envs"}}
        unfinished = [
            remote_train_agent.remote(sample_batch[agent_id]["training_batch"], agent_map[agent_id], agent_id, **train_config)
            for agent_id in trainable_agents
        ]

        while unfinished:
            finished, unfinished = ray.wait(unfinished)
            for training_stats, agent, agent_id in ray.get(finished):
                agent_map.update({agent_id: agent})
                sample_batch[agent_id]["training_stats"] = {**training_stats}

        for agent_id in chain(trainable_agents, opponent_agents or []):
            rollout_stats = {}
            episode_batches = sample_batch[agent_id]["training_batch"].split_by_episode()
            episode_returns = np.stack([episode_batch["returns"][-1] for episode_batch in episode_batches]).mean()
            rollout_stats["returns"] = episode_returns
            sample_batch[agent_id]["rollout_stats"] = rollout_stats

        return agent_map, sample_batch

    def island_rollout_step(
        self,
        agent_map: dict[AgentID, BaseAgent],
        sample_batch: dict[AgentID, dict[str, SampleBatch | Any]],
        trainable_agents: Sequence[AgentID],
        opponent_agents: Sequence[AgentID] | None = None,
    ) -> tuple[dict[AgentID, BaseAgent], dict[AgentID, dict[str, Any]]]:
        """Conduct a single rollout on the island without training.

        Parameters
        ----------
        agent_map : dict[AgentID, BaseAgent]
            A mapping from agent ID to a nn.Module instance
        sample_batch : dict[AgentID, dict[str, SampleBatch | Any]]
            A mapping from agent id to their sample batch
        trainable_agents : Sequence[AgentID]
            Iterable of trainable agent IDs
        opponent_agents : Sequence[AgentID] | None, optional
            Opponent agent names in the environment, default is :data:`python:None`

        Returns
        -------
        dict[AgentID, BaseAgent]
            The agents used for rollout
        dict[AgentID, dict[str, Any]]
            The mapping from agent id to info dict, which includes the rollout
        """
        torch.manual_seed(self.seed + self.tape_index)
        rollout_results = {}
        for agent_id in chain(trainable_agents, opponent_agents or []):
            agent_batch = sample_batch[str(agent_id)]["extra_info"]["rollout_buffer"].copy()
            rollout_stats = {}
            episode_batches = agent_batch.split_by_episode()
            episode_returns = np.stack([episode_batch["returns"][-1] for episode_batch in episode_batches]).mean()
            rollout_stats["returns"] = episode_returns
            rollout_results[agent_id] = {"extra_info": {"rollout_buffer": agent_batch}, "rollout_stats": rollout_stats}
        return agent_map, rollout_results

    def rollout_island(
        self,
        envs: list,
        agent_map: dict[AgentID, BaseAgent],
        trainable_agents: Sequence[AgentID],
        opponent_agents: Sequence[AgentID] | None = None,
        reward_metrics: dict[AgentID, Sequence[str]] | None = None,
        reward_metric_gains: dict[AgentID, Sequence[float]] | None = None,
        rollout_config: dict | None = None,
        visualize: bool = False,
        **kwargs,
    ) -> tuple[dict[AgentID, BaseAgent], dict[AgentID, dict[str, Any]], list]:
        """Gather a rollout from the island without training.

        Parameters
        ----------
        envs : list
            Environments to use for experience gathering
        agent_map : dict[AgentID, BaseAgent]
            Mapping from agent ids to their classes
        trainable_agents : Sequence[AgentID]
            The agents which are trainable
        opponent_agents : Sequence[AgentID] | None, optional
            The agents on opposing teams, :data:`python:None`
        reward_metrics : dict[AgentID, Sequence[str]] | None, optional
            The rewards metrics for each agent, :data:`python:None`
        reward_metric_gains : dict[AgentID, Sequence[float]] | None, optional
            The gains for each reward metric for each agent, :data:`python:None`
        rollout_config : dict | None, optional
            The configuration to use when gathering experiences, :data:`python:None`
        visualize : bool, optional
            Whether to visualize the experiences, :data:`python:False`
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        tuple[dict[AgentID, BaseAgent], dict[AgentID, dict[str, Any]], list]
            The agents, the rollouts for the agents, and the visualization
        """
        rollout_config = self._rollout_config | kwargs | (rollout_config or {})
        for agent in agent_map.values():
            agent.train(mode=False)

        # with Timer("rollout_island_rollout", "mean"):
        sample_batch, rendered_episodes = self.gather_parallel_rollouts(
            envs=envs,
            agent_map=agent_map,
            trainable_agents=trainable_agents,
            opponent_agents=opponent_agents,
            reward_metrics=reward_metrics,
            reward_metric_gains=reward_metric_gains,
            visualize=visualize,
            **rollout_config,
        )
        agent_map, rollout_results = self.island_rollout_step(
            agent_map=agent_map,
            sample_batch=sample_batch,
            trainable_agents=trainable_agents,
            opponent_agents=opponent_agents,
        )
        return agent_map, rollout_results, rendered_episodes

    def train_island(
        self,
        envs: list,
        agent_map: dict[AgentID, BaseAgent],
        trainable_agents: Sequence[AgentID],
        opponent_agents: Sequence[AgentID] | None = None,
        reward_metrics: dict[AgentID, Sequence[str]] | None = None,
        reward_metric_gains: dict[AgentID, Sequence[float]] | None = None,
        rollout_config: dict | None = None,
        train_config: dict | None = None,
        visualize: bool = False,
        **kwargs,
    ) -> tuple[dict[AgentID, BaseAgent], dict[AgentID, dict[str, Any]], list]:
        """Gather a rollout from the island with training.

        Parameters
        ----------
        envs : list
            List of environments to use for experience collection
        agent_map : dict[AgentID, BaseAgent]
            Mapping of agent ids to their objects
        trainable_agents : Sequence[AgentID]
            Sequence of trainable agent ids
        opponent_agents : Sequence[AgentID] | None, optional
            Sequence of opponent agent ids, :data:`python:None`
        reward_metrics : dict[AgentID, Sequence[str]] | None, optional
            Reward metrics for each agent, :data:`python:None`
        reward_metric_gains : dict[AgentID, Sequence[float]] | None, optional
            Multipliers for reward metrics for each agent, :data:`python:None`
        rollout_config : dict | None, optional
            Configuration to use when gathering experiences, :data:`python:None`
        train_config : dict | None, optional
            Configuration to use when training, :data:`python:None`
        visualize : bool, optional
            Whether to visualize the trajectory, :data:`python:False`
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        tuple[dict[AgentID, BaseAgent], dict[AgentID, dict[str, Any]], list]
            The trained agents, training info, and trajectory visualizations
        """
        rollout_config = self._rollout_config | kwargs | (rollout_config or {})
        train_config = self._train_config | kwargs | (train_config or {})
        for agent in agent_map.values():
            agent.train(mode=False)

        # with Timer("train_island_rollout", "mean"):
        sample_batch, rendered_episodes = self.gather_parallel_rollouts(
            envs=envs,
            agent_map=agent_map,
            trainable_agents=trainable_agents,
            opponent_agents=opponent_agents,
            reward_metrics=reward_metrics,
            reward_metric_gains=reward_metric_gains,
            visualize=visualize,
            **rollout_config,
        )

        for agent_id, agent in agent_map.items():
            if agent_id in trainable_agents:
                agent.reset_optimizer()  # TODO: ??
                agent.train(mode=True)

        agent_map, sample_batch = self.island_training_step(
            agent_map=agent_map,
            sample_batch=sample_batch,
            trainable_agents=trainable_agents,
            opponent_agents=opponent_agents,
            **train_config,
        )
        return agent_map, sample_batch, rendered_episodes

    def __call__(
        self,
        envs: list,
        train: bool = False,
        visualize: bool = False,
        **kwargs,
    ) -> tuple[dict[AgentID, BaseAgent], dict[AgentID, dict[str, Any]], list[np.ndarray]]:
        """
        Call the training method or the rollout method.

        Parameters
        ----------
        envs : list
            The envs to use for rollouts and training
        train : bool, optional
            Whether to call the training method, :data:`python:False`
        visualize : bool
            Whether to visualize the trajectories
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        agent_map : dict[AgentID, BaseAgent]
            The agent map passed to the function.
        result_dict : dict[AgentID, dict[str, Any]]
            Dictionary of info for each agent.
        rendered_episodes : list[np.ndarray]
            The rendered episodes collected from the runner.
        """
        logger = logging.getLogger("ray")
        logger.debug("before BaseRunner.__call__")
        log_agent_params(kwargs["agent_map"], logger.debug)

        if train:
            agent_map, result_dict, rendered_episodes = self.train_island(envs=envs, visualize=visualize, **self._train_config, **kwargs)
        else:
            agent_map, result_dict, rendered_episodes = self.rollout_island(
                envs=envs,
                visualize=visualize,
                **self._rollout_config,
                **kwargs,
            )

        logger.debug("after BaseRunner.__call__")
        log_agent_params(agent_map, logger.debug)

        self.tape_index += 1
        return agent_map, result_dict, rendered_episodes


class SingleAgentRLRunner(BaseRunner):
    """Runner class for RL algorithms where a single algorithm controls and updates all trainable agents."""

    def island_training_step(  # noqa: PLR0915
        self,
        agent_map: dict[AgentID, BaseAgent],
        sample_batch: dict[AgentID, dict[str, SampleBatch | Any]],
        trainable_agents: Sequence[AgentID],
        opponent_agents: Sequence[AgentID] | None = None,
        device: str | torch.device = "cpu",
        **kwargs,
    ) -> tuple[dict[AgentID, BaseAgent], dict[AgentID, dict[str, Any]]]:
        """Conduct a single rollout on the island, with training.

        Parameters
        ----------
        agent_map : dict[AgentID, BaseAgent]
            A mapping from agent ID to a nn.Module instance
        sample_batch : dict[AgentID, dict[str, SampleBatch | Any]]
            A mapping from agent id to their sample batch
        trainable_agents : Iterable[AgentID]
            Iterable of trainable agent IDs
        opponent_agents : Sequence[AgentID] | None, optional
            Opponent agent names in the environment, default is :data:`python:None`
        device : str | torch.device
            The device to use for training
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        agent_map : dict[AgentID, BaseAgent]
            The agents used for training
        sample_batch : dict[AgentID, dict[str, Any]]
            A mapping from agent id to their training results.
        """
        torch.manual_seed(self.seed + self.tape_index)

        def render_debug_panel(d: dict):
            table = Table(title="Runner remote_train_agent() stats")

            table.add_column("Metric", justify="left", style="cyan", no_wrap=False)
            table.add_column("value", justify="center", style="green")

            for k, v in d.items():
                if isinstance(v, torch.Tensor):
                    v = f"{v.detach().cpu().item():.3f}"  # noqa: PLW2901
                elif isinstance(v, np.ndarray):
                    v = f"{v:.3f}"  # noqa: PLW2901
                else:
                    v = f"{v}"  # noqa: PLW2901

                table.add_row(k, v)

            console = Console()
            console.print(table)

        @ray.remote
        def remote_train_agent(
            batch: SampleBatch,
            agent: BaseAgent,
            agent_id: AgentID,
            **kwargs,
        ) -> tuple[dict, BaseAgent, AgentID]:
            logger = logging.getLogger("ray")

            batch.set_training(True)
            tb_builder = SampleBatchBuilder()
            training_infos: defaultdict[str, list[np.ndarray]] = defaultdict(list)
            training_stats = {}
            batch = egocentric_shuffle(batch, agent.np_random)
            index_start = 0
            batch_size = kwargs.get("sgd_minibatch_size", batch.count)
            if batch_size > batch.count:
                msg = f"Minibatch size ({batch_size}) must be less than the total batch size ({batch.count})"
                raise RuntimeError(msg)

            for _sgd_iter in range(kwargs.get("num_sgd_iter", 1)):
                if _sgd_iter % 5 == 0:
                    logger.debug(f"\t ========= sgd iter {_sgd_iter}")

                # perform minibatch SGD
                index_end = index_start + batch_size
                if index_end > batch.count:
                    batch = egocentric_shuffle(batch, agent.np_random)
                    index_start = 0
                    index_end = index_start + batch_size

                train_batch = batch.slice(index_start, index_end)
                tb_builder.add_batch(train_batch)
                train_batch = tb_builder.build_and_reset()
                train_batch.set_get_interceptor(functools.partial(convert_to_torch_tensor, device=device))

                _, infos = agent.learn(train_batch=train_batch, **kwargs)
                for key, val in infos.items():
                    training_infos[key].append(val)

                if _sgd_iter % 25 == 0:
                    render_debug_panel(infos)

            training_stats = {key: np.stack(sval, axis=-1) for key, sval in training_infos.items()}
            training_stats |= agent.post_training_hook(training_stats=training_stats)

            return training_stats, agent, agent_id

        train_config = {key: value for key, value in kwargs.items() if key not in {"env", "envs"}}
        # since this is a single agent rl, all agents are controlled by the same policy, so we aggregate the losses
        # and backpropagate to the first agent
        sbb = SampleBatchBuilder()
        first_agent = None
        for agent_id in trainable_agents:
            if first_agent is None:
                first_agent = agent_id
            sbb.add_batch(sample_batch[agent_id]["training_batch"])
        training_batch = sbb.build_and_reset()
        training_stats, agent, _agent_id = ray.get(
            remote_train_agent.remote(training_batch, agent_map[first_agent], first_agent, **train_config),
        )

        for agent_id in trainable_agents:
            agent_map.update({agent_id: agent})
            sample_batch[agent_id]["training_stats"] = {**training_stats}

        for agent_id in chain(trainable_agents, opponent_agents or []):
            rollout_stats = {}
            episode_batches = training_batch.split_by_episode()
            episode_returns = np.stack([episode_batch["returns"][-1] for episode_batch in episode_batches]).mean()
            rollout_stats["returns"] = episode_returns
            sample_batch[agent_id]["rollout_stats"] = rollout_stats

        return agent_map, sample_batch
