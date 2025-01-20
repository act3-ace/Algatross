"""Runner class for SMACv2 environments."""

from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Literal

import torch

from gymnasium.spaces import Dict

from algatross.agents.base import BaseAgent
from algatross.environments.runners import BaseRunner
from algatross.environments.smac.smac_wrapper import SMACV2RllibEnv
from algatross.environments.utilities import episode_hash
from algatross.utils.sample_batch_builder import SampleBatchBuilder
from algatross.utils.types import AgentID

if TYPE_CHECKING:
    import numpy as np

try:
    from ray.rllib.core import Columns  # type: ignore[attr-defined, unused-ignore]
except ImportError:
    from algatross.utils.compatibility import Columns  # type: ignore[assignment, unused-ignore]


class SMACV2Runner(BaseRunner):
    """Runner class for MPE Simple Spread."""

    @staticmethod
    @torch.no_grad()
    def rollout_parallel_env(  # type: ignore[override] # noqa: D102, PLR0915, PLR0914
        env: SMACV2RllibEnv,
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
        **kwargs,
    ) -> dict[AgentID, SampleBatchBuilder]:
        episode_done = False

        sb: dict[AgentID, defaultdict[str, list]] = {agent_id: defaultdict(list) for agent_id in agent_map}
        rb: dict[AgentID, SampleBatchBuilder] = {agent_id: SampleBatchBuilder() for agent_id in agent_map}

        next_obs, _infos = env.reset()

        episode_num = 0
        step = 0
        n_steps = 0
        episode_id = episode_hash(episode_num)
        actions: dict[AgentID, torch.Tensor | np.ndarray] = {}
        logits: dict[AgentID, torch.Tensor | np.ndarray] = {}
        values: dict[AgentID, torch.Tensor | np.ndarray] = {}
        logp: dict[AgentID, torch.Tensor | np.ndarray] = {}

        obs_feature_names = env.get_obs_feature_names()

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
            for agent_id, agent in agent_map.items():
                agent_obs = obs[agent_id]
                if isinstance(agent_obs, dict | Dict) and "obs" in agent_obs:
                    torch_obs = torch.from_numpy(agent_obs["obs"]).unsqueeze(0)
                    action_mask = agent_obs.get("action_mask", None)
                    if action_mask is not None:
                        action_mask = torch.from_numpy(action_mask).unsqueeze(0)
                else:
                    torch_obs = torch.from_numpy(agent_obs).unsqueeze(0)
                    action_mask = None
                ag_logits = agent.actor(torch_obs)  # type: ignore[attr-defined]
                ag_actions, ag_values, dist = agent.get_actions_and_values(torch_obs, logits=ag_logits, action_mask=action_mask)
                ag_logp = dist.log_prob(ag_actions)
                logits[agent_id] = ag_logits.numpy()
                values[agent_id] = ag_values.numpy()
                logp[agent_id] = ag_logp.numpy()
                actions[agent_id] = ag_actions.numpy()[0]

            next_obs, rewards, terms, truncs, infos = env.step(actions)
            episode_done = any(terms.values()) or any(truncs.values())

            for agent_id in next_obs:
                agent_obs = obs[agent_id]
                if isinstance(agent_obs, dict | Dict) and "obs" in agent_obs:
                    agent_obs = agent_obs["obs"]
                # only store the rollouts for trainable agents
                sb[agent_id][Columns.T].append(step)
                sb[agent_id][Columns.OBS].append(agent_obs)
                sb[agent_id][Columns.ACTIONS].append([actions[agent_id]])
                sb[agent_id][Columns.REWARDS].append([rewards[agent_id]])
                sb[agent_id][Columns.NEXT_OBS].append(next_obs[agent_id])
                sb[agent_id][Columns.VF_PREDS].append(values[agent_id][0])
                sb[agent_id][Columns.ACTION_LOGP].append(logp[agent_id][0])
                sb[agent_id][Columns.ACTION_DIST_INPUTS].append(logits[agent_id][0])
                sb[agent_id][Columns.TERMINATEDS].append(terms[agent_id])
                sb[agent_id][Columns.TRUNCATEDS].append(truncs[agent_id])
                for info_key, info in infos[agent_id].items():
                    sb[agent_id][info_key].append([info])
                for obs_feat, ob in zip(obs_feature_names, agent_obs, strict=True):
                    if "own_" in obs_feat:
                        sb[agent_id][obs_feat].append([ob])
                sb[agent_id][Columns.INFOS].append({})
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

                if episode_num == max_episodes:
                    break

                # reset the episode data
                step = 0
                episode_num += 1
                episode_done = False
                next_obs, _infos = env.reset()
                episode_id = episode_hash(episode_num)
                sb = {agent_id: defaultdict(list) for agent_id in agent_map}

            else:
                step += 1

            # reset the step data
            n_steps += 1
            actions.clear()
            logits.clear()
            values.clear()
            logp.clear()
        if not finalized:
            # our last episode didn't finish but we still need to finalize
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
        return rb

    @staticmethod
    @torch.no_grad()
    def rollout_aec_env(  # type: ignore[override] # noqa: D102
        env: SMACV2RllibEnv,
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
        **kwargs,
    ) -> dict[AgentID, SampleBatchBuilder]:
        raise NotImplementedError

    def gather_batch_data(  # noqa: D102
        self,
        env: SMACV2RllibEnv,
        sample_batch: dict[AgentID, defaultdict[str, list]],
        agent_map: dict[AgentID, BaseAgent],
        trainable_agents: Sequence[AgentID],
        opponent_agents: Sequence[AgentID] | None = None,
        reportable_agent: str | int | None = None,
        **kwargs,
    ):
        super().gather_batch_data(env, sample_batch, agent_map, trainable_agents, opponent_agents, reportable_agent, **kwargs)
        extra_info: dict[str, float] = defaultdict(float)
        for feat, value in zip(env.get_state_feature_names(), env.get_state(), strict=True):
            if "ally_health_level" in feat:
                extra_info["ally_health_level"] += value
            elif "ally_health" in feat:
                extra_info["ally_health"] += value
            elif "ally_shield" in feat:
                extra_info["ally_shield"] += value
            elif "enemy_health" in feat:
                extra_info["enemy_health"] += value
            elif "enemy_shield" in feat:
                extra_info["enemy_shield"] += value
        for agent_id in sample_batch:
            for feat, value in extra_info.items():
                sample_batch[agent_id][feat].append([value])
