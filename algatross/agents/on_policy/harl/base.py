"""Base classes for on-policy HARL agents."""

from abc import abstractmethod
from collections.abc import Mapping
from typing import Literal

import numpy as np

from ray.rllib import SampleBatch

import torch

from torch.optim import Optimizer

from gymnasium.spaces import Space
from harl.algorithms.actors.off_policy_base import OffPolicyBase
from harl.algorithms.actors.on_policy_base import OnPolicyBase
from harl.common.buffers.off_policy_buffer_ep import OffPolicyBufferEP
from harl.common.buffers.off_policy_buffer_fp import OffPolicyBufferFP
from harl.common.buffers.on_policy_actor_buffer import OnPolicyActorBuffer
from harl.utils.trans_tools import _t2n  # noqa: PLC2701

from algatross.agents.torch_base import TorchBaseAgent, TorchOnPolicyMARLAgent
from algatross.configs.harl.agents import HARLAgentConfig
from algatross.environments.utilities import episode_hash
from algatross.utils.types import PlatformID

try:
    from ray.rllib.core import Columns  # type: ignore[attr-defined]
except ImportError:
    from algatross.utils.compatibility import Columns  # type: ignore[unused-ignore, assignment]


class TorchBaseHARLActor(TorchBaseAgent):
    """A base class for torch actors using HARL."""

    harl_actor: OnPolicyBase | OffPolicyBase
    actor_buffer: OnPolicyActorBuffer | OffPolicyBufferEP | OffPolicyBufferFP
    critic_buffer: OffPolicyBufferEP | OffPolicyBufferFP | None

    @property
    def actor(self) -> OnPolicyBase | OffPolicyBase:  # noqa: D102
        return self.harl_actor

    def lr_decay(self, episode: int, episodes: int):  # noqa: D102
        self.harl_actor.lr_decay(episode, episodes)

    def get_actions(self, step: int, **kwargs) -> tuple:  # type: ignore[override] # noqa: D102
        return self.harl_actor.get_actions(obs=self.actor_buffer.obs[step], **kwargs)

    def reset_optimizer(self):  # noqa: D102
        self.harl_actor.actor_optimizer.load_state_dict(self._initial_optimizer_state)

    @abstractmethod
    def clear_buffer(self):  # noqa: D102
        pass

    @abstractmethod
    def warmup(self, /, **kwargs):  # noqa: D102
        pass

    @abstractmethod
    def insert(self, /, **kwargs):  # noqa: D102
        pass

    @abstractmethod
    def prep_rollout(self):  # noqa: D102
        pass

    @abstractmethod
    def prep_training(self):  # noqa: D102
        pass

    @abstractmethod
    def after_update(self):  # noqa: D102
        pass


class TorchOnPolicyBaseHARLActor(TorchBaseHARLActor):
    """
    A single torch actor for on-policy HARL algorithms.

    Parameters
    ----------
    obs_space : dict[PlatformID, gymnasium.spaces.Space]
        The observation space for each platform of this agent.
    act_space : dict[PlatformID, gymnasium.spaces.Space]
        The action space for each platform of this agent.
    critic_outs : int, optional
        The number of outputs for the critic network, default is 1.
    optimizer_class : str | type[torch.optim.Optimizer], optional
        The optimizer to use with this module, default is :class:`~torch.optim.adam.Adam`
    optimizer_kwargs : dict | None, optional
        Keyword arguments to pass to the optimizer constructor, default is :data:`python:None`
    `**kwargs`
        Additional keyword arguments.
    """

    harl_actor: OnPolicyBase
    """The actor for this off-policy algorithm."""
    actor_buffer: OnPolicyActorBuffer
    """The buffer for the actor."""

    def __init__(
        self,
        *,
        obs_space: Space,
        act_space: Space,
        critic_outs: int = 1,
        optimizer_class: str | type[Optimizer] = torch.optim.Adam,
        optimizer_kwargs: dict | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            obs_space=obs_space,
            act_space=act_space,
            critic_outs=critic_outs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            **kwargs,
        )
        self.actor_buffer = OnPolicyActorBuffer(args=kwargs, obs_space=obs_space, act_space=act_space)
        self.actor_buffer.infos = np.zeros((self.actor_buffer.episode_length + 1, self.actor_buffer.n_rollout_threads), dtype=np.object_)
        self.critic_buffer = None

    @property
    def actor(self) -> OnPolicyBase:
        """
        The actor for this off-policy algorithm.

        Returns
        -------
        OnPolicyBase
            The actor for this off-policy algorithm
        """
        return self.harl_actor

    @property
    def critic(self):  # noqa: D102
        return None

    @property
    def state_type(self) -> Literal["EP", "FP"]:
        """
        The RNN state type.

        Either:

        - :python:`"EP"` for *episode_provided*
        - :python:`"FP"` for *feature_provided*

        Returns
        -------
        Literal["EP", "FP"]
            The RNN state type.
        """
        return self._state_type

    def learn(  # type: ignore[override] # noqa: D102
        self,
        /,
        advantages: np.ndarray | torch.Tensor,
        state_type: Literal["EP", "FP"],
        **kwargs,
    ) -> tuple[torch.Tensor, dict]:
        train_infos = self.harl_actor.train(self.actor_buffer, advantages, state_type)
        return train_infos["policy_loss"], train_infos

    def evaluate_actions(self, obs, **kwargs):  # noqa: D102
        return self.harl_actor.evaluate_actions(obs=obs, **kwargs)

    def act(self, obs, **kwargs):  # noqa: D102
        return self.harl_actor.act(obs=obs, **kwargs)

    def prep_training(self):  # noqa: D102
        self.train(True)
        self.harl_actor.prep_training()

    def prep_rollout(self):  # noqa: D102
        self.train(False)
        self.harl_actor.prep_rollout()

    def warmup(self, /, obs: np.ndarray, infos: np.ndarray, available_actions: np.ndarray | None, **kwargs):  # type: ignore[override] # noqa: D102
        self.actor_buffer.obs[0] = obs.copy()
        self.actor_buffer.infos[0] = infos.copy()
        if self.actor_buffer.available_actions is not None:
            self.actor_buffer.available_actions[0] = available_actions.copy()

    def insert(self, infos: np.ndarray, **kwargs):  # type: ignore[override] # noqa: D102
        self.actor_buffer.infos[self.actor_buffer.step + 1] = infos.copy()
        self.actor_buffer.insert(**kwargs)

    def after_update(self):  # noqa: D102
        self.actor_buffer.infos[0] = self.actor_buffer.infos[-1].copy()
        self.actor_buffer.after_update()

    def clear_buffer(self):  # noqa: D102
        self.actor_buffer.factor = None
        self.actor_buffer.step = 0

        self.actor_buffer.obs.fill(0.0)
        self.actor_buffer.actions.fill(0.0)
        self.actor_buffer.action_log_probs.fill(0.0)
        self.actor_buffer.rnn_states.fill(0.0)
        self.actor_buffer.masks.fill(1.0)
        self.actor_buffer.active_masks.fill(1.0)
        self.actor_buffer.infos.fill({})

        if self.actor_buffer.available_actions is not None:
            self.actor_buffer.available_actions.fill(1.0)

    def get_actions(self, step: int, **kwargs) -> tuple:  # type: ignore[override] # noqa: D102
        return self.harl_actor.get_actions(
            obs=self.actor_buffer.obs[step],
            rnn_states_actor=self.actor_buffer.rnn_states[step],
            masks=self.actor_buffer.masks[step],
            available_actions=self.actor_buffer.available_actions[step] if self.discrete else None,
            deterministic=kwargs.get("deterministic", False),
        )


class TorchOnPolicyHARLAgent(TorchOnPolicyMARLAgent):
    """Torch agent class for on-policy HARL algorithms."""

    platforms: Mapping[PlatformID, TorchOnPolicyBaseHARLActor]
    default_config: type[HARLAgentConfig] = HARLAgentConfig

    @property
    def actor(self) -> list[TorchOnPolicyBaseHARLActor]:  # type: ignore[override] # noqa: D102
        return list(self.platforms.values())

    def train(self) -> dict:
        """Train the model.

        Returns
        -------
        dict
            The training infos
        """
        train_infos = {}

        # factor is used for considering updates made by previous agents
        factor = np.ones((self.runner_config["episode_length"], self.runner_config["n_rollout_threads"], 1), dtype=np.float32)

        # compute advantages
        advantages = self.critic_buffer.returns[:-1] - (
            self.critic_buffer.value_preds[:-1]
            if self.value_normalizer is None
            else self.value_normalizer.denormalize(self.critic_buffer.value_preds[:-1])
        )

        # normalize advantages for FP
        if self.state_type == "FP":
            active_masks_collector = [agent.actor_buffer.active_masks for agent in self.actors.values()]
            active_masks_array = np.stack(active_masks_collector, axis=2)
            advantages_copy = advantages.copy()
            advantages_copy[active_masks_array[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        # update critic
        critic_train_info = self.critic.train(self.critic_buffer, self.value_normalizer)

        platform_order = list(enumerate(self.actors))
        if not self.algorithm_config["fixed_order"]:
            np.random.default_rng().shuffle(platform_order)

        for platform_idx, platform_id in platform_order:
            self.actors[platform_id].actor_buffer.update_factor(factor)

            # the following reshaping combines the first two dimensions (i.e. episode_length and n_rollout_threads) to form a batch
            available_actions = (
                None
                if self.actors[platform_id].actor_buffer.available_actions is None
                else self.actors[platform_id]
                .actor_buffer.available_actions[:-1]
                .reshape(-1, *self.actors[platform_id].actor_buffer.available_actions.shape[2:])
            )

            # compute action log probs for the actor before update.
            old_actions_logprob, _, _ = self.actors[platform_id].evaluate_actions(
                obs=self.actors[platform_id].actor_buffer.obs[:-1].reshape(-1, *self.actors[platform_id].actor_buffer.obs.shape[2:]),
                rnn_states_actor=self.actors[platform_id]
                .actor_buffer.rnn_states[0:1]
                .reshape(-1, *self.actors[platform_id].actor_buffer.rnn_states.shape[2:]),
                action=self.actors[platform_id].actor_buffer.actions.reshape(-1, *self.actors[platform_id].actor_buffer.actions.shape[2:]),
                masks=self.actors[platform_id].actor_buffer.masks[:-1].reshape(-1, *self.actors[platform_id].actor_buffer.masks.shape[2:]),
                available_actions=available_actions,
                active_masks=self.actors[platform_id]
                .actor_buffer.active_masks[:-1]
                .reshape(-1, *self.actors[platform_id].actor_buffer.active_masks.shape[2:]),
            )

            # update actor
            if self.state_type == "EP":
                _loss, actor_train_info = self.actors[platform_id].learn(advantages=advantages.copy(), state_type="EP")  # type: ignore[call-arg]
            elif self.state_type == "FP":
                _loss, actor_train_info = self.actors[platform_id].learn(advantages=advantages[:, :, platform_idx].copy(), state_type="FP")  # type: ignore[call-arg]

            # compute action log probs for updated agent
            new_actions_logprob, _, _ = self.actors[platform_id].evaluate_actions(
                obs=self.actors[platform_id].actor_buffer.obs[:-1].reshape(-1, *self.actors[platform_id].actor_buffer.obs.shape[2:]),
                rnn_states_actor=self.actors[platform_id]
                .actor_buffer.rnn_states[0:1]
                .reshape(-1, *self.actors[platform_id].actor_buffer.rnn_states.shape[2:]),
                action=self.actors[platform_id].actor_buffer.actions.reshape(-1, *self.actors[platform_id].actor_buffer.actions.shape[2:]),
                masks=self.actors[platform_id].actor_buffer.masks[:-1].reshape(-1, *self.actors[platform_id].actor_buffer.masks.shape[2:]),
                available_actions=available_actions,
                active_masks=self.actors[platform_id]
                .actor_buffer.active_masks[:-1]
                .reshape(-1, *self.actors[platform_id].actor_buffer.active_masks.shape[2:]),
            )

            # update factor for next agent
            factor *= _t2n(
                getattr(torch, self.runner_config["action_aggregation"])(
                    torch.exp(new_actions_logprob - old_actions_logprob),
                    dim=-1,
                ).reshape(self.runner_config["episode_length"], self.runner_config["n_rollout_threads"], 1),
            )
            train_infos[platform_id] = {"training_stats/actor": actor_train_info, "training_stats/critic": critic_train_info}

        return train_infos

    def buffers_to_sample_batch(self, envs: list[int] | None = None) -> dict[PlatformID, SampleBatch]:
        """Convert the experiences stored in the buffers to a SampleBatch.

        Parameters
        ----------
        envs : list[int] | None, optional
            The list of envs to buffer, :data:`python:None`

        Returns
        -------
        dict[PlatformID, SampleBatch]
            The updated sample batch
        """
        reserved_keys = {
            Columns.OBS,
            Columns.NEXT_OBS,
            Columns.REWARDS,
            Columns.ADVANTAGES,
            Columns.VF_PREDS,
            Columns.VALUE_TARGETS,
            Columns.ACTION_LOGP,
            Columns.ACTIONS,
            Columns.EPS_ID,
            Columns.STATE_IN,
            Columns.STATE_OUT,
            Columns.INFOS,
        }
        platform_batches = {}
        if envs is None:
            envs = list(range(self.n_rollout_threads))
        n_envs = len(envs)
        rewards = np.concatenate(np.split(self.critic_buffer.rewards[:, envs], n_envs, axis=1)).squeeze(1)
        vf_preds = np.concatenate(np.split(self.critic_buffer.value_preds[:-1, envs], n_envs, axis=1)).squeeze(1)
        seq_lens = np.full(self.critic_buffer.episode_length * n_envs, fill_value=self.critic_buffer.recurrent_n)
        episode_ids = None

        for platform, actor in self.actors.items():
            rnn_states_in = np.concatenate(np.split(actor.actor_buffer.rnn_states[:-1, envs], n_envs, axis=1)).squeeze(1)
            rnn_states_out = np.concatenate(np.split(actor.actor_buffer.rnn_states[1:, envs], n_envs, axis=1)).squeeze(1)
            infos = np.concatenate(np.split(actor.actor_buffer.infos[:-1, envs], n_envs, axis=1))
            all_info_keys = [set(inf[0].keys()) for inf in infos]
            if all_info_keys:
                # info provided for the entire episode
                eps_info_keys = all_info_keys[0].intersection(*all_info_keys[1:])
                eps_info_keys -= reserved_keys
                # info provided only for certain steps of the episode
                step_info_keys = all_info_keys[0].union(*all_info_keys[1:]) - eps_info_keys
                step_info_keys -= reserved_keys
            else:
                step_info_keys = set()
                eps_info_keys = set()

            step_infos = np.array([
                {key: inf[0][key] for key in step_info_keys if key in inf[0] and key != Columns.REWARDS} for inf in infos
            ])
            eps_infos = {key: np.array([inf[0][key] for inf in infos]) for key in eps_info_keys}

            episode_ts = eps_infos.pop("step")
            if episode_ids is None:
                new_episodes = np.concatenate([np.argwhere(episode_ts == 0).squeeze(), np.array([episode_ts.shape[0]])])
                episode_lens = new_episodes[1:] - new_episodes[:-1]
                episode_ids = np.concatenate([
                    np.full(episode_len, fill_value=episode_hash(step + eps_idx), dtype=np.int64)
                    for eps_idx, (episode_len, step) in enumerate(zip(episode_lens, new_episodes[:-1], strict=True))
                ])

            batch = {
                Columns.T: episode_ts,
                Columns.OBS: np.concatenate(np.split(actor.actor_buffer.obs[:-1, envs], n_envs, axis=1)).squeeze(1),
                Columns.NEXT_OBS: np.concatenate(np.split(actor.actor_buffer.obs[1:, envs], n_envs, axis=1)).squeeze(1),
                Columns.ACTIONS: np.concatenate(np.split(actor.actor_buffer.actions[:, envs], n_envs, axis=1)).squeeze(1),
                Columns.REWARDS: rewards.copy(),
                Columns.VF_PREDS: vf_preds.copy(),
                Columns.ACTION_LOGP: np.concatenate(np.split(actor.actor_buffer.action_log_probs[:, envs], n_envs, axis=1)).squeeze(1),
                Columns.SEQ_LENS: seq_lens.copy(),
                Columns.STATE_IN: rnn_states_in.copy(),
                Columns.STATE_OUT: rnn_states_out.copy(),
                Columns.INFOS: step_infos,
                Columns.EPS_ID: episode_ids.copy(),
                "masks": np.concatenate(np.split(actor.actor_buffer.masks[:-1, envs], n_envs, axis=1)).squeeze(1),
                "active_masks": np.concatenate(np.split(actor.actor_buffer.active_masks[:-1, envs], n_envs, axis=1)).squeeze(1),
                **eps_infos,
            }
            if actor.actor_buffer.available_actions is not None:
                batch["available_actions"] = np.concatenate(
                    np.split(actor.actor_buffer.available_actions[:-1, envs], n_envs, axis=1),
                ).squeeze(1)
            platform_batches[platform] = SampleBatch(**batch)
        return platform_batches
