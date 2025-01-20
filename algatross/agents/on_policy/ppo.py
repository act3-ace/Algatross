"""A module containing clean-rl style PPO agents, rollout, and training operations."""

# torch.autograd.set_detect_anomaly(True)  # noqa: ERA001
import logging

from collections.abc import Sequence
from typing import Any

import numpy as np

from ray.rllib.policy.sample_batch import SampleBatch

import torch
import torch.autograd

from torch import nn
from torch.distributions.distribution import Distribution
from torch.optim.adam import Adam

import gymnasium

from algatross.agents.torch_base import TorchActorCriticModule, TorchBaseAgent
from algatross.environments.utilities import compute_advantage, explained_var
from algatross.models.layers import BiasLayer
from algatross.utils.exceptions import InitNotCalledError

try:
    from ray.rllib.core import Columns  # type: ignore[attr-defined]
except ImportError:
    from algatross.utils.compatibility import Columns  # type: ignore[assignment, unused-ignore]


class TorchPPOAgent(TorchBaseAgent, TorchActorCriticModule):
    """The main PPO agent which is used to make value and action predictions.

    Parameters
    ----------
    obs_space : gymnasium.spaces.Space
        The observation space for each platform of this agent.
    act_space : gymnasium.spaces.Space
        The action space for each platform of this agent.
    seed : Any
        The seed for randomness
    critic_outs : int, optional
        The number of outputs for the critic network, default is 1.
    optimizer_class : str | type[torch.optim.Optimizer], optional
        The optimizer to use with this module, default is :class:`~torch.optim.adam.Adam`
    optimizer_kwargs : dict | None, optional
        Keyword arguments to pass to the optimizer constructor, default is :data:`python:None`
    shared_encoder : bool, optional
        Whether the actor and critic share a single encoder, default is :data:`python:False`.
    free_log_std : bool, optional
        Whether the actor uses a single free log standard deviation parameter, default is :data:`python:True`.
    entropy_coeff : float, optional
        The coefficient to apply to the entropy term in the loss, default is 0.0
    kl_target : float, optional
        The target value for KL-divergence, default is 0.2.
    kl_coeff : float, optional
        The initial value for the coefficient of the KL-divergence term in the loss, default is 0.2.
    vf_coeff : float, optional
        The coefficient for the value network loss term, default is 1.0.
    logp_clip_param : float, optional
        The clip parameter for the log-probabilities, default is 0.2.
    vf_clip_param : float | None, optional
        The clip parameter for the value function loss, default is :data:`python:None`.
    actor_grad_clip : float | None, optional
        The clip parameter for the actor gradients, default is :data:`python:None`,
    critic_grad_clip : float | None, optional
        The clip parameter for the critic gradients, default is :data:`python:None`
    `**kwargs`
        Additional keyword arguments.
    """

    initial_entropy_coeff: float = 0.0
    """The initial coefficient to apply to the entropy term in the loss."""
    initial_kl_target: float = 0.2
    """The initial target value for KL-divergence."""
    initial_kl_coeff: float = 0.2
    """The initial value for the coefficient of the KL-divergence term in the loss."""
    initial_vf_coeff: float = 1.0
    """The initial coefficient for the value network loss term."""
    initial_logp_clip_param: float = 0.2
    """The initial clip parameter for the log-probabilities."""
    initial_vf_clip_param: float | None = None
    """The clip parameter for the value function loss."""
    actor_grad_clip: float | None = None
    """The clip parameter for the actor gradients."""
    critic_grad_clip: float | None = None
    """The clip parameter for the critic gradients."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        obs_space: gymnasium.spaces.Space,
        act_space: gymnasium.spaces.Space,
        seed: Any,  # noqa: ANN401
        critic_outs: int = 1,
        optimizer_class: str | type[torch.optim.Optimizer] = Adam,
        optimizer_kwargs: dict | None = None,
        shared_encoder: bool = False,
        free_log_std: bool = True,
        entropy_coeff: float = 0.0,
        kl_target: float = 0.2,
        kl_coeff: float = 0.2,
        vf_coeff: float = 1.0,
        logp_clip_param: float = 0.2,
        vf_clip_param: float | None = None,
        actor_grad_clip: float | None = None,
        critic_grad_clip: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            obs_space=obs_space,
            act_space=act_space,
            critic_outs=critic_outs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            shared_encoder=shared_encoder,
            free_log_std=free_log_std,
            seed=seed,
            **kwargs,
        )

        self.initial_entropy_coeff = entropy_coeff
        self.initial_kl_coeff = kl_coeff

        self.initial_kl_target = kl_target
        self.initial_vf_coeff = vf_coeff
        self.initial_logp_clip_param = logp_clip_param
        self.initial_vf_clip_param = 0 if vf_clip_param is None else vf_clip_param

        self.actor_grad_clip = actor_grad_clip
        self.critic_grad_clip = critic_grad_clip

        if actor_grad_clip:
            # clip during backprop:
            for p in self.actor.parameters():
                p.register_hook(lambda grad: torch.clamp(grad, -actor_grad_clip, actor_grad_clip))

        if critic_grad_clip:
            for p in self.critic.parameters():
                p.register_hook(lambda grad: torch.clamp(grad, -critic_grad_clip, critic_grad_clip))

        self.reset_buffers()
        logger = logging.getLogger("ray")

        def _grad_to_str(grad):
            return f"min={grad[0].min()}, max={grad[0].max()}, {'' if len(grad) == 1 else grad[1:]}, dtype={grad[0].dtype}"

        def _module_to_str(module):
            return f"module: min={[torch.min(k.data) for k in module.parameters()]}, max={[torch.max(k.data) for k in module.parameters()]}"

        def _save_output(module, grad_input, grad_output):
            logger.debug(module)
            logger.debug(f"grad_input:  {_grad_to_str(grad_input)}")
            logger.debug(f"grad_output: {_grad_to_str(grad_output)}")
            logger.debug(f"module: {_module_to_str(module)}")

        def _critic_output(module, grad_input, grad_output):
            logger.debug("I am critic!!")
            _save_output(module, grad_input, grad_output)

        def _actor_output(module, grad_input, grad_output):
            logger.debug("I am actor!!")
            _save_output(module, grad_input, grad_output)

        if self.debug:
            for module in self.critic.modules():
                module.register_full_backward_hook(_critic_output)

            for module in self.actor.modules():
                module.register_full_backward_hook(_actor_output)

    def build_networks(  # noqa: D102
        self,
        *,
        observation_size: int | Sequence[int],
        value_network_outs: int,
        action_size: int,
        shared_encoder: bool = False,
        free_log_std: bool = True,
        **kwargs,
    ):
        actor_encoder = nn.Sequential(
            *self._make_encoder_layers(observation_size),
            self._layer_init(nn.Linear(64, 64), generator=self.torch_generator),
            nn.Tanh(),
        )
        critic_encoder = (
            actor_encoder
            if shared_encoder
            else nn.Sequential(
                *self._make_encoder_layers(observation_size),
                self._layer_init(nn.Linear(64, 64), generator=self.torch_generator),
                nn.Tanh(),
            )
        )

        if free_log_std:
            actor_head = nn.Sequential(
                self._layer_init(nn.Linear(64, int(action_size) // 2), generator=self.torch_generator),
                BiasLayer(int(action_size) // 2),
            )
        else:
            actor_head = nn.Sequential(self._layer_init(nn.Linear(64, int(action_size)), generator=self.torch_generator))

        critic_head = self._layer_init(nn.Linear(64, value_network_outs), generator=self.torch_generator)

        self._actor = nn.Sequential(actor_encoder, actor_head)
        self._critic = nn.Sequential(critic_encoder, critic_head)

    def _make_encoder_layers(self, observation_size: int | Sequence[int]) -> list[nn.Module]:
        encoder_layers: list[nn.Module] = []
        if isinstance(observation_size, int):
            encoder_layers.extend([self._layer_init(nn.Linear(int(observation_size), 64), generator=self.torch_generator), nn.Tanh()])
        elif len(observation_size) == 2:  # noqa: PLR2004
            encoder_layers.extend([
                nn.Conv1d(observation_size[0], observation_size[0], (3,)),
                nn.MaxPool1d((3,), 1),
                nn.LazyBatchNorm1d(),
                nn.Flatten(),
                nn.LazyLinear(64),
                nn.Tanh(),
            ])
        elif len(observation_size) == 3:  # noqa: PLR2004
            encoder_layers.extend([
                nn.Conv2d(observation_size[0], observation_size[0], (3, 3)),
                nn.MaxPool2d((3, 3), 1),
                nn.LazyBatchNorm2d(),
                nn.Flatten(),
                nn.LazyLinear(64),
                nn.Tanh(),
            ])
        elif len(observation_size) == 4:  # noqa: PLR2004
            encoder_layers.extend([
                nn.Conv3d(observation_size[0], observation_size[0], (3, 3, 3)),
                nn.MaxPool3d((3, 3, 3), 1),
                nn.LazyBatchNorm3d(),
                nn.Flatten(),
                nn.LazyLinear(64),
                nn.Tanh(),
            ])
        else:
            msg = f"Could not make a linear or convolutional encoder with input: {observation_size}"
            raise ValueError(msg)
        return encoder_layers

    def _check_logits(self, logits):  # noqa: PLR6301
        if torch.any(torch.isnan(logits)):
            logger = logging.getLogger("ray")
            logger.warning("Found NaN in logits!")

        return logits

    def get_values(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Get the critic network value predictions.

        Parameters
        ----------
        x : torch.Tensor
            The observation or state for which we would like to produce an action.
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        torch.Tensor
            The value network prediction from the critic network.

        Raises
        ------
        InitNotCalledError
            If the agents :python:`__init__` method was not called.
        """
        # TODO: make sure torch.manual_seed is set before calling
        if not self._init_called:
            raise InitNotCalledError(self)
        return self.critic(x)

    def get_action_dist(self, x: torch.Tensor, logits: torch.Tensor | None = None, **kwargs) -> Distribution:
        """Generate the action distribution from the input.

        Parameters
        ----------
        x : torch.Tensor
            The observation or state for which we would like to produce an action.
        logits : torch.Tensor | None, optional
            The action distribution logits, :data:`python:None`
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        Distribution
            The action distribution constructed from the actor networks prediction.

        Raises
        ------
        InitNotCalledError
            If the agents :python:`__init__` method was not called
        """
        # TODO: make sure torch.manual_seed is set before calling
        if not self._init_called:
            raise InitNotCalledError(self)
        logits = self.actor(x) if logits is None else logits
        if self.debug:
            logits = self._check_logits(logits)

        return self.dist_class(logits)  # type: ignore[arg-type]

    def get_actions(
        self,
        x: torch.Tensor | dict[str, torch.Tensor],
        logits: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, Distribution]:
        """Get the action distribution and value predictions from the agent.

        Parameters
        ----------
        x : torch.Tensor
            The input observation
        logits : torch.Tensor | None, optional
            The logits used to construct the action distribution, :data:`python:None`, in which case a new action distribution is
            constructed from the actor network output.
        `*args`
            Additional positional arguments.
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        tuple[torch.Tensor, Distribution]
            A tuple of actions, value predictions, and the action distribution used to generate the actions.

        Raises
        ------
        InitNotCalledError
            If the agents :python:`__init__` method was not called.
        """
        if not self._init_called:
            raise InitNotCalledError(self)

        if logits is not None and self.debug:
            logits = self._check_logits(logits)

        dist = self.get_action_dist(x) if logits is None else self.dist_class(logits)  # type: ignore[arg-type]
        action = dist.sample() if self.training else dist.mean
        action = self._act_type_map(action)
        return action, dist

    def train(self, mode: bool = True) -> None:  # noqa: D102
        self.actor.train(mode)
        self.critic.train(mode)

    def get_actions_and_values(
        self,
        x: torch.Tensor,
        action: torch.Tensor | None = None,
        logits: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, Distribution]:
        """Get the action distribution and value predictions from the agent.

        Parameters
        ----------
        x : torch.Tensor
            The input observation
        action : torch.Tensor | None, optional
            The actions for which we want to compute the logits, :data:`python:None`, in which case new actions and distributions are
            created from the input observations or logits if provided
        logits : torch.Tensor | None, optional
            The logits used to construct the action distribution, :data:`python:None`, in which case a new action distribution is
            constructed from the actor network output.
        action_mask : torch.Tensor, optional
            The valid action mask for discrete action spaces, default is :data:`python:None`
        `*args`
            Additional positional arguments.
        `**kwargs`
            Additional keword arguments.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, Distribution]
            A tuple of actions, value predictions, and the action distribution used to generate the actions.

        Raises
        ------
        InitNotCalledError
            If the agents :python:`__init__` method was not called.
        """
        if not self._init_called:
            raise InitNotCalledError(self)

        if logits is not None and self.debug:
            logits = self._check_logits(logits)
        dist = self.get_action_dist(x) if logits is None else self.dist_class(logits)  # type: ignore[arg-type]
        value = self.critic(x)

        if action_mask is not None:
            logits = action_mask.log() + logits
        if action is None:
            action, dist = self.get_actions(x=x, logits=logits, **kwargs)
        else:
            action, dist = (
                self._act_type_map(action),
                self.get_action_dist(x=x, logits=logits, **kwargs) if logits is None else self.dist_class(logits),  # type: ignore[arg-type]
            )

        return action, value, dist

    def update_kl(self, mean_kl: torch.Tensor, **kwargs):
        """Update the KL loss coefficient.

        Parameters
        ----------
        mean_kl : torch.Tensor
            The mean KL divergence from the most recent training epoch
        `**kwargs`
            Additional keyword arguments.

        Raises
        ------
        InitNotCalledError
            If the agents :python:`__init__` method was not called.
        """
        if not self._init_called:
            raise InitNotCalledError(self)
        if mean_kl > 2.0 * self.kl_target:
            self.kl_coeff *= 1.5
        elif mean_kl < 0.5 * self.kl_target:
            self.kl_coeff *= 0.5

    def loss(self, train_batch: SampleBatch, **kwargs) -> tuple[torch.Tensor, dict[str, np.ndarray]]:
        """
        Calculate the loss from the training batch for this agent.

        Parameters
        ----------
        train_batch : SampleBatch
            The training batch of data for this agent.
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        tuple[torch.Tensor, dict[str, np.ndarray]]

            - loss : :class:`~torch.Tensor`
                The training loss
            - infos : :class:`dict` [ :class:`str` , :class:`~np.ndarray`]
                The training info or stats

        Raises
        ------
        InitNotCalledError
            If the agents :python:`__init__` method was not called.
        """
        if not self._init_called:
            raise InitNotCalledError(self)
        return torch_ppo_loss(agent=self, train_batch=train_batch, **kwargs)

    def postprocess_batch(self, batch: SampleBatch, **kwargs) -> SampleBatch:  # noqa: D102
        super().postprocess_batch(batch, **kwargs)
        batch[Columns.ADVANTAGES] = (batch[Columns.ADVANTAGES] - batch[Columns.ADVANTAGES].mean()) / max(
            batch[Columns.ADVANTAGES].std(),
            1e-8 if batch[Columns.ADVANTAGES].dtype in {torch.float32, np.float32} else 1e-16,
        )
        return batch

    def process_episode(  # noqa: D102
        self,
        episode_batch: SampleBatch,
        rewards: np.ndarray,
        gamma: float = 0.99,
        gae_lambda: float = 1.0,
        **kwargs,
    ) -> SampleBatch:
        episode_batch = super().process_episode(episode_batch=episode_batch, rewards=rewards, **kwargs)
        adv, val = compute_advantage(
            rewards,
            episode_batch[Columns.VF_PREDS],
            (
                np.expand_dims(episode_batch[Columns.VF_PREDS][-1], axis=0)
                if not episode_batch[SampleBatch.DONES].any()
                else np.zeros([1, *episode_batch[Columns.VF_PREDS][-1].shape])
            ),
            gae_lambda=gae_lambda,
            gamma=gamma,
        )
        episode_batch[Columns.ADVANTAGES] = adv
        episode_batch[Columns.VALUE_TARGETS] = val
        return episode_batch

    def reset_buffers(self):
        """Reset the parameter buffers to their initial values."""
        # register entropy and KL coeff as buffers for updating
        self.register_buffer("entropy_coeff", torch.tensor(self.initial_entropy_coeff))
        self.register_buffer("kl_coeff", torch.tensor(self.initial_kl_coeff))

        self.register_buffer("kl_target", torch.tensor(self.initial_kl_target))
        self.register_buffer("vf_coeff", torch.tensor(self.initial_vf_coeff))
        self.register_buffer("logp_clip_param", torch.tensor(self.initial_logp_clip_param))
        self.register_buffer("vf_clip_param", torch.tensor(self.initial_vf_clip_param))

    def load_flat_params(self, flat_params: torch.Tensor | np.ndarray, **kwargs):  # noqa: D102
        super().load_flat_params(flat_params)
        self.reset_buffers()

    def post_training_hook(self, **kwargs) -> dict:  # noqa: D102
        results = super().post_training_hook(**kwargs)
        self.update_kl(mean_kl=kwargs.get("training_stats", {}).get("kl_loss", np.array(0, dtype=np.float32)).mean())
        return results


def torch_ppo_loss(agent: TorchPPOAgent, train_batch: SampleBatch, **kwargs) -> tuple[torch.Tensor, dict[str, np.ndarray]]:
    """PPO loss function.

    Parameters
    ----------
    agent : TorchPPOAgent
        The agent to use when calculating the loss
    train_batch : SampleBatch
        The batch to use when calculating the loss
    `**kwargs`
        Additional keyword arguments.

    Returns
    -------
    tuple[torch.Tensor, dict[str, np.ndarray]]
        The loss and training info dict
    """
    agent.train(True)
    prev_dist = agent.dist_class(train_batch[Columns.ACTION_DIST_INPUTS])
    _, curr_val, curr_dist = agent.get_actions_and_values(x=train_batch[Columns.OBS].requires_grad_(), action=train_batch[Columns.ACTIONS])
    curr_logprobs = curr_dist.log_prob(train_batch[Columns.ACTIONS])
    curr_entropy = curr_dist.entropy()

    # entropy
    entropy_bonus = curr_entropy.neg() if agent.entropy_coeff else torch.tensor(0.0, device=curr_logprobs.device, dtype=curr_logprobs.dtype)

    # surrogate loss
    ratio = (curr_logprobs - train_batch[Columns.ACTION_LOGP]).exp()
    ratio_clip = torch.clamp(ratio, 1 - agent.logp_clip_param, 1 + agent.logp_clip_param)

    surrogate_loss = torch.minimum(train_batch[Columns.ADVANTAGES] * ratio_clip, train_batch[Columns.ADVANTAGES] * ratio).neg()

    # kl loss
    kl_loss = (
        torch.distributions.kl_divergence(prev_dist, curr_dist)
        if agent.kl_coeff > 0
        else torch.tensor(0.0, device=curr_logprobs.device, dtype=curr_logprobs.dtype)
    )

    # vf loss
    vf_loss = nn.functional.mse_loss(curr_val, train_batch[Columns.VALUE_TARGETS], reduction="none")
    vf_loss = torch.clamp(vf_loss, torch.tensor(0.0), agent.vf_clip_param) if agent.vf_clip_param > 0 else vf_loss

    total_loss: torch.Tensor = (surrogate_loss + entropy_bonus + (kl_loss * agent.kl_coeff) + (vf_loss * agent.vf_coeff)).mean()

    stats: dict[str, np.ndarray] = {}
    with torch.no_grad():
        stacked = (
            torch.stack([
                vf_loss.mean(),
                surrogate_loss.mean(),
                total_loss,
                kl_loss.mean(),
                curr_entropy.mean(),
                agent.kl_coeff,
                agent.entropy_coeff,
                explained_var(curr_val, train_batch[Columns.VALUE_TARGETS]),
            ])
            .cpu()
            .numpy()
        )

        stats["critic_loss"] = stacked[0]
        stats["actor_loss"] = stacked[1]
        stats["total_loss"] = stacked[2]
        stats["kl_loss"] = stacked[3]
        stats["entropy"] = stacked[4]
        stats["kl_coeff"] = stacked[5]
        stats["entropy_coeff"] = stacked[6]
        stats["vf_explained_var"] = stacked[6]

    return total_loss, stats
