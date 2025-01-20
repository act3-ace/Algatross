"""A module containing clean-rl style PPO agents, rollout, and training operations."""

from collections.abc import Sequence

import numpy as np

from ray.rllib.policy.sample_batch import SampleBatch

import torch

from torch import nn
from torch.distributions import Distribution

import gymnasium

from algatross.agents.torch_base import TorchBaseAgent
from algatross.environments.utilities import explained_var
from algatross.models.layers import BiasLayer

try:
    from ray.rllib.core import Columns  # type: ignore[attr-defined, unused-ignore]
except ImportError:
    from algatross.utils.compatibility import Columns  # type: ignore[assignment, unused-ignore]


class MAPPOCentralCritic(nn.Module):
    """
    PPOAgent the main PPO agent which is used to make value and action predictions.

    Parameters
    ----------
    centralized_obs_space : gymnasium.spaces.Space
        The centralized observation space for this critic
    critic_outs : int, optional
        The number of outputs for this critic, default is 1
    optimizer_class : str | type[torch.optim.Optimizer], optional
        The optimizer to use with this module, default is :class:`~torch.optim.adam.Adam`
    optimizer_kwargs : dict | None, optional
        Keyword arguments to pass to the optimizer constructor, default is :data:`python:None`
    `*args`
        Additional positional arguments.
    `**kwargs`
        Additional keyword arguments.
    """

    critic: nn.Module
    """The centralized critic module."""
    critic_outs: int = 1
    """The output size of the critic network."""
    optimizer_class: type[torch.optim.Optimizer] = torch.optim.Adam
    """The class of the optimizer used by this network."""
    optimizer_kwargs: dict
    """Keyword arguments to pass to the optimizer constructor."""
    optimizer: torch.optim.Optimizer
    """The optimizer for this network."""

    def __init__(
        self,
        centralized_obs_space: gymnasium.spaces.Space,
        critic_outs: int = 1,
        optimizer_class: str | type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: dict | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        if isinstance(centralized_obs_space, gymnasium.spaces.Box):
            cent_in_size = np.prod(centralized_obs_space.shape)
        elif isinstance(centralized_obs_space, gymnasium.spaces.Discrete):
            cent_in_size = centralized_obs_space.n
        elif isinstance(centralized_obs_space, gymnasium.spaces.MultiDiscrete):
            cent_in_size = np.prod(centralized_obs_space.nvec)

        critic_encoder = nn.Sequential(
            self._layer_init(nn.Linear(int(cent_in_size), 64)),
            nn.Tanh(),
            self._layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )
        critic_head = self._layer_init(nn.Linear(64, critic_outs))

        self.critic = nn.Sequential(critic_encoder, critic_head)

        optimizer_class = getattr(torch.optim, optimizer_class) if isinstance(optimizer_class, str) else optimizer_class
        optimizer_kwargs = optimizer_kwargs or {}
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.reset_optimizer()

    @staticmethod
    def _layer_init(layer: nn.Module, std: float | np.ndarray | None = None, bias_const: float = 0.0) -> nn.Module:
        if std is None:
            std = np.sqrt(2)
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_values(self, x: torch.Tensor) -> torch.Tensor:
        """Get the critic network value predictions.

        Parameters
        ----------
        x : torch.Tensor
            The observation or state for which we would like to produce an action.

        Returns
        -------
        torch.Tensor
            The value network prediction from the critic network.
        """
        return self.critic(x)

    def reset_optimizer(self):
        """
        Reset the optimizer to its initial state.

        Reinitializes the optimizer so the gradient buffers are cleared. This is necessary for optimizers with momentum
        whenever the network parameters are forcibly changed.
        """
        self.optimizer = self.optimizer_class(list(self.parameters()), **self.optimizer_kwargs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.critic(inputs)


class TorchMAPPOAgent(TorchBaseAgent):
    """
    PPOAgent the main PPO agent which is used to make value and action predictions.

    Parameters
    ----------
    obs_space : gymnasium.spaces.Space
        The observation space for each platform of this agent.
    act_space : gymnasium.spaces.Space
        The action space for each platform of this agent.
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

    dist_class: type[Distribution]
    """The distribution class to use for action sampling."""

    def __init__(
        self,
        *,
        obs_space: gymnasium.spaces.Space,
        act_space: gymnasium.spaces.Space,
        centralized_critic: nn.Module,
        critic_outs: int = 1,
        shared_encoder: bool = False,
        free_log_std: bool = True,
        entropy_coeff: float = 0.0,
        kl_target: float = 0.2,
        kl_coeff: float = 0.2,
        vf_coeff: float = 1.0,
        logp_clip_param: float = 0.2,
        vf_clip_param: float | None = None,
        optimizer_class: str | type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: dict | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            obs_space=obs_space,
            act_space=act_space,
            critic_outs=critic_outs,
            shared_encoder=shared_encoder,
            free_log_std=free_log_std,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            **kwargs,
        )

        # register entropy and KL coeff as buffers for updating
        self.register_buffer("entropy_coeff", torch.tensor(entropy_coeff))
        self.register_buffer("kl_coeff", torch.tensor(kl_coeff))

        self.register_buffer("kl_target", torch.tensor(kl_target))
        self.register_buffer("vf_coeff", torch.tensor(vf_coeff))
        self.register_buffer("logp_clip_param", torch.tensor(logp_clip_param))
        self.register_buffer("vf_clip_param", torch.tensor(0 if vf_clip_param is None else vf_clip_param))

        self.centralized_critic = centralized_critic

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
        actor_encoder = nn.Sequential(*self._make_encoder_layers(observation_size), self._layer_init(nn.Linear(64, 64)), nn.Tanh())
        critic_encoder = (
            actor_encoder
            if shared_encoder
            else nn.Sequential(*self._make_encoder_layers(observation_size), self._layer_init(nn.Linear(64, 64)), nn.Tanh())
        )

        if free_log_std:
            actor_head = nn.Sequential(self._layer_init(nn.Linear(64, int(action_size) // 2)), BiasLayer(int(action_size) // 2))
        else:
            actor_head = nn.Sequential(self._layer_init(nn.Linear(64, int(action_size))))

        critic_head = self._layer_init(nn.Linear(64, value_network_outs))

        self._actor = nn.Sequential(actor_encoder, actor_head)
        self._critic = nn.Sequential(critic_encoder, critic_head)

    def _make_encoder_layers(self, observation_size: int | Sequence[int]) -> list[nn.Module]:
        encoder_layers: list[nn.Module] = []
        if isinstance(observation_size, int):
            encoder_layers.extend([self._layer_init(nn.Linear(int(observation_size), 64)), nn.Tanh()])
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

    def get_values(self, x: torch.Tensor, centralized_obs: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        """Get the critic network value predictions.

        Parameters
        ----------
        x : torch.Tensor
            The observation or state for which we would like to produce an action.
        centralized_obs : torch.Tensor, optional
            The centralized observations, default is :data:`python:None`
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        torch.Tensor
            The value network prediction from the critic network.
        """
        values = self.critic(x)
        if centralized_obs is not None:
            values += self.centralized_critic(centralized_obs)
        return values

    def get_action_dist(self, x: torch.Tensor, **kwargs) -> Distribution:
        """Generate the action distribution from the input.

        Parameters
        ----------
        x : torch.Tensor
            The observation or state for which we would like to produce an action.
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        Distribution
            The action distribution constructed from the actor networks prediction.
        """
        logits = self.actor(x)
        return self.dist_class(logits)

    def get_actions_and_values(
        self,
        x: torch.Tensor,
        action: torch.Tensor | None = None,
        logits: torch.Tensor | None = None,
        centralized_obs: torch.Tensor | None = None,
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
        centralized_obs : torch.Tensor, optional
            The centralized observations, default is :data:`python:None`
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, Distribution]
            A tuple of actions, value predictions, and the action distribution used to generate the actions.
        """
        dist = self.get_action_dist(x) if logits is None else self.dist_class(logits)  # type: ignore[arg-type]
        value = self.get_values(x, centralized_obs=centralized_obs)
        if action is None and self.training:
            action = dist.sample()
        elif action is None:
            action = dist.mean
        action = self._act_type_map(action)
        return action, value, dist

    def update_kl(self, mean_kl: torch.Tensor):
        """Update the KL loss coefficient.

        Parameters
        ----------
        mean_kl : torch.Tensor
            The mean KL divergence from the most recent training epoch
        """
        if mean_kl > 2.0 * self.kl_target:
            self.kl_coeff *= 1.5
        elif mean_kl < 0.5 * self.kl_target:
            self.kl_coeff *= 0.5

    def reset_optimizer(self):
        """
        Reset the optimizer to its initial state.

        Reinitializes the optimizer so the gradient buffers are cleared. This is necessary for optimizers with momentum
        whenever the network parameters are forcibly changed.
        """
        self.optimizer = self.optimizer_class([*list(self.actor.parameters()), *list(self.critic.parameters())], **self.optimizer_kwargs)
        self.centralized_critic.reset_optimizer()
        self.centralized_optimizer = self.centralized_critic.optimizer

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
        """
        return torch_ma_ppo_loss(agent=self, train_batch=train_batch, **kwargs)

    @property
    def actor(self) -> nn.Module:
        """
        The actor for this agent.

        Returns
        -------
        nn.Module
            The actor for this agent
        """
        return self._actor

    @property
    def critic(self) -> nn.Module:
        """
        The central critic network.

        Returns
        -------
        nn.Module
            The central critic network
        """
        return self._critic


def torch_ma_ppo_loss(agent: TorchMAPPOAgent, train_batch: SampleBatch, **kwargs) -> tuple[torch.Tensor, dict[str, np.ndarray]]:
    """PPO loss function.

    Parameters
    ----------
    agent : TorchMAPPOAgent
        The agent being trained
    train_batch : SampleBatch
        The batch of experiences for this agent
    `**kwargs`
        Additional keyword arguments.

    Returns
    -------
    tuple[torch.Tensor, dict[str, np.ndarray]]
        The training loss and the loss info
    """
    agent.train(True)
    prev_dist = agent.dist_class(train_batch[Columns.ACTION_DIST_INPUTS])
    _, curr_val, curr_dist = agent.get_actions_and_values(
        x=train_batch[Columns.OBS].requires_grad_(),
        action=train_batch[Columns.ACTIONS],
        centralized_obs=train_batch[f"centralized_{Columns.OBS}"].requires_grad_(),
    )
    curr_logprobs = curr_dist.log_prob(train_batch[Columns.ACTIONS])
    curr_entropy = curr_dist.entropy()

    # entropy
    entropy_bonus: torch.Tensor = (
        curr_entropy.neg() if agent.entropy_coeff else torch.tensor(0.0, device=curr_logprobs.device, dtype=curr_logprobs.dtype)
    )

    # surrogate loss
    ratio = (curr_logprobs - train_batch[Columns.ACTION_LOGP]).exp()
    ratio_clip = torch.clamp(ratio, 1 - agent.logp_clip_param, 1 + agent.logp_clip_param)

    surrogate_loss = torch.minimum(train_batch[Columns.ADVANTAGES] * ratio_clip, train_batch[Columns.ADVANTAGES] * ratio).neg()

    # kl loss
    kl_loss: torch.Tensor = (
        torch.distributions.kl_divergence(prev_dist, curr_dist)
        if agent.kl_coeff > 0
        else torch.tensor(0.0, device=curr_logprobs.device, dtype=curr_logprobs.dtype)
    )

    # vf loss
    vf_loss: torch.Tensor = nn.functional.mse_loss(curr_val, train_batch[Columns.VALUE_TARGETS], reduction="none")
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
                explained_var(curr_val, train_batch[Columns.REWARDS]),
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
