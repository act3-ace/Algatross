"""Module containing HAPPO Agents."""

from collections.abc import Mapping, Sequence
from copy import deepcopy

import torch

from torch.optim import Optimizer

from gymnasium.spaces import Space
from harl.algorithms.actors.happo import HAPPO
from harl.algorithms.critics.v_critic import VCritic

from algatross.agents.on_policy.harl.base import TorchOnPolicyBaseHARLActor, TorchOnPolicyHARLAgent
from algatross.configs.harl.agents import HAPPOAgentConfig
from algatross.utils.merge_dicts import merge_dicts
from algatross.utils.types import AgentID, PlatformID


class TorchHAPPOActorBase(TorchOnPolicyBaseHARLActor):
    """
    HARL PPO torch base actor.

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
        optimizer_kwargs = optimizer_kwargs or {}
        # use the same arguments as PPO so map them to the equivalent HAPPO keys
        happo_kwargs = deepcopy(kwargs)
        happo_kwargs = merge_dicts(happo_kwargs, kwargs.get("runner_config") or {})
        happo_kwargs = merge_dicts(happo_kwargs, kwargs.get("model_config") or {})
        happo_kwargs = merge_dicts(happo_kwargs, kwargs.get("algorithm_config") or {})

        super().__init__(
            obs_space=obs_space,
            act_space=act_space,
            critic_outs=critic_outs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            **happo_kwargs,
        )

    def build_networks(self, *, obs_space: Space, act_space: Space, **kwargs):  # type: ignore[override] # noqa: D102
        device = kwargs.get("device", "cpu")
        if isinstance(device, str):
            device = torch.device(device)
        self.harl_actor = HAPPO(args=kwargs, obs_space=obs_space, act_space=act_space, device=device)
        self._initial_optimizer_state = self.harl_actor.actor_optimizer.state_dict()


class TorchHAPPOAgent(TorchOnPolicyHARLAgent):
    """
    HARL PPO Torch Agent.

    Parameters
    ----------
    platforms : Sequence[PlatformID]
        The platforms controlled by this agent
    obs_spaces : dict[PlatformID, gymnasium.spaces.Space]
        The observation space for each platform of this agent.
    act_spaces : dict[PlatformID, gymnasium.spaces.Space]
        The action space for each platform of this agent.
    shared_obs_spaces : gymnasium.spaces.Space
        The shared observation space of this agent.
    device : str, optional
        The device to create this agent on, default is :python:`"cpu"`
    `**kwargs`
        Additional keyword arguments.
    """

    platforms: Mapping[AgentID, TorchHAPPOActorBase]
    """A mapping from agents to the platforms which control them."""
    default_config: type[HAPPOAgentConfig] = HAPPOAgentConfig
    """The default configuration if one isn't provided."""

    def __init__(
        self,
        platforms: Sequence[PlatformID],
        obs_spaces: dict[PlatformID, Space],
        act_spaces: dict[PlatformID, Space],
        shared_obs_space: Space,
        device: str = "cpu",
        **kwargs,
    ):
        kwargs.setdefault("actor_class", TorchHAPPOActorBase)
        kwargs.setdefault("critic_class", VCritic)
        if not issubclass(kwargs["actor_class"], TorchHAPPOActorBase):
            msg = f"`actor_class` for {self.__class__.__name__} must subclass `TorchHAPPOActorBase`"
            raise TypeError(msg)
        if not issubclass(kwargs["critic_class"], VCritic):
            msg = f"`critic_class` for {self.__class__.__name__} must subclass `VCritic`"
            raise TypeError(msg)
        super().__init__(
            platforms=platforms,
            obs_spaces=obs_spaces,
            act_spaces=act_spaces,
            shared_obs_space=shared_obs_space,
            device=device,
            **kwargs,
        )

    @property
    def actor(self) -> list[TorchHAPPOActorBase]:  # type: ignore[override]
        """
        The actors for each platform controlled by this agent.

        Returns
        -------
        list[TorchHAPPOActorBase]
            The actors for each platform controlled by this agent
        """
        return list(self.platforms.values())
