"""Base classes for off policy HARL agents."""

from typing import Literal

import torch

from torch.optim import Optimizer

from gymnasium import Space
from harl.algorithms.actors.off_policy_base import OffPolicyBase
from harl.common.buffers.off_policy_buffer_base import OffPolicyBufferBase
from harl.common.buffers.off_policy_buffer_ep import OffPolicyBufferEP
from harl.common.buffers.off_policy_buffer_fp import OffPolicyBufferFP

from algatross.agents.torch_base import TorchBaseAgent


class TorchOffPolicyBaseHARLActor(TorchBaseAgent):
    """
    Base class for HARL off policy torch actors.

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

    harl_actor: OffPolicyBase
    """The actor for this off-policy algorithm."""
    actor_buffer: OffPolicyBufferBase
    """The buffer for the actor."""
    critic_buffer: OffPolicyBufferBase
    """The buffer for the critic."""

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
        state_type = kwargs.get("state_type")
        if state_type == "EP":
            self.actor_buffer = OffPolicyBufferEP(
                args=kwargs,
                share_obs_space=kwargs.get("share_obs_space"),
                num_agents=len(obs_space),  # type: ignore[arg-type]
                obs_spaces=obs_space,
                act_spaces=act_space,
            )
        elif state_type == "FP":
            self.actor_buffer = OffPolicyBufferFP(
                args=kwargs,
                share_obs_space=kwargs.get("share_obs_space"),
                num_agents=len(obs_space),  # type: ignore[arg-type]
                obs_spaces=obs_space,
                act_spaces=act_space,
            )
        else:
            msg = f"Invalid state type selection: `{state_type}`, expected `EP` or `FP`"
            raise ValueError(msg)

        self.critic_buffer = self.actor_buffer
        self._initial_optimizer_state = self.harl_actor.actor_optimizer.state_dict()

    @property
    def actor(self) -> OffPolicyBase:
        """
        The actor for this off-policy algorithm.

        Returns
        -------
        OffPolicyBase
            The actor for this off-policy algorithm
        """
        return self.harl_actor

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

    def get_target_actions(self, obs, **kwargs):  # noqa: D102
        return self.harl_actor.get_target_actions(obs=obs, **kwargs)

    def soft_update(self):  # noqa: D102
        self.harl_actor.soft_update()

    def turn_on_grad(self):  # noqa: D102
        self.harl_actor.turn_on_grad()

    def turn_off_grad(self):  # noqa: D102
        self.harl_actor.turn_off_grad()

    def evaluate_actions(  # noqa: D102
        self,
        obs,
        rnn_states_actor,
        action,
        masks,
        available_actions=None,
        active_masks=None,
    ):
        return self.harl_actor.evaluate_actions(
            obs,
            rnn_states_actor,
            action,
            masks,
            available_actions=available_actions,
            active_masks=active_masks,
        )

    def clear_buffer(self):  # noqa: D102
        self.actor_buffer.cur_size = 0  # current occupied size of buffer
        self.actor_buffer.idx = 0  # current index to insert

        def clear_buffer(buf, fill_value):
            for b in buf:
                b.fill(fill_value)

        clear_buffer(self.actor_buffer.obs, 0.0)
        clear_buffer(self.actor_buffer.share_obs, 0.0)
        clear_buffer(self.actor_buffer.actions, 0.0)

        clear_buffer(self.actor_buffer.next_obs, 0.0)
        clear_buffer(self.actor_buffer.next_share_obs, 0.0)
        clear_buffer(self.actor_buffer.next_available_actions, 0.0)

        clear_buffer(self.actor_buffer.rewards, 0.0)
        clear_buffer(self.actor_buffer.dones, 0.0)
        clear_buffer(self.actor_buffer.terms, 0.0)

        clear_buffer(self.actor_buffer.valid_transitions, 1.0)
        clear_buffer(self.actor_buffer.available_actions, 0.0)
