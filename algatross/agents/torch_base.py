"""Base agent for Torch framework."""

import logging

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict
from functools import cached_property
from itertools import chain
from typing import Any, Literal

import numpy as np

from ray.rllib import SampleBatch

import torch

from torch import nn
from torch.distributions import Categorical, Multinomial, Normal
from torch.distributions.distribution import Distribution

import gymnasium

from harl.algorithms.critics.v_critic import VCritic
from harl.common.buffers.on_policy_critic_buffer_ep import OnPolicyCriticBufferEP
from harl.common.buffers.on_policy_critic_buffer_fp import OnPolicyCriticBufferFP
from harl.common.valuenorm import ValueNorm
from harl.utils.trans_tools import _t2n  # noqa: PLC2701

from algatross.agents.base import BaseAgent
from algatross.configs.harl.agents import HARLAgentConfig
from algatross.utils.exceptions import InitNotCalledError
from algatross.utils.merge_dicts import merge_dicts
from algatross.utils.random import get_torch_generator_from_numpy
from algatross.utils.stats import calc_grad_norm
from algatross.utils.types import PlatformID


class TorchActorCriticModule(nn.Module):
    """An actor/critic torch module."""

    _actor: nn.Module
    _critic: nn.Module

    @property
    def actor(self) -> nn.Module:
        """
        Return the actor module.

        Returns
        -------
        nn.Module
            The actor module
        """
        return self._actor

    @actor.setter
    def actor(self, module):
        if not isinstance(module, nn.Module):
            msg = f"Actor ({type(module)}) must inherit from torch.nn.Module"
            raise TypeError(msg)
        self._actor = module

    @property
    def critic(self) -> nn.Module:
        """
        Return the critic module.

        Returns
        -------
        nn.Module
            The critic module
        """
        return self._critic

    @critic.setter
    def critic(self, module):
        if not isinstance(module, nn.Module):
            msg = f"Actor ({type(module)}) must inherit from torch.nn.Module"
            raise TypeError(msg)
        self._critic = module

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward actor call.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        torch.Tensor
            The result of the forward :attr:`actor` call
        """
        return self._actor(x, **kwargs)


def found_infinite_gradients(optimizer):  # noqa: D103
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None and (torch.isinf(param.grad).any() or torch.isnan(param.grad).any()):
                return True
    return False


class TorchBaseAgent(BaseAgent, nn.Module):
    """
    A base agent for torch network models.

    Parameters
    ----------
    obs_space : gymnasium.spaces.Space
        The observation space of this agent.
    act_space : gymnasium.spaces.Space
        The action space of this agent.
    seed : Any
        The seed for randomness.
    critic_outs : int, optional
        The number of outputs for the critic network, default is 1.
    optimizer_class : str | type[torch.optim.Optimizer], optional
        The optimizer to use with this module, default is :class:`~torch.optim.adam.Adam`
    optimizer_kwargs : dict | None, optional
        Keyword arguments to pass to the optimizer constructor, default is :data:`python:None`
    dtype : Any, optional
        The datatype for this agent, default is ``torch.float32``
    device : str, optional
        The physical device to use for this agent, default is :python:`"cpU"`
    `**kwargs`
        Additional keyword arguments.
    """

    dist_class: type[Distribution]
    """The class of distribution used for action sampling."""
    optimizer_class: type[torch.optim.Optimizer]
    """The class of optimizer to use with this module."""
    optimizer_kwargs: dict
    """Keyword arguments to pass to the optimizer constructor."""
    torch_generator: torch.Generator
    """The random generator used for torch functions."""
    optimizer: torch.optim.Optimizer
    """The optimizer used by this agent."""
    dtype: Any = torch.float32
    """The datatype for this agent."""
    # TODO: move to desired device during island iter, then move back to cpu. For now always be on cpu
    device: str = "cpu"
    """The physical device to use for this agent."""

    def __init__(
        self,
        *,
        obs_space: gymnasium.spaces.Space,
        act_space: gymnasium.spaces.Space,
        seed: Any,  # noqa: ANN401
        critic_outs: int = 1,
        optimizer_class: str | type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: dict | None = None,
        dtype: Any = torch.float32,  # noqa: ANN401
        device: str = "cpu",
        **kwargs,
    ) -> None:
        nn.Module.__init__(self)
        if dtype != torch.float32:
            msg = "Agents currently only run on cpu, and cpu only supports float32 operations"
            raise NotImplementedError(msg)

        BaseAgent.__init__(
            self,
            obs_space=obs_space,
            act_space=act_space,
            critic_outs=critic_outs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            seed=seed,
            dtype=dtype,
            device=device,
            **kwargs,
        )
        optimizer_class = getattr(torch.optim, optimizer_class) if isinstance(optimizer_class, str) else optimizer_class
        optimizer_kwargs = optimizer_kwargs or {}
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.reset_optimizer()

    def __getstate__(self) -> dict:
        """Get the agent state.

        Returns
        -------
        dict
            The agent state

        Raises
        ------
        NotImplementedError
            If the agent doesn't have a ``torch_generator`` attribute
        """
        mod_state = self.__dict__.copy()

        mod_state.pop("_compiled_call_impl", None)
        mod_state.pop("torch_generator", None)

        if not hasattr(self, "torch_generator"):
            msg = f"{self.__class__.__name__} definition must have a torch_generator attribute!"
            raise NotImplementedError(msg)

        return {"torch_generator_state": self.torch_generator.get_state()} | mod_state

    def __setstate__(self, state):
        """
        Set the agent state.

        Parameters
        ----------
        state : dict
            The state dict to set for this agent
        """
        torch_generator_state = state.pop("torch_generator_state")
        self.torch_generator = torch.Generator()  # state will be acquired next line
        self.torch_generator.set_state(torch_generator_state)
        for k in state:
            setattr(self, k, state[k])

    def initialize_random_generators(self, seed: Any = None) -> None:  # noqa: ANN401
        """
        Initialize the random number generators.

        Parameters
        ----------
        seed : Any, optional
            The seed for randomness, default is :data:`python:None`
        """
        super().initialize_random_generators(seed)
        self.torch_generator = get_torch_generator_from_numpy(self.np_random)[0]

    def build_networks(self, *, observation_size: "int | Sequence[int]", value_network_outs: int, action_size: int, **kwargs):
        """
        Build the prediction networks.

        Parameters
        ----------
        observation_size : int | Sequence[int]
            The size of the observation space
        value_network_outs : int
            The number of outputs from the value network
        action_size : int
            The size of the action outputs
        `**kwargs`
            Additional keyword arguments.
        """
        raise NotImplementedError

    @classmethod
    def _layer_init(
        cls,
        layer: nn.Module,
        std: float | np.ndarray | None = None,
        bias_const: float = 0.0,
        generator: torch.Generator | None = None,
        **kwargs,
    ) -> nn.Module:
        if std is None:
            std = np.sqrt(2)
        if int(torch.__version__.split(".")[0]) > 1:
            nn.init.orthogonal_(layer.weight, std, generator=generator)
        else:
            nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer

    @classmethod
    def get_dist_class(cls, dist_class_name: Literal["normal", "categorical", "multinomial"], **kwargs) -> type[Distribution]:
        """Get the action distribution class from the class name.

        Parameters
        ----------
        dist_class_name : Literal["normal", "categorical", "multinomial"]
            The string name for the type of distribution class
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        type[Distribution]
            The class type.

        Raises
        ------
        ValueError
            If an invalid class name is given.
        """
        dist_class: type[Distribution]
        if dist_class_name == "normal":
            # class PartialMultivariateNormal(MultivariateNormal):
            class PartialNormal(Normal):
                def __init__(self, logits, validate_args=None):
                    # chunk [B, logits] into [B, mean], [B, covariance]
                    loc, cov = torch.chunk(logits, 2, dim=logits.ndim - 1)
                    if logits.ndim == 1:
                        loc, cov = loc[None], cov[None]
                    super().__init__(loc=loc, scale=cov.exp(), validate_args=validate_args)

            dist_class = PartialNormal
        elif dist_class_name == "categorical":

            class PartialCategorical(Categorical):
                def __init__(self, logits, probs=None, validate_args=None):
                    if logits.ndim == 1:
                        logits = logits[None]
                    super().__init__(probs, logits, validate_args=validate_args)

                @property
                def mean(self):
                    # for eval we need the index of the largest value.
                    return self.mode

            dist_class = PartialCategorical
        elif dist_class_name == "multinomial":

            class PartialMultinomial(Multinomial):
                def __init__(self, logits, total_count=1, probs=None, validate_args=None):
                    if logits.ndim == 1:
                        logits = logits[None]
                    super().__init__(total_count, probs, logits, validate_args=validate_args)

            dist_class = PartialMultinomial
        else:
            msg = f"Invalid distribution class name: {dist_class_name}. Expected `normal`, `categorical`, `multinomial`"
            raise ValueError(msg)
        return dist_class

    def load_flat_params(self, flat_params: torch.Tensor | np.ndarray, **kwargs):
        """
        Load a flat vector of parameters into the agent.

        Buffers are left unchanged

        Parameters
        ----------
        flat_params : torch.Tensor | np.ndarray
            The vector of parameters to unflatten into the network parameters.
        `**kwargs`
            Additional keyword arguments.

        Raises
        ------
        InitNotCalledError
            If the agents :python:`__init__` method has not been called.
        RuntimeError
            Any RuntimeError raised by ray
        """
        if not self._init_called:
            raise InitNotCalledError(self)
        reshaped = {}
        pointer = 0
        vec: torch.Tensor = (torch.from_numpy(flat_params.copy()) if isinstance(flat_params, np.ndarray) else flat_params).view(-1)
        for param_name, param in self.named_parameters():
            # The length of the parameter
            num_param = param.numel()
            # Slice the vector, reshape it, and replace the old data of the parameter
            try:
                reshaped[param_name] = vec[pointer : pointer + num_param].view_as(param).data
            except RuntimeError:
                logger = logging.getLogger("ray")
                msg = (
                    "\t Critical mis-match during parameter loading. Please include this if filing an issue:"
                    f"\t\t param_name: {param_name}"
                    f"\t\t expected param.shape: {param.shape}"
                    f"\t\t from vec.shape: {vec.shape}"
                    f"\t\t pointer : pointer + num_param: {pointer} : {pointer + num_param}  (diff of {(pointer + num_param) - pointer})",
                )
                logger.exception(msg)
                raise

            # Increment the pointer
            pointer += num_param

        self.load_state_dict(reshaped, strict=False)

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
        NotImplementedError
            if sub classes did not override this method
        """
        raise NotImplementedError

    def get_actions(self, x: torch.Tensor, logits: torch.Tensor | None = None, *args, **kwargs) -> tuple[torch.Tensor, Distribution]:
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
        NotImplementedError
            if sub classes did not override this method
        """
        raise NotImplementedError

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

        Raises
        ------
        NotImplementedError
            if sub classes did not override this method
        """
        raise NotImplementedError

    def get_actions_and_values(
        self,
        x: torch.Tensor,
        action: torch.Tensor | None = None,
        logits: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, Distribution]:
        """Get the action distribution and value predictions from the agent.

        Parameters
        ----------
        x : torch.Tensor
            The input observation
        action : torch.Tensor | None, optional
            The actions for which we want to compute the logits, :data:`python:None`, in which case new actions and distributions
            are created from the input observations or logits if provided
        logits : torch.Tensor | None, optional
            The logits used to construct the action distribution, :data:`python:None`, in which case a new action distribution is
            constructed
            from the actor network output.
        `*args`
            Additional positional arguments.
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, Distribution]
            A tuple of actions, value predictions, and the action distribution used to generate the actions.

        Raises
        ------
        NotImplementedError
            if sub classes did not override this method
        """
        raise NotImplementedError

    @cached_property
    def _act_type_map(self) -> "Callable[[torch.Tensor], torch.Tensor]":
        # cached_property so the if/else condition is only ever checked once
        return self._act_long_map if self.discrete else self._act_float_map

    @staticmethod
    def _act_float_map(x: torch.Tensor) -> torch.Tensor:
        """Return the action tensor.

        This determines what will end up in SampleBatch, and is the magnitude used during agent.learn().
        The environment may process the action values further without our knowledge.

        Parameters
        ----------
        x : torch.Tensor
            The observation tensor $x$.

        Returns
        -------
        torch.Tensor
            The action tensor.
        """
        return x

    @staticmethod
    def _act_long_map(x: torch.Tensor) -> torch.Tensor:
        # convert to longs
        return x.long()

    def reset_optimizer(self):
        """
        Reset the optimizer to its initial state.

        Reinitializes the optimizer so the gradient buffers are cleared. This is necessary for optimizers with momentum
        whenever the network parameters are forcibly changed.
        """
        self.optimizer = self.optimizer_class(list(self.parameters()), **self.optimizer_kwargs)

    def loss(self, train_batch: SampleBatch, **kwargs) -> tuple[torch.Tensor, dict[str, np.ndarray]]:  # noqa: ARG002
        """
        Calculate the loss from the training batch for this agent.

        Parameters
        ----------
        train_batch : SampleBatch
            Training batch of data for this agent.
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        tuple[torch.Tensor, dict[str, np.ndarray]]
            The training loss, the training info or stats

        Raises
        ------
        InitNotCalledError
            If the agents init method was not called
        NotImplementedError
            If the method is not overridden by subclasses
        """  # noqa: DOC202
        if not self._init_called:
            raise InitNotCalledError(self)
        raise NotImplementedError

    def postprocess_batch(self, batch: SampleBatch, **kwargs) -> SampleBatch:
        """Run postprocessing on a batch of data from multiple episodes.

        Parameters
        ----------
        batch : SampleBatch
            The batch to postprocess

        Returns
        -------
        SampleBatch
            The postprocessed sample batch
        `**kwargs`
            Additional keyword arguments.

        Raises
        ------
        InitNotCalledError
            If the agent has not been initialized
        """
        if not self._init_called:
            raise InitNotCalledError(self)
        return batch

    def process_episode(
        self,
        episode_batch: SampleBatch,
        rewards: np.ndarray,  # noqa: ARG002
        **kwargs,
    ) -> SampleBatch:
        """Run processing on a batch of data from a single episode.

        Parameters
        ----------
        episode_batch : SampleBatch
            The single episode batch to process
        rewards : np.ndarray,
            The sequence of rewards for this agent
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        SampleBatch
            The processed episode batch

        Raises
        ------
        InitNotCalledError
            If the agents :python:`__init__` method was not called.
        """
        if not self._init_called:
            raise InitNotCalledError(self)
        return episode_batch

    def train(self, mode: bool = True) -> "TorchBaseAgent":  # noqa: D102
        return nn.Module.train(self, mode)

    @property
    def flat_parameters(self) -> torch.Tensor:  # noqa: D102
        if not self._init_called:
            raise InitNotCalledError(self)
        return torch.nn.utils.parameters_to_vector(self.parameters()).detach()

    def learn(self, /, train_batch: SampleBatch, **kwargs) -> tuple[torch.Tensor, dict]:  # noqa: D102
        if not self._init_called:
            raise InitNotCalledError(self)
        self.optimizer.zero_grad(True)
        loss, infos = self.loss(train_batch=train_batch)
        logger = logging.getLogger("ray")
        loss.backward()
        if self.debug and found_infinite_gradients(self.optimizer):
            logger.error("Optimizer had inf gradients!")
            logger.error(f"loss={loss}")

        infos["grad_norm"] = calc_grad_norm(self.optimizer).cpu().numpy()

        self.optimizer.step()

        return loss, infos


class TorchBaseMARLAgent(ABC):  # noqa: PLR0904
    """
    A base agent class which controls multiple actors in the environment.

    Parameters
    ----------
    platforms : Sequence[PlatformID]
        The platforms controlled by this agent
    obs_spaces : dict[PlatformID, gymnasium.spaces.Space]
        The observation space for each platform of this agent.
    act_spaces : dict[PlatformID, gymnasium.spaces.Space]
        The action space for each platform of this agent.
    actor_class : type[TorchBaseAgent], optional
        The default class to use for the individual agents, default is :class:`~algatross.agents.torch_base.TorchBaseAgent`
    shared_obs_spaces : gymnasium.spaces.Space
        The shared observation space of this agent.
    runner_config : dict | None, optional
        The configuration to pass to the runner, default is :data:`python:None`.
    config : HARLAgentConfig | dict | None, optional
        The config for this agent, default is HARLAgentConfig.
    device : str, optional
        The device to create this agent on, default is :python:`"cpu"`
    `**kwargs`
        Additional keyword arguments.
    """

    default_config: type[HARLAgentConfig] = HARLAgentConfig
    """The default configuration dataclass to use when none is specified."""
    platforms: Mapping[PlatformID, TorchBaseAgent]
    """A mapping from platforms to the agents in this class which control them."""
    device: torch.device
    """The device this agent is on."""
    use_valuenorm: bool
    """Whether to use a value normalizer."""
    use_linear_lr_decay: bool
    """Whether to use a learning rate decay."""
    model_config: dict
    """The configuration dict of the models."""
    algorithm_config: dict
    """The configuration dict of the algorithm."""
    runner_config: dict
    """The configuration dict of the runner."""

    _recurrent: bool
    _recurrent_n: int
    _rnn_hidden_size: int
    _state_type: Literal["EP", "FP"]
    _n_rollout_threads: int

    def __init__(
        self,
        platforms: Sequence[PlatformID],
        obs_spaces: dict[PlatformID, gymnasium.spaces.Space],
        act_spaces: dict[PlatformID, gymnasium.spaces.Space],
        actor_class: type[TorchBaseAgent],
        shared_obs_space: gymnasium.spaces.Space,
        runner_config: dict | None = None,
        config: HARLAgentConfig | dict | None = None,
        device: str = "cpu",
        **kwargs,
    ):
        config = config or {}
        config = deepcopy(config if isinstance(config, dict) else asdict(config))
        config = merge_dicts(asdict(self.default_config()), config)

        runner_config = runner_config or {}
        tmp_runner_config = config.pop("runner_config", None) or {}
        tmp_runner_config = tmp_runner_config if isinstance(tmp_runner_config, dict) else asdict(tmp_runner_config)
        model_config = config.pop("model_config", None) or {}
        algorithm_config = config.pop("algorithm_config", None) or {}

        self.runner_config = merge_dicts(tmp_runner_config, runner_config)
        self.model_config = model_config if isinstance(model_config, dict) else asdict(model_config)
        self.algorithm_config = algorithm_config if isinstance(algorithm_config, dict) else asdict(algorithm_config)
        self.config = merge_dicts(config, kwargs)

        self.device = torch.device(device)

        self._state_type = self.runner_config.get("state_type", "EP")
        self._n_rollout_threads = self.runner_config.get("n_rollout_threads", 20)
        self.use_valuenorm = self.runner_config.get("use_valuenorm")
        self.use_linear_lr_decay = self.runner_config.get("use_linear_lr_decay")
        self.share_params = self.algorithm_config.get("share_params")
        self.platforms = self.construct_platforms(
            actor_class=actor_class,
            obs_spaces=obs_spaces,  # type: ignore[arg-type]
            act_spaces=act_spaces,  # type: ignore[arg-type]
            runner_config=runner_config,
            algorithm_config=algorithm_config,
            model_config=model_config,
            **kwargs,
        )
        self.shared_obs_space = shared_obs_space
        self._num_platforms = len(platforms)
        self._recurrent_n = self.model_config["recurrent_n"]
        self._rnn_hidden_size = self.model_config["hidden_sizes"][-1]
        self._recurrent = self._recurrent_n > 0

    def construct_platforms(
        self,
        obs_spaces: gymnasium.spaces.Space,
        act_spaces: gymnasium.spaces.Space,
        actor_class: type[TorchBaseAgent],
        platforms: Sequence[PlatformID],
        runner_config: dict,
        algorithm_config: dict,
        model_config: dict,
        **kwargs,
    ) -> Mapping[PlatformID, TorchBaseAgent]:
        """Construct the platforms controlled by this agent.

        Parameters
        ----------
        obs_spaces : gymnasium.spaces.Space
            Observation spaces in the environment
        act_spaces : gymnasium.spaces.Space
            Action spaces in the environment
        actor_class : type[TorchBaseAgent]
            Base class for the actors controlled by this agent.
        platforms : Sequence[PlatformID]
            The platform IDs controlled by this agent.
        runner_config : dict
            Config for the runner.
        algorithm_config : dict
            Config for the algorithm.
        model_config : dict
            Config for the model.
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        Mapping[PlatformID, TorchBaseAgent]
            Mapping from platform ID to its corresponding controller agent.
        """
        sorted_platforms = sorted(platforms)
        constructed = {}

        if self.share_params:
            agent = actor_class(obs_space=obs_spaces[sorted_platforms[0]], act_space=act_spaces[sorted_platforms[0]], **kwargs)  # type: ignore[index]
            for platform_id in sorted_platforms:
                constructed[platform_id] = agent
        else:
            for platform_id in sorted_platforms:
                constructed[platform_id] = actor_class(
                    obs_space=obs_spaces[platform_id],  # type: ignore[index]
                    act_space=act_spaces[platform_id],  # type: ignore[index]
                    runner_config=runner_config,
                    algorithm_config=algorithm_config,
                    model_config=model_config,
                    **kwargs,
                )
        return constructed

    @abstractmethod
    def act(self, **kwargs):
        """
        Get a single action from the actor.

        Parameters
        ----------
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        Any
            A single action from the actor
        """

    @abstractmethod
    def get_values(self, **kwargs):
        """
        Return a generator which yields value predictions for each actor.

        Parameters
        ----------
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        Any
            The values from the agent
        """

    @abstractmethod
    def get_actions(self, **kwargs):
        """
        Return a generator which yields action predictions for each actor.

        Parameters
        ----------
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        Any
            The actions from the agent
        """

    @abstractmethod
    def get_actions_and_values(self, **kwargs):
        """
        Return a generator which yields action and value predictions for each actor.

        Parameters
        ----------
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        Any
            The actions ang values from the network
        """

    def postprocess_batch(self, agent_batches: dict[PlatformID, SampleBatch], **kwargs) -> dict[PlatformID, SampleBatch]:  # noqa: D102
        for platform_id, actor in self.platforms.items():
            agent_batches[platform_id] = actor.postprocess_batch(agent_batches[platform_id], **kwargs)

        return agent_batches

    def process_episode(  # noqa: D102
        self,
        agent_episode_batches: dict[PlatformID, SampleBatch],
        rewards: dict[PlatformID, np.ndarray],
        **kwargs,
    ) -> dict[PlatformID, SampleBatch]:
        for agent_id, agent in self.actors.items():
            agent_episode_batches[agent_id] = agent.process_episode(agent_episode_batches[agent_id], rewards[agent_id], **kwargs)

        return agent_episode_batches

    def learn(self, **kwargs):
        """
        Update the actor(s) and critic(s).

        Parameters
        ----------
        `**kwargs`
            Additional keyword arguments.
        """
        for actor in self.actors.values():
            if hasattr(actor, "learn"):
                actor.learn(**kwargs)

    @abstractmethod
    def clear_buffer(self):
        """Clear any buffers stored with the agent."""

    def reset_optimizer(self):  # noqa: D102
        for agent in self.actors.values():
            agent.reset_optimizer()

    def lr_decay(self, step, steps):  # noqa: ARG002, PLR6301
        """
        Decay the learning rates of the networks.

        Parameters
        ----------
        step : int
            The current step
        steps : int
            The total steps
        """
        return

    @abstractmethod
    def insert(self, data, /, **kwargs):
        """Insert experiences into the buffers.

        Parameters
        ----------
        data : Any
            The data to insert.
        `**kwargs`
            Additional keyword arguments.
        """

    @abstractmethod
    def warmup(self, **kwargs):
        """
        Run operations when rollouts begin.

        Parameters
        ----------
        `**kwargs`
            Additional keyword arguments.
        """

    @abstractmethod
    def compute(self):
        """Run operations which computes the returns in the critic buffers."""

    @abstractmethod
    def prep_rollout(self):
        """Run operations before the rollouts are gathered."""

    @abstractmethod
    def prep_training(self):
        """Run operations before the networks are updated."""

    @abstractmethod
    def after_update(self):
        """Run operations after the networks have been updated."""

    @property
    def actors(self) -> Mapping[PlatformID, TorchBaseAgent]:
        """
        A mapping from each platform to the actor which controls it.

        Returns
        -------
        Mapping[PlatformID, TorchBaseAgent]
            A mapping from each platform to the actor which controls it
        """
        return self.platforms

    @property
    def recurrent_n(self) -> int:
        """
        The length of the recurrent state sequence.

        Returns
        -------
        int
            The length of the recurrent state sequence
        """
        return self._recurrent_n

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
            The RNN state type
        """
        return self._state_type

    @property
    def rnn_hidden_size(self) -> int:
        """
        The hidden side of each RNN layer.

        Returns
        -------
        int
            The hidden side of each RNN layer.
        """
        return self._rnn_hidden_size

    @property
    def recurrent(self) -> bool:
        """
        Whether or not this agent uses a recurrent network.

        Returns
        -------
        bool
            Whether or not this agent uses a recurrent network
        """
        return self._recurrent

    @property
    def num_platforms(self) -> int:
        """
        The number of platforms controlled by this agent.

        Returns
        -------
        int
            The number of platforms controlled by this agent
        """
        return self._num_platforms

    @property
    def n_rollout_threads(self) -> int:
        """
        The number of threads to use for rollout.

        Returns
        -------
        int
            The number of threads to use for rollout
        """
        return self._n_rollout_threads


class TorchOnPolicyMARLAgent(TorchBaseMARLAgent):
    """
    Agent controling multiple on-policy actors.

    Parameters
    ----------
    platforms : Sequence[PlatformID]
        The platforms this agent controls
    obs_spaces : dict[PlatformID, gymnasium.spaces.Space]
        The observation spaces for each platform
    act_spaces : dict[PlatformID, gymnasium.spaces.Space]
        The action spaces for each platform
    actor_class : type[TorchBaseAgent]
        The class to use for the actors
    critic_class : type[VCritic]
        The class to use for the value critic
    shared_obs_space : gymnasium.spaces.Space
        The shared observation space
    device : str, optional
        The device to place this agent onto, default is :python:`"cpu"`
    `**kwargs`
        Additional keyword arguments.
    """

    critic_buffer: OnPolicyCriticBufferEP | OnPolicyCriticBufferFP
    """The buffer for the critic network."""
    value_normalizer: ValueNorm | None = None
    """A function to use to normalize critic values."""

    def __init__(
        self,
        platforms: Sequence[PlatformID],
        obs_spaces: dict[PlatformID, gymnasium.spaces.Space],
        act_spaces: dict[PlatformID, gymnasium.spaces.Space],
        actor_class: type[TorchBaseAgent],
        critic_class: type[VCritic],
        shared_obs_space: gymnasium.spaces.Space,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(
            platforms=platforms,
            obs_spaces=obs_spaces,
            act_spaces=act_spaces,
            actor_class=actor_class,
            shared_obs_space=shared_obs_space,
            critic_class=critic_class,
            device=device,
            **kwargs,
        )

        args = {**self.runner_config, **self.model_config, **self.algorithm_config, **kwargs}

        if self.state_type == "EP":
            # EP stands for Environment Provided, as phrased by MAPPO paper.
            # In EP, the global states for all platforms are the same.
            self.critic_buffer = OnPolicyCriticBufferEP(args, self.shared_obs_space)
        elif self.state_type == "FP":
            # FP stands for Feature Pruned, as phrased by MAPPO paper.
            # In FP, the global states for all platforms are different, and thus needs the dimension of the number of platforms.
            self.critic_buffer = OnPolicyCriticBufferFP(args, self.shared_obs_space, self.num_platforms)
        else:
            raise NotImplementedError

        self._critic = critic_class(args, self.shared_obs_space, device=self.device)
        self.value_normalizer = ValueNorm(1, device=self.device) if self.use_valuenorm else None

    @property
    def actor(self) -> list[TorchBaseAgent]:
        """
        Each actor managed by this multi-agent class.

        Returns
        -------
        list[TorchBaseAgent]
            Each actor managed by this multi-agent class
        """
        return list(self.actors.values())

    @property
    def critic(self) -> VCritic:
        """
        The critic for this agent.

        Returns
        -------
        VCritic
            The critic for this agent
        """
        return self._critic

    def prep_rollout(self):
        """Prepare for rollout."""
        for module in chain(self.actors.values(), [self.critic]):
            if hasattr(module, "prep_rollout"):
                module.prep_rollout()

    def prep_training(self):
        """Prepare for training."""
        for module in chain(self.actors.values(), [self.critic]):
            if hasattr(module, "prep_training"):
                module.prep_training()

    def after_update(self):
        """Do the necessary data operations after an update.

        After an update, copy the data at the last step to the first position of the buffer.
        This will be used for then generating new actions.
        """
        for buffered in chain(self.actors.values(), [self.critic_buffer]):
            if hasattr(buffered, "after_update"):
                buffered.after_update()

    def act(self, **kwargs):  # noqa: D102
        for actor in self.actors.values():
            yield actor.act(**kwargs)

    def get_actions(self, **kwargs):  # noqa: D102
        for actor in self.actors.values():
            yield actor.get_actions(**kwargs)

    def get_values(self, **kwargs):  # noqa: D102
        step = kwargs.get("step")
        if self.state_type == "EP":
            value, rnn_state_critic = self.critic.get_values(
                self.critic_buffer.share_obs[step],
                self.critic_buffer.rnn_states_critic[step],
                self.critic_buffer.masks[step],
            )
            # (n_threads, dim)  # noqa: ERA001
            values = _t2n(value)
            rnn_states_critic = _t2n(rnn_state_critic)
        elif self.state_type == "FP":
            value, rnn_state_critic = self.critic.get_values(
                np.concatenate(self.critic_buffer.share_obs[step]),
                np.concatenate(self.critic_buffer.rnn_states_critic[step]),
                np.concatenate(self.critic_buffer.masks[step]),
            )  # concatenate (n_threads, n_agents, dim) into (n_threads * n_agents, dim)
            # split (n_threads * n_agents, dim) into (n_threads, n_agents, dim)
            values = np.array(np.split(_t2n(value), self.n_rollout_threads))
            rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))

        for _actor in self.actors:
            yield values, rnn_states_critic

    def get_actions_and_values(self, **kwargs):  # noqa: D102
        step = kwargs.get("step")
        if self.state_type == "EP":
            value, rnn_state_critic = self.critic.get_values(
                self.critic_buffer.share_obs[step],
                self.critic_buffer.rnn_states_critic[step],
                self.critic_buffer.masks[step],
            )
            # (n_threads, dim)  # noqa: ERA001
            values = _t2n(value)
            rnn_states_critic = _t2n(rnn_state_critic)
        elif self.state_type == "FP":
            value, rnn_state_critic = self.critic.get_values(
                np.concatenate(self.critic_buffer.share_obs[step]),
                np.concatenate(self.critic_buffer.rnn_states_critic[step]),
                np.concatenate(self.critic_buffer.masks[step]),
            )  # concatenate (n_threads, n_agents, dim) into (n_threads * n_agents, dim)
            # split (n_threads * n_agents, dim) into (n_threads, n_agents, dim)
            values = np.array(np.split(_t2n(value), self.n_rollout_threads))
            rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))
        for actor in self.actors.values():
            actions, logprobs, rnn_states_actor = actor.get_actions(**kwargs)
            yield values, actions, logprobs, rnn_states_actor, rnn_states_critic

    def clear_buffer(self):
        """Clear the actor buffers and the critic buffer."""
        for actor in self.actors.values():
            if hasattr(actor, "clear_buffer"):
                actor.clear_buffer()

        def clear_buffer(buf, fill_value):
            for b in buf:
                b.fill(fill_value)

        clear_buffer(self.critic_buffer.share_obs, 0.0)
        clear_buffer(self.critic_buffer.rnn_states_critic, 0.0)
        clear_buffer(self.critic_buffer.value_preds, 0.0)
        clear_buffer(self.critic_buffer.returns, 0.0)
        clear_buffer(self.critic_buffer.rewards, 0.0)
        clear_buffer(self.critic_buffer.masks, 1.0)
        clear_buffer(self.critic_buffer.bad_masks, 1.0)

        self.critic_buffer.step = 0

    def lr_decay(self, step, steps):
        """
        Decay the learning rate linearly for each actor.

        Parameters
        ----------
        step : int
            The current step
        steps : int
            The total number of steps
        """
        if self.use_linear_lr_decay:
            actors = [next(iter(self.actors.values())), self.critic] if self.share_params else chain(self.actors.values(), [self.critic])
            for actor in actors:
                if hasattr(actor, "lr_decay"):
                    actor.lr_decay(step, steps)

    def warmup(self, /, obs: np.ndarray, share_obs: np.ndarray, infos: np.ndarray, available_actions: np.ndarray | None, **kwargs):  # type: ignore[override]
        """
        Call each of the the actors' warmup functions and copy the initial observations into the critic buffer.

        Parameters
        ----------
        obs : np.ndarray
            The observations
        share_obs : np.ndarray
            The shared observations
        infos : np.ndarray
            The environment info
        available_actions : np.ndarray | None
            The available actions for the next timestep
        `**kwargs`
            Additional keyword arguments.
        """
        for actor_idx, actor in enumerate(self.actors.values()):
            if hasattr(actor, "warmup"):
                actor.warmup(
                    obs=obs[:, actor_idx].copy(),
                    infos=infos[:, actor_idx].copy(),
                    available_actions=available_actions if available_actions.ndim == 1 else available_actions[:, actor_idx],
                    **kwargs,
                )
        if self.state_type == "EP":
            self.critic_buffer.share_obs[0] = share_obs[:, 0].copy()
        elif self.state_type == "FP":
            self.critic_buffer.share_obs[0] = share_obs.copy()

    def insert(self, data: tuple, **kwargs):  # type: ignore[override]
        """
        Insert new data into the actor and critic buffers.

        Parameters
        ----------
        data : tuple
            The data to insert as a tuple with the following entries:

            - ``obs`` - :python:`(n_threads, n_agents, obs_dim)`
            - ``share_obs`` - :python:`(n_threads, n_agents, share_obs_dim)`
            - ``rewards`` - :python:`(n_threads, n_agents, 1)`
            - ``dones`` - :python:`(n_threads, n_agents)`
            - ``infos`` - list, shape: :python:`(n_threads, n_agents)`
            - ``available_actions`` - :python:`(n_threads, )` of :data:`python:None` or :python:`(n_threads, n_agents, action_number)`
            - ``values`` - EP: :python:`(n_threads, dim)`, FP: :python:`(n_threads, n_agents, dim)`
            - ``actions`` - :python:`(n_threads, n_agents, action_dim)`
            - ``action_log_probs`` - :python:`(n_threads, n_agents, action_dim)`
            - ``rnn_states`` - :python:`(n_threads, n_agents, dim)`
            - ``rnn_states_critic`` - EP: :python:`(n_threads, dim)`, FP: :python:`(n_threads, n_agents, dim)`

        `**kwargs`
            Additional keyword arguments.
        """
        (
            obs,  # (n_threads, n_agents, obs_dim)
            share_obs,  # (n_threads, n_agents, share_obs_dim)
            rewards,  # (n_threads, n_agents, 1)
            dones,  # (n_threads, n_agents)
            infos,  # list, shape: (n_threads, n_agents)
            available_actions,  # (n_threads, ) of None or (n_threads, n_agents, action_number)
            values,  # EP: (n_threads, dim), FP: (n_threads, n_agents, dim)
            actions,  # (n_threads, n_agents, action_dim)
            action_log_probs,  # (n_threads, n_agents, action_dim)
            rnn_states,  # (n_threads, n_agents, dim)
            rnn_states_critic,  # EP: (n_threads, dim), FP: (n_threads, n_agents, dim)
        ) = data

        dones_env = np.all(dones, axis=1)  # if all agents are done, then env is done
        n_dones_env = dones_env.sum()
        rnn_states[dones_env] = np.zeros(  # if env is done, then reset rnn_state to all zero
            (n_dones_env, self.num_platforms, self.recurrent_n, self.rnn_hidden_size),
            dtype=np.float32,
        )

        # If env is done, then reset rnn_state_critic to all zero
        if self.state_type == "EP":
            rnn_states_critic[dones_env] = np.zeros((n_dones_env, self.recurrent_n, self.rnn_hidden_size), dtype=np.float32)
        elif self.state_type == "FP":
            rnn_states_critic[dones_env] = np.zeros(
                (n_dones_env, self.num_platforms, self.recurrent_n, self.rnn_hidden_size),
                dtype=np.float32,
            )

        # masks use 0 to mask out threads that just finish.
        # this is used for denoting at which point should rnn state be reset
        masks = np.ones((self.n_rollout_threads, self.num_platforms, 1), dtype=np.float32)
        masks[dones_env] = np.zeros((n_dones_env, self.num_platforms, 1), dtype=np.float32)

        # active_masks use 0 to mask out agents that have died
        active_masks = np.ones((self.n_rollout_threads, self.num_platforms, 1), dtype=np.float32)
        active_masks[dones] = np.zeros((dones.sum(), 1), dtype=np.float32)
        active_masks[dones_env] = np.ones((n_dones_env, self.num_platforms, 1), dtype=np.float32)

        # bad_masks use 0 to denote truncation and 1 to denote termination
        if self.state_type == "EP":
            bad_masks = np.array([[0.0] if "bad_transition" in info[0] and info[0]["bad_transition"] else [1.0] for info in infos])
        elif self.state_type == "FP":
            bad_masks = np.array([
                [
                    [0.0] if "bad_transition" in info[agent_id] and info[agent_id]["bad_transition"] else [1.0]
                    for agent_id in range(self.num_platforms)
                ]
                for info in infos
            ])

        for actor_id, actor in enumerate(self.actors.values()):
            if hasattr(actor, "insert"):
                actor.insert(
                    obs=obs[:, actor_id],
                    rnn_states=rnn_states[:, actor_id],
                    actions=actions[:, actor_id],
                    action_log_probs=action_log_probs[:, actor_id],
                    masks=masks[:, actor_id],
                    active_masks=active_masks[:, actor_id],
                    available_actions=available_actions[:, actor_id] if available_actions[0] is not None else None,
                    infos=infos[:, actor_id],
                )

        if self.state_type == "EP":
            self.critic_buffer.insert(share_obs[:, 0], rnn_states_critic, values, rewards[:, 0], masks[:, 0], bad_masks)
        elif self.state_type == "FP":
            self.critic_buffer.insert(share_obs, rnn_states_critic, values, rewards, masks, bad_masks)

    def compute(self):
        """Update the returns in the critic buffer."""
        if self.state_type == "EP":
            next_value, _ = self.critic.get_values(
                self.critic_buffer.share_obs[-1],
                self.critic_buffer.rnn_states_critic[-1],
                self.critic_buffer.masks[-1],
            )
            next_value = _t2n(next_value)
        elif self.state_type == "FP":
            next_value, _ = self.critic.get_values(
                np.concatenate(self.critic_buffer.share_obs[-1]),
                np.concatenate(self.critic_buffer.rnn_states_critic[-1]),
                np.concatenate(self.critic_buffer.masks[-1]),
            )
            next_value = np.array(np.split(_t2n(next_value), self.n_rollout_threads))
        self.critic_buffer.compute_returns(next_value, self.value_normalizer)

    def train(self) -> dict:
        """Training procedure for MAPPO.

        Returns
        -------
        dict
            The training info
        """
        train_infos = {}

        # compute advantages
        if self.value_normalizer is not None:
            advantages = self.critic_buffer.returns[:-1] - self.value_normalizer.denormalize(self.critic_buffer.value_preds[:-1])
        else:
            advantages = self.critic_buffer.returns[:-1] - self.critic_buffer.value_preds[:-1]

        # normalize advantages for FP
        if self.state_type == "FP":
            active_masks_collector = [actor.actor_buffer.active_masks for actor in self.actors.values()]
            active_masks_array = np.stack(active_masks_collector, axis=2)
            advantages_copy = advantages.copy()
            advantages_copy[active_masks_array[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        # update critic
        critic_train_info = self.critic.train(self.critic_buffer, self.value_normalizer)

        # update actors
        if getattr(self, "share_param", False):
            actor = self.actor[0]
            train_info = actor.share_param_train(actor.actor_buffer, advantages.copy(), self.num_platforms, self.state_type)
            train_infos.update({
                actor_id: {"training_stats/actor": train_info, "training_stats/critic": critic_train_info} for actor_id in self.actors
            })
        else:
            for actor_idx, (actor_id, actor) in enumerate(self.actors.items()):
                if self.state_type == "EP":
                    train_info = actor.learn(  # type: ignore[call-arg]
                        advantages=advantages.copy(),
                        state_type="EP",
                    )
                elif self.state_type == "FP":
                    # TODO: fix this typing
                    train_info = actor.learn(  # type: ignore[call-arg]
                        advantages=advantages[:, :, actor_idx].copy(),
                        state_type="FP",
                    )
                train_infos[actor_id] = {"training_stats/actor": train_info, "training_stats/critic": critic_train_info}

        return train_infos
