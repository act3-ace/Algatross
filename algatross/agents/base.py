"""Base agent class."""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np

from ray.rllib import SampleBatch

import gymnasium

from algatross.utils.random import resolve_seed
from algatross.utils.types import PlatformID


class BaseAgent(ABC):  # noqa: PLR0904
    """
    A base agent to use when creating agents for different frameworks.

    Parameters
    ----------
    obs_space : gymnasium.spaces.Space
        The observation space for this agent
    act_space : gymnasium.spaces.Space
        The action space for this agent
    seed : Any, optional
        The random seed for generators, default is :data:`python:None`
    critic_outs : int
        The number of outputs of the critic network, default is 1
    debug : bool
        Whether to enable debugging, default is :data:`python:False`
    `**kwargs`
        Additional keyword arguments.
    """

    dist_class: type
    """The class of distribution used for sampling actions."""
    genome: np.ndarray | None = None
    """The genome representing the agent, typically the weights of the neural network."""
    debug: bool = False
    """Whether debugging is enabled, default is :data:`python:False`."""
    dtype: Any = None
    """The datatype for this agent."""
    device: str = "cpu"
    """The physical device to use for this agent."""

    _recurrent: bool
    _recurrent_n: int
    _rnn_hidden_size: int
    _state_type: Literal["EP", "FP"]
    _init_called: bool = False

    def __init__(
        self,
        *,
        obs_space: gymnasium.spaces.Space,
        act_space: gymnasium.spaces.Space,
        seed: Any = None,  # noqa: ANN401
        critic_outs: int = 1,
        debug: bool = False,
        dtype: Any = None,  # noqa: ANN401
        device: str = "cpu",
        **kwargs,
    ) -> None:
        in_size, _, actor_out_size, discrete_act, dist_class_name = self.get_obs_and_action_size(obs_space, act_space)
        self.discrete = discrete_act
        self.dist_class = self.get_dist_class(dist_class_name)
        self.initialize_random_generators(seed=seed)

        self._recurrent_n = kwargs.get("recurrent_n", 0)
        self._rnn_hidden_size = -1
        self._state_type = kwargs.get("state_type", "EP")
        self._recurrent = self._recurrent_n > 0
        self.debug = debug
        self.dtype = dtype
        self.device = device

        self.build_networks(
            obs_space=obs_space,
            act_space=act_space,
            observation_size=in_size,
            value_network_outs=critic_outs,
            action_size=actor_out_size,
            **kwargs,
        )
        self._init_called = True

    @staticmethod
    def get_obs_and_action_size(
        obs_space: gymnasium.spaces.Space,
        act_space: gymnasium.spaces.Space,
    ) -> tuple[int | Sequence[int], bool, int, bool, Literal["normal", "categorical", "multinomial"]]:
        """Determine the observation input size and action output size from the spaces.

        Parameters
        ----------
        obs_space : gymnasium.spaces.Space
            The agents observation space
        act_space : gymnasium.spaces.Space
            The agents action space

        Returns
        -------
        in_size : int | Sequence[int]
            The observation input size
        discrete_obs : bool
            Whether the observations are discrete
        actor_out_size : int
            The output shape of the action space
        discrete_act : bool
            Whether the actions are discrete
        dist_class_name : Literal["normal", "categorical", "multinomial"]
            The name of the type of distribution class
        """
        dist_class_name: Literal["normal", "categorical", "multinomial"]
        err_msg = "BaseAgent must only receive observations from one agent at a time. Use BaseMultiAgent instead."
        err_msg += f"\nSpace: {obs_space}"

        def obs_size(space):
            in_size = None
            discrete_obs = True
            if isinstance(space, gymnasium.spaces.Box):
                in_size = int(np.prod(space.shape))
                discrete_obs = False
            elif isinstance(space, gymnasium.spaces.Discrete):
                in_size = int(space.n)
            elif isinstance(space, gymnasium.spaces.MultiDiscrete):
                in_size = int(np.prod(space.nvec))
            elif isinstance(space, dict | gymnasium.spaces.Dict):
                obs = space.get("obs", None)
                if len(space) != 1 and obs is None:
                    raise RuntimeError(err_msg)
                if obs is not None:
                    discrete_obs, in_size = obs_size(obs)
                else:
                    discrete_obs, in_size = next(map(obs_size, space.values()))
            elif isinstance(space, Sequence | gymnasium.spaces.Tuple):
                if len(space) != 1:
                    raise RuntimeError(err_msg)
                discrete_obs, in_size = next(map(obs_size, space))
            if in_size is None:
                raise ValueError(type(space))
            return discrete_obs, in_size

        discrete_obs, in_size = obs_size(obs_space)

        def act_size(space):
            discrete_act = True
            if isinstance(space, gymnasium.spaces.Box):
                # box output mean / std
                actor_out_size = int(2 * np.prod(space.shape))
                dist_class_name = "normal"
                discrete_act = False
            elif isinstance(space, gymnasium.spaces.Discrete):
                # discrete output logits
                actor_out_size = int(space.n)
                dist_class_name = "categorical"
            elif isinstance(space, gymnasium.spaces.MultiDiscrete):
                # discrete output logits
                actor_out_size = int(np.prod(space.nvec))
                dist_class_name = "multinomial"
            elif isinstance(space, dict | gymnasium.spaces.Dict):
                if len(space) != 1:
                    raise RuntimeError(err_msg)
                actor_out_size, discrete_act, dist_class_name = next(map(act_size, space.values()))
            elif isinstance(space, Sequence | gymnasium.spaces.Tuple):
                if len(space) != 1:
                    raise RuntimeError(err_msg)
                actor_out_size, discrete_act, dist_class_name = next(map(act_size, space))
            return actor_out_size, discrete_act, dist_class_name

        actor_out_size, discrete_act, dist_class_name = act_size(act_space)

        return in_size, discrete_obs, actor_out_size, discrete_act, dist_class_name

    @classmethod
    @abstractmethod
    def _layer_init(cls, layer, std, bias_const, **kwargs):
        pass

    @classmethod
    @abstractmethod
    def get_dist_class(cls, dist_class_name, **kwargs):
        """
        Get the action distribution class from the class name.

        Parameters
        ----------
        dist_class_name : str
            A string specifying the name of the distribution class
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        Any
            The class constructor for the distribution
        """

    def initialize_random_generators(self, seed: Any) -> None:  # noqa: ANN401
        """
        Initialize the random generators.

        .. important:: to always be called before build_networks.

        Parameters
        ----------
        seed : Any
            The random seed to use for initialization
        """
        self.seed = seed
        self.np_random = resolve_seed(seed=seed)

    @abstractmethod
    def build_networks(self, *, observation_size: int | Sequence[int], value_network_outs: int, action_size: int, **kwargs) -> None:
        """
        Build the prediction networks.

        Parameters
        ----------
        observation_size : int | Sequence[int]
            The size of the observation space(s)
        value_network_outs : int
            The number of outputs for the value network
        action_size : int
            The size of the action space
        `**kwargs`
            Additional keyword arguments.
        """

    @abstractmethod
    def load_flat_params(self, flat_params, **kwargs):
        """
        Load a flat vector of parameters into the agent.

        Buffers are left unchanged

        Parameters
        ----------
        flat_params : Any
            The flat network parameters.
        `**kwargs`
            Additional keyword arguments.
        """

    @abstractmethod
    def get_values(self, x, *args, **kwargs):
        """
        Get the critic network value predictions.

        Parameters
        ----------
        x : Any
            The input tensor to make predictions on.
        `*args`
            Additional positional arguments.
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        Any
            The value predictions
        """

    @abstractmethod
    def get_actions(self, x, logits=None, **kwargs):
        """
        Get the action distribution and value predictions from the agent.

        Parameters
        ----------
        x : Any
            The input to get the actions for
        logits : Any, optional
            The logits for the action distribution if already calculated, default is :data:`python:None`
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        Any
            The actions based on the input
        """

    @abstractmethod
    def get_action_dist(self, x, **kwargs):
        """
        Generate the action distribution from the input.

        Parameters
        ----------
        x : Any
            The input to generate an action distribution for
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        Any
            The action distribution
        """

    @abstractmethod
    def get_actions_and_values(self, x, action=None, logits=None, **kwargs):
        """
        Get the action distribution and value predictions from the agent.

        Parameters
        ----------
        x : Any
            The input to get predictions for
        action : Any, optional
            The actions if already calculated, default is :data:`python:None`
        logits : Any, optional
            The logits if already calculated, default is :data:`python:None`
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        Any
            The action and value predictions from the agent
        """

    @abstractmethod
    def loss(self, train_batch, **kwargs):
        """
        Calculate the loss from the training batch for this agent.

        Parameters
        ----------
        train_batch : SampleBatch
            A batch of data to calculate the loss on.
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        Any
            The batch loss
        """

    @abstractmethod
    def postprocess_batch(self, batch: SampleBatch, **kwargs) -> SampleBatch:
        """Run postprocessing on a batch of data from multiple episodes.

        Parameters
        ----------
        batch : SampleBatch
            The batch to postprocess
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        SampleBatch
            The postprocessed batch
        """

    @abstractmethod
    def process_episode(self, episode_batch: SampleBatch, rewards: np.ndarray, **kwargs) -> SampleBatch:
        """
        Run processing on a batch of data from a single episode.

        Parameters
        ----------
        episode_batch : SampleBatch
            The batch for the episode data.
        rewards : np.ndarray
            The episode rewards
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        SampleBatch
            The processed episode.
        """

    def post_training_hook(self, **kwargs) -> dict:  # noqa: PLR6301
        """
        Run operations after training completes.

        Parameters
        ----------
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        dict
            A dictionary of info from the hooks
        """
        return {}

    @abstractmethod
    def train(self, mode: bool = True) -> "BaseAgent":
        """Modify the agents state by making its weights trainable and changing how predictions are made.

        Parameters
        ----------
        mode : bool, optional
            Enable or disable draining, :data:`python:True`

        Returns
        -------
        BaseAgent
            The trained agent (self)
        """

    @abstractmethod
    def reset_optimizer(self):
        """Reset the optimizer state."""

    @property
    @abstractmethod
    def flat_parameters(self):
        """Return the agents trainable parameters as a flattened vector."""

    @abstractmethod
    def learn(self, /, train_batch: SampleBatch, **kwargs) -> tuple[Any, dict]:
        """
        Update (learn) on the agent weights using the inputs.

        Parameters
        ----------
        train_batch : SampleBatch
            The training batch to learn on.
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        Any
            The training loss
        dict
            Any additional training info.
        """

    @property
    def recurrent_n(self) -> int:  # noqa: D102
        return self._recurrent_n

    @property
    def state_type(self) -> Literal["EP", "FP"]:
        """Return the RNN state type.

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
    def rnn_hidden_size(self) -> int:  # noqa: D102
        return self._rnn_hidden_size

    @property
    def recurrent(self) -> bool:
        """
        Whether this agent uses an RNN.

        Returns
        -------
        bool
            Whether this agent uses an RNN.
        """
        return self._recurrent


class BaseMultiAgent:  # noqa: PLR0904
    """
    Base class for multiagent algorithms where one agent controls multiple platforms.

    Parameters
    ----------
    platforms : Sequence[PlatformID]
        The platforms controlled by this agent
    obs_spaces : dict[PlatformID, gymnasium.spaces.Space]
        The observation space for this agent.
    act_spaces : dict[PlatformID, gymnasium.spaces.Space]
        The action space for this agent.
    critic_outs : int, optional
        The number of outputs for the critic network, default is 1.
    actor_class : type[BaseAgent], optional
        The default class to use for the individual agents, default is :class:`~algatross.agents.base.BaseAgent`
    seed : Any, optional
        The seed for randomness, default is :data:`python:None`
    `**kwargs`
        Additional keyword arguments.
    """

    platforms: Mapping[PlatformID, BaseAgent]
    """A mapping from platforms to the agents in this class which control them."""
    seed: Any
    """The random seed used for initialization."""
    np_random: np.random.Generator
    """The seeded random number generator."""

    def __init__(
        self,
        *,
        obs_spaces: gymnasium.spaces.Space,
        act_spaces: gymnasium.spaces.Space,
        critic_outs: int = 1,
        platforms: Sequence[PlatformID],
        actor_class: type[BaseAgent] = BaseAgent,
        seed: Any = None,  # noqa: ANN401
        **kwargs,
    ) -> None:
        self.initialize_random_generators(seed=seed)
        self.platforms = self.construct_platforms(
            obs_space=obs_spaces,  # type: ignore[arg-type]
            act_space=act_spaces,  # type: ignore[arg-type]
            actor_class=actor_class,
            platforms=platforms,
            critic_outs=critic_outs,
            **kwargs,
        )
        self._recurrent_n = kwargs.get("recurrent_n", 0)
        self._rnn_hidden_size = {plat_id: plat.rnn_hidden_size for plat_id, plat in self.platforms.items()}
        self._state_type = kwargs.get("state_type", "EP")
        self._recurrent = self._recurrent_n > 0

    def construct_platforms(  # noqa: PLR6301
        self,
        obs_space: dict[PlatformID, gymnasium.spaces.Space],
        act_space: dict[PlatformID, gymnasium.spaces.Space],
        actor_class: type[BaseAgent],
        platforms: Sequence[PlatformID],
        critic_outs: int = 1,
        **kwargs,
    ) -> Mapping[PlatformID, BaseAgent]:
        """Construct the platforms controlled by this agent.

        Parameters
        ----------
        obs_space : dict[PlatformID, gymnasium.spaces.Space]
            Observation spaces in the environment
        act_space : dict[PlatformID, gymnasium.spaces.Space]
            Action spaces in the environment
        actor_class : type[TorchBaseAgent]
            Base class for the actors controlled by this agent.
        platforms : Sequence[PlatformID]
            The platform IDs controlled by this agent.
        critic_outs : int
            Number of outputs for the critic.
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        Mapping[PlatformID, BaseAgent]
            Mapping from platform ID to its corresponding controller agent.
        """
        sorted_platforms = sorted(platforms)
        constructed = {}
        for platform in sorted_platforms:
            constructed[platform] = actor_class(
                obs_space=obs_space[platform],
                act_space=act_space[platform],
                critic_outs=critic_outs,
                **kwargs,
            )
        return constructed

    def initialize_random_generators(self, seed: Any = None) -> None:  # noqa: ANN401
        """
        Initialize random geenrators.

        .. important:: to always be called before build_networks.

        Parameters
        ----------
        seed : Any, optional
            The initial random seed for this agent, default is :data:`python:None`
        """
        self.seed = seed
        self.np_random = resolve_seed(seed=seed)

    def load_flat_params(self, flat_params_dict, **kwargs):
        """
        Load flattened parameters for each platform.

        Parameters
        ----------
        flat_params_dict : dict
            The dictionary of flat parameters for each platform
        `**kwargs`
            Additional keyword arguments.
        """
        for platform, flat_params in flat_params_dict.items():
            self.platforms[platform].load_flat_params(flat_params, **kwargs)

    def get_values(self, x_dict: dict[PlatformID, np.ndarray], *args, **kwargs) -> dict[PlatformID, np.ndarray]:
        """Get values for each platform.

        Parameters
        ----------
        x_dict : dict[PlatformID, np.ndarray]
            Dictionary of platforms and their observations
        `*args`
            Additional positional arguments.
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        dict[PlatformID, np.ndarray]
            The value predictions from each platforms critic
        """
        return {platform: self.platforms[platform].get_values(x, *args, **kwargs) for platform, x in x_dict.items()}

    def get_actions(
        self,
        x_dict: dict[PlatformID, np.ndarray],
        logits_dict: dict[PlatformID, np.ndarray] | None = None,
        **kwargs,
    ) -> dict[PlatformID, np.ndarray]:
        """Get actions for each platform.

        Parameters
        ----------
        x_dict : dict[PlatformID, np.ndarray]
            Dictionary of platforms and their observations
        logits_dict : dict[PlatformID, np.ndarray] | None, optional
            Dictionary of action distribution logits, :data:`python:None`
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        dict[PlatformID, np.ndarray]
            The actions to take for each platform
        """
        logits_dict = logits_dict or {}
        return {platform: self.platforms[platform].get_actions(x, logits_dict.get(platform), **kwargs) for platform, x in x_dict.items()}

    def get_action_dist(self, x_dict: dict[PlatformID, np.ndarray], **kwargs) -> dict[PlatformID, Any]:
        """Get action distributions for each platform.

        Parameters
        ----------
        x_dict : dict[PlatformID, np.ndarray]
            Dictionary of platforms and their observations
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        dict[PlatformID, Any]
            Action distributions for each platform
        """
        return {platform: self.platforms[platform].get_action_dist(x, **kwargs) for platform, x in x_dict.items()}

    def get_actions_and_values(
        self,
        x_dict: dict[PlatformID, np.ndarray],
        action_dict: dict[PlatformID, np.ndarray] | None = None,
        logits_dict: dict[PlatformID, np.ndarray] | None = None,
        **kwargs,
    ) -> dict[PlatformID, np.ndarray]:
        """Get actions and values for each platform.

        Parameters
        ----------
        x_dict : dict[PlatformID, np.ndarray]
            Dictionary of platforms and their observations
        action_dict : dict[PlatformID, np.ndarray] | None, optional
            Dictionary of actions for each platform, :data:`python:None`
        logits_dict : dict[PlatformID, np.ndarray] | None, optional
            Dictionary of action distribution logits for each platform, :data:`python:None`
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        dict[PlatformID, np.ndarray]
            The actions and values for each platform
        """
        logits_dict = logits_dict or {}
        action_dict = action_dict or {}
        return {
            platform: self.platforms[platform].get_actions_and_values(
                x,
                logits=logits_dict.get(platform),
                action=action_dict.get(platform),
                **kwargs,
            )
            for platform, x in x_dict.items()
        }

    def loss(self, train_batch_dict: Mapping[PlatformID, SampleBatch], **kwargs) -> Mapping[PlatformID, dict]:
        """Compute the loss for each platform.

        Parameters
        ----------
        train_batch_dict : Mapping[PlatformID, SampleBatch]
            Dictionary of training batches for each platform.
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        Mapping[PlatformID, dict]
            Mapping from platforms to loss info
        """
        return {platform: self.platforms[platform].loss(train_batch, **kwargs) for platform, train_batch in train_batch_dict.items()}

    def postprocess_batch(self, batch_dict: dict[PlatformID, SampleBatch], **kwargs) -> dict[PlatformID, SampleBatch]:
        """Postprocess the batches for each platform.

        Parameters
        ----------
        batch_dict : dict[PlatformID, SampleBatch]
            A dictionary of :py:class:`~ray.rllib.SampleBatch` for each platform
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        dict[PlatformID, SampleBatch]
            A dictionary of postprocessed batches.
        """
        return {platform: self.platforms[platform].postprocess_batch(batch, **kwargs) for platform, batch in batch_dict.items()}

    def process_episode(
        self,
        episode_batch_dict: dict[PlatformID, SampleBatch],
        rewards: dict[PlatformID, np.ndarray],
        **kwargs,
    ) -> dict[PlatformID, SampleBatch]:
        """Process the episode for each of the platforms.

        Parameters
        ----------
        episode_batch_dict : dict[PlatformID, SampleBatch]
            A batch of experiences for each platform
        rewards : dict[PlatformID, np.ndarray]
            The rewards for each platform
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        dict[PlatformID, SampleBatch]
            The postprocessed batch for the platform
        """
        return {
            platform: self.platforms[platform].postprocess_batch(episode_batch, rewards=rewards, **kwargs)
            for platform, episode_batch in episode_batch_dict.items()
        }

    def post_training_hook(self, **kwargs) -> dict:
        """
        Run operations after training has occurred.

        Parameters
        ----------
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        dict
            The postprocessed data
        """
        return {platform: self.platforms[platform].post_training_hook(**kwargs) for platform in self.platforms}

    def train(self, mode: bool = True) -> "BaseMultiAgent":
        """Set the training mode for each of the platforms.

        Parameters
        ----------
        mode : bool, optional
            Whether or not to set the agent to trainable, :data:`python:True`

        Returns
        -------
        BaseMultiAgent
            The agent itself
        """
        for platform in self.platforms.values():
            platform.train(mode)
        return self

    def reset_optimizer(self):
        """Reset the optimizer for each of the platforms."""
        for platform in self.platforms.values():
            platform.reset_optimizer()

    @property
    def flat_parameters(self) -> Mapping:
        """
        A mapping from each platform to their flattened parameters.

        Returns
        -------
        Mapping
            The flat parameters of each platform
        """
        return {platform_id: platform.flat_parameters for platform_id, platform in self.platforms.items()}

    @property
    def actor(self):
        """
        Return the actor network for each platform.

        Returns
        -------
        dict[PlatformID, Any]
            The actor network for each platform
        """
        return {platform_id: platform.actor for platform_id, platform in self.platforms.items()}

    @property
    def critic(self):
        """
        Return the critic network for each platform.

        Returns
        -------
        dict[PlatformID, Any]
            The critic network for each platform.
        """
        return {platform_id: platform.critic for platform_id, platform in self.platforms.items()}

    def learn(self, /, train_batch_dict: dict[PlatformID, SampleBatch], **kwargs) -> dict[PlatformID, tuple[Any, dict]]:
        """Update (learn) on the agent weights using the inputs for each platform.

        Parameters
        ----------
        train_batch_dict : dict[PlatformID, SampleBatch]
            The batch of experiences on which to learn for each platform.
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        dict[PlatformID, tuple[Any, dict]]
            The results from training on each platform.
        """
        return {
            platform: self.platforms[platform].learn(train_batch=train_batch, **kwargs)
            for platform, train_batch in train_batch_dict.items()
        }

    @property
    def recurrent_n(self) -> int:  # noqa: D102
        return self._recurrent_n

    @property
    def state_type(self) -> Literal["EP", "FP"]:
        """Return the RNN state type.

        Either:

        - :python:`"EP"` for *episode_provided*
        - :python:`"FP"` for *feature_provided*

        Returns
        -------
        Literal["EP", "FP"]
            The type of RNN state
        """
        return self._state_type

    @property
    def rnn_hidden_size(self) -> Mapping[PlatformID, int]:
        """The hidden side of each RNN layer.

        Returns
        -------
        Mapping[PlatformID, int]
            The hidden size of each RNN layer
        """
        return self._rnn_hidden_size

    @property
    def recurrent(self) -> bool:
        """
        Whether this agent uses an RNN.

        Returns
        -------
        bool
            Whether this agent uses an RNN
        """
        return self._recurrent
