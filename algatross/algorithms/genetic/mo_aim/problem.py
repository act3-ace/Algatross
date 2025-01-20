"""A module of uder-defined-problems for the MO-AIM algorithm."""

import copy
import functools
import logging

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, MutableMapping, Sequence
from typing import Any, Literal

import numpy as np

from ray.rllib.policy.sample_batch import SampleBatch

import torch

from pettingzoo.utils.env import AECEnv, ParallelEnv

from algatross.agents.base import BaseAgent
from algatross.algorithms.genetic.mo_aim.configs import BehaviorClassificationConfig
from algatross.environments.runners import BaseRunner
from algatross.utils.debugging import log_agent_params
from algatross.utils.merge_dicts import filter_keys, flatten_dicts
from algatross.utils.random import resolve_seed
from algatross.utils.types import AgentID, ConstructorData, MOAIMIslandInhabitant, NumpyRandomSeed, OptimizationTypeEnum, PlatformID


class UDP(ABC):  # noqa: PLR0904
    """
    ABC for MO-AIM UDPs which define the base methods and properties subclasses should inherit or define.

    Parameters
    ----------
    trainer_constructor_data : ConstructorData
        Constructors for the trainer objects
    training_agents : Sequence[AgentID]
        The agents which will be trainable on this problem.
    conspecific_data_keys : list[str]
        The keys of data to extract from the batch to use as conspecific data.
    fitness_metric_keys : Sequence[str]
        The keys of data to use for fitness metrics
    fitness_metric_optimization_type : Literal["min", "max"] | list[Literal["min", "max"]]
        The type of optimization for each  fitness metric (minimization/maximization)
    fitness_multiplier : np.ndarray | None, optional
        The multiplier (gain/weighting) for each fitness metric, default is :data:`python:None`.
    fitness_reduce_fn : Callable | str | None, optional
        The function to call in order to reduce a vector of fitness metric data into a scalar, by default :python:`"sum"`
    agents_to_platforms : MutableMapping[AgentID, list[PlatformID]] | None, optional
        Mapping from agents to platforms which control them, :data:`python:None`.
    ally_teams : MutableMapping[str, list[AgentID]] | None, optional
        Mapping from ally team names to agents on the team, :data:`python:None`.
    opponent_teams : MutableMapping[str, list[AgentID]] | None, optional
        Mapping from opponent team names to agents on the team, :data:`python:None`.
    seed : NumpyRandomSeed | None, optional
        The seed for randomness, :data:`python:None`.
    max_envs : int, optional
        The maximum number of parallel environments to run, by default 2.
    `*args`
        Additional positional arguments
    `**kwargs`
        Additional keyword arguments
    """

    trainer_class: Callable | type[BaseRunner] | type
    """The class to use when training the agents."""
    trainer_config: dict
    """The configuration dictionary for the trainer class."""
    optimization_type: OptimizationTypeEnum
    """The overall type of optimization for this problem.

    E.G. RL is usually maximization, function optimization and operations research is typically minimization.
    """
    allied_teams: MutableMapping[str, list[AgentID]]
    """The mapping from allied team names to the agents on the team."""
    opponent_teams: MutableMapping[str, list[AgentID]]
    """The mapping from opponent team names to the agents on the team."""
    fitness_metric_keys: Sequence[str]
    """The keys of data to use for fitness metrics."""
    fitness_multiplier: np.ndarray
    """The array of multipliers to apply to each of the fitness metrics."""
    behavior_classification_config: BehaviorClassificationConfig
    """The configuration to use when setting up behavior classification."""
    runner: BaseRunner
    """The runner to use on this problem."""
    logger: logging.Logger
    """The logger for this problem."""

    _conspecific_data_keys: list[str]
    _env: ParallelEnv | AECEnv
    _training_agents: set[AgentID]
    _opponent_agents: set[AgentID]
    _rollout_buffers: dict[AgentID, list[SampleBatch]]
    _fitness_sign: np.ndarray
    _agents_to_platforms: MutableMapping[AgentID, list[PlatformID]]

    def __init__(
        self,
        trainer_constructor_data: ConstructorData,
        training_agents: Sequence[AgentID],
        conspecific_data_keys: list[str],
        fitness_metric_keys: Sequence[str],
        fitness_metric_optimization_type: Literal["min", "max"] | list[Literal["min", "max"]],
        fitness_multiplier: np.ndarray | None = None,
        fitness_reduce_fn: Callable | str | None = "sum",
        agents_to_platforms: MutableMapping[AgentID, list[PlatformID]] | None = None,
        ally_teams: MutableMapping[str, list[AgentID]] | None = None,
        opponent_teams: MutableMapping[str, list[AgentID]] | None = None,
        seed: NumpyRandomSeed | None = None,
        max_envs: int = 2,
        *args,
        **kwargs,
    ) -> None:
        self.logger = logging.getLogger("ray")
        self.logger.info(f"UDP got seed={seed}")
        # self._numpy_generator = get_generators(seed, seed_global=False)[0]  # noqa: ERA001
        self._numpy_generator = resolve_seed(seed)  # type: ignore[arg-type]
        self._n_envs = int(max_envs)

        self._agents_to_platforms = agents_to_platforms or {}
        self.agent_templates_map: dict[AgentID, BaseAgent] = {}

        self.trainer_class = trainer_constructor_data.constructor
        self.trainer_config = trainer_constructor_data.config
        self.trainer_config["seed"] = self._numpy_generator  # .spawn(1).pop()

        self._conspecific_data_keys = conspecific_data_keys

        self.allied_teams = ally_teams or {"allies_0": list(training_agents)}
        self.opponent_teams = opponent_teams or {}

        self._ally_agents = set()
        self._opponent_agents = set()

        self._training_agents = set(training_agents)
        for team in self.allied_teams.values():
            self._ally_agents.update(set(team))
        for team in self.opponent_teams.values():
            self._opponent_agents.update(set(team))
        self.fitness_metric_keys = fitness_metric_keys
        if isinstance(fitness_metric_optimization_type, str):
            fitness_opt_enum = [OptimizationTypeEnum(fitness_metric_optimization_type.lower())] * len(fitness_metric_keys)
        else:
            # use zip(..., strict=True) to ensure equal length iterables
            fitness_opt_enum = [
                OptimizationTypeEnum(fo_type.lower())
                for _, fo_type in zip(self.fitness_metric_keys, fitness_metric_optimization_type, strict=True)
            ]

        self._fitness_metric_opt_enum = fitness_opt_enum
        self._fitness_sign = np.array([1 if opt_type == OptimizationTypeEnum.MAX else -1 for opt_type in fitness_opt_enum])
        self._fitness_reduce_fn: Callable = (
            functools.partial(
                functools.partial(lambda x, y, *x_args, **x_kwargs: getattr(y, x)(*x_args, **x_kwargs), fitness_reduce_fn),
                axis=-1,
            )
            if isinstance(fitness_reduce_fn, str)
            else fitness_reduce_fn
        )

        if fitness_multiplier is None:
            fitness_multiplier = np.array([1.0] * len(self.fitness_metric_keys), dtype=np.float32)

        fitness_multiplier = np.array([0.0 if fm is None else fm for fm in fitness_multiplier], dtype=np.float32)

        self.fitness_multiplier = fitness_multiplier * self.fitness_sign

        self.behavior_classification_config = BehaviorClassificationConfig(
            **filter_keys(BehaviorClassificationConfig, **kwargs.get("behavior_classification", {})),
        )

        self._rollout_buffers = {}
        # TODO: this runner is permanent
        self.runner = self.get_runner()

    def get_runner(self):  # noqa: D102
        if issubclass(self.trainer_class, BaseRunner):
            # share the same class
            runner = self.trainer_class(
                n_envs=self._n_envs,
                train_config=self.trainer_config.get("train_config", {}),
                rollout_config=self.trainer_config.get("rollout_config", {}),
                seed=self._numpy_generator,
            )
        else:
            msg = f"Trainer class ({self.trainer_class}) must inherit from {BaseRunner}"
            raise TypeError(msg)
        return runner

    def get_state(self) -> dict:
        """Get the UDP state.

        Returns
        -------
        dict
            The UDP state
        """
        ignore_keys = ["train_rl", "rollout_rl", "runner", "cleanrl_trainer", "_rollout_buffers", "agent_templates_map"]
        state = copy.deepcopy({k: v for k, v in self.__dict__.items() if k not in ignore_keys})
        # TODO: we may have the case where runner is ephemeral within evolve(), so this should get called seperately.
        state["runner_state"] = self.runner.get_state()
        return state

    def set_state(self, state: dict):
        """
        Set the UDP state.

        Parameters
        ----------
        state : dict
            The state to set for the problem.
        """
        for key, value in state.items():
            if key != "runner_state":
                setattr(self, key, value)

        if state.get("runner_state") is None:
            logger = logging.getLogger("ray")
            logger.warning("No runner state found in the checkpoint for this island! Results may be non-determinstic.")
        else:
            self.runner.set_state(state["runner_state"])

    def initialize_agent_templates(self, agents: dict[AgentID, BaseAgent]):
        """Initialize the template agents used in rollouts.

        Parameters
        ----------
        agents : dict[AgentID, BaseAgent]
            MutableMapping from agent ids to the base agent objects.
        """
        self.agent_templates_map = dict(agents.items())

    def set_solution_dim(self, agents: dict[AgentID, BaseAgent]):
        """
        Set the agents solution dim for the problem.

        Parameters
        ----------
        agents : dict[AgentID, BaseAgent]
            MutableMapping from agent ids to the base agent objects.
        """
        self._solution_dim = {
            agent_id: np.prod(agent.flat_parameters.shape) for agent_id, agent in agents.items() if agent_id in self.training_agents
        }

    def load_team(self, team: MutableMapping[AgentID, MOAIMIslandInhabitant]) -> dict[AgentID, BaseAgent]:
        """
        Load the flat team weights into initialized agents.

        Parameters
        ----------
        team : MutableMapping[AgentID, MOAIMIslandInhabitant]
            A mapping from AgentID to island inhabitant

        Returns
        -------
        dict[AgentID, BaseAgent]
            The loaded agents
        """
        logger = logging.getLogger("ray")
        loaded_agent_map = copy.deepcopy({agent_id: self.agent_templates_map[agent_id] for agent_id in team})

        for agent_id, inhabitant in team.items():
            logger.debug(f"load team agent {agent_id}")
            loaded_agent_map[agent_id].load_flat_params(inhabitant.genome)
            self._solution_dim[agent_id] = np.prod(loaded_agent_map[agent_id].flat_parameters.shape)
            loaded_agent_map[agent_id].reset_optimizer()

        return loaded_agent_map

    def set_trainable_agents(self, agents: Sequence[AgentID], agent_map: dict, specified_only: bool = True):
        """
        Set the given agents as trainable.

        Parameters
        ----------
        agents : Sequence[AgentID]
            The IDs of the agents to set as trainable
        agent_map : dict
            The mapping of agent names to agents
        specified_only : bool, optional
            Whether only the agents specified should be trainable, :data:`python:True`. If False, any currently trainable agents
            ares still trainable.
        """
        self.validate_trainable_agents(agents, agent_map)
        self._training_agents = set(agents).union(set() if specified_only else self.training_agents)

    def set_ally_agents(self, agents: Sequence[AgentID], agent_map: dict, specified_only: bool = True):
        """
        Set the given agents as allied.

        Parameters
        ----------
        agents : Sequence[AgentID]
            The IDs of the agents to set as allied
        agent_map : dict
            The mapping of agent names to agents
        specified_only : bool, optional
            Whether only the agents specified should be trainable, :data:`python:True`. If False, any currently trainable agents
            ares still trainable.
        """
        self.validate_trainable_agents(agents, agent_map)
        self._ally_agents = set(agents).union(set() if specified_only else self.ally_agents)

    def set_opponent_agents(self, agents: Sequence[AgentID], agent_map: dict, specified_only: bool = True):
        """
        Set the given agents as opponents.

        Parameters
        ----------
        agents : Sequence[AgentID]
            The IDs of the agents to set as trainable
        agent_map : dict
            The mapping of agent names to agents
        specified_only : bool, optional
            Whether only the agents specified should be trainable, :data:`python:True`. If False, any currently trainable agents
            ares still trainable.
        """
        self.validate_opponent_agents(agents, agent_map)
        self._opponent_agents = set(agents).union(set() if specified_only else self.opponent_agents)

    @abstractmethod
    def fitness(self, team: dict[AgentID, MOAIMIslandInhabitant], train: bool = True, **kwargs) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Calculate the fitness for the genome(s).

        Parameters
        ----------
        team : dict[AgentID, MOAIMIslandInhabitant]
            The dictonary of agent names and inhabitants to train
        train : bool, Optional
            Whether to train the algorithm as well as evaluate fitness, defaults to :data:`python:True`
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        tuple[np.ndarray, dict[str, Any]]
            The fitness of the given genome(s) and any additional info
        """

    def _calc_fitness(self, rollout_data: dict[AgentID, list[SampleBatch]]) -> tuple[np.ndarray, dict[str, Any]]:
        # Team fitness [F]:
        # > for each agent: stack fitness metrics
        # > [F] <- [A]x[F] (mean)
        # >   for each fitness metric: stack rollout by episode
        # >   [A, F] <- [E]x[A, F] (mean)
        # >     for each episode: stack the metric over the trajectory
        # >     [E, A, F] <- [E, A, F]x[T] (sum)
        f = np.stack([
            np.stack([
                np.stack([f[fitness_metric].sum() for f in rollout_data[agent_id]]).mean(axis=0, keepdims=False)
                for fitness_metric in self.fitness_metric_keys
            ])
            for agent_id in self.training_agents
        ])
        assert f.shape[-1] == len(self.fitness_metric_keys)  # noqa: S101
        f = f if self.fitness_multiplier is None else f * self.fitness_multiplier[None]
        mean_f = f.mean(axis=0, keepdims=False)
        infos = {
            f"agent/{agent_id}": {fitness_metric: agent_f[idx] for idx, fitness_metric in enumerate(self.fitness_metric_keys)}
            for agent_id, agent_f in zip(self.training_agents, f, strict=True)
        }
        infos.update({fitness_metric: mean_f[idx] for idx, fitness_metric in enumerate(self.fitness_metric_keys)})
        return f if self._fitness_reduce_fn is None else self._fitness_reduce_fn(f), infos

    def fitness_from_sample_batch(self, sample_batch: SampleBatch) -> np.ndarray:
        """
        Calculate the fitness from a batch of rollout data.

        Parameters
        ----------
        sample_batch : SampleBatch
            The rollout data.

        Returns
        -------
        np.ndarray
            The fitness
        """
        f = np.stack([
            np.stack([f[fitness_metric].mean() for f in sample_batch.split_by_episode()]).mean(axis=0, keepdims=False)
            for fitness_metric in self.fitness_metric_keys
        ])
        assert f.shape[0] == len(self.fitness_metric_keys)  # noqa: S101
        f = f if self.fitness_multiplier is None else f * self.fitness_multiplier[None]
        return f if self._fitness_reduce_fn is None else self._fitness_reduce_fn(f)

    @property
    def conspecific_data_keys(self) -> list[str]:
        """The keys to use for retrieval of conspecific data from a SampleBatch.

        Returns
        -------
        list[str]
            The keys to use for retrieval of conspecific data from a SampleBatch.
        """
        return self._conspecific_data_keys

    @property
    def rollout_buffers(self) -> dict[AgentID, list[SampleBatch]]:
        """A mapping from AgentIDs to their corresponding rollout buffer.

        Returns
        -------
        dict[AgentID, list[SampleBatch]]
            A mapping from AgentIDs to their corresponding rollout buffer.
        """
        return self._rollout_buffers

    @property
    def training_agents(self) -> set[AgentID]:
        """The agents currently set as trainable.

        Returns
        -------
        set[AgentID]
            The agents currently set as trainable.
        """
        return self._training_agents

    @training_agents.setter
    def training_agents(self, agents: Sequence[AgentID]):
        self.set_trainable_agents(agents, agent_map=self.agent_templates_map)

    @property
    def ally_agents(self) -> set[AgentID]:
        """
        The agents currently set as allies.

        Returns
        -------
        set[AgentID]
            The agents currently set as allies.
        """
        return self._ally_agents

    @ally_agents.setter
    def ally_agents(self, agents: Sequence[AgentID]):
        self.set_ally_agents(agents, self.agent_templates_map)

    @property
    def opponent_agents(self) -> set[AgentID]:
        """
        The agents currently set as opponents.

        Returns
        -------
        set[AgentID]
            The agents currently set as opponents.
        """
        return self._opponent_agents

    @opponent_agents.setter
    def opponent_agents(self, agents: Sequence[AgentID]):
        self.set_opponent_agents(agents, agent_map=self.agent_templates_map)

    @property
    def solution_dim(self) -> dict[AgentID, int]:
        """The dimension of the solution space, i.e. the size of the genome or decision variable.

        Returns
        -------
        dict[AgentID, int]
            The dimension of the solution space, i.e. the size of the genome or decision variable.
        """
        return self._solution_dim

    @property
    def fitness_sign(self) -> np.ndarray:
        """The sign of each component of the fitness function.

        since RL is typically formulated as a maximization problem, and ML is classically a minimization problem, we track
        a sign for each objective function which assumes a minimization formulation. The fitness from the rollout is then
        multiplied by this sign before possibly being reduced to a scalar.

        This way the fitness can be passed as-is to a minimizing algorithm or optimizer (such as a classical ML algorithm),
        or multiplied by -1 to be used with a maximizing algorithm or optimizer (such as an RL algorithm).

        Returns
        -------
        np.ndarray
            An array containing the sign by which each objective in the fitness function is multiplied. A 1 corresponds to
            An objective which we would like to minimize while a -1 corresponds to an objective we wish to maximize.
        """
        return self._fitness_sign

    @abstractmethod
    def validate_trainable_agents(self, agents: Sequence[AgentID] | None = None, agent_map: dict | None = None):
        """Run some validation on the trainable agents.

        Parameters
        ----------
        agents : Sequence[AgentID] | None, optional
            The iterable of AgentIDs we wish to validate, :data:`python:None`
        agent_map : dict | None, optional
            The agent map to validate, default is :data:`python:None`
        """

    @abstractmethod
    def validate_ally_agents(self, agents: Sequence[AgentID] | None = None, agent_map: dict | None = None):
        """
        Run some validation on the ally agents.

        Parameters
        ----------
        agents : Sequence[AgentID] | None, optional
            The iterable of AgentIDs we wish to validate, :data:`python:None`
        agent_map : dict | None, optional
            The agent map to validate, default is :data:`python:None`
        """

    @abstractmethod
    def validate_opponent_agents(self, agents: Sequence[AgentID] | None = None, agent_map: dict | None = None):
        """
        Run some validation on the opponent agents.

        Parameters
        ----------
        agents : Sequence[AgentID] | None, optional
            The iterable of AgentIDs we wish to validate, :data:`python:None`
        agent_map : dict | None, optional
            The agent map to validate, default is :data:`python:None`
        """

    @abstractmethod
    def validate_decision_variable(self, team: dict[AgentID, MOAIMIslandInhabitant]):
        """Run some validation on the decision variable.

        Parameters
        ----------
        team : dict[AgentID, MOAIMIslandInhabitant]
            The decision variable we wish to validate.
        """

    @abstractmethod
    def validate_opponent_variable(self, x: np.ndarray | list[np.ndarray]):
        """
        Run some validation on the opponent variables.

        Parameters
        ----------
        x : np.ndarray | list[np.ndarray]
            The opponent variables we wish to validate.
        """


class MOAIMIslandUDP(UDP):
    """
    MOAIMIslandUDP a UDP for MO-AIM Islands.

    Parameters
    ----------
    trainer_constructor_data : ConstructorData
        Constructors for the trainer objects
    training_agents : Sequence[AgentID]
        The agents which will be trainable on this problem.
    conspecific_data_keys : list[str]
        The keys of data to extract from the batch to use as conspecific data.
    fitness_metric_keys : Sequence[str]
        The keys of data to use for fitness metrics
    fitness_metric_optimization_type : Literal["min", "max"] | Sequence[Literal["min", "max"]]
        The type of optimization for each  fitness metric (minimization/maximization)
    fitness_multiplier : np.ndarray | None, optional
        The multiplier (gain/weighting) for each fitness metric, default is :data:`python:None`.
    fitness_reduce_fn : Callable | str | None, optional
        The function to call in order to reduce a vector of fitness metric data into a scalar, by default :python:`"sum"`
    agents_to_platforms : MutableMapping[AgentID, Sequence[PlatformID]] | None, optional
        Mapping from agents to platforms which control them, :data:`python:None`.
    ally_teams : MutableMapping[str, Sequence[AgentID]] | None, optional
        Mapping from ally team names to agents on the team, :data:`python:None`.
    opponent_teams : MutableMapping[str, Sequence[AgentID]] | None, optional
        Mapping from opponent team names to agents on the team, :data:`python:None`.
    seed : NumpyRandomSeed | None, optional
        The seed for randomness, :data:`python:None`.
    max_envs : int, optional
        The maximum number of parallel environments to run, by default 2.
    `*args`
        Additional positional arguments
    `**kwargs`
        Additional keyword arguments
    """

    _conspecific_weights: np.ndarray

    def fitness(  # type: ignore[override] # noqa: D102, PLR0915
        self,
        # x: np.ndarray,
        team: dict[AgentID, MOAIMIslandInhabitant],
        envs: list,
        train: bool = True,
        training_iterations: int = 1,
        visualize: bool = False,
        **kwargs,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        logger = logging.getLogger("ray")
        logger.debug(f"Team: {team.keys()}")

        self.validate_decision_variable(team)
        # initilat team as cleanrl agents
        loaded_team = self.load_team(team)

        for agent in loaded_team.values():
            # set all agents to eval mode. Let runner handle switching
            agent.train(mode=False)

        infos = defaultdict(list)
        # agent_map: dict = self._agent_map  # noqa: ERA001
        results: dict[AgentID, dict[str, SampleBatch | Any]]

        rollout_buffers: dict[AgentID, list[SampleBatch]] = defaultdict(list)

        # TODO: handle ephemeral runner's tape index
        # runner = self.get_runner()  # noqa: ERA001

        for _it in range(training_iterations):
            logger.info(f"\t === UDP iteration {_it}")
            logger.debug("agent map UDP")
            log_agent_params(loaded_team, logger.debug)

            loaded_team, results, rendered_episodes = self.runner(
                agent_map=loaded_team,
                trainable_agents=self.training_agents,
                opponent_agents=self.opponent_agents,
                reward_metrics=dict.fromkeys(loaded_team, self.fitness_metric_keys),
                reward_metric_gains=dict.fromkeys(loaded_team, self.fitness_multiplier),
                train=train,
                visualize=visualize,
                envs=envs,
                **kwargs,
            )
            flat = flatten_dicts({
                f"agent/{agent_id}/{'training' if train and 'training_stats' in res else 'rollout'}": res[
                    "training_stats" if train and "training_stats" in res else "rollout_stats"
                ]
                for agent_id, res in results.items()
            })
            for path, info in flat.items():
                infos[path].append(copy.deepcopy(info))
                if "total_loss" in path:
                    logger.debug(path)
                    logger.debug(info)

            infos["rendered_episodes"].extend(rendered_episodes)
            # self.logger.info(f"Size of UDP infos {total_size(infos)} bytes")  # noqa: ERA001

        if train:
            # evaluate the agent once
            for agent in loaded_team.values():
                agent.train(False)

            _, results, _ = self.runner(
                agent_map=loaded_team,
                trainable_agents=self.training_agents,
                opponent_agents=self.opponent_agents,
                reward_metrics=dict.fromkeys(loaded_team, self.fitness_metric_keys),
                reward_metric_gains=dict.fromkeys(loaded_team, self.fitness_multiplier),
                train=False,
                visualize=False,
                envs=envs,
                **kwargs,
            )

            flat = flatten_dicts({
                f"agent/{agent_id}/{'training' if train and 'training_stats' in res else 'rollout'}": res[
                    "training_stats" if train and "training_stats" in res else "rollout_stats"
                ]
                for agent_id, res in results.items()
            })
            for path, info in flat.items():
                infos[path].append(copy.deepcopy(info))

        def collect_rollout_data(
            rollout_buffers=rollout_buffers,
            results=results,
            trajectory_length=self.behavior_classification_config.trajectory_length,
            num_samples=self.behavior_classification_config.num_samples,
        ):
            for agent_id in self.training_agents:
                if len(rollout_buffers[agent_id]) < num_samples:
                    rollout_buffers[agent_id].extend([
                        traj[-trajectory_length:].copy()
                        for traj in results[agent_id]["extra_info"]["rollout_buffer"].split_by_episode()
                        if len(traj) >= trajectory_length
                    ])

        collect_rollout_data()
        while any(len(x) < self.behavior_classification_config.num_samples for x in rollout_buffers.values()):
            _, results, _ = self.runner(
                agent_map=loaded_team,
                trainable_agents=self.training_agents,
                opponent_agents=self.opponent_agents,
                reward_metrics=dict.fromkeys(loaded_team, self.fitness_metric_keys),
                reward_metric_gains=dict.fromkeys(loaded_team, self.fitness_multiplier),
                train=False,
                envs=envs,
                **kwargs,
            )
            collect_rollout_data()

        for path, info in infos.items():
            if path == "rendered_episodes":
                continue

            stacked = np.stack(info)
            if len(stacked.shape) > 1:
                infos[path] = np.concatenate(stacked).copy()
            else:
                infos[path] = stacked.copy()

        # back to inhabitant format
        for agent_id, agent in team.items():  # type: ignore[assignment]
            if agent_id in self.training_agents:
                if kwargs.get("debug", False):
                    torch_bad_weight = (
                        torch.any(torch.isinf(loaded_team[agent_id].flat_parameters)).item()
                        or torch.any(torch.isnan(loaded_team[agent_id].flat_parameters)).item()
                    )
                    agent.genome = loaded_team[agent_id].flat_parameters.numpy()  # TODO: (need numpy() or get weird attr bug)
                    np_bad_weight = np.any(np.isnan(agent.genome)) or np.any(np.isinf(agent.genome))
                    if np_bad_weight or torch_bad_weight:
                        cause = "np" if np_bad_weight else "torch"
                        msg = f"Bad weight found in agent {agent_id}, stopping!! caused by {cause}"
                        logger.info(msg)
                else:
                    agent.genome = loaded_team[agent_id].flat_parameters.numpy()

        # make sure we clear the buffers before each rollout
        self._rollout_buffers.clear()
        self._rollout_buffers.update(rollout_buffers.items())
        fitness, extra_infos = self._calc_fitness(self._rollout_buffers)
        # self.logger.info(f"Size of UDP {total_size(self)} bytes")  # noqa: ERA001
        infos.update(extra_infos)
        del loaded_team
        del results
        # TODO: handle this case
        # del runner

        infos["teams"] = team  # type: ignore[assignment]
        return fitness, infos

    @property
    def conspecific_weights(self) -> np.ndarray:
        """The conspecific weights for the species on this island.

        Returns
        -------
        np.ndarray
            The conspecific weight vector
        """
        return self._conspecific_weights

    def validate_trainable_agents(self, agents: Sequence[AgentID] | None = None, agent_map: dict | None = None):  # noqa: D102, ARG002
        agents_len = len(self._training_agents if agents is None else agents)
        if agents_len != 1:
            msg = f"Invalid number of trainable agents. Expected 1, got {agents_len}"
            raise ValueError(msg)

    def validate_ally_agents(self, agents: Sequence[AgentID] | None = None, agent_map: dict | None = None):  # noqa: D102
        agents_len = len(self._ally_agents if agents is None else agents)
        if agents_len >= len(agent_map) or agents_len > len(self._ally_agents):
            msg = f"Invalid number of opponent agents. Expected at most {len(agent_map) - 1} or {len(self._ally_agents)}, got {agents_len}"
            raise ValueError(msg)

    def validate_opponent_agents(self, agents: Sequence[AgentID] | None = None, agent_map: dict | None = None):  # noqa: D102
        agents_len = len(self._opponent_agents if agents is None else agents)
        if agents_len >= len(agent_map) or agents_len > len(self._opponent_agents):
            msg = (
                f"Invalid number of opponent agents. Expected at most {len(agent_map) - 1} or "
                f"{len(self._opponent_agents)}, got {agents_len}"
            )
            raise ValueError(msg)

    def validate_decision_variable(self, team: dict[AgentID, MOAIMIslandInhabitant]):  # noqa: D102
        if set(team.keys()).intersection(set(self.training_agents)) != set(self.training_agents):
            msg = f"Invalid number of decision variables. Expected at least {list(self.training_agents)}, got {list(team)} instead"
            raise ValueError(msg)

        wrong_shapes = {}
        for agent_id, agent in team.items():
            if agent_id not in self.training_agents:
                continue

            if self.solution_dim[agent_id] != np.prod(agent.genome.shape):
                wrong_shapes[agent_id] = (self.solution_dim[agent_id], np.prod(agent.genome.shape))
        if wrong_shapes:
            msg = "Invalid decision variable shape. Expected:" + "\n".join([
                f"\tAgent {agent_id} to have shape {correct}, got {actual} instead" for agent_id, (correct, actual) in wrong_shapes.items()
            ])
            raise ValueError(msg)

    def validate_opponent_variable(self, x: np.ndarray | list[np.ndarray]):  # noqa: D102
        if len(x) != len(self.opponent_agents):
            msg = f"Invalid opponent variable. Expected {len(self.opponent_agents)} variables, got {len(x)} instead"
            raise ValueError(msg)


class MOAIMMainlandUDP(UDP):
    """MOAIMIslandUDP a UDP for MO-AIM Mainlands.

    Parameters
    ----------
    trainer_constructor_data : ConstructorData
        Constructors for the trainer objects
    training_agents : Sequence[AgentID]
        The agents which will be trainable on this problem.
    conspecific_data_keys : list[str]
        The keys of data to extract from the batch to use as conspecific data.
    fitness_metric_keys : Sequence[str]
        The keys of data to use for fitness metrics
    fitness_metric_optimization_type : Literal["min", "max"] | Sequence[Literal["min", "max"]]
        The type of optimization for each  fitness metric (minimization/maximization)
    fitness_multiplier : np.ndarray | None, optional
        The multiplier (gain/weighting) for each fitness metric, default is :data:`python:None`.
    fitness_reduce_fn : Callable | str | None, optional
        The function to call in order to reduce a vector of fitness metric data into a scalar, by default :python:`"sum"`
    agents_to_platforms : MutableMapping[AgentID, Sequence[PlatformID]] | None, optional
        Mapping from agents to platforms which control them, :data:`python:None`.
    ally_teams : MutableMapping[str, Sequence[AgentID]] | None, optional
        Mapping from ally team names to agents on the team, :data:`python:None`.
    opponent_teams : MutableMapping[str, Sequence[AgentID]] | None, optional
        Mapping from opponent team names to agents on the team, :data:`python:None`.
    seed : NumpyRandomSeed | None, optional
        The seed for randomness, :data:`python:None`.
    max_envs : int, optional
        The maximum number of parallel environments to run, by default 2.
    `*args`
        Additional positional arguments
    `**kwargs`
        Additional keyword arguments
    """

    def fitness(  # type: ignore[override] # noqa: D102
        self,
        team: dict[AgentID, MOAIMIslandInhabitant],
        envs: list,
        train: bool = True,
        visualize: bool = False,
        **kwargs,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        self.validate_decision_variable(team)
        # load weights and update agent ID map
        infos = defaultdict(list)
        # agent_weight_map = {agent.name: agent.genome.flatten() for agent in team.values()}  # noqa: ERA001
        # self._agent_id_map.clear()  # noqa: ERA001
        loaded_team = self.load_team(team)
        rollout_buffers: dict[AgentID, list[SampleBatch]] = defaultdict(list)
        for agent in loaded_team.values():
            agent.train(mode=train)

        # generate the rollout and update the local buffers
        _, results, rendered_episodes = self.runner(
            agent_map=loaded_team,
            trainable_agents=self.training_agents,
            opponent_agents=self.opponent_agents,
            reward_metrics=dict.fromkeys(loaded_team, self.fitness_metric_keys),
            reward_metric_gains=dict.fromkeys(loaded_team, self.fitness_multiplier),
            visualize=visualize,
            train=False,
            envs=envs,
            **kwargs,
        )

        infos["rendered_episodes"].extend(rendered_episodes)

        def collect_rollout_data(
            rollout_buffers=rollout_buffers,
            results=results,
            trajectory_length=self.behavior_classification_config.trajectory_length,
            num_samples=self.behavior_classification_config.num_samples,
        ):
            for agent_id in self.training_agents:
                if len(rollout_buffers[agent_id]) < num_samples:
                    rollout_buffers[agent_id].extend([
                        traj[len(traj) - trajectory_length : len(traj)]
                        for traj in results[agent_id]["extra_info"]["rollout_buffer"].split_by_episode()
                        if len(traj) >= trajectory_length
                    ])

        collect_rollout_data()

        while any(len(x) < self.behavior_classification_config.num_samples for x in rollout_buffers.values()):
            _, results, _ = self.runner(
                agent_map=loaded_team,
                trainable_agents=self.training_agents,
                opponent_agents=self.opponent_agents,
                reward_metrics=dict.fromkeys(loaded_team, self.fitness_metric_keys),
                reward_metric_gains=dict.fromkeys(loaded_team, self.fitness_multiplier),
                train=False,
                visualize=False,
                envs=envs,
                **kwargs,
            )
            collect_rollout_data()

        del collect_rollout_data

        self._rollout_buffers.clear()
        self._rollout_buffers.update(rollout_buffers)

        flat = flatten_dicts({
            f"agent/{agent_id}/{'training' if train and 'training_stats' in res else 'rollout'}": res[
                "training_stats" if train and "training_stats" in res else "rollout_stats"
            ]
            for agent_id, res in results.items()
        })
        for path, info in flat.items():
            infos[path] = info
        fitness, extra_infos = self._calc_fitness(self._rollout_buffers)
        infos.update(extra_infos)
        # self.logger.info(f"Size of UDP {total_size(self)} bytes")  # noqa: ERA001
        # TODO: handle this case
        # del runner

        return fitness.mean(axis=0), infos

    def validate_trainable_agents(self, agents: Sequence[AgentID] | None = None, agent_map: dict | None = None):  # noqa: D102, ARG002
        agents_len = len(self._training_agents if agents is None else agents)
        if agents_len < 1:
            msg = f"Invalid number of trainable agents. Expected at least 1, got {agents_len}"
            raise ValueError(msg)

    def validate_ally_agents(self, agents: Sequence[AgentID] | None = None, agent_map: dict | None = None):  # noqa: D102
        agents_len = len(self._ally_agents if agents is None else agents)
        if agents_len >= len(agent_map) or agents_len > len(self._ally_agents):
            msg = f"Invalid number of opponent agents. Expected at most {len(agent_map) - 1} or {len(self._ally_agents)}, got {agents_len}"
            raise ValueError(msg)

    def validate_opponent_agents(self, agents: Sequence[AgentID] | None = None, agent_map: dict | None = None):  # noqa: D102
        agents_len = len(self._opponent_agents if agents is None else agents)
        if agents_len >= len(agent_map) or agents_len > len(self._opponent_agents):
            msg = (
                f"Invalid number of opponent agents. Expected at most {len(agent_map) - 1} or "
                f"{len(self._opponent_agents)}, got {agents_len}"
            )
            raise ValueError(msg)

    def validate_decision_variable(self, team: dict[AgentID, MOAIMIslandInhabitant]):  # noqa: D102
        if set(team.keys()).intersection(set(self.training_agents)) != set(self.training_agents):
            msg = f"Invalid number of decision variables. Expected at least {list(self._training_agents)}, got {list(team)}"
            raise ValueError(msg)

    def validate_opponent_variable(self, x: np.ndarray | list[np.ndarray]):  # noqa: D102
        if len(x) != len(self.opponent_agents):
            msg = f"Invalid opponent variable. Expected {len(self.opponent_agents)} variables, got {len(x)} instead"
            raise ValueError(msg)

    @property
    def teammate_buffers(self) -> dict[AgentID, list[SampleBatch]]:
        """The rollout buffers for each teammate.

        Returns
        -------
        dict[AgentID, list[SampleBatch]]
            A mapping from teammate ID to its rollout buffer.
        """
        return {k: self._rollout_buffers[k].copy() for k in self.training_agents}

    @property
    def fitness_sign(self) -> np.ndarray:  # noqa: D102
        # return the negative of the fitness since this is a minimization problem
        return -self._fitness_sign
