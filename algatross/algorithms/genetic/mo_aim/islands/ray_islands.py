"""Module containing MO-AIM user-defined islands."""

import copy
import logging

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

import ray

from ray.tune.registry import ENV_CREATOR, _global_registry  # noqa: PLC2701
from ray.util.queue import Queue

from algatross.actors.ray_actor import RayActor
from algatross.algorithms.genetic.mo_aim.islands.base import MOAIMIslandUDI, MOAIMMainlandUDI, RemoteUDI
from algatross.algorithms.genetic.mo_aim.population import MOAIMIslandPopulation, MOAIMMainlandPopulation, PopulationServer
from algatross.algorithms.genetic.mo_aim.problem import UDP
from algatross.environments.runners import ManagedEnvironmentsContext
from algatross.utils.types import AgentID, ConstructorData, IslandID, MainlandID, PlatformID

logger = logging.getLogger("ray")


class RayRemoteUDI:
    """
    A class which adds ray context information to RemoteUDIs.

    Subclasses must inherit from this class first in order to have the correct behavior.

    Parameters
    ----------
    `*args` : list
        Additional positional arguments.
    `**kwargs` : dict
        Additional keyword arguments.
    """

    _context: ray.runtime_context.RuntimeContext

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._context = ray.get_runtime_context()
        self._namespace = self._context.namespace
        self._actor_name = self._context.get_actor_name()

    def __repr__(self) -> str:  # noqa: D105
        return f"{self._actor_name} {RemoteUDI.__repr__(self)}"  # type: ignore[arg-type]


class RayMOAIMIslandUDI(MOAIMIslandUDI):
    """
    A MOAIMIslandUDI implemented in the Ray backend.

    Parameters
    ----------
    env_name : str
        The name of the environment to pull from the global registry.
    `**kwargs` : dict
        Additional keyword arguments.
    """

    def __init__(self, /, env_name: str, **kwargs):
        self.env_name = env_name
        super().__init__(**kwargs)

    def evolve(
        self,
        pop: MOAIMIslandPopulation | MOAIMMainlandPopulation,
        n: int = 1,
        epoch: int = 0,
        **kwargs,
    ) -> tuple[MOAIMIslandPopulation | MOAIMMainlandPopulation, dict[str, Any]]:
        """
        Evolve the population and returns the results as a remote object reference.

        See :meth:`~algatross.algorithms.genetic.mo_aim.islands.base.RemoteUDI.evolve` for more details.

        Parameters
        ----------
        pop : MOAIMIslandPopulation | MOAIMMainlandPopulation
            The population evolving on this island
        n : int, optional
            The number of iterations to evolve for each epoch, by default 1
        epoch : int, optional
            The number of epochs to evolve per call , by default 0
        `**kwargs` : dict
            Additional keyword arguments.

        Returns
        -------
        tuple[MOAIMIslandPopulation | MOAIMMainlandPopulation, dict[str, Any]]
            The evolved population and evolve info
        """
        return super().evolve(pop, n, epoch, **kwargs)


class RayMOAIMMainlandUDI(MOAIMMainlandUDI):
    """
    RayMOAIMMainlandUDI a MOAIMMainlandUDI implemented in the Ray backend.

    Parameters
    ----------
    env_name : str
        The name of the environment to pull from the global registry.
    `**kwargs` : dict
        Additional keyword arguments.
    """

    def __init__(self, /, env_name: str, **kwargs):
        self.env_name = env_name
        super().__init__(**kwargs)

    def evolve(
        self,
        pop: MOAIMIslandPopulation | MOAIMMainlandPopulation,
        n: int = 1,
        epoch: int = 0,
        **kwargs,
    ) -> tuple[MOAIMIslandPopulation | MOAIMMainlandPopulation, dict[str, Any]]:
        """
        Evolve evolves the population and returns the results as a remote object reference.

        See :meth:`~algatross.algorithms.genetic.mo_aim.islands.base.RemoteUDI.evolve` for more details.

        Parameters
        ----------
        pop : MOAIMIslandPopulation | MOAIMMainlandPopulation
            The population to evolve
        n : int, optional
            The number of iterations per epoch, by default 1
        epoch : int, optional
            The number of epochs to evolve, by default 0
        `**kwargs` : dict
            Additional keyword arguments.

        Returns
        -------
        tuple[MOAIMIslandPopulation | MOAIMMainlandPopulation, dict[str, Any]]
            The evolved population and evolve info
        """
        return super().evolve(pop, n, epoch, **kwargs)


@ray.remote
class IslandServer(RayActor):
    """
    A remote container for asyncronously evolving islands and manlands.

    Parameters
    ----------
    island_id : IslandID | MainlandID
        The id of this island in the archipelago.
    island_constructor : ConstructorData
        The constructor for the island.
    algorithm_constructor : ConstructorData
        The constructor for the algorithm.
    problem_constructor : ConstructorData
        The constructor for the problem.
    population_server : PopulationServer
        The population server.
    seed : np.random.Generator
        The random seed to use on this island.
    log_queue : Queue | None, optional
        The logging queue for log messages, default is :data:`python:None`.
    result_queue : Queue | None, optional
        The queue for learning results, default is :data:`python:None`.
    agents_to_platforms : Mapping[AgentID, Sequence[PlatformID]] | None, optional
        The mapping from agents in the environment to the IDs of the platforms they control.
    `**kwargs` : dict
        Additional keyword arguments.
    """

    island: RemoteUDI
    """The island object to use for evolving the population."""
    env: Callable
    """The environment running on this island from the global registry."""
    island_id: IslandID | MainlandID
    """The id of this island in the archipelago."""
    population_server: PopulationServer
    """A handle to the population server."""
    max_trajectory_length: int
    """The maximum length of trajectory to use for experience collection."""
    agents: dict
    """The mapping to the agents running on this island."""
    current_epoch: int
    """The number of times this island has been stepped."""

    _agents_to_platforms: Mapping[AgentID, Sequence[PlatformID]]
    _numpy_generator: np.random.Generator

    def __init__(
        self,
        island_id: IslandID | MainlandID,
        island_constructor: ConstructorData,
        algorithm_constructor: ConstructorData,
        problem_constructor: ConstructorData,
        population_server: PopulationServer,
        seed: np.random.Generator,
        log_queue: Queue | None = None,
        result_queue: Queue | None = None,
        agents_to_platforms: Mapping[AgentID, Sequence[PlatformID]] | None = None,
        **kwargs,
    ):
        RayActor.__init__(self, log_queue=log_queue, result_queue=result_queue, **kwargs)
        algo = algorithm_constructor.construct()
        prob = problem_constructor.construct()
        self.island: RemoteUDI = island_constructor.construct(algo=algo, prob=prob)

        # let runner seed env
        # maybe it will create env too
        self.env: Callable = _global_registry.get(ENV_CREATOR, self.island.env_name)
        self.island.set_island_id(island_id)
        self.island_id = island_id
        self.population_server = population_server
        self.max_trajectory_length = self.island.max_trajectory_length  # type: ignore[attr-defined]

        with ManagedEnvironmentsContext(env_list=[self.env]) as mec_envs:
            temp_env = mec_envs[0]
            agent_ids = temp_env.get_agent_ids()
            self._agents_to_platforms = agents_to_platforms or {agent_id: [agent_id] for agent_id in agent_ids}
            self._platforms_to_agents = {}
            missing_platforms = dict.fromkeys(agent_ids, True)
            for agent_id, platforms in self._agents_to_platforms.items():
                for platform_id in platforms:
                    self._platforms_to_agents[platform_id] = agent_id
                    missing_platforms[platform_id] = False
            if any(missing_platforms.values()):
                missed = [platform_id for platform_id, missing in missing_platforms.items() if missing]
                msg = f"The agent to platform map is missing a contoller agents (policies) for the following platforms: {missed}."
                raise ValueError(msg)
            agents = {}
            agent_seed = iter(seed.spawn(len(self._agents_to_platforms)))
            for agent_id, platform_ids in self._agents_to_platforms.items():
                obs_space = {platform_id: temp_env.observation_space[platform_id] for platform_id in platform_ids}
                act_space = {platform_id: temp_env.action_space[platform_id] for platform_id in platform_ids}
                agents[agent_id] = self.island.agent_constructors[agent_id].construct(
                    platforms=platform_ids,
                    obs_space=obs_space,
                    act_space=act_space,
                    seed=next(agent_seed),
                )

        as_platform_dict = {platform_id: agents[agent_id] for platform_id, agent_id in self._platforms_to_agents.items()}
        self.island.problem.initialize_agent_templates(as_platform_dict)
        self.island.problem.set_solution_dim(as_platform_dict)
        self.agents = agents
        self.current_epoch = -1

    def get_state(self) -> dict:
        """Get the server state.

        Returns
        -------
        dict
            The island servers state
        """
        # TODO: re-enable
        return copy.deepcopy(
            {
                "island": self.island.get_state(),
                "island_id": self.island_id,
                "_agents_to_platforms": self._agents_to_platforms,
                "_platforms_to_agents": self._platforms_to_agents,
                "agents": self.agents,
                "current_epoch": self.current_epoch,
            },
        )

    def set_state(self, state: dict):
        """
        Set the island server state.

        Parameters
        ----------
        state : dict
            The state dict to set for the island.
        """
        self.island.set_state(state["island"])
        self.island.set_island_id(state["island_id"])
        self.island_id = state["island_id"]
        for key in ["_agents_to_platforms", "_platforms_to_agents", "agents", "current_epoch"]:
            value = state[key]
            setattr(self, key, value)

        self.env = _global_registry.get(ENV_CREATOR, self.island.env_name)()

    def evolve(
        self,
        n: int = 1,
        **kwargs,
    ) -> tuple[
        IslandID | MainlandID,
        dict[str, Any],
    ]:
        """
        Evolve the ``pop`` on the ``island`` for the given number of iterations and returns the results.

        Parameters
        ----------
        n : int, optional
            The number of times to run the inner evolution loop, by default 1
        `**kwargs` : dict
            Additional keyword arguments.

        Returns
        -------
        IslandID | MainlandID
            The ID of the island in the archipelago
        dict[str, Any]
            The results from evolving the population
        """
        self.current_epoch += 1
        pop: MOAIMIslandPopulation | MOAIMMainlandPopulation = ray.get(  # type: ignore[assignment]
            self.population_server.get_population_to_evolve.remote(  # type: ignore[attr-defined]
                self.island_id,
            ),
        )
        self.log(f"{self.island} starting epoch {self.current_epoch}: evolving for {n} iterations...", logger_name="ray")
        kwargs.setdefault("max_trajectory_length", self.max_trajectory_length)
        try:
            kwargs.pop("ray_context", None)

            pop, result = self.island.evolve(pop, n, self.current_epoch, envs=[self.env], problem=self.island.problem, **kwargs)
            result |= ray.get(
                self.population_server.set_population.remote(self.island_id, pop, problem=self.island.problem),  # type: ignore[attr-defined]
            )

        except Exception as e:
            msg = "Unrecoverable exception occured. Debug info:"
            self.log(msg, level=logging.ERROR, logger_name="ray")
            msg = f"Island: {self.island}, island_id={self.island_id}, epoch={self.current_epoch}"
            self.log(msg, level=logging.ERROR, logger_name="ray")
            raise e  # noqa: TRY201

        return self.island_id, {f"{self.island.island_type}/{self.island_id}": {"epoch": self.current_epoch, **result}}

    def get_island_id(self) -> IslandID:
        """Get the island ID for this server.

        Returns
        -------
        IslandID
            The island ID for this server.
        """
        return self.island_id

    def get_island(self) -> RemoteUDI:
        """Get the island for this server.

        Returns
        -------
        RemoteUDI
            The island for this server.
        """
        return self.island

    def get_problem(self) -> UDP:
        """Get this islands problem.

        Returns
        -------
        UDP
            This islands problem.
        """
        return self.island.problem

    def get_agents(self) -> dict:
        """Get this islands agents.

        Returns
        -------
        dict
            This islands agents.
        """
        return self.agents
