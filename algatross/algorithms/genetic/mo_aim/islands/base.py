"""Module containing MO-AIM user-defined islands."""

import copy
import logging

from abc import ABC
from contextlib import suppress
from typing import Any, overload

import numpy as np

from algatross.algorithms.genetic.mo_aim.algorithm import UDA, MOAIMIslandUDA, MOAIMMainlandUDA
from algatross.algorithms.genetic.mo_aim.population import MOAIMIslandPopulation, MOAIMMainlandPopulation
from algatross.algorithms.genetic.mo_aim.problem import UDP, MOAIMIslandUDP, MOAIMMainlandUDP
from algatross.utils.types import AgentID, ConstructorData, IslandID, IslandTypeStr

logger = logging.getLogger("ray")


class RemoteUDI(ABC):
    """ABC for defining methods which every remote UDI must implement.

    Parameters
    ----------
    algo : MOAIMIslandUDA
        The algorithm for this island.
    prob : MOAIMIslandUDP
        The problem for this island.
    agent_constructors : dict[AgentID, ConstructorData]
        The constructors for each agent
    `*args`
        Additional positional arguments
    `**kwargs`
        Additional keyword arguments
    """

    algorithm: UDA
    """The algorithm for this island."""
    problem: UDP
    """The problem for this island."""
    island_id: IslandID
    """The id of this island in the archipelago."""
    island_type: IslandTypeStr
    """The type of island this represents :python:`"island"`, or :python:`"mainland"`."""
    env_name: str = ""
    """The name of the environment this island is working on."""
    agent_constructors: dict[AgentID, ConstructorData]
    """The constructors for each agent."""

    _numpy_generator: np.random.Generator

    @overload
    def __init__(
        self,
        algo: MOAIMIslandUDA,
        prob: MOAIMIslandUDP,
        agent_constructors: dict[AgentID, ConstructorData],
        *args,
        **kwargs,
    ) -> None: ...

    @overload
    def __init__(
        self,
        algo: "MOAIMMainlandUDA",
        prob: "MOAIMMainlandUDP",
        agent_constructors: dict[AgentID, ConstructorData],
        *args,
        **kwargs,
    ) -> None: ...

    @overload
    def __init__(self, algo: UDA, prob: UDP, agent_constructors: dict[AgentID, ConstructorData], *args, **kwargs) -> None: ...

    def __init__(self, algo, prob, agent_constructors, *args, **kwargs) -> None:
        self.algorithm = algo
        self.problem = prob
        self.agent_constructors = agent_constructors
        for key, value in kwargs.items():
            if not key.startswith("_"):
                with suppress(AttributeError):
                    setattr(self, key, value)

    def set_state(self, state: dict):
        """
        Set the state.

        Parameters
        ----------
        state : dict
            The state to set for the island.
        """
        for key, value in state.items():
            if key not in {"problem", "_problem", "algorithm"}:
                setattr(self, key, value)

        self.problem.set_state(state["problem"])
        self.algorithm.set_state(state["algorithm"])

    def get_state(self) -> dict:
        """Get the state for pickling.

        Returns
        -------
        dict
            The UDIs state
        """
        state = copy.deepcopy({k: v for k, v in self.__dict__.items() if k not in {"problem", "_problem", "algorithm"}})
        state["problem"] = self.problem.get_state()
        state["algorithm"] = self.algorithm.get_state()
        return state

    def set_island_id(self, island_id: IslandID) -> bool:
        """Set the ID of the remote island.

        Parameters
        ----------
        island_id : IslandID
            The ID of the remote island to set

        Returns
        -------
        bool
            True if the actor is ready
        """
        self.island_id = island_id
        return self.get_actor_alive_status()

    @staticmethod
    def get_actor_alive_status() -> bool:
        """Check whether the actor is alive and ready.

        Returns
        -------
        bool
            Always True. Therefore a call to this method, such as :func:`ray.get` or :func:`ray.wait` will return if the
            actor is alive. Otherwise they will timeout or raise an error.
        """
        return True

    def _get_obj_name_recurse(self, name: str, obj: Any) -> Any:  # noqa: ANN401
        """
        Recursively get the attribute specified by ``name`` from ``obj``.

        Parameters
        ----------
        name : str
            The name of the attribute to get
        obj : Any
            The object holding the attribute. this is only sued for recursion and shouldn't be passed by
            external routines.

        Returns
        -------
        Any
            The value of the attribute.
        """
        name = name.split(".", maxsplit=1)  # type: ignore[assignment]
        recurse = len(name) > 1
        next_name = name[1] if recurse else ""
        name = name[0]
        obj = self if obj is None else obj
        return obj, name, next_name, recurse

    def get_remote_attr(self, name: str, obj: Any | None = None, /) -> Any:  # noqa: ANN401
        """
        Get_remote_attr enable geting offset an attribute from this instance if it's a remote worker.

        Parameters
        ----------
        name : str
            The attribute name
        obj : Any | None, optional
            The object from which to retrieve ``name``, :data:`python:None`. This is only used by recursion and
            should not be passed by external routines.

        Returns
        -------
        Any
            The attribute values
        """
        o_obj, o_name, next_name, recurse = self._get_obj_name_recurse(name, obj)
        next_obj = getattr(o_obj, o_name)
        if recurse:
            next_obj = self.get_remote_attr(next_name, next_obj)
        return next_obj

    def set_remote_attr(self, name: str, value: Any, obj: Any | None = None, /):  # noqa: ANN401
        """
        Set_remote_attr enables setting of an attribute on this instance if it's a remote actor.

        Parameters
        ----------
        name : str
            The name of the attribute to set
        value : Any
            The value to set the attribute to.
        obj : Any | None, optional
            The object to which ``value`` will be assigned, :data:`python:None`. This is only used by recursion and
            should not be passed by external routines.
        """
        o_obj, o_name, next_name, recurse = self._get_obj_name_recurse(name, obj)
        if recurse:
            self.set_remote_attr(next_name, value, o_obj)
        if hasattr(o_obj, o_name):
            setattr(o_obj, o_name, value)

    def evolve(
        self,
        pop: MOAIMIslandPopulation | MOAIMMainlandPopulation,
        n: int = 1,
        epoch: int = 0,
        **kwargs,
    ) -> tuple[MOAIMIslandPopulation | MOAIMMainlandPopulation, dict[str, Any]]:
        """
        Evolve runs the algorithms :meth:`~algatross.algorithms.genetic.mo_aim.algorithm.UDA.run_evolve` method ``n`` times with callbacks.

        Calls the :meth:`~algatross.algorithms.genetic.mo_aim.algorithm.UDA.on_evolve_begin`,
        :meth:`~algatross.algorithms.genetic.mo_aim.algorithm.UDA.on_evolve_step`, and
        :meth:`~algatross.algorithms.genetic.mo_aim.algorithm.UDA.on_evolve_end` callbacks.

        Parameters
        ----------
        pop : MOAIMIslandPopulation | MOAIMMainlandPopulation
            The population to evolve
        n : int, optional
            The number of times to have the algorithm evolve the population, by default 1
        epoch : int, optional
            The current epoch of this island, default is 0
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        MOAIMIslandPopulation | MOAIMMainlandPopulation
            The evolved population.
        dict[str, Any]
            A dictionary of results from the evolution steps.
        """
        pop, results = self.algorithm.run_evolve(pop, n, **kwargs)  # type: ignore[assignment]
        return pop, {"epoch": epoch, **results}

    def evaluate(self, pop: MOAIMIslandPopulation | MOAIMMainlandPopulation, envs: list, problem: UDP, **kwargs) -> dict[str, Any]:
        """
        Run evaluation on a population.

        Parameters
        ----------
        envs : list
            The list of environments to use for evaluation.
        problem : UDP
            The problem the population is working on.
        pop : MOAIMIslandPopulation | MOAIMMainlandPopulation
            The population to evaluate.
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        dict[str, Any]]
            The info gathered by evaluation callbacks.
        """
        return self.algorithm.run_evaluate(pop=pop, envs=envs, problem=problem, **kwargs)

    def __repr__(self) -> str:  # noqa: D105
        return f"<{self.__class__.__name__}> island_id={self.island_id}"


class MOAIMIslandUDI(RemoteUDI):
    """A RemoteUDI controlling evolution on the MO-AIM Islands."""

    algorithm: MOAIMIslandUDA
    """The algorithm for this island."""
    problem: MOAIMIslandUDP
    """The problem for this island."""
    island_type = "island"


class MOAIMMainlandUDI(RemoteUDI):
    """A RemoteUDI controlling evolution on the MO-AIM Mainlands."""

    algorithm: "MOAIMMainlandUDA"
    """The algorithm for this island."""
    problem: "MOAIMMainlandUDP"
    """The problem for this island."""
    island_type = "mainland"
