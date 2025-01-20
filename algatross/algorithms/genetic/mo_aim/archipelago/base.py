"""
A module containing  archipelago base class(es) for use with different parallelization backends.

The architecture is heavily derived from PaGMO2s generalized island model.
"""

import logging

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

import torch

from algatross.algorithms.genetic.mo_aim.islands.base import MOAIMIslandUDI, MOAIMMainlandUDI, RemoteUDI
from algatross.algorithms.genetic.mo_aim.topology import MOAIMTopology
from algatross.utils.types import IslandID, IslandTypeStr, MainlandID, NumpyRandomSeed

logger = logging.getLogger("ray")


class RemoteMOAIMArchipelago(ABC):
    """
    ABC for remote MO-AIM archipelagos.

    initialize the archipelago to contain the islands and mainlands as remote actors and store connections in the toplogy.

    Parameters
    ----------
    islands : Sequence[Callable[[Any], MOAIMIslandUDI]]
        A sequence of callables for constructing islands
    mainlands : Sequence[Callable[[Any], MOAIMMainlandUDI]]
        A sequence of callables for constructing mainlands
    island_configs : Sequence[Mapping]
        A sequence of configurations to pass to the island constructors
    mainland_configs : Sequence[Mapping]
        A sequence of configurations to pass to the mainland constructors
    t : MOAIMTopology
        The topology fore storing the connection weights for migration between the islands.
    conspecific_utility_keys : list[str]
        A list of keys to use for conspecific utility calculation
    seed : NumpyRandomSeed, optional
        A seed to pass to the random generators default is :data:`python:None`
    `**kwargs`
        Additional keyword arguments.
    """

    _conspecific_utility_keys: list[str]
    _topology: MOAIMTopology
    _archipelago: dict
    _island: dict
    _mainland: dict

    def __len__(self) -> int:
        """
        __len__ returns the number of islands and mainlands in the problem.

        Returns
        -------
        int
            The total number of islands and mainlands
        """
        return len(self.archipelago)

    @abstractmethod
    def __init__(
        self,
        islands: Sequence[Callable[[Any], MOAIMIslandUDI]],
        mainlands: Sequence[Callable[[Any], MOAIMMainlandUDI]],
        island_configs: Sequence[Mapping],
        mainland_configs: Sequence[Mapping],
        t: MOAIMTopology,
        conspecific_utility_keys: list[str],
        seed: NumpyRandomSeed | None = None,
        **kwargs,
    ) -> None:
        pass

    @property
    def conspecific_shape(self) -> list[int]:
        """Return the shape of the conspecific data.

        Returns
        -------
        list[int]
            The conspecific data shape
        """
        return [len(self._conspecific_utility_keys)]

    @property
    def n_islands(self) -> int:
        """The number of Islands in the archipelago.

        Returns
        -------
        int
            The number of islands in the archipelago
        """
        return len(self.island)

    @property
    def n_mainlands(self) -> int:
        """The number of Mainlands in the archipelago.

        Returns
        -------
        int
            The number of Mainlands in the archipelago.
        """
        return len(self.mainland)

    @property
    def island(self) -> dict[IslandID, Any]:
        """
        Island a dictionary mapping IDs to UDI objects.

        Returns
        -------
        dict[IslandID, Any]
            A mapping from island IDs to UDI objects.
        """
        return self._island

    @property
    def mainland(self) -> dict[MainlandID, Any]:
        """
        Mainland a dictionary mapping IDs to UDI objects.

        Returns
        -------
        dict[MainlandID, Any]
            A mapping from mainland IDs to UDI objects
        """
        return self._mainland

    @property
    def archipelago(self) -> dict[IslandID | MainlandID, Any]:
        """
        Archipelago the conglomoration of island and mainland UDIs.

        Returns
        -------
        dict[IslandID | MainlandID, Any]
            A mapping from island ID or Mainland ID to the associated UDI object.
        """
        return self._archipelago

    @property
    def topology(self) -> MOAIMTopology:
        """
        Topology returns the archipelagos topology.

        Returns
        -------
        MOAIMTopology
            The topology for this archipelago
        """
        return self._topology

    def push_back(self, udi: RemoteUDI | list[RemoteUDI], island_type: IslandTypeStr, **kwargs):
        """
        Push back the udi(s) to the archipelago and topology.

        Parameters
        ----------
        udi : RemoteUDI | list[RemoteUDI]
            The UDI object(s), future(s) [Dask], or object reference(s) [Ray] to be pushed back to the archipelago and topology
        island_type : IslandTypeStr, optional
            The type of island to which the UDI(s) refer, by default "island"
        `**kwargs`
            Additional keyword arguments.
        """
        if not isinstance(udi, list | tuple):
            udi = [udi]
        msg = f"pushing back {island_type}"
        suffix = f"s {len(self)} thru {len(self) + len(udi) - 1}" if len(udi) > 1 else f" {len(self)}"
        logger.info(f"{msg}{suffix}")
        self.topology.push_back(len(udi), island_type)
        isl_container = self._island if island_type == "island" else self._mainland if island_type == "mainland" else {}
        update_dict = {len(self) + idx: isl for idx, isl in enumerate(udi)}
        isl_container.update(update_dict)
        self._archipelago.update(update_dict)

    @abstractmethod
    def _init_island_pops(
        self,
        warmup_generations: int = 10,
        warmup_iterations: int = 5,
        target_population_size: int = 20,
        *args,
        **kwargs,
    ):
        """
        Run warmup training on the islands to initialize a population up to a certain size.

        Parameters
        ----------
        warmup_generations : int, optional
            The  number of QD generations to run on the island per warmup iteration. Defaults to 10
        warmup_iterations : int, optional
            The number of warmup iterations (epochs) to run on the island. Defaults to 5
        target_population_size : int, optional
            The target size for the island population, by default 20
        `*args`
            Additional positional arguments.
        `**kwargs`
            Additional keyword arguments.
        """

    @abstractmethod
    def _init_mainland_pops(self, *args, **kwargs):
        """Initialize a brand new mainland population.

        The new population is drawn from the islands using the initial softmax distribution.

        Parameters
        ----------
        `*args`
            Additional positional arguments.
        `**kwargs`
            Additional keyword arguments.
        """

    @staticmethod
    @abstractmethod
    def _wait(futures: list[Any]):
        """
        _wait blocks execution until the futures finish execution.

        This is a framework dependent method which can be called when we need to re-sync the actors but don't care
        about the results in the futures.

        Parameters
        ----------
        futures : list[Any]
            The futures to wait upon for task completion.
        """

    @staticmethod
    @abstractmethod
    def _gather(futures: list[Any]) -> list[Any]:
        """
        _gather blocks execution until the futures finish execution and returns the results in order.

        This is a framework dependent method which can be called when we need to re-sync the actors and need the
        results from the futures.

        Parameters
        ----------
        futures : list[Any]
            The futures to wait upon for task completion.

        Returns
        -------
        list[Any]
            The results from the futures.
        """

    @abstractmethod
    def evolve(self, epochs: int = 1, **kwargs):
        """
        Evolve synchronously evolves the entire archipelago through ``n`` MO-AIM iterations.

        Synchronous in this context means that each Island and Mainland is evolved ``n`` times. At which point evolution
        pauses for migration to occur.

        In contrast, asynchronous does not pause evolution on the Islands and Mainlands but instead uses buffers to add
        and remove migrants from a queue.

        Parameters
        ----------
        epochs : int, optional
            The number of MO-AIM epochs to evolve the Islands and Mainlands, by default 1
        `**kwargs`
            Additional keyword arguments.
        """

    @abstractmethod
    def migrate(self, destinations: list[IslandID | MainlandID] | IslandID | MainlandID | None = None):
        """
        Migrate synchronously migrates the individuals from the islands to the mainland according to the topology.

        Parameters
        ----------
        destinations : list[IslandID | MainlandID] | IslandID | MainlandID | None
            The destinations (buffers) the population server should be migrating (clearing).
        """

    @abstractmethod
    def train_topology(
        self,
        results: dict[IslandID | MainlandID, dict[str, Any]],
        mainlands: list[MainlandID] | MainlandID | None = None,
    ):
        """
        Update the edge weights in the topology.

        Updates the edge weights using a softmax loss with entropy regularization.

        Parameters
        ----------
        results : dict[IslandID | MainlandID, dict[str, Any]]
            The results to use for training the softmax.
        mainlands : list[MainlandID] | MainlandID | None, optional
            The list of mainlands to update, :data:`python:None`
        """

    def stack_conspecific_data(self, data_dict: dict[IslandID, np.ndarray]) -> torch.Tensor:
        """Take a dictionary of island ids with their conspecific data and return them stacked into a tensor.

        If any island IDs are not found in the dictionary a tensor of zeros is used instead.

        Parameters
        ----------
        data_dict : dict[IslandID, np.ndarray]
            A dictionary of IslandIDs and conspecific data for the respective island.

        Returns
        -------
        torch.Tensor
            The stacked conspecific data with shape [I, C] where I is the number of islands in the archipelago and C is the length of the
            conspecific data.
        """
        return torch.from_numpy(np.stack([data_dict.get(key, 0) for key in sorted(self.island)]))
