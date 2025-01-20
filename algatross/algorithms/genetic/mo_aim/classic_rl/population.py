"""Population classes for classic RL algorithms."""

import logging

from asyncio import QueueEmpty
from contextlib import suppress
from copy import deepcopy
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import numpy as np

import ray

from ray.rllib import SampleBatch
from ray.util.queue import Empty

from algatross.algorithms.genetic.mo_aim.algorithm import TeamID
from algatross.algorithms.genetic.mo_aim.classic_rl.problem import MOAIMRLUDP
from algatross.algorithms.genetic.mo_aim.configs import MOAIMRLPopulationConfig
from algatross.algorithms.genetic.mo_aim.population import MOAIMPopulation
from algatross.utils.cloud import indicated_ray_get, indicated_ray_put
from algatross.utils.merge_dicts import filter_keys
from algatross.utils.queue import MultiQueue
from algatross.utils.random import resolve_seed
from algatross.utils.types import AgentID, IslandID, MOAIMIslandInhabitant, MainlandID, MigrantData, NumpyRandomSeed, RolloutData

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger("ray")


class MOAIMRLPopulation(MOAIMPopulation):
    """
    MOAIM population for working with classic RL algorithms.

    Parameters
    ----------
    seed : NumpyRandomSeed | None, optional
        The random seed for the bit generators on this island, default is :data:`python:None`.
    island_id : IslandID | None, optional
        The ID of this island in the archipelago.
    migrant_queue : MultiQueue | None, optional
        The queue of migrants for the archipelago, default is :data:`python:None`.
    `**kwargs`
        Additional keyword arguments.
    """

    config: MOAIMRLPopulationConfig  # type: ignore[assignment]
    """The configuration for this RL population."""
    _current_flat_params: np.ndarray

    def __init__(
        self,
        *,
        seed: NumpyRandomSeed | None = None,
        island_id: IslandID | None = None,
        migrant_queue: MultiQueue | None = None,
        **kwargs,
    ):
        self.rollout_buffer = []  # type: ignore[var-annotated]
        self.migrant_buffer = []  # type: ignore[var-annotated]
        namespace = kwargs.pop("namespace") if "namespace" in kwargs else None
        self.migrant_queue = (
            MultiQueue(queue_keys=[island_id], actor_options={"name": f"MigrantQueue {island_id}", "namespace": namespace, "num_cpus": 0.0})
            if migrant_queue is None
            else migrant_queue
        )
        config_dict = deepcopy(kwargs)
        config_dict["seed"] = config_dict.get("seed") or seed
        config_dict.pop("training_agents", None)
        self.config = MOAIMRLPopulationConfig(**filter_keys(MOAIMRLPopulationConfig, **config_dict))
        self.storage_path = self.config.storage_path
        if island_id is not None:
            self.island_id = island_id
        self._numpy_generator = resolve_seed(seed)  # type: ignore[arg-type]

        self._opposing_team_candidates = {}

        self._current_best_fitness = -np.inf
        self._current_flat_params = None

        self.training_teams_cache: Mapping = None
        self.opposing_teams_cache: Mapping = None

    def setup(self, problem: MOAIMRLUDP, **kwargs):
        """
        Set the current flattened agent parameters.

        Parameters
        ----------
        problem : MOAIMRLUDP
            The problem being solved by the population.
        `**kwargs`
            Additional keyword arguments.
        """
        for agent_id in problem.training_agents:
            self._current_flat_params = problem.agent_templates_map[agent_id].flat_parameters
            break

    @property
    def population_size(self) -> int:  # noqa: D102
        return 1

    @property
    def max_population_size(self) -> int:  # noqa: D102
        return 1

    def get_teams_for_training(  # type: ignore[override] # noqa: D102
        self,
        problem: MOAIMRLUDP,
        remote: bool = False,
        **kwargs,
    ) -> dict | ray.ObjectRef[dict]:
        teams: Mapping[TeamID, Mapping[AgentID, MOAIMIslandInhabitant]] = {TeamID(0): {}}
        for name in sorted(problem.ally_agents):
            ind = MOAIMIslandInhabitant(
                name=name,
                team_id=TeamID(0),
                inhabitant_id=str(uuid4()),
                genome=self._current_flat_params,
                current_island_id=self.island_id,
                conspecific_utility_dict={},
                island_id=self.island_id,
            )
            teams[TeamID(0)][ind.name] = ind  # type: ignore[index]
        self.training_teams_cache = teams
        return ray.put(self.training_teams_cache) if remote else self.training_teams_cache  # type: ignore[return-value]

    def get_state(self) -> dict:
        """Get the population state.

        Returns
        -------
        dict
            The population state
        """
        # avoid copying ray actor. migrant queue should be cleared by next epoch anyway.
        state = deepcopy({k: v for k, v in self.__dict__.items() if k not in {"migrant_queue", "problem", "_problem"}})
        state["training_teams_cache"], state["training_teams_cache_indicator"] = indicated_ray_get(state["training_teams_cache"])
        state["opposing_teams_cache"], state["opposing_teams_cache_indicator"] = indicated_ray_get(state["opposing_teams_cache"])
        return state

    def set_state(self, state: dict):  # noqa: D102
        ignore_keys = ["migrant_queue", "problem", "_problem", "training_teams_cache_indicator", "opposing_teams_cache_indicator"]
        state["training_teams_cache"] = indicated_ray_put(state["training_teams_cache"], state["training_teams_cache_indicator"])
        state["opposing_teams_cache"] = indicated_ray_put(state["opposing_teams_cache"], state["opposing_teams_cache_indicator"])
        for key, value in state.items():
            if key not in ignore_keys:
                setattr(self, key, value)

    def _construct_data_samples(self, **kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        obj_batch = []
        sol_batch = []
        traj_batch = []

        for inhabitant in self.migrant_buffer:
            obj_batch.append(inhabitant.objective)
            sol_batch.append(inhabitant.solution)
            traj_batch.append(inhabitant.rollout)

        # obj_batch: N x Obj
        obj_batch = np.stack(obj_batch)
        # sol_batch: N x Sol
        sol_batch = np.stack(sol_batch)
        # traj_batch: N x B x T x Obs  or  B x T x Obs (see below)
        traj_batch = np.stack(traj_batch)

        return obj_batch, sol_batch, traj_batch  # type: ignore[return-value]

    def add_migrants_to_buffer(
        self,
        migrants: dict[MainlandID, list[tuple[MOAIMIslandInhabitant | None, RolloutData | SampleBatch]]] | None = None,
        *,
        problem: MOAIMRLUDP,
        **kwargs,
    ) -> dict[str, Any]:
        """Add immigrants to the populations migrant buffer.

        Parameters
        ----------
        migrants : list[np.ndarray]
            The genomes of thefreedictionary migrants to add to the current population.
        problem : MOAIMRLUDP
            The problem this population is solving
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        dict[str, Any]
            A dictionary of info from this step
        """
        results: dict[str, Any] = {}

        def extend_buffer(migrants, migrant_buffer=self.migrant_buffer):
            if migrants:
                immigrants: list[RolloutData] = []
                for migrant_list in migrants.values():
                    for migrant in migrant_list:
                        if migrant[0] is None or isinstance(migrant[1], RolloutData):
                            immigrants.append(migrant[1])
                        elif isinstance(migrant[1], SampleBatch):
                            immigrants.append(
                                RolloutData(
                                    rollout=self.conspecific_rollout_from_sample_batch(migrant[1], problem, **kwargs),
                                    objective=problem.fitness_from_sample_batch(migrant[1]),
                                    solution=migrant[0].genome,
                                ),
                            )
                migrant_buffer.extend(immigrants)

        extend_buffer(migrants=migrants)
        extended: list[MigrantData] = []
        new_migrants = []

        with suppress(QueueEmpty, Empty):
            while True:
                new_migrants.append(self.migrant_queue.get_nowait(key=self.island_id)[self.island_id])

        for new_migrant in new_migrants:
            extended.extend(list(new_migrant))
            extend_buffer(migrants=new_migrant)
        logger.debug(f"Pulled migrants from mainlands {sorted(extended)} into island {self.island_id} buffer")
        return results

    def add_from_buffer(self, **kwargs):
        """
        Set the current weights based on the incoming migrant if the objective is improved.

        Parameters
        ----------
        `**kwargs`
            Additional keyword arguments.
        """
        if not self.migrant_buffer:
            return

        obj_batch, sol_batch, _traj_batch = self._construct_data_samples(include_archive_data=False, **kwargs)
        while obj_batch.ndim > 1:
            obj_batch = obj_batch.mean(axis=-1)
        best_migrant = np.argmax(obj_batch)

        if obj_batch[best_migrant] > self._current_best_fitness:
            self._current_flat_params = sol_batch[best_migrant]

        self.migrant_buffer.clear()
