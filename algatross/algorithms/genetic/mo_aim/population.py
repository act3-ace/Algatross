"""Module containing methods for controlling Populations on the islands & mainlands in MO-AIM."""

import copy
import dataclasses
import logging
import traceback

from abc import ABC, abstractmethod
from asyncio import QueueEmpty
from collections import defaultdict
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from contextlib import suppress
from copy import deepcopy
from itertools import chain, cycle
from typing import Any
from uuid import uuid4

import numpy as np

import ray

from ray.rllib.policy.sample_batch import SampleBatch
from ray.util.queue import Empty, Queue

import torch

from ribs.archives import ArchiveBase
from ribs.emitters import EmitterBase, GaussianEmitter

from algatross.actors.ray_actor import RayActor
from algatross.algorithms.genetic.mo_aim.configs import MOAIMIslandPopulationConfig, MOAIMMainlandPopulationConfig
from algatross.algorithms.genetic.mo_aim.problem import UDP, MOAIMIslandUDP, MOAIMMainlandUDP
from algatross.configs.pyribs.archive_config import UnstructuredArchiveConfig
from algatross.configs.pyribs.emitter_config import GaussianEmitterConfig, RandomEmitterConfig
from algatross.models.encoders.base import BaseEncoder
from algatross.models.encoders.pca import PCAEncoder
from algatross.quality_diversity.archives.unstructured import _UNSTRUCTURED_ARCHIVE_FIELDS, UnstructuredArchive
from algatross.quality_diversity.archives.wrappers import FileArchiveWrapper
from algatross.quality_diversity.emitters.random import RandomEmitter
from algatross.utils.cloud import indicated_ray_get, indicated_ray_put
from algatross.utils.merge_dicts import filter_keys, merge_dicts
from algatross.utils.queue import MultiQueue
from algatross.utils.random import get_torch_generator_from_numpy, resolve_seed
from algatross.utils.stats import summarize
from algatross.utils.types import (
    AgentID,
    ConstructorData,
    InhabitantID,
    IslandID,
    IslandSample,
    MOAIMIslandInhabitant,
    MainlandID,
    MigrantData,
    MigrantFlock,
    MigrantQueue,
    NumpyRandomSeed,
    RolloutData,
    TeamID,
)

logger = logging.getLogger("ray")

MOAIM_ISLAND_POPULATION_CONFIG_DEFAULTS = {
    "archive_base_class": UnstructuredArchive,
    "archive_config": dataclasses.asdict(UnstructuredArchiveConfig()),
    "use_result_archive": False,
    "result_archive_base_class": UnstructuredArchive,
    "result_archive_config": dataclasses.asdict(UnstructuredArchiveConfig()),
    "emitter_base_class": GaussianEmitter,
    "emitter_config": dataclasses.asdict(GaussianEmitterConfig()),
    "random_emitter_base_class": RandomEmitter,
    "random_emitter_config": dataclasses.asdict(RandomEmitterConfig()),
    "qd_samples": 5,
    "qd_experience_buffer_size": 1_000,
}


def _make_archive_owner(archive: ArchiveBase):
    """
    _make_archive_owner make this process the owner of the memory locations of the buffers in the archive.

    Since the memory is created in a local process and moved to a remote actor, the PopulationServer doesn't own
    the memory and thus the archive is not writeable. We have to create an archive whose memory is owned by the curreny actor

    Parameters
    ----------
    archive : ArchiveBase
        The archive with its buffers' memory locations owned by the current process.
    """
    # TODO: create the archive *on* the remote worker so the population owns this memory
    if archive is not None:
        for field, arr in archive._store._fields.items():  # noqa: SLF001
            archive._store._fields[field] = np.require(arr, requirements=["W", "O"])  # noqa: SLF001

        for prop, arr in archive._store._props.items():  # noqa: SLF001
            if isinstance(arr, np.ndarray):
                archive._store._props[prop] = np.require(arr, requirements=["W", "O"])  # noqa: SLF001


class MOAIMPopulation(ABC):
    """ABC which defines the attributes and methods populations of MO-AIM algorithms must implement."""

    config: MOAIMIslandPopulationConfig | MOAIMMainlandPopulationConfig
    """The configuration of this population."""

    _numpy_generator: np.random.Generator
    _torch_generator: torch.Generator
    _island_id: IslandID | MainlandID | None = None
    _opposing_team_candidates: dict[IslandID | MainlandID, list[list[MOAIMIslandInhabitant]]]
    _opposing_softmax_weights: dict[IslandID | MainlandID, torch.Tensor]
    _opposing_islands: dict[int, IslandID | MainlandID]
    _opposing_teams: dict[IslandID | MainlandID, list[dict[AgentID, MOAIMIslandInhabitant]]]
    _opposing_team_map: dict[TeamID, IslandID | MainlandID]

    @property
    @abstractmethod
    def population_size(self) -> int:
        """
        The size of the current population of islanders.

        Returns
        -------
        int
            The size of the current population of islanders.
        """

    @property
    @abstractmethod
    def max_population_size(self) -> int:
        """
        The maximum size of the current population of islanders.

        Returns
        -------
        int
            The maximum size of the current population of islanders.
        """

    @property
    def island_id(self) -> IslandID | MainlandID | None:
        """
        The ID in the archipelago of this island.

        Returns
        -------
        IslandID | MainlandID | None
            The ID in the archipelago of this island.
        """
        return self._island_id

    @island_id.setter
    def island_id(self, new_id: IslandID | MainlandID):
        self._island_id = new_id

    @abstractmethod
    def get_teams_for_training(self, **kwargs) -> dict | ray.ObjectRef:
        """
        Get a dictionary of teams to be trained.

        Parameters
        ----------
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        dict | ray.ObjectRef
            The teams to use for training or a reference to them in the object store.
        """

    def set_state(self, state: dict):
        """
        Set the population state.

        Parameters
        ----------
        state : dict
            The population state to set.
        """
        for key, value in state.items():
            if key in {"problem", "_problem"}:
                continue

            setattr(self, key, value)

    def get_state(self) -> dict:
        """
        Get the population state.

        Returns
        -------
        dict
            The populations state
        """
        return copy.deepcopy({k: getattr(self, k) for k in self.__dict__})

    def get_teams_for_competition(self, *, problem: MOAIMIslandUDP | MOAIMMainlandUDP, **kwargs) -> dict:
        """
        Get the teams to be used to compete as opponents on other islands/mainlands.

        Parameters
        ----------
        problem : MOAIMIslandUDP | MOAIMMainlandUDP
            The problem this population is solving
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        dict
            Dictionary mapping team IDs to team members
        """
        opposing_teams: dict[TeamID, dict[AgentID, MOAIMIslandInhabitant]] = {}
        self._opposing_team_map = {}
        if self._opposing_islands:
            self.sample_opposing_teams(len(problem.opponent_agents))
            for isl, teams in self._opposing_teams.items():
                for team_id, team in enumerate(teams, len(opposing_teams)):
                    opposing_teams[team_id] = team
                    self._opposing_team_map[team_id] = isl
        return opposing_teams

    @abstractmethod
    def add_migrants_to_buffer(self, migrants: Any, *args, **kwargs) -> dict[str, Any]:  # noqa: ANN401
        """
        Add migrants to the populations incoming migration buffer.

        Parameters
        ----------
        migrants : Any
            The migrants to add to the buffer
        `*args`
            Additional positional arguments.
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        dict[str, Any]
            The results of adding the migrants to the buffer
        """

    def set_opposing_team_candidates(self, opposing_team_candidates: dict[IslandID | MainlandID, list[list[MOAIMIslandInhabitant]]]):
        """
        Set the candidates to be used as opponents from other mainlands or islands.

        Parameters
        ----------
        opposing_team_candidates : dict[IslandID  |  MainlandID, list[list[MOAIMIslandInhabitant]]]
            Dictionary of island ids and the candidate opponent teams from each.
        """
        if getattr(self, "_opposing_team_candidates", None) is None:
            self._opposing_team_candidates = {}
        if getattr(self, "_opposing_softmax_weights", None) is None:
            self._opposing_softmax_weights = dict.fromkeys(opposing_team_candidates, torch.tensor(1.0, dtype=torch.float64))

        incoming_teams = set(self._opposing_team_candidates) - set(opposing_team_candidates)

        if len(incoming_teams):
            self._opposing_softmax_weights.update(dict.fromkeys(incoming_teams, torch.tensor(1.0, dtype=torch.float64)))

        self._opposing_team_candidates.update(opposing_team_candidates)
        self._opposing_islands = dict(enumerate(self._opposing_team_candidates))
        self._opposing_teams = {}

    def sample_opposing_teams(self, n_opponents: int):
        """
        Generate opposing team members.

        Parameters
        ----------
        n_opponents : int
            Number of opponents
        """
        self._opposing_teams.clear()

        sampled = 0
        opposing_teams = defaultdict(list)
        while sampled < n_opponents and self._opposing_softmax_weights:
            island_ids, counts = torch.multinomial(
                torch.softmax(torch.stack(list(self._opposing_softmax_weights.values())), dim=0),
                1,
                replacement=True,
                generator=self._torch_generator,
            ).unique(return_counts=True)
            for isl_idx, count in zip(island_ids, counts, strict=True):
                isl = self._opposing_islands[int(isl_idx)]
                new_teams = list(self._numpy_generator.choice(self._opposing_team_candidates[isl], size=int(count), replace=True))  # type: ignore[arg-type]
                opposing_teams[isl].extend(new_teams)
                sampled += sum(len(new_team) for new_team in new_teams)
        self._opposing_teams.update(opposing_teams)

    def update_opponent_softmax(self, update_increment: float | torch.Tensor, opposing_islands: list[IslandID | MainlandID]):
        """
        Update the softmax function used for sampling opponents.

        Parameters
        ----------
        update_increment : float | torch.Tensor
            Amount to update softmax weights
        opposing_islands : list[IslandID  |  MainlandID]
            Islands' weights being updated
        """
        for isl in opposing_islands:
            self._opposing_softmax_weights[isl] += update_increment


# TODO: make island population based on "Inhabitants" class
class MOAIMIslandPopulation(MOAIMPopulation):  # noqa: PLR0904
    """
    MOAIMIslandPopulation a population evolving on a MO-AIM Island.

    Parameters
    ----------
    seed : NumpyRandomSeed | None, optional
        The random seed to use for this populations RNGs, default is :data:`python:None`.
    island_id : IslandID | None, optional
        The id for this island in the archipelago, default is :data:`python:None`.
    migrant_queue : MultiQueue | None, optional
        The queue to use for migrants coming into the island, default is :data:`python:None`.
    `**kwargs`
        Additional keyword arguments.
    """

    config: MOAIMIslandPopulationConfig
    """The config for this population."""
    archive: ArchiveBase
    """The archive used for storing elites in this population."""
    archive_config: dict
    """The config used to set up the archive."""
    result_archive: ArchiveBase | None = None
    """The separate result archive storing elites in this population."""
    result_archive_config: dict | None = None
    """The config used to set up the result archive."""
    emitters: list[EmitterBase]
    """The emitters for the archives in this population."""
    emitter_config: dict
    """The config for the emitters in this population."""
    random_emitter: EmitterBase
    """The emitter used to retrieve random elites unchanged from the archive."""
    random_emitter_config: dict
    """The config used to set up the random emitter."""
    solution_dim: int
    """The dimensionality of the solutions in the archive."""
    rollout_buffer: list
    """The rollout buffers for the elites in this population."""
    migrant_buffer: list[RolloutData]
    """The buffer of migrants waiting to come to the island after having been moved out of the queue."""
    migrant_queue: MultiQueue
    """The queue of migrants coming to the island."""
    encoder: BaseEncoder
    """The encoder for converting data to measures for elites in behavior space."""

    _numpy_generator: np.random.Generator

    def __init__(
        self,
        *,
        seed: NumpyRandomSeed | None = None,
        island_id: IslandID | None = None,
        migrant_queue: MultiQueue | None = None,
        **kwargs,
    ):
        self.rollout_buffer = []
        # TODO: implement this as a ray.util.Queue
        self.migrant_buffer = []
        namespace = kwargs.pop("namespace") if "namespace" in kwargs else None
        self.migrant_queue = (
            MultiQueue(queue_keys=[island_id], actor_options={"name": f"MigrantQueue {island_id}", "namespace": namespace, "num_cpus": 0.0})
            if migrant_queue is None
            else migrant_queue
        )
        config_dict = merge_dicts(MOAIM_ISLAND_POPULATION_CONFIG_DEFAULTS, kwargs)
        config_dict["seed"] = config_dict.get("seed") or seed
        config_dict.pop("training_agents", None)
        self.config = MOAIMIslandPopulationConfig(**filter_keys(MOAIMIslandPopulationConfig, **config_dict))
        self.storage_path = self.config.storage_path
        if island_id is not None:
            self.island_id = island_id
        # self._numpy_generator, _ = get_generators(seed=, seed_global=False)
        self._numpy_generator = resolve_seed(seed)  # type: ignore[arg-type]
        self._torch_generator = get_torch_generator_from_numpy(self._numpy_generator)[0]

        self._opposing_team_candidates = {}
        self.encoder = None
        self.archive = None
        self.result_archive = None
        self.training_teams_cache: dict = None
        self.opposing_teams_cache: dict = None

    def setup(self, *, problem: MOAIMIslandUDP, **kwargs):
        """
        Run some additional setup for the population.

        Initializes the archives and encoder.

        Parameters
        ----------
        problem : MOAIMIslandUDP
            The problem this population is solving
        `**kwargs`
            Additional keyword arguments.
        """
        self.solution_dim = next(iter((problem.solution_dim if self.config.solution_dim is None else self.config.solution_dim).values()))

        self.archive_config = (
            self.config.archive_config if isinstance(self.config.archive_config, dict) else dataclasses.asdict(self.config.archive_config)
        )
        extra_fields = self.archive_config.get("extra_fields", {})
        extra_fields["trajectory"] = ((), np.float32)
        for k in _UNSTRUCTURED_ARCHIVE_FIELDS:
            # remove unallowed keys, potentially stored due to a checkpoint
            if k in extra_fields:
                extra_fields.pop(k)

        self.archive_config["extra_fields"] = extra_fields
        self.archive_config["solution_dim"] = self.solution_dim
        # TODO: Update this to be compatible with different versions of numpy
        self.archive_config["seed"] = self._numpy_generator.spawn(1).pop()
        self.emitter_config = (
            self.config.emitter_config if isinstance(self.config.emitter_config, dict) else dataclasses.asdict(self.config.emitter_config)
        )
        # TODO: Update this to be compatible with different versions of numpy
        self.emitter_config["seed"] = self._numpy_generator.spawn(1).pop()
        if self.emitter_config.get("initial_solutions") is None:
            self.emitter_config["initial_solutions"] = self._numpy_generator.normal(
                0,
                1,
                (self.emitter_config["batch_size"], self.solution_dim),
            )

        self.random_emitter_config = (
            self.config.random_emitter_config
            if isinstance(self.config.random_emitter_config, dict)
            else dataclasses.asdict(self.config.random_emitter_config)
        )
        # TODO: Update this to be compatible with different versions of numpy
        self.random_emitter_config["seed"] = self._numpy_generator.spawn(1).pop()
        if self.random_emitter_config.get("initial_solutions") is None:
            self.random_emitter_config["initial_solutions"] = self._numpy_generator.normal(
                0,
                1,
                (self.random_emitter_config["batch_size"], self.solution_dim),
            )

        if self.config.use_result_archive:
            self.result_archive_config = (
                self.config.result_archive_config
                if isinstance(self.config.result_archive_config, dict)
                else dataclasses.asdict(self.config.result_archive_config)
            )
            extra_fields = self.result_archive_config.get("extra_fields", {})
            extra_fields["trajectory"] = ((), np.float32)
            self.result_archive_config["extra_fields"] = extra_fields
            self.result_archive_config["solution_dim"] = self.solution_dim
            # TODO: Update this to be compatible with different versions of numpy
            self.result_archive_config["seed"] = self._numpy_generator.spawn(1).pop()

        # default to 1 sample per migrant if set to 0
        self.config.max_samples_per_migrant = self.config.max_samples_per_migrant or 1

        self.setup_qd()

    def setup_encoder(self, encoder_class: type[BaseEncoder], encoder_config: dict, *args, **kwargs):
        """Instantiate the encoder used for behavior classification for QD.

        Parameters
        ----------
        encoder_class : type[BaseEncoder]
            The base class of the encoder.
        encoder_config : dict
            The configuration dictionary to be passed to the encoders :python:`__init__` method
        `*args`
            Additional positional arguments.
        `**kwargs`
            Additional keyword arguments.
        """
        # TODO: figure out a way to determine the shape beforehand and make the lr configurable
        self.encoder = encoder_class(**encoder_config)

    def setup_qd(self):
        """Set up the QD archives."""
        # create the archive and possibly result archive
        self.archive_visualizer = self.archive_config.pop("visualizer", None)
        self.visualize_archive = self.archive_config.pop("visualize", False) and self.archive_visualizer is not None
        self.result_archive_visualizer = None
        self.visualize_result_archive = False
        # self.archive = self.config.archive_base_class(**self.archive_config)  # noqa: ERA001
        # self.archive._store = ArrayStore.from_raw_dict(
        #     self.archive._store.as_raw_dict(),  # noqa: ERA001
        #     storage_path=self.storage_path / "archive" / f"island_{self.island_id}",  # noqa: ERA001
        # )  # noqa: RUF100, ERA001
        self.archive = FileArchiveWrapper(
            archive=self.config.archive_base_class(**self.archive_config),
            storage_path=self.storage_path / "archive",
            name=f"island_{self.island_id}",
        )

        if self.config.use_result_archive:
            self.result_archive_config = (
                self.config.result_archive_config
                if isinstance(self.config.result_archive_config, dict)
                else dataclasses.asdict(self.config.result_archive_config)
            )
            self.result_archive_visualizer = self.result_archive_config.pop("visualizer", None)
            self.visualize_result_archive = (
                self.result_archive_config.pop("visualize", False) and self.result_archive_visualizer is not None
            )
            # self.visualize_result_archive = (
            #     self.result_archive_config.pop("visualize", False) and self.result_archive_visualizer is not None  # noqa: ERA001
            # )  # noqa: RUF100, ERA001
            # self.result_archive._store = ArrayStore.from_raw_dict(
            #     self.result_archive._store.as_raw_dict(),  # noqa: ERA001
            #     storage_path=self.storage_path / "result_archive" / f"island_{self.island_id}",  # noqa: ERA001
            # )  # noqa: RUF100, ERA001
            self.result_archive = FileArchiveWrapper(
                archive=self.config.result_archive_base_class(**self.result_archive_config),
                storage_path=self.storage_path / "result_archive",
                name=f"island_{self.island_id}",
            )

        # create the emitters
        self.emitters = [self.config.emitter_base_class(self.archive, **self.emitter_config)]
        self.random_emitter = self.config.random_emitter_base_class(self.archive, **self.random_emitter_config)

    def get_state(self) -> dict:
        """Get the population state.

        Returns
        -------
        dict
            The populations state
        """
        # avoid copying ray actor. migrant queue should be cleared by next epoch anyway.
        state = copy.deepcopy({k: v for k, v in self.__dict__.items() if k not in {"migrant_queue", "problem", "_problem"}})
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

    def refresh_archives(self, **kwargs):
        """
        Update the embeddings and add each existing solution back under the new embedding.

        Also updates the result archive if applicable.

        Parameters
        ----------
        `**kwargs`
            Additional keyword arguments.
        """
        self.update_embeddings(self.archive, **kwargs)

        if self.result_archive:
            self.update_embeddings(self.result_archive)

    def update_embeddings(self, archive: ArchiveBase, **kwargs):
        """
        Recalculate the embedding for elites in the archive and return them under the ner embedding.

        Parameters
        ----------
        archive : ArchiveBase
            The archive which will have its embeddings updated.
        `**kwargs`
            Additional keyword arguments.
        """
        logger.debug(f"Updating embeddings ({archive})")
        if len(archive):
            fields = archive.data()
            objs = fields.pop("objective")
            sols = fields.pop("solution")
            traj = fields.pop("trajectory")
            fields.pop("index")
            fields.pop("measures")
            fields.pop("threshold")
            # concatenate the metadata which should have shape [B, C] where B is the number of samples relating to this member
            # and C is the size of the conspecific data.
            meas = self.get_measure_embedding(traj, **kwargs)

            _make_archive_owner(archive)

            archive.clear()
            archive.add(sols, objs, meas, trajectory=traj, **fields)

    def _construct_data_samples(self, **kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        obj_batch = None
        sol_batch = None
        traj_batch = None

        for ix, inhabitant in enumerate(self.migrant_buffer):
            if obj_batch is None:  # initialize
                n = len(self.migrant_buffer)
                # obj_batch: N x Obj.shape
                obj_batch = np.zeros((n, *np.asarray(inhabitant.objective).shape))
                # sol_batch: N x Sol.shape
                sol_batch = np.zeros((n, *inhabitant.solution.shape))
                # traj_batch: N x B x T x Obs  or  B x T x Obs (see below)
                traj_batch = np.zeros((n, *inhabitant.rollout.shape))

            obj_batch[ix] = inhabitant.objective
            sol_batch[ix] = inhabitant.solution
            traj_batch[ix] = inhabitant.rollout

        return obj_batch, sol_batch, traj_batch

    def add_from_buffer(self, **kwargs):
        """
        Add new solutions waiting in the migrant buffer to the archives then clears the buffer.

        Parameters
        ----------
        `**kwargs`
            Additional keyword arguments.
        """
        if self.archive is None:
            self.setup(**kwargs)

        if not self.migrant_buffer:
            return

        obj_batch, sol_batch, traj_batch = self._construct_data_samples(include_archive_data=False, **kwargs)
        meas_batch = self.get_measure_embedding(traj_batch, **kwargs)
        _make_archive_owner(self.archive)
        # update the shape for the trajectory field since we didn't know it when the archive was constructed
        if traj_batch.shape[1:] != self.archive._store._fields["trajectory"].shape[1:]:  # noqa: SLF001
            self.archive._store._fields["trajectory"] = np.empty((self.archive._store.capacity, *traj_batch.shape[1:]))  # noqa: SLF001
            if "field_desc" in self.archive._store.__dict__:  # noqa: SLF001
                del self.archive._store.__dict__["field_desc"]  # noqa: SLF001

        self.archive.add(sol_batch, obj_batch, meas_batch, trajectory=traj_batch)

        if self.result_archive:
            _make_archive_owner(self.result_archive)
            if traj_batch.shape[1:] != self.result_archive._store._fields["trajectory"].shape[1:]:  # noqa: SLF001
                self.result_archive._store._fields["trajectory"] = np.empty(  # noqa: SLF001
                    (self.result_archive._store.capacity, *traj_batch.shape[1:]),  # noqa: SLF001
                )
                if "field_desc" in self.result_archive._store.__dict__:  # noqa: SLF001
                    del self.result_archive._store.__dict__["field_desc"]  # noqa: SLF001
            self.result_archive.add(sol_batch, obj_batch, meas_batch, trajectory=traj_batch)
        self.migrant_buffer.clear()

    def get_measure_embedding(self, conspecific_data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Get the embedding from the conspecific data.

        Parameters
        ----------
        conspecific_data : np.ndarray
            The conspecific data for which to obtain a measure.
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        np.ndarray
            The embedded conspecific data.
        """
        if self.encoder is None:
            self.setup_encoder(PCAEncoder, {"sample_input": conspecific_data, "n_components": self.archive.measure_dim})

        with torch.no_grad():
            # zero dimension is batch, first dimension is for multiple samples of the same policy
            data = torch.from_numpy(conspecific_data)
            meas = torch.stack([self.encoder.encode(d) for d in data])
            while len(meas.shape) > 2:  # noqa: PLR2004
                meas = torch.mean(meas, dim=1)
            batch_size = conspecific_data.shape[0]
            if meas.shape != (batch_size, self.archive.measure_dim):
                meas = meas.reshape(batch_size, self.archive.measure_dim)

        return meas.numpy()

    def train_encoder(self, **kwargs) -> dict[str, Any]:
        """
        Train the encoder.

        Parameters
        ----------
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        dict[str, Any]
            A dictionary of results from training the encoder
        """
        # use all existing data for training

        conspecific_data = self._construct_data_samples(**kwargs)[-1]
        if not np.prod(conspecific_data.shape):
            return {}
        train_data: torch.Tensor = torch.from_numpy(conspecific_data)
        if not train_data.requires_grad:
            train_data.requires_grad_(True)

        if self.encoder is None:
            self.setup_encoder(PCAEncoder, {"sample_input": train_data, "n_components": self.archive.measure_dim})

        encoder_loss = None
        # TODO: make these stop conditions configurable
        train_ind = list(range(len(train_data)))
        self._numpy_generator.shuffle(train_ind)
        batch = train_data[train_ind].reshape(int(np.prod(conspecific_data.shape[:2]).item()), -1)
        encoder_loss = self.encoder.fit(batch).detach().numpy()
        return {"train_encoder": summarize({"encoder_loss": np.mean(encoder_loss)})}

    def ask(self, batch_size: int | None = None, remote: bool = False, **kwargs) -> np.ndarray:
        """
        Ask ask the emitters for a new batch of candidates.

        Parameters
        ----------
        batch_size : int | None, optional
            The size of the batch, :data:`python:None`
        remote : bool, optional
            Whether to return the batch as a remote object reference, :data:`python:True`
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        np.ndarray
            The batch of candidates.
        """
        if self.archive is None:
            self.setup(**kwargs)

        batch = None
        em = self._numpy_generator.choice(self.emitters)
        if batch_size:
            while True:
                batch = em.ask() if batch is None else np.concatenate([batch, em.ask()])
                if batch.shape[0] >= batch_size:
                    batch = batch[:batch_size]
                    break
                em = self._numpy_generator.choice(self.emitters)
        else:
            batch = em.ask()

        return ray.put(batch) if remote else batch

    def add_migrants_to_buffer(
        self,
        migrants: dict[MainlandID, list[tuple[MOAIMIslandInhabitant | None, RolloutData | SampleBatch]]] | None = None,
        update_archives: bool = False,
        *,
        problem: MOAIMIslandUDP,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Add immigrants to the populations migrant buffer.

        Parameters
        ----------
        migrants : list[np.ndarray]
            The genomes of thefreedictionary migrants to add to the current population.
        update_archives : bool, optional
            Whether to call :meth:`update_archives` after the migrants are added.
        problem : MOAIMIslandUDP
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

        if update_archives:
            results.update(self.update_archives(**kwargs))
        return {"add_migrants_to_buffer": results}

    @staticmethod
    def conspecific_rollout_from_sample_batch(sample_batch: SampleBatch, *, problem: MOAIMIslandUDP, **kwargs) -> np.ndarray:
        """
        Calculate the conspecific data from a SampleBatch.

        Parameters
        ----------
        sample_batch : SampleBatch
            The SampleBatch containing conspecific data.
        problem : MOAIMIslandUDP
            The problem this population is solving
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        np.ndarray
            The conspecific data from the sample batch.
        """
        cdata = [
            np.concatenate([np.atleast_1d(batch[key]).reshape(len(batch), -1) for key in problem.conspecific_data_keys], axis=-1)
            for batch in sample_batch.split_by_episode()
            if len(batch) > problem.behavior_classification_config.trajectory_length
        ]
        # zero-pad the beginning if slice_len < 0 (slicing from end)
        # otherwise zero-pad end (slicing from beginning)
        return np.stack([
            cd[-problem.behavior_classification_config.trajectory_length :]
            for cd in cdata[-problem.behavior_classification_config.num_samples :]
        ])

    def update_archives(self, **kwargs) -> dict[str, Any]:
        """
        Update this populations archives.

        trains the encoder, refreshes the archives, and clears and migrants or offspring in the queue.

        Returns
        -------
        dict[str, Any]
            A dictionary of info from this step.
        `**kwargs`
            Additional keyword arguments.

        Raises
        ------
        RuntimeError
            Number of episodes is less than the number of components needed for the encoder
        """
        logger.debug("Updating archives")
        if self.archive is None:
            self.setup(**kwargs)

        results: dict[str, Any] = {}
        pop_before = len(self.archive)
        if self.result_archive is not None:
            res_pop_before = len(self.result_archive)
        train_results = self.train_encoder(**kwargs)
        if train_results:
            results["migrants_in_queue"] = len(self.migrant_buffer)

            self.refresh_archives(**kwargs)
            try:
                self.add_from_buffer(**kwargs)
            except AttributeError as err:
                traceback.format_exc()

                msg = "Probably num episodes < num components, try increasing iterations or batch size."
                raise RuntimeError(msg) from err

        pop_after = len(self.archive)
        obj_after = self.archive.data("objective")
        k_neighbors = min(len(self.archive), self.archive.k_neighbors)
        novelty, _indices = self.archive._cur_kd_tree.query(  # noqa: SLF001
            self.archive.data("measures"),
            k=k_neighbors,
            **self.archive._ckdtree_query_kwargs,  # noqa: SLF001
        )
        novelty = novelty[:, None] if k_neighbors == 1 else novelty
        results["archive"] = summarize({
            "population_change": pop_after - pop_before,
            "population_total": pop_after,
            "fitness": obj_after if np.prod(obj_after.shape) > 1 else np.zeros(1),
            "novelty": np.mean(novelty, axis=1),
        })
        if self.visualize_archive and kwargs.get("visualize_archive", True):
            results["archive/heatmap"] = self.archive_visualizer(self.archive)

        if self.result_archive is not None:
            res_pop_after = len(self.result_archive)
            res_obj_after = self.result_archive.data("objective")
            novelty, _indices = self.result_archive._cur_kd_tree.query(  # noqa: SLF001
                self.result_archive.data("measures"),
                k=k_neighbors,
                **self.result_archive._ckdtree_query_kwargs,  # noqa: SLF001
            )
            novelty = novelty[:, None] if k_neighbors == 1 else novelty
            results["result_archive"] = summarize({
                "population_change": res_pop_after - res_pop_before,
                "population_total": res_pop_after,
                "fitness": res_obj_after if np.prod(res_obj_after.shape) > 1 else np.zeros(1),
                "novelty": np.mean(novelty, axis=1),
            })
            if self.visualize_result_archive and kwargs.get("visualize_archive", True):
                results["result_archive/heatmap"] = self.result_archive_visualizer(self.result_archive)

        return {"update_archives": results}

    def build_batch(
        self,
        *,
        emitter: EmitterBase,
        problem: MOAIMIslandUDP,
        batch_size: int | None = None,
        team_id: int = 0,
        do_batched: bool = False,
        teammates: Sequence[np.ndarray] | None = None,
        names: Sequence[AgentID] | None = None,
        **kwargs,
    ) -> dict[TeamID, dict[AgentID, MOAIMIslandInhabitant]]:
        """
        Build a batch from the emitter.

        Parameters
        ----------
        emitter : EmitterBase
            The emitter from which we wish to sample.
        problem : MOAIMIslandUDP
            The problem this population is solving.
        batch_size : int | None
            The desired batch size.
        team_id : int, optional
            The starting ID of the team, by default 0. The size of the batch returned will be at most :python:`batch_size - team_id`.
        do_batched : bool, optional
            Whether we will stop at the batch size, :data:`python:False`. If True then as soon as the :python:`batch_size` is met the loop
            will exit.
        teammates : Sequence[np.ndarray] | None, optional
            The genomes of teammates to put on the team, :data:`python:None`. If None then teams are singletons.
        names : Sequence[AgentID] | None, optional
            The names to use for the agents.
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        dict[TeamID, dict[AgentID, MOAIMIslandInhabitant]]
            The dictionary of teams
        """
        if self.archive is None:
            self.setup(problem=problem, **kwargs)

        teams = {}
        name = iter(names or [f"agent_{idx}" for idx in range(len(teammates or []))])
        genomes = emitter.ask()
        for genome in genomes:
            ind = MOAIMIslandInhabitant(
                name=next(name),
                team_id=team_id,
                inhabitant_id=str(uuid4()),
                genome=genome,
                island_id=self.island_id,
                current_island_id=self.island_id,
                conspecific_utility_dict=dict(zip(problem.fitness_metric_keys, problem.fitness_multiplier, strict=True)),
            )
            team = {ind.name: ind}
            if teammates is not None:
                team.update({
                    ally.name: ally
                    for ally in [
                        MOAIMIslandInhabitant(
                            name=next(name),
                            team_id=team_id,
                            inhabitant_id=str(uuid4()),
                            genome=t,
                            island_id=self.island_id,
                            current_island_id=self.island_id,
                            conspecific_utility_dict=dict(zip(problem.fitness_metric_keys, problem.fitness_multiplier, strict=True)),
                        )
                        for tm_id, t in enumerate(teammates)
                    ]
                })
            teams[team_id] = team
            name = iter(names or [f"agent_{idx}" for idx in range(1 + len(teammates or []))])
            team_id += 1
            if do_batched and team_id >= batch_size:
                break

        return teams

    def get_random_teammates(self, n_teammates: int = 0) -> list[np.ndarray]:
        """
        Get ``n_teammates`` from the :attr:`~algatross.algorithms.genetic.mo_aim.population.MOAIMIslandPopulation.random_emitter`.

        Parameters
        ----------
        n_teammates : int, optional
            The number of teammates to sample from the random emitter, by default 0

        Returns
        -------
        list[np.ndarray]
            The list of teammates sampled from the
            :attr:`~algatross.algorithms.genetic.mo_aim.population.MOAIMIslandPopulation.random_emitter`.
        """
        return [self.random_emitter.ask() for _ in range(n_teammates)]

    def _get_teams_from_archive(
        self,
        problem: MOAIMIslandUDP,
        batch_size: int | None = None,
        randomized: bool = True,
        remote: bool = False,
        names: list | None = None,
        num_teammates: int = 0,
        **kwargs,
    ) -> dict | ray.ObjectRef:
        """
        Get a team of allies or opponents from the archive for training.

        Constructs teams of agents. The number of teams is determined by ``batch_size`` where it is assumed that each team
        contains one trainable agent. The remaining agents on each team are chosen by the
        :attr:`~algatross.algorithms.genetic.mo_aim.population.MOAIMIslandPopulation.random_emitter`.

        Parameters
        ----------
        problem : MOAIMIslandUDP
            The problem this population is solving.
        batch_size : int | None, optional
            The number of teams we wish to train, :data:`python:None`. If None then the batch size is determined by the batch size of the
            chosen emitters.
        randomized : bool, optional
            Whether to randomize the choice of emitter, :data:`python:True`. If True then a emitters are selected at random to build the
            teams up to the batch size. If True and no batch size is given then the batch size will equal the number of emitters. If False
            then each emitter is sampled in order until the ``batch_size`` is met. If no batch size is given then each emitter is sampled
            exactly once. The total batch size is therefore determined by the batch size of each emitter.
        remote : bool, optional
            Whether to return the teams as a remote object reference, :data:`python:True`.
        names : list, optional
            Names to use for the agents, default is :data:`python:None`.
        num_teammates : int, optional
            The number of teammates for the team, default is 0.
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        dict | ray.ObjectRef
            The a dict with ``batch_size`` teams.
        """
        if self.archive is None:
            self.setup(problem=problem, **kwargs)

        logger.debug(f"\t\t== Update teams for training: {names}")
        teams: dict[TeamID, list] = {}
        do_batched = batch_size is not None
        n_emitters = len(self.emitters)
        names = sorted(names or [])
        if randomized:
            batch_size = n_emitters
            emitter = self.random_emitter_generator()
        elif do_batched:
            emitter = cycle(self.emitters)  # type: ignore[assignment]
        else:
            emitter = iter(self.emitters)  # type: ignore[assignment]

        # convenience function for checking if while look should continue
        def should_continue(teams, do_batched, n_emitters=n_emitters, batch_size=batch_size):
            if do_batched:
                return len(teams) < batch_size
            return len(teams) < n_emitters

        with suppress(StopIteration):
            while should_continue(teams, do_batched, n_emitters):
                teammates = self.get_random_teammates(num_teammates)
                teams.update(
                    self.build_batch(  # type: ignore[arg-type]
                        emitter=next(emitter),
                        batch_size=batch_size,
                        team_id=len(teams),
                        do_batched=do_batched,
                        teammates=teammates,
                        names=names,
                        problem=problem,
                        **kwargs,
                    ),
                )

        return ray.put(teams) if remote else teams

    def step_training_teams_cache(
        self,
        new_teams: dict,
        problem: MOAIMIslandUDP,  # noqa: ARG002
        remote: bool = False,
        **kwargs,
    ) -> dict:
        """Update the cache of training teams.

        Parameters
        ----------
        new_teams : dict
            The new teams to enter into the cache.
        problem : MOAIMIslandUDP
            The problem this population is working on.
        remote : bool, optional
            Whether to put the objects into the object store.
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        dict
            The updated :attr:`~algatross.algorithms.genetic.mo_aim.population.MOAIMIslandPopulation.training_teams_cache`
        """
        updated = self.training_teams_cache
        if remote or isinstance(updated, ray.ObjectRef):
            updated = ray.get(updated)  # type: ignore[call-overload]

        for team in updated:
            for agent_k in updated[team]:
                updated[team][agent_k] = new_teams[team][agent_k]

        if remote:
            updated = ray.put(updated)

        self.training_teams_cache = updated

        return {"updated": "Success"}

    def get_teams_for_training(  # type: ignore[override]
        self,
        problem: MOAIMIslandUDP,
        batch_size: int | None = None,
        randomized: bool = True,
        remote: bool = False,
        refresh: bool = True,
        **kwargs,
    ) -> dict[TeamID, dict[AgentID, MOAIMIslandInhabitant]]:
        """Get the teams to be used for training.

        Parameters
        ----------
        problem : MOAIMIslandUDP
            The problem this population is working on.
        batch_size : int | None, optional
            The batch size of agents to pull from the archive, :data:`python:None`.
        randomized : bool, optional
            Whether to randomize the choice of emitter, :data:`python:True`.
        remote : bool, optional
            Whether to return the objects as remote :class:`~ray.ObjectRef`, :data:`python:False`.
        refresh : bool, optional
            Whether to pull from the :attr:`~algatross.algorithms.genetic.mo_aim.population.MOAIMIslandPopulation.training_teams_cache`
            or conjure a fresh batch, :data:`python:True`.
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        dict[TeamID, dict[AgentID, MOAIMIslandInhabitant]]
            _description_
        """
        if refresh or self.training_teams_cache is None:
            self.training_teams_cache = self._get_teams_from_archive(  # type: ignore[assignment]
                problem=problem,
                batch_size=batch_size,
                randomized=randomized,
                remote=remote,
                names=list(problem.ally_agents),
                num_teammates=self.config.team_size - len(problem.training_agents) if self.config.team_size else 0,
                **kwargs,
            )

        return self.training_teams_cache

    def get_teams_for_evaluation(
        self,
        problem: MOAIMIslandUDP,
        remote: bool = False,
        num_island_elites: int = 1,
        elites_per_island_team: int = 1,
        **kwargs,
    ) -> dict[TeamID, dict[AgentID, MOAIMIslandInhabitant]]:
        """
        Get elite ally agents to use in evaluation.

        Parameters
        ----------
        problem : MOAIMIslandUDP
            The problem this population is solving.
        remote : bool, optional
            Whether to handle objects as remote references, by default :data:`python:False`
        num_island_elites : int, optional
            The number of island elites to fetch, by default 1
        elites_per_island_team : int, optional
            The number of elites per team, by default 1
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        dict[TeamID, dict[AgentID, MOAIMIslandInhabitant]]
            The dictionary of teams.
        """
        # im just gonna cram together build_batch with _get_from_archive and Fix It Later®
        logger = logging.getLogger("ray")
        num_allies = len(list(problem.ally_agents))
        teams = {}

        if elites_per_island_team > num_allies:
            msg = (
                f"Requested {elites_per_island_team} elites per team but team can only have {num_allies} allies. "
                f"Defaulting to num_island_elites={num_allies}"
            )
            logger.warning(msg)

        elites_per_island_team = min(num_allies, elites_per_island_team)

        if num_island_elites < elites_per_island_team:
            msg = f"num_island_elites must be >= elites_per_island_team. Defaulting to num_island_elites={elites_per_island_team}"
            logger.warning(msg)

        num_island_elites = max(num_island_elites, elites_per_island_team)

        data = self.archive.data()
        indices = np.arange(len(data["objective"]))
        # largest -> smallest
        best_indices = sorted(indices, key=lambda ix: data["objective"][ix], reverse=True)
        best_solutions = list(data["solution"][best_indices[:num_island_elites]])

        team_id = 0
        while best_solutions:
            names = iter(problem.ally_agents)
            team = {}

            for _ in range(elites_per_island_team):
                if len(best_solutions) == 0:
                    break

                elite_genome = best_solutions.pop()

                elite_inhabitant = MOAIMIslandInhabitant(
                    name=next(names),
                    team_id=team_id,
                    inhabitant_id=str(uuid4()),
                    genome=elite_genome,
                    island_id=self.island_id,
                    current_island_id=self.island_id,
                    conspecific_utility_dict=dict(zip(problem.fitness_metric_keys, problem.fitness_multiplier, strict=True)),
                )
                team.update({elite_inhabitant.name: elite_inhabitant})

            # TODO: different strategies
            random_fillers = self.get_random_teammates(num_allies - len(team))
            if len(random_fillers):
                logger.warning(f"Not enough elites to fill the team, filling with {len(random_fillers)} random genome(s)!")

            for random_genome in random_fillers:
                random_inhabitant = MOAIMIslandInhabitant(
                    name=next(names),
                    team_id=team_id,
                    inhabitant_id=str(uuid4()),
                    genome=random_genome,
                    island_id=self.island_id,
                    current_island_id=self.island_id,
                    conspecific_utility_dict=dict(zip(problem.fitness_metric_keys, problem.fitness_multiplier, strict=True)),
                )
                team.update({random_inhabitant.name: random_inhabitant})

            teams[team_id] = team.copy()
            team_id += 1

        if remote:
            teams = ray.put(teams)

        return teams

    def get_teams_for_competition(  # type: ignore[override]
        self,
        *,
        problem: MOAIMIslandUDP,
        batch_size: int | None = None,
        randomized: bool = True,
        remote: bool = False,
        refresh: bool = True,
        **kwargs,
    ) -> dict[TeamID, dict[AgentID, MOAIMIslandInhabitant]]:
        """
        Get the teams to be used to compete as opponents on other islands/mainlands.

        Parameters
        ----------
        problem : MOAIMIslandUDP
            The problem this population is solving
        batch_size : int, optional
            The team batch size, default is :data:`python:None`
        randomized : bool, optional
            Whether to randomize the emitters, default is :data:`python:True`
        remote : bool = False
            Whether to return the result as a :class:`~ray.ObjectRef`
        refresh : bool = True
            Whether to refresh the :attr:`~algatross.algorithms.genetic.mo_aim.population.MOAIMIslandPopulation.opposing_teams_cache`
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        dict[TeamID, dict[AgentID, MOAIMIslandInhabitant]]
            Dictionary mapping team IDs to team members
        """
        opposing_teams: dict[TeamID, dict[AgentID, MOAIMIslandInhabitant]] = {}
        self._opposing_team_map = {}
        if self._opposing_islands:
            self.sample_opposing_teams(len(problem.opponent_agents))
            for isl, teams in self._opposing_teams.items():
                for team_id, team in enumerate(teams, len(opposing_teams)):
                    opposing_teams[team_id] = team
                    self._opposing_team_map[team_id] = isl
            return opposing_teams
        if refresh or self.opposing_teams_cache is None:
            self.opposing_teams_cache = self._get_teams_from_archive(  # type: ignore[assignment]
                problem=problem,
                batch_size=batch_size,
                randomized=randomized,
                remote=remote,
                names=list(problem.opponent_agents),
                num_teammates=max(len(problem.opponent_agents) - 1, 0),
                **kwargs,
            )

        return self.opposing_teams_cache

    def random_emitter_generator(self) -> Generator[EmitterBase, None, None]:
        """
        Indefinitely return a randomly chosen emitter from the list of emitters.

        Yields
        ------
        EmitterBase
            An emitter randomly chosen from the emitters list
        """
        while True:
            yield self._numpy_generator.choice(self.emitters)

    @property
    def population_size(self) -> int:  # noqa: D102
        return len(self.archive or [])

    @property
    def max_population_size(self) -> int:  # noqa: D102
        return self.archive.cells


class MOAIMMainlandPopulation(MOAIMPopulation):  # noqa: PLR0904
    """
    Apopulation evolving on a MO - AIM mainland.

    The population itself are teams rather than individual policies. Teams are actually a set of ids which indicate
    which of the individual policies belong to the team.

    Parameters
    ----------
    training_agents : Sequence[AgentID]
        The agents which are to be trained in this population
    seed : NumpyRandomSeed | None = None
        The random number generator seed
    island_id : MainlandID | None = None
        The id for this island
    `**kwargs`
        Additional keyword arguments.
    """

    config: MOAIMMainlandPopulationConfig
    """The config for this population."""
    free_agents: dict[InhabitantID, MOAIMIslandInhabitant]
    """Dict[InhabitantID, MOAIMIslandInhabitant]."""
    island_teams: dict[IslandID, set[TeamID]]
    """The set of teams which contain agents from the given island."""
    elite_teams: list[TeamID]
    """The list of elite teams."""
    non_elite_teams: list[TeamID]
    """The list of non-elite teams."""
    rollout_buffers: dict[InhabitantID, list[SampleBatch]]
    """The rollout buffers for each inhabitant."""
    n_teams: int
    """The number of teams on the island."""
    n_elites: int
    """The number of elites to keep each generation."""
    n_non_elites: int
    """The maximum number of non-elites."""
    team_size: int
    """The size of each team on the island."""

    _agents: dict
    _teams: dict
    _opposing_teams: dict[MainlandID, list[dict[AgentID, MOAIMIslandInhabitant]]]
    _f: dict[TeamID, np.ndarray | None]
    _problem: MOAIMMainlandUDP

    def __init__(
        self,
        # prob: MOAIMMainlandUDP,
        *,
        training_agents: Sequence[AgentID],
        seed: NumpyRandomSeed | None = None,
        island_id: MainlandID | None = None,
        **kwargs,
    ):
        config_dict = {**kwargs}
        config_dict["seed"] = config_dict.get("seed") or seed
        self.config: MOAIMMainlandPopulationConfig = MOAIMMainlandPopulationConfig(
            **filter_keys(MOAIMMainlandPopulationConfig, **config_dict),
        )
        self.storage_path = self.config.storage_path
        self.free_agents = {}
        self._agents = {}
        self.island_teams = {}
        self.n_teams = self.config.n_teams
        self.n_elites = int(np.floor(self.config.elite_proportion * self.config.n_teams))
        self.n_non_elites = int(self.config.n_teams - self.n_elites)
        self._elites_to_return: MigrantFlock = []
        self._teams_to_compete: dict[TeamID, dict[AgentID, MOAIMIslandInhabitant]] = {}
        self._elites: list[MOAIMIslandInhabitant] = []
        self._non_elites: list[MOAIMIslandInhabitant] = []
        self.elite_teams: list[TeamID] = []
        self.non_elite_teams = list(range(self.config.n_teams))
        team_id = 0
        team = {}
        teams = {}
        non_elites = []
        name = iter(cycle(sorted(training_agents)))
        self.team_size = len(training_agents)
        for idx in range(int(self.config.n_teams * self.team_size)):
            a = MOAIMIslandInhabitant(
                name=next(name),
                team_id=team_id,
                island_id=-1,
                current_island_id=self.island_id,
                inhabitant_id=str(uuid4()),
                genome=np.empty(1),
                conspecific_utility_dict={},
            )
            self._agents[a.inhabitant_id] = a
            non_elites.append(a)
            team[a.name] = a
            if idx + 1 == self.team_size:
                teams[team_id] = team
                team_id += 1
                assert len(team) == len(  # noqa: S101
                    training_agents,
                ), f"Partial team {team_id - 1}: {list(team)} {training_agents}"

        self.non_elites = non_elites
        self._teams = teams
        self._opposing_teams = {}
        self.f = dict.fromkeys(range(self.config.n_teams))
        self.rollout_buffers = {}

        if island_id is not None:
            self.island_id = island_id

        # self._numpy_generator = get_generators(seed=, seed_global=False)[0]
        seed = self.config.seed
        self._numpy_generator = resolve_seed(seed)  # type: ignore[arg-type]
        self._torch_generator = get_torch_generator_from_numpy(self._numpy_generator)[0]

    @MOAIMPopulation.island_id.setter  # type: ignore[attr-defined]
    def island_id(self, new_id: IslandID | MainlandID):  # noqa: D102
        self._island_id = new_id
        for agent in chain(self._agents.values(), self._elites, self._non_elites, self.free_agents.values()):
            agent.current_island_id = new_id

    def init_teams(self, reset: bool = True, *, problem: MOAIMMainlandUDP, **kwargs) -> dict[str, Any]:
        """Initialize the teams by randomly grouping inhabitants from the free-agents.

        Parameters
        ----------
        reset : bool
            Whether to re-shuffle the teams after initialization
        problem : MOAIMMainlandUDP
            The problem the population is working on
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        dict[str, Any]
            A dictionary of info from this step

        Raises
        ------
        ValueError
            If the incoming agents can't be grouped into the correct number of teams
        """
        results: dict[str, Any] = {}
        results["elite_teams/ids"] = self.elite_teams
        results["elite_teams/count"] = len(self.elite_teams)
        results["non_elite_teams/ids"] = self.non_elite_teams
        results["non_elite_teams/count"] = len(self.non_elite_teams)
        replaced_count = 0
        if self.non_elite_teams and self.free_agents.keys():
            teammates = list(self.free_agents)
            self._numpy_generator.shuffle(teammates)
            try:
                new_teams = np.split(np.array(teammates), len(self.non_elite_teams))
            except ValueError as err:
                msg = f"Could not split the {len(teammates)} free agents into {len(self.non_elite_teams)} teams"
                raise ValueError(msg) from err
            for team_idx, new_team in zip(self.non_elite_teams, new_teams, strict=True):
                team = {}
                name = iter(problem.training_agents)
                for teammate_id in new_team:
                    replaced_count += 1
                    # update the team_id for the agent
                    teammate = self.free_agents[teammate_id]
                    # add them to the set of agents
                    self._agents[teammate.inhabitant_id] = MOAIMIslandInhabitant(
                        name=next(name),
                        team_id=team_idx,
                        inhabitant_id=teammate.inhabitant_id,
                        island_id=teammate.island_id,
                        current_island_id=self.island_id,
                        genome=teammate.genome.copy(),
                        conspecific_utility_dict=teammate.conspecific_utility_dict,
                    )
                    team[self._agents[teammate_id].name] = self._agents[teammate_id]
                msg = (
                    f"New team ({team_idx}) does not match the expected team size:\n\tteam: {list(team)}"
                    f"\n\ttraining agents: {problem.training_agents}"
                    f"\nNon-elite teams: {list(self.non_elite_teams)}"
                )
                assert len(team) == len(problem.training_agents), msg  # noqa: S101
                self._teams[team_idx] = team
            if reset:
                self.reset_non_elites()
            self.free_agents.clear()
        results["replaced_individuals/count"] = replaced_count
        return {"init_teams": results}

    def sync_teams(self, *, problem: MOAIMMainlandUDP, **kwargs) -> dict:
        """Sync the teams according to how agents have been assigned.

        Parameters
        ----------
        problem : MOAIMMainlandUDP
            The problem this population is working on
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        dict
            The synced teams

        Raises
        ------
        ValueError
            if there is a discrepency in the team members
        """
        teams: dict[TeamID, dict[AgentID, MOAIMIslandInhabitant]] = defaultdict(dict)
        elites = []
        non_elites = []

        # make sure agents Team ID matches the team to which it is assigned
        for team_id, team in self.teams.items():
            for team_name, agent in team.items():
                agent.name = team_name
                agent.team_id = team_id
                self._agents[agent.inhabitant_id] = agent
        if not all(len(team) == len(problem.training_agents) for team in self.teams.values()):
            wrong_teams = []
            right_teams = []
            for team_id, team in self.teams.items():
                if len(team) != len(problem.training_agents):
                    wrong_teams.append((team_id, len(team), list(team)))
                else:
                    right_teams.append((team_id, len(team), list(team)))
            msg = f"Teams {[info[0] for info in wrong_teams]} do not have the correct number of trainable agents."
            msg += "".join([f"\n\tteam: {info[0]}\n\tsize: {info[1]}\n\tagents: {info[2]}\n" for info in wrong_teams])
            raise ValueError(msg)

        # make sure agent is in the correct group (elite/non-elite)
        for inhabitant_id, agent in self.agents.items():
            agent.inhabitant_id = inhabitant_id
            teams[agent.team_id][agent.name] = agent
            if agent.team_id in self.elite_teams:
                elites.append(agent)
            else:
                non_elites.append(agent)
        self._elites = elites
        self._non_elites = non_elites
        self._teams = dict(teams.items())
        return teams

    def reset_non_elites(self):
        """Clear the containers for non-elite individuals and teams."""
        self.non_elites.clear()
        self.non_elite_teams.clear()

    def replace_non_elite(self, ind: MOAIMIslandInhabitant, clear_buffer: bool = False):
        """Replace a non-elite with a migrant.

        Parameters
        ----------
        ind : MOAIMIslandInhabitant
            The individual which is replacing a non-elite
        clear_buffer : bool, optional
            Whether to clear the buffer for the inhabitant id
        """
        replaced = self.non_elites.pop()
        ind.name = replaced.name
        # remove the old agent
        self.free_agents.pop(replaced.inhabitant_id, None)
        self._agents.pop(replaced.inhabitant_id, None)
        self._agents.pop(ind.inhabitant_id, None)

        # add the new agent
        self.free_agents[ind.inhabitant_id] = ind

        # make sure the buffer from the last inhabitant with that same ID isnt used for the new inhabitant
        if clear_buffer:
            self.rollout_buffers.pop(ind.inhabitant_id, None)
            self.rollout_buffers.pop(replaced.inhabitant_id, None)

    def add_migrants_to_buffer(
        self,
        migrants: MigrantQueue,
        conspecific_utility_dicts: dict[IslandID, dict[str, float]],
        reset_non_elites: bool = True,
        *args,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Add migrants from the queue to the mainland.

        Parameters
        ----------
        migrants : MigrantQueue
            The queue of migrants to incorporate.
        conspecific_utility_dicts : dict[IslandID, dict[str, float]]
            The dict of conspecific utility weights for each island
        reset_non_elites : bool, optional
            Whether to reset the non-elites and non-elite teams, default is :data:`python:True`
        `*args`
            Additional positional arguments.
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        dict[str, Any]
            A dictionary of info from this step
        """
        replaced = 0
        if reset_non_elites:
            self.non_elites = list(self.agents.values())
        if migrants:
            for island_id, manifest in migrants.items():
                for data in manifest:
                    if isinstance(data[0], np.ndarray):
                        genome = data[0]
                    elif isinstance(data[0], MOAIMIslandInhabitant):
                        genome = data[0].genome
                    elif isinstance(data[1], RolloutData):
                        genome = data[1].solution
                    ind = MOAIMIslandInhabitant(
                        name=-1,
                        team_id=-1,
                        inhabitant_id=str(uuid4()),
                        genome=genome,
                        island_id=island_id,
                        current_island_id=self.island_id,
                        conspecific_utility_dict=conspecific_utility_dicts[island_id],
                    )
                    self.replace_non_elite(ind, clear_buffer=True)
                    replaced += 1

        return {}

    def update_teams_from_results(self, update_dict: dict[str, Any], *, problem: MOAIMMainlandUDP, **kwargs) -> dict[str, Any]:
        """Update the teams from an update dictionary.

        Parameters
        ----------
        update_dict : dict[str, Any]
            The dictionary of update information. Must contain the following keys:

                - ``elite_teams``
                    A list of TeamIDs of the elite teams
                - ``non_elite_teams``
                    A list of TeamIDs of the non-elite teams
                - ``free_agents``
                    The dictionary of free agents (dict[InhabitantID, MOAIMIslandInhabitant])
                - ``team_fitness``
                    The dictionary of fitness values for each team
                - ``elites``
                    The list of elite inhabitants
                - ``non_elites``
                    The list of non-elite inhabitants
        problem : MOAIMMainlandUDP
            The problem being solved by this population
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        dict[str, Any]
            A dictionary of info from this step
        """
        results = {}
        updates = update_dict
        self.elites = updates["elites"].copy()
        self.non_elites = updates["non_elites"].copy()
        self.elite_teams = updates["elite_teams"].copy()
        self.non_elite_teams = updates["non_elite_teams"].copy()
        self.free_agents = updates["free_agents"].copy()
        for team_id, team_f in updates["team_fitness"].items():
            self.set_f(team_id, team_f.copy())

        results.update(self.init_teams(reset=False, problem=problem))

        return {"update_teams_from_results": results}

    def set_elites_to_return(self):
        """Add the elites from the most recent training epoch to the outgoing elites."""
        elite_teams: dict[TeamID, dict[AgentID, MOAIMIslandInhabitant]] = defaultdict(dict)
        for elite in self.elites:
            elite_copy = MOAIMIslandInhabitant(
                name=elite.name,
                team_id=elite.team_id,
                inhabitant_id=elite.inhabitant_id,
                genome=elite.genome.copy(),
                island_id=elite.island_id,
                current_island_id=self.island_id,
                conspecific_utility_dict=elite.conspecific_utility_dict.copy(),
            )
            self._elites_to_return.append((elite_copy, self.rollout_buffers[elite.inhabitant_id]))
            elite_teams[elite.team_id][elite.name] = elite_copy
        self._teams_to_compete = dict(elite_teams)

    def reset_teams(self, *, problem: MOAIMMainlandUDP, **kwargs):
        """Reset the teams so that all agents are randomly grouped and all elites/non-elites are reset.

        Parameters
        ----------
        problem : MOAIMMainlandUDP
            The problem this population is working on
        `**kwargs`
            Additional keyword arguments.

        Raises
        ------
        ValueError
            if there is a discrepency in the team members
        """
        self.elites.clear()
        self.elite_teams.clear()
        self.free_agents.clear()

        teammates = list(
            zip(sorted(list(range(self.n_teams - self.n_elites)) * self.team_size), cycle(sorted(problem.training_agents)), strict=False),
        )
        teams: dict[TeamID, dict[AgentID, MOAIMIslandInhabitant]] = defaultdict(dict)
        self._numpy_generator.shuffle(teammates)

        agents = list(self.agents.values())
        self._agents.clear()
        for agent in agents:
            team_id, name = teammates.pop()
            self.agents[agent.inhabitant_id] = MOAIMIslandInhabitant(
                name=name,
                team_id=team_id,
                inhabitant_id=agent.inhabitant_id,
                genome=agent.genome.copy(),
                island_id=agent.island_id,
                current_island_id=self.island_id,
                conspecific_utility_dict=agent.conspecific_utility_dict.copy(),
            )
            teams[team_id].update({name: self._agents[agent.inhabitant_id]})

        if not all(len(team) == len(problem.training_agents) for team in teams.values()):
            wrong_teams = []
            right_teams = []
            for team_id, team in self.teams.items():
                if len(team) != len(problem.training_agents):
                    wrong_teams.append((team_id, len(team), list(team)))
                else:
                    right_teams.append((team_id, len(team), list(team)))
            msg = f"Teams {[info[0] for info in wrong_teams]} do not have the correct number of trainable agents."
            msg += f"\nExpected training agents: {list(problem.training_agents)}"
            msg += "".join([f"\n\tteam: {info[0]}\n\tsize: {info[1]}\n\tagents: {info[2]}\n" for info in wrong_teams])
            raise ValueError(msg)

        self.non_elites = list(self._agents.values())
        self.non_elite_teams = list(range(self.config.n_teams))
        self.teams = dict(teams)
        for team_id in self.non_elite_teams:
            self.set_f(team_id, np.inf)

        self.rollout_buffers.clear()

    def get_teams_for_training(self, remote: bool = False, **kwargs) -> dict | ray.ObjectRef:
        """Get the teams we wish to train.

        Parameters
        ----------
        remote : bool, optional
            Whether to return the teams as a remote object reference, :data:`python:True`
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        dict | ray.ObjectRef
            The teams we wish to train.
        """
        return ray.put(dict(self.teams)) if remote else dict(self.teams)

    def get_population_size(self) -> int:
        """
        Get the current population size residing on the island.

        Returns
        -------
        int
            The current population size residing on the island.
        """
        return len(self.agents)

    def get_max_population_size(self) -> int:
        """Get the maximum population size this island can contain.

        Returns
        -------
        int
            The maximum population size this island can contain
        """
        return int(self.config.n_teams * self.team_size)

    def set_f(self, team_id: TeamID, f: float | np.ndarray):
        """Set the fitness value for the team.

        Parameters
        ----------
        team_id : TeamID
            The ID of the team whose fitness value is being updated.
        f : float | np.ndarray
            The new fitness value of the team.
        """
        self.f[team_id] = np.asarray(f).astype(np.float64)

    @property
    def teams(self) -> dict[TeamID, dict[AgentID, MOAIMIslandInhabitant]]:
        """
        The mapping from team IDs to the individuals on the team.

        Returns
        -------
        dict[TeamID, dict[AgentID, MOAIMIslandInhabitant]]
            The mapping of team IDs to individuals on the team.
        """
        return self._teams

    @teams.setter
    def teams(self, other: dict[TeamID, dict[AgentID, MOAIMIslandInhabitant]]):
        self._teams = other

    @property
    def agents(self) -> dict[InhabitantID, MOAIMIslandInhabitant]:
        """
        The individuals on the island.

        Returns
        -------
        dict[InhabitantID, MOAIMIslandInhabitant]
            The pool of agents (individuals) on the island.
        """
        return self._agents

    @agents.setter
    def agents(self, other: Iterable | Mapping):
        self._agents = dict(other)

    @property
    def elites(self) -> list[MOAIMIslandInhabitant]:
        """
        The list of elite inhabitants.

        Returns
        -------
        list[MOAIMIslandInhabitant]
            The list of elite inhabitants.
        """
        return self._elites

    @elites.setter
    def elites(self, other: Sequence):
        self._elites = list(other)

    @property
    def non_elites(self) -> list[MOAIMIslandInhabitant]:
        """The list of non-elite inhabitants.

        Returns
        -------
        list[MOAIMIslandInhabitant]
            The list f non-elite inhabitants.
        """
        return self._non_elites

    @non_elites.setter
    def non_elites(self, other: Sequence):
        self._non_elites = list(other)

    def get_elites_to_return(
        self,
        island_id: IslandID | Sequence[IslandID] | None = None,
        remote: bool = False,
    ) -> MigrantQueue | MigrantFlock:
        """Get the elites which need to return to the island.

        Parameters
        ----------
        island_id : IslandID | Sequence[IslandID] | None, optional
            The ID of the island(s) whose elites we wish to return, :data:`python:None`. If None then the elites for all islands are
            returned as a dictionary. If a sequence then the dictionary contains elites only for the islands specified
        remote : bool, optional
            Whether to return the value as a remote object reference, default is :data:`python:True`

        Returns
        -------
        MigrantQueue | MigrantFlock
            The MigrantQueue (if None was given for the ``island_id`` ) or the MigrantFlock for the island.
        """
        single = False
        if isinstance(island_id, IslandID):
            single = True
            island_id = [island_id]

        returned_indices = []
        island_returnees = defaultdict(list)
        if island_id is None or not single:
            for elite_idx, (elite, elite_buffer) in enumerate(self._elites_to_return):
                if island_id is None or elite.island_id in island_id:  # type: ignore[union-attr]
                    island_returnees[elite.island_id].append((elite, elite_buffer))  # type: ignore[union-attr]
                    returned_indices.append(elite_idx)
        else:
            for elite_idx, elite in enumerate(self._elites_to_return):  # type: ignore[assignment]
                if elite.island_id in island_id:  # type: ignore[union-attr]
                    island_returnees[elite.island_id].append((elite, elite_buffer))  # type: ignore[union-attr]
                    returned_indices.append(elite_idx)

        # clear the returned elites from the buffer
        for elite_idx in returned_indices[::-1]:
            self._elites_to_return.pop(elite_idx)
        return ray.put(island_returnees) if remote else island_returnees

    def get_elite_teams_to_compete(self):  # noqa: D102
        returned_teams = list(self._teams_to_compete.values())
        self._teams_to_compete.clear()
        return returned_teams

    def get_replenishment_size(self) -> int:
        """Get the number of migrants needed in order to replenish this population.

        Returns
        -------
        int
            The number of migrants needed in order to replenish this population.
        """
        return len(self.non_elites)

    def set_rollout_buffers(self, team_buffers: dict[TeamID, dict[AgentID, list[SampleBatch]]]):
        """Set the :attr:`~algatross.algorithms.genetic.mo_aim.population.MOAIMMainlandPopulation.rollout_buffers` property.

        Parameters
        ----------
        team_buffers : dict[TeamID, dict[AgentID, list[SampleBatch]]]
            The mapping from teams, to a mapping of teammate ids and their respective rollout buffers.
        """
        self.rollout_buffers.clear()
        for team_id in self.elite_teams:
            teammates: dict[AgentID, MOAIMIslandInhabitant] = self.teams[team_id]
            buffer = {teammates[teammate].inhabitant_id: teammate_buffer for teammate, teammate_buffer in team_buffers[team_id].items()}
            self.rollout_buffers.update(buffer)

    @property
    def population_size(self) -> int:
        """The current size of the population on the mainland.

        Returns
        -------
        int
            The current size of the population on the mainland.
        """
        return self.get_population_size()

    @property
    def max_population_size(self) -> int:
        """Get the maximum size of the population this mainland can contain.

        Returns
        -------
        int
            The maximum size of the population this mainland can contain.
        """
        return self.get_max_population_size()


@ray.remote  # noqa: PLR0904
class PopulationServer(RayActor):
    """
    A remote container for serving populations to and from islands.

    Parameters
    ----------
    island_populations : dict[IslandID, ConstructorData],
        The island populations and their constructors
    mainland_populations : dict[MainlandID, ConstructorData]
        The mainland populations and their constructors
    seed : NumpyRandomSeed | None, optional
        The seed for randomness.
    log_queue : Queue | None, optional
        The queue for logging messages, default is :data:`python:None`.
    result_queue : Queue | None, optional
        The result queue for reporting results, default is :data:`python:None`.
    `**kwargs`
        Additional keyword arguments.
    """

    island_populations: dict[IslandID, MOAIMIslandPopulation]
    """Copies of the island populations."""
    mainland_populations: dict[IslandID, MOAIMMainlandPopulation]
    """Copies of the mainland populations."""
    migrant_queues: dict[IslandID, MultiQueue | MigrantQueue]
    """Queues for migrants moving between islands and mainlands."""
    conspecific_utilities_dicts: dict[IslandID, dict[str, float]]
    """The dictionary of conspecific utilities for the islands."""
    competitive_teams: dict[MainlandID, list[dict[AgentID, MOAIMIslandInhabitant]]]
    """The teams to use in competition between the mainlands."""
    returned_elite_indices: dict[MainlandID, list[int]]
    """The indices of the elites returning from each mainland."""
    logger: logging.Logger
    """The message logger for ``ray``."""

    _context: ray.runtime_context.RuntimeContext

    def __init__(
        self,
        island_populations: dict[IslandID, ConstructorData],
        mainland_populations: dict[MainlandID, ConstructorData],
        seed: NumpyRandomSeed | None = None,
        log_queue: Queue | None = None,
        result_queue: Queue | None = None,
        **kwargs,
    ):
        RayActor.__init__(self, log_queue=log_queue, result_queue=result_queue, **kwargs)
        island_migrant_queue = MultiQueue(queue_keys=list(island_populations), actor_options={"name": "MigrantQueue"})
        self.island_populations = {
            idx: isl.construct(island_id=idx, namespace=self._namespace, migrant_queue=island_migrant_queue)
            for idx, isl in island_populations.items()
        }
        self.logger = logging.getLogger("ray")
        self.mainland_populations = {idx: isl.construct(island_id=idx) for idx, isl in mainland_populations.items()}

        # Always create new migrant queues
        self.migrant_queues: dict[IslandID, MultiQueue | MigrantQueue] = {
            isl: pop.migrant_queue for isl, pop in self.island_populations.items()
        }
        self.migrant_queues.update({idx: defaultdict(list) for idx in self.mainland_populations})

        self.conspecific_utilities_dicts: dict[IslandID, dict[str, float]] = {}
        self.competitive_teams: dict[MainlandID, list[dict[AgentID, MOAIMIslandInhabitant]]] = {}
        self.returned_elite_indices: dict[MainlandID, list[int]] = {}
        self._numpy_generator = resolve_seed(seed)  # type: ignore[arg-type]
        self._torch_generator = get_torch_generator_from_numpy(self._numpy_generator)[0]
        self._topology: Callable[[Sequence[int], torch.Generator], dict[MainlandID, list[IslandSample]]] | None = None

    def set_conspecific_data(self, island_id: IslandID, *, problem: MOAIMIslandUDP, **kwargs):
        """Set the conspecific data for the island.

        Parameters
        ----------
        island_id : IslandID
            The island to update
        problem : MOAIMIslandUDP
            The problem containing the conspecific data information
        `**kwargs`
            Additional keyword arguments.
        """
        self.island_populations[island_id].setup(problem=problem, **kwargs)
        self.conspecific_utilities_dicts[island_id] = dict(zip(problem.fitness_metric_keys, problem.fitness_multiplier, strict=True))

    @property
    def archipelago_populations(self) -> dict[IslandID | MainlandID, MOAIMIslandPopulation | MOAIMMainlandPopulation]:
        """The populations for the whole archipelago.

        Returns
        -------
        dict[IslandID | MainlandID, MOAIMIslandPopulation | MOAIMMainlandPopulation]
            The all populations in the archipelago.
        """
        return {**self.island_populations, **self.mainland_populations}

    @property
    def topology(self) -> Callable[[Sequence[int], torch.Generator], dict[MainlandID, list[IslandSample]]]:
        """
        The callable which defines the migration between islands and mainlands.

        Returns
        -------
        Callable[[Sequence[int], torch.Generator], dict[MainlandID, list[IslandSample]]]
            A topology function which takes a sequence of migration sizes and returns a dictionary of mainland ids and
            migrants moving to the mainlands.
        """
        return self._topology

    def get_random_generators(self):  # noqa: D102
        return self._numpy_generator, self._torch_generator

    def set_state(self, state: dict) -> None:
        """
        Set the state.

        Parameters
        ----------
        state : dict
            The PopulationServer state to set.
        """
        for idx, pop_state in state["island_populations"].items():
            self.island_populations[idx].set_state(pop_state)

        for idx, pop_state in state["mainland_populations"].items():
            self.mainland_populations[idx].set_state(pop_state)

        self.conspecific_utilities_dicts = state["conspecific_utilities_dicts"]
        self.competitive_teams = state["competitive_teams"]
        self.returned_elite_indices = state["returned_elite_indices"]
        self._topology = state["topology"]

        self._numpy_generator.bit_generator.state = state["numpy_generator_state"]
        self._torch_generator.set_state(state["torch_generator_state"])

    def get_state(self) -> dict:
        """Get the state.

        Returns
        -------
        dict
            The PopulationServer state
        """
        server_state: dict[str, Any] = {}
        server_state["torch_generator_state"] = self._torch_generator.get_state()
        server_state["numpy_generator_state"] = self._numpy_generator.bit_generator.state
        server_state["conspecific_utilities_dicts"] = self.conspecific_utilities_dicts
        server_state["competitive_teams"] = self.competitive_teams
        server_state["returned_elite_indices"] = self.returned_elite_indices
        server_state["topology"] = self._topology

        island_id: IslandID
        island_pop: MOAIMIslandPopulation
        server_state["island_populations"] = {}

        for island_id, island_pop in self.island_populations.items():
            server_state["island_populations"][island_id] = island_pop.get_state()

        mainland_id: MainlandID
        mainland_pop: MOAIMMainlandPopulation
        server_state["mainland_populations"] = {}

        for mainland_id, mainland_pop in self.mainland_populations.items():
            server_state["mainland_populations"][mainland_id] = mainland_pop.get_state()

        return server_state

    def set_topology(self, top: Callable[[Sequence[int], torch.Generator], dict[MainlandID, list[IslandSample]]]):
        """
        Set the stored topology for the population.

        Parameters
        ----------
        top : Callable[[Sequence[int], torch.Generator], dict[MainlandID, list[IslandSample]]]
            A topology function which takes a sequence of migration sizes and returns a dictionary of mainland ids and
            migrants moving to the mainlands.
        """
        self._topology = top

    def initialize_mainlands(self, top: Callable[[Sequence[int], torch.Generator], dict[MainlandID, list[IslandSample]]]):
        """
        Set the topology used for sampling populations and fill the mainlands with an initial random population.

        Parameters
        ----------
        top : Callable[[Sequence[int], torch.Generator], dict[MainlandID, list[IslandSample]]]
            A topology function which takes a sequence of migration sizes and returns a dictionary of mainland ids and
            migrants moving to the mainlands.
        """
        self._topology = top
        for island, population in self.mainland_populations.items():
            self.migrate_to_mainlands(island)
            population.add_migrants_to_buffer(
                self.get_migrants_for(island, False),
                conspecific_utility_dicts=self.conspecific_utilities_dicts,
            )
            self.mainland_populations[island] = population

    def update_archives(self, islands: IslandID | Sequence[IslandID] | None) -> dict[str, Any]:
        """Update the archives of the island.

        Parameters
        ----------
        islands : IslandID | Sequence[IslandID] | None
            The island or islands whos archives need to be updated. If None then all islands' archives are updated.

        Returns
        -------
        dict[str, Any]
            A dictionary of info from this step
        """
        if islands is None:
            islands = list(self.island_populations)
        elif isinstance(islands, IslandID):
            islands = [islands]

        results = {}

        for isl in islands:
            results[f"island/{isl}"] = self.island_populations[isl].update_archives()

        return {"population_server/update_archives": results}

    def get_teams_for_training(
        self,
        island: IslandID | MainlandID,
        batch_size: int | None = None,
        **kwargs,
    ) -> dict[TeamID, list[MOAIMIslandInhabitant]]:
        """Get the teams from the island to use for training.

        Parameters
        ----------
        island : IslandID | MainlandID
            The ID of the island for which we would like to obtain a dict
        batch_size : int | None, optional
            The number of migrants in the dict, :data:`python:None`. In which case the default batch settings of the respective
            island is used.
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        dict[TeamID, list[MOAIMIslandInhabitant]]
            A dictionary of teams to use for training.
        """
        return self.archipelago_populations[island].get_teams_for_training(batch_size=batch_size, **kwargs)  # type: ignore[return-value]

    def add_migrants_to_buffer(self, island: IslandID | MainlandID, migrants: MigrantQueue, *, problem: UDP, **kwargs) -> dict[str, Any]:
        """Add migrants to the islands local buffer.

        Parameters
        ----------
        island : IslandID | MainlandID
            The ID of the island to update.
        migrants : MigrantQueue
            The queue of migrants to use for update.
        problem : UDP
            The problem this population is working on
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        dict[str, Any]
            A dictionary of info from this step
        """
        results = {}
        if island in self.archipelago_populations:
            key = "mainland" if island in self.mainland_populations else "island"
            migrants = ray.get(migrants) if isinstance(migrants, ray.ObjectRef) else migrants
            results[f"{key}/{island}"] = self.archipelago_populations[island].add_migrants_to_buffer(
                migrants,  # type: ignore[arg-type]
                conspecific_utility_dicts=self.conspecific_utilities_dicts,
                update_archives=False,
                problem=problem,  # type: ignore[arg-type]
            )
        return {"population_server/add_migrants_to_buffer": results}

    def set_migrants_for(
        self,
        island_to: IslandID | MainlandID,
        island_from: IslandID | MainlandID | Sequence[IslandID | MainlandID],
        migrants: MigrantFlock | Sequence[IslandID | MainlandID],
    ):
        """Set the migrant queue coming from ``island_from`` going to ``island_to``.

        Parameters
        ----------
        island_to : IslandID | MainlandID
            The island whither the migrants are emigrating.
        island_from : IslandID | MainlandID | Sequence[IslandID | MainlandID]
            The island whence the migrants are emigrating.
        migrants : MigrantFlock | Sequence[IslandID | MainlandID]
            The flock of migrants in migration.
        """
        batch = (
            dict(zip(island_from, migrants, strict=True))
            if isinstance(island_from, Sequence)
            else dict(zip([island_from], [migrants], strict=True))
        )
        if batch:
            logger.debug(
                f"Sending migrants from {sorted(island_from) if isinstance(island_from, Sequence) else island_from} to {island_to}",
            )
            if island_to in self.mainland_populations:
                for fr, mig in batch.items():
                    self.migrant_queues[island_to][fr] += mig  # type: ignore[operator, index]
            else:
                self.migrant_queues[island_to].put_nowait(batch, key=island_to)  # type: ignore[union-attr]

    def get_migrants_for(self, island_to: IslandID | MainlandID, remote: bool = False) -> MigrantQueue:
        """Get the migrant queue for the given island, return a remote object reference if desired.

        Parameters
        ----------
        island_to : IslandID | MainlandID
            The ID of the island whose migrant queue we wish to retrieve.
        remote : bool, optional
            Whether to return a remote object reference or the local buffer, :data:`python:True`

        Returns
        -------
        MigrantQueue
            The migrant queue for the island.
        """
        if island_to in self.mainland_populations:
            queue = ray.put(dict(self.migrant_queues[island_to])) if remote else dict(self.migrant_queues[island_to])  # type: ignore[arg-type]
            self.migrant_queues[island_to].clear()  # type: ignore[union-attr]
        else:
            queue = self.migrant_queues[island_to]
        return queue

    def migrate_to_mainlands(self, mainlands: list[MainlandID] | MainlandID | None) -> dict[str, Any]:
        """Replenish the mainlands' populations by sampling from the islands according to the softmax function.

        Parameters
        ----------
        mainlands : list[MainlandID] | MainlandID | None
            The list of mainlands to receive immigrants. Default is None, in which case all mainlands are migrated.

        Returns
        -------
        dict[str, Any]
            A dictionary of info from this step
        """
        flocks = self.topology(list(self.get_replenishment_size().values()), self._torch_generator)
        results = {}
        mainlands_to_migrate = list(self.mainland_populations) if mainlands is None else mainlands
        mainlands_to_migrate = set([mainlands_to_migrate] if not isinstance(mainlands_to_migrate, list) else mainlands_to_migrate)  # type: ignore[assignment]
        for mainland, samples in flocks.items():
            if mainland not in mainlands_to_migrate:  # type: ignore[operator]
                continue
            ml_results = {}
            total_flock_size = 0
            migration_dict: dict[IslandID | MainlandID, list[tuple[MOAIMIslandInhabitant, None]]] = {}
            for sample in samples:
                total_flock_size += sample.migrants
                migrants = self.island_populations[sample.island].ask(sample.migrants, remote=False)
                migration_dict[sample.island] = [(m, None) for m in migrants]
                ml_results[f"island/{sample.island}/flock/size"] = sample.migrants
            self.set_migrants_for(mainland, list(migration_dict), list(migration_dict.values()))  # type: ignore[arg-type]
            ml_results["total_flock_size"] = total_flock_size
            results[f"mainland/{mainland}"] = ml_results
        return results

    def get_population_size(self, island: IslandID | MainlandID) -> int:
        """
        Get the the current size of the population on the island.

        Parameters
        ----------
        island : IslandID | MainlandID
            The ID of the island whose population size we wish to know.

        Returns
        -------
        int
            The current size of the population on the island
        """
        return self.archipelago_populations[island].population_size

    def get_max_population_size(self, island: IslandID | MainlandID) -> int:
        """Get the maximum size of the population the island can hold.

        Parameters
        ----------
        island : IslandID | MainlandID
            The ID of the island whose max size we're wanting to know.

        Returns
        -------
        int
            The maximum capacity of the island.
        """
        return self.archipelago_populations[island].max_population_size

    def get_population_to_evolve(self, island: IslandID | MainlandID) -> MOAIMIslandPopulation | MOAIMMainlandPopulation:
        """Get the population to evolve from the server for the given island.

        Parameters
        ----------
        island : IslandID | MainlandID
            The ID of the island whose entire population is being evolved.

        Returns
        -------
        MOAIMIslandPopulation | MOAIMMainlandPopulation
            The islands population.
        """
        # create a deepcopy of the competitors
        self.archipelago_populations[island].set_opposing_team_candidates({
            island_id: [[deepcopy(inhabitant) for inhabitant in team.values()] for team in teams]
            for island_id, teams in self.competitive_teams.items()
        })
        return self.archipelago_populations[island]

    def get_archipelago_populations(self):  # noqa: D102
        return self.archipelago_populations

    def set_population(
        self,
        island: IslandID | MainlandID,
        population: MOAIMIslandPopulation | MOAIMMainlandPopulation,
        *,
        problem: MOAIMIslandUDP | MOAIMMainlandUDP,
        **kwargs,
    ) -> dict:
        """Set the population of the island.

        Moves the memory from the ``population`` archives to the current process.

        Parameters
        ----------
        island : IslandID | MainlandID
            The ID of the island to which the population is being assigned
        population : MOAIMIslandPopulation | MOAIMMainlandPopulation
            The population to apply.
        problem : MOAIMIslandUDP | MOAIMMainlandUDP
            The problem the population is working on.
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        dict
            The result info from setting the population
        """
        key = "island" if island in self.island_populations else "mainland"
        msg = f"Setting {key} {island} population"
        log_results = {}
        if island in self.island_populations and isinstance(population, MOAIMIslandPopulation):
            self.log(msg, logger_name="ray")
            _make_archive_owner(population.archive)

            if population.result_archive:
                # move the memory
                _make_archive_owner(population.result_archive)

            # move all the elites coming from the mainlands onto the islands and then add them to the archive
            self.island_populations[island] = population
        elif island in self.mainland_populations and isinstance(population, MOAIMMainlandPopulation):
            self.log(msg, logger_name="ray")
            # extract the elites from the population
            elites: MigrantQueue = population.get_elites_to_return(island_id=None, remote=False)  # type: ignore[assignment]
            competitive_teams = population.get_elite_teams_to_compete()
            if competitive_teams:
                # only add the competitive teams if the list is non-empty
                self.competitive_teams[island] = competitive_teams
            for island_id, migrants in elites.items():
                self.set_migrants_for(island_id, island, migrants)
            log_results.update(self.migrate_to_mainlands(island))
            log_results.update(
                population.add_migrants_to_buffer(
                    self.get_migrants_for(island, False),
                    conspecific_utility_dicts=self.conspecific_utilities_dicts,
                    problem=problem,
                    reset_non_elites=False,
                ),
            )
            self.mainland_populations[island] = population
        log_results["current_population"] = self.archipelago_populations[island].population_size
        return {"population_server/set_population": log_results}

    def get_replenishment_size(self) -> dict[MainlandID, int]:
        """Get the size of the population of the mainlands which needs to be replenished.

        Returns
        -------
        dict[MainlandID, int]
            The mapping from mainland IDs to the number of migrants needed in order to replenish its population.
        """
        return {idx: pop.get_replenishment_size() for idx, pop in self.mainland_populations.items()}
