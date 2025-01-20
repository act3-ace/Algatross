"""A module containing implementations of MO-AIM archipelago(s) for Ray parallelization backend."""

import logging
import logging.handlers
import multiprocessing

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np

import ray

from ray.util.queue import Queue

import torch

from algatross.actors.ray_actor import RayActor
from algatross.algorithms.genetic.mo_aim.archipelago.base import RemoteMOAIMArchipelago
from algatross.algorithms.genetic.mo_aim.islands.ray_islands import IslandServer
from algatross.algorithms.genetic.mo_aim.population import PopulationServer
from algatross.algorithms.genetic.mo_aim.topology import MOAIMTopology
from algatross.environments.common.rendering import make_movies
from algatross.environments.mpe.rendering import get_frame_functional as mpe_frame_func
from algatross.utils.io import load_checkpoint, save_checkpoint
from algatross.utils.random import get_torch_generator_from_numpy, resolve_seed
from algatross.utils.types import ConstructorData, ConstructorDataDict, IslandID, MainlandID, NumpyRandomSeed

logger = logging.getLogger("ray")


# TODO: use ray.put and remote object references to place/pull migrants to/from queue rather than pulling to local worker with `get`
# @ray.remote
class RayMOAIMArchipelago(RemoteMOAIMArchipelago, RayActor):
    """
    A MO-AIM archipelago in the Ray framework.

    Parameters
    ----------
    island_constructors : ConstructorDataDict
        The constructors for the islands.
    mainland_constructors : ConstructorDataDict
        The constructors for the mainlands.
    conspecific_utility_keys : list[str]
        Keys to use for conspecific utility
    topology : MOAIMTopology | None, optional
        The topology for the archipelago, if pre-constructed, default is :data:`python:None`
    seed : NumpyRandomSeed | None, optional
        The seed for randomness, default is :data:`python:None`
    n_workers : int | None, optional
        The number of workers to use, default is :data:`python:None`.
    log_queue : Queue | None, optional
        The queue for log messages, default is :data:`python:None`.
    result_queue : Queue | None, optional
        The queue for results, default is :data:`python:None`.
    from_checkpoint_folder : str | None, optional
        The folder to use if loading from a checkpoint
    `**kwargs`
        Additional keyword arguments
    """

    population_server: PopulationServer
    """The population server for the archipelago."""
    island_servers: dict[IslandID | MainlandID, IslandServer]
    """The mapping to the server for each island."""

    _actor_name: str = ""

    def __len__(self) -> int:
        """Return the number of islands and mainlands in the problem.

        Returns
        -------
        int
            The total number of islands and mainlands
        """
        return len(self.archipelago)

    def __init__(  # noqa: PLR0915
        self,
        island_constructors: ConstructorDataDict,
        mainland_constructors: ConstructorDataDict,
        conspecific_utility_keys: list[str],
        topology: MOAIMTopology | None = None,
        seed: NumpyRandomSeed | None = None,
        n_workers: int | None = None,
        log_queue: Queue | None = None,
        result_queue: Queue | None = None,
        from_checkpoint_folder: str | None = None,
        **kwargs,
    ) -> None:
        # Pass log_queue=None since this object should be local and therefore logs can be gathered directly
        RayActor.__init__(self, log_queue=None, result_queue=result_queue, **kwargs)
        self._initial_epoch = 0

        if from_checkpoint_folder:
            from_state = load_checkpoint(from_checkpoint_folder)
            self._initial_epoch = from_state["epoch"]
            self.log(f"Resume from epoch {self._initial_epoch}", logger_name="ray")
        else:
            from_state = {}

        self._island: dict[int, IslandServer] = {}
        self._mainland: dict[int, IslandServer] = {}
        self._archipelago: dict[int, IslandServer] = {}
        if from_state.get("topology", None) is not None:
            self._topology = from_state["topology"]
        elif topology is not None:
            self._topology = topology
        else:
            self._topology = MOAIMTopology()

        self._conspecific_utility_keys = conspecific_utility_keys

        self.log(f"Archipelago was given seed={seed}", logger_name="ray", level=logging.DEBUG)
        self._numpy_generator = resolve_seed(seed)  # type: ignore[arg-type]
        self._torch_generator = get_torch_generator_from_numpy(self._numpy_generator)[0]

        self.log(f"Connected to local client {self._context.gcs_address} using namespace={self._namespace}", logger_name="ray")

        islands_constructors = self.get_island_constructors(
            island_constructors=island_constructors["constructors"],
            problem_constructors=island_constructors["problem_constructors"],
            algorithm_constructors=island_constructors["algorithm_constructors"],
            population_constructors=island_constructors["population_constructors"],
            seed=self._numpy_generator,
            **kwargs,
        )
        mainlands_constructors = self.get_island_constructors(
            island_constructors=mainland_constructors["constructors"],
            problem_constructors=mainland_constructors["problem_constructors"],
            algorithm_constructors=mainland_constructors["algorithm_constructors"],
            population_constructors=mainland_constructors["population_constructors"],
            seed=self._numpy_generator,
            **kwargs,
        )
        arch_size = len(islands_constructors) + len(mainlands_constructors)
        n_workers = arch_size if n_workers is None else min(arch_size, n_workers)

        # evenly distribute load across available resources
        resources = ray.available_resources()
        n_workers = int(min(n_workers, resources.get("CPU", multiprocessing.cpu_count())))
        # cpus_per_worker = kwargs.get("cpus_per_worker", int((resources["CPU"] - 1) / n_workers))  # noqa: ERA001

        self.extra_init_info = kwargs

        self.log("Setting up population server", logger_name="ray")
        island_populations = {idx: isl["population"] for idx, isl in enumerate(islands_constructors)}
        mainland_populations = {idx: ml["population"] for idx, ml in enumerate(mainlands_constructors, len(island_populations))}
        self.population_server = PopulationServer.options(name="Population Server", namespace=self._namespace).remote(  # type: ignore[attr-defined]
            island_populations=island_populations,
            mainland_populations=mainland_populations,
            seed=self._numpy_generator.spawn(1).pop(),
            log_queue=log_queue,
            **kwargs,
        )
        if from_state.get("population_server", None) is not None:
            self.population_server.set_state.remote(from_state["population_server"])  # type: ignore[attr-defined]

        self.log("Setting up island servers", logger_name="ray")
        # TODO: Need to make this compatible with older numpy versions
        child_seed = iter(self._numpy_generator.spawn(len(islands_constructors) + len(mainlands_constructors)))
        islands_state = from_state.get("islands", None)
        mainlands_state = from_state.get("mainlands", None)

        islands: dict[IslandID, IslandServer] = {
            island_id: IslandServer.options(name=f"Island Server {island_id}", **self._child_actor_options).remote(  # type: ignore[attr-defined]
                island_constructor=isl["island"],
                algorithm_constructor=isl["algorithm"],
                problem_constructor=isl["problem"],
                population_server=self.population_server,
                island_id=island_id,
                log_queue=log_queue,
                seed=next(child_seed),
                **kwargs,
            )
            for island_id, isl in enumerate(islands_constructors)
        }
        mainlands: dict[MainlandID, IslandServer] = {
            island_id: IslandServer.options(name=f"Island Server {island_id}", **self._child_actor_options).remote(  # type: ignore[attr-defined]
                island_constructor=isl["island"],
                algorithm_constructor=isl["algorithm"],
                problem_constructor=isl["problem"],
                population_server=self.population_server,
                island_id=island_id,
                log_queue=log_queue,
                seed=next(child_seed),
                **kwargs,
            )
            for island_id, isl in enumerate(mainlands_constructors, len(islands))
        }

        # deal with checkpoint
        if islands_state is not None:
            n_from_islands = len(islands_state)
            n_to_islands = len(islands)
            assert (  # noqa: S101
                n_from_islands == n_to_islands
            ), f"Checkpoint mis-match of islands: ckpt had {n_from_islands}, need {n_to_islands}"
            for ix, island_id in enumerate(islands):
                assert islands_state[ix]["island_id"] == island_id, "Mis-matching islands!"  # noqa: S101
                islands[island_id].set_state.remote(islands_state[ix])  # type: ignore[attr-defined]

        if mainlands_state is not None:
            n_from_mainlands = len(mainlands_state)
            n_to_mainlands = len(mainlands)
            assert (  # noqa: S101
                n_from_mainlands == n_to_mainlands
            ), f"Checkpoint mis-match of mainlands: ckpt had {n_from_mainlands}, need {n_to_mainlands}"
            for ix, mainland_id in enumerate(mainlands):
                assert mainlands_state[ix]["island_id"] == mainland_id, "Mis-matching mainlands!"  # noqa: S101
                mainlands[mainland_id].set_state.remote(mainlands_state[ix])  # type: ignore[attr-defined]

        self._island = islands
        self._mainland = mainlands
        self._archipelago.update(self._island)
        self._archipelago.update(self._mainland)

        if from_state.get("population_server", None) is None:
            # run setup() and setup_qd() on freshly created island populations
            unfinished = [
                self.population_server.set_conspecific_data.remote(isl_id, problem=ray.get(isl_server.get_island.remote()).problem)  # type: ignore[attr-defined]
                for isl_id, isl_server in self._island.items()
            ]
            while unfinished:
                finished, unfinished = ray.wait(unfinished)
                ray.get(finished)

        self.island_servers = {**islands, **mainlands}
        self._init_topology()

        if islands_state is None:
            # run warmup training on islands
            self._init_island_pops(**kwargs)

        # must always refresh mainlands pop from island agents
        self._init_mainland_pops(**kwargs)
        self._archipelago_epoch = 0
        self.log("Setup finished", logger_name="ray")

    @staticmethod
    def get_island_constructors(
        island_constructors: Sequence[ConstructorData],
        problem_constructors: Sequence[ConstructorData],
        algorithm_constructors: Sequence[ConstructorData],
        population_constructors: Sequence[ConstructorData],
        seed: np.random.Generator,
        **kwargs,
    ) -> list[dict[Literal["island", "algorithm", "problem", "population"], ConstructorData]]:
        """Get the constructors for the islands and their populations.

        Parameters
        ----------
        island_constructors : Sequence[ConstructorData]
            Constructors which build the islands
        problem_constructors : Sequence[ConstructorData],
            Constructors which build the problems
        algorithm_constructors : Sequence[ConstructorData],
            Constructors which build the algorithms
        population_constructors : Sequence[ConstructorData],
            Constructors which build the populations
        seed : np.random.Generator
            A numpy random number generator
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        list[dict[Literal["island", "algorithm", "problem", "population"], ConstructorData]]
            A list of dictionaries, one for each island, containing the constructors for the islands' populations and the islands
            themselves.
        """
        constructors = []
        for isl_c, prb_c, alg_c, pop_c in zip(
            island_constructors,
            problem_constructors,
            algorithm_constructors,
            population_constructors,
            strict=True,
        ):
            seeds = seed.spawn(4)

            prb_kwargs = kwargs | prb_c.config
            prb_kwargs.update({"seed": np.random.default_rng(seed=seeds.pop())})

            alg_kwargs = kwargs | alg_c.config
            alg_kwargs.update({"seed": np.random.default_rng(seed=seeds.pop())})

            pop_kwargs = kwargs | pop_c.config
            pop_kwargs.update({"seed": np.random.default_rng(seed=seeds.pop()), "training_agents": prb_kwargs["training_agents"]})

            isl_kwargs = kwargs | isl_c.config
            isl_kwargs.update({"seed": np.random.default_rng(seed=seeds.pop())})

            constructs: dict[Literal["island", "algorithm", "problem", "population"], ConstructorData] = {
                "island": ConstructorData(constructor=isl_c.constructor, config=isl_kwargs),
                "algorithm": ConstructorData(constructor=alg_c.constructor, config=alg_kwargs),
                "problem": ConstructorData(constructor=prb_c.constructor, config=prb_kwargs),
                "population": ConstructorData(constructor=pop_c.constructor, config=pop_kwargs),
            }
            constructors.append(constructs)
        return constructors

    def _init_topology(self):
        """
        _init_topology fully initializes the topology by pushing back all Islands and Mainlands at once.

        This is in contrast to a piece-meal method where each island and mainland is pushed back individually.
        """
        self.log("Setting up topology", logger_name="ray")
        self.topology.set_archipelago(list(self._island), list(self._mainland))
        ray.get(self.population_server.set_topology.remote(self.topology.get_softmax_fn()))

    def _init_island_pops(  # type: ignore[override]
        self,
        warmup_generations: int = 10,
        warmup_iterations: int = 1,
        **kwargs,
    ):
        """Run warmup training on the islands to initialize a population up to a certain size.

        Parameters
        ----------
        warmup_generations : int, optional
            The  number of QD generations to run on the island per warmup iteration. Defaults to 10
        warmup_iterations : int, optional
            The number of warmup iterations (epochs) to run on the island. Defaults to 1
        `**kwargs`
            Additional keyword arguments.
        """
        if warmup_iterations == 0:
            return

        self.log("Initializing island populations", logger_name="ray")

        current_epoch = dict.fromkeys(self.island, 0)
        remaining_this_epoch = []
        kwargs = dict(kwargs.items())
        kwargs["visualize"] = False

        # iterate through the results until all futures have returned
        unfinished = [
            island.evolve.remote(warmup_generations, **kwargs)  # type: ignore[attr-defined]
            for island_id, island in self.island_servers.items()
            if island_id in self.island
        ]

        # iterate through the results until all futures have returned
        while unfinished:
            finished, unfinished = ray.wait(unfinished)
            for island_id, train_result in ray.get(finished):
                self.log_result(train_result)

                if current_epoch[island_id] < warmup_iterations:
                    # add the island back into the queue
                    remaining_this_epoch.append(self.island_servers[island_id])

                current_epoch[island_id] += 1

                if min(current_epoch.values()) < warmup_iterations and remaining_this_epoch:
                    # submit another task to the actor pool
                    isl = remaining_this_epoch.pop(self._numpy_generator.integers(len(remaining_this_epoch)))
                    unfinished.append(isl.evolve.remote(warmup_generations, **kwargs))  # type: ignore[attr-defined]

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
        self.log("Initializing mainland populations", logger_name="ray")
        ray.get(self.population_server.initialize_mainlands.remote(self.topology.get_softmax_fn()))

    @property
    def island(self) -> dict[IslandID, ray.ObjectRef]:  # noqa: D102
        return super().island

    @property
    def mainland(self) -> dict[MainlandID, ray.ObjectRef]:  # noqa: D102
        return super().mainland

    @property
    def archipelago(self) -> dict[IslandID | MainlandID, ray.ObjectRef]:  # noqa: D102
        return super().archipelago

    @property
    def initial_epoch(self) -> int:  # noqa: D102
        return self._initial_epoch

    def evolve(  # type: ignore[override] # noqa: D102
        self,
        island_iterations: int = 1,
        mainland_iterations: int = 1,
        epochs: int = 1,
        ckpt_interval_epochs: int = 25,  # noqa: ARG002
        softmax_reset_interval: int | None = None,
        **kwargs,
    ):
        self.log("Evolving archipelago...", logger_name="ray")
        current_epoch = dict.fromkeys(self.archipelago, 0)
        remaining_this_epoch = []
        conspecific_utility = {}
        unfinished = [
            island.evolve.remote(  # type: ignore[attr-defined]
                island_iterations if island_id in self.island else mainland_iterations,
                **kwargs,
            )
            for island_id, island in self.island_servers.items()
        ]

        # iterate through the results until all futures have returned
        while unfinished:
            finished, unfinished = ray.wait(unfinished)
            for island_id, train_result in ray.get(finished):
                self.log_result(train_result)

                if current_epoch[island_id] < epochs:
                    # add the island back into the queue
                    remaining_this_epoch.append((island_id, self.island_servers[island_id]))

                if island_id in self.mainland:
                    conspecific_utility[island_id] = {
                        k.split("/")[1]: v
                        for k, v in train_result[f"mainland/{island_id}"]["algorithm"].get("conspecific_utility", {}).items()
                    }
                    if softmax_reset_interval and current_epoch[island_id] % softmax_reset_interval == 0:
                        self.topology.reset_softmax_for(island_id)
                    self.log_result({
                        f"archipelago/0/mainland/{island_id}": {
                            "epoch": self._archipelago_epoch * epochs + current_epoch[island_id],
                            **self.train_topology(conspecific_utility, island_id),
                        },
                    })
                    ray.get(self.population_server.set_topology.remote(self.topology.get_softmax_fn()))  # type: ignore[attr-defined]

                current_epoch[island_id] += 1

                if min(current_epoch.values()) < epochs and remaining_this_epoch:
                    # submit another task to the actor pool
                    next_idx, isl = remaining_this_epoch.pop(self._numpy_generator.integers(len(remaining_this_epoch)))
                    n = island_iterations if next_idx in self.island else mainland_iterations
                    unfinished.append(isl.evolve.remote(n, **kwargs))  # type: ignore[attr-defined]
                yield True

            if len(unfinished) == 0:
                yield False

        self._archipelago_epoch += 1

    def save_checkpoint(
        self,
        storage_path: str,
        epoch: int,
        config_path: str,
        config_patches: list | None = None,
        config_parents: list | None = None,
    ) -> None:
        """Save a checkpoint of the current archipelago as folder.

        Parameters
        ----------
        storage_path : str
            Folder path to save serialized objects.
        epoch : int
            Epoch associated with the checkpoint.
        config_path : str
            Path to configuration file associated with this run.
        config_patches : list[str]
            Path to patch files which modify island configuration data (optional).
        config_parents : list[str]
            The parent files from potentially many checkpoints (optional).
        """
        config_patches = config_patches or []
        config_parents = config_parents or []
        save_checkpoint(
            storage_path=storage_path,
            config_path=config_path,
            config_patches=config_patches,
            config_parents=config_parents,
            population_server=self.population_server,
            epoch=epoch,
            topology=self.topology,
            islands=[isl_server for _, isl_server in self._island.items()],  # includes UDA, UDP
            mainlands=[isl_server for _, isl_server in self._mainland.items()],  # include UDA, UDP
        )

    def render_episodes(
        self,
        storage_path: str | Path,
        max_episodes: int = 5,
        render_islands: bool = False,
        render_mainlands: bool = False,
        **kwargs,
    ):
        """
        Render a certain number of episodes from mainlands and islands.

        Parameters
        ----------
        storage_path : str | Path
            The location to store the rendered episodes.
        max_episodes : int, optional
            The maximum number of episodes to render, by default 5
        render_islands : bool, optional
            Whether to render the island episodes, by default :data:`python:False`
        render_mainlands : bool, optional
            Whether to render the mainland episodes, by default :data:`python:False`
        `**kwargs`
            Additional keyword arguments.

        Raises
        ------
        ValueError
            If the returned island or mainland id is not in the archipelago.
        """
        if isinstance(storage_path, str):
            storage_path = Path(storage_path)

        work: dict[IslandID, IslandServer] = {}
        if render_islands:
            work.update(self._island)
        if render_mainlands:
            work.update(self._mainland)

        kwargs |= {
            "visualize": True,
            "rollout_config": {
                "batch_mode": "complete_episodes",
                "batch_size": 1_000 * max_episodes,  # only needed to bypass rollout_length check in runner
            },
        }

        unfinished = [
            island_server.evaluate.remote(**kwargs)  # type: ignore[attr-defined]
            for island_id, island_server in work.items()
        ]
        while unfinished:
            finished, unfinished = ray.wait(unfinished)
            for island_id, rollout_result in ray.get(finished):
                # assert island_id in self._island, "Bad island!"
                if island_id in self._island:
                    island_name = "island"
                elif island_id in self._mainland:
                    island_name = "mainland"
                else:
                    msg = f"Got a weird island ID: {island_id}. Archipelago changed?"
                    raise ValueError(msg)

                iter_rendered_episodes_map = rollout_result[f"{island_name}/{island_id}"]["iter_rendered_episode_map"]

                # iter 0
                # TODO: do remotely
                team_to_episodes = iter_rendered_episodes_map[0]
                for team in team_to_episodes:
                    episodes = team_to_episodes[team]
                    self.log(
                        make_movies(
                            episodes,
                            storage_path / f"island-{island_id}" / f"team-{team}",
                            get_frame_functional=mpe_frame_func,
                            fps=15,
                        ),
                    )

    @staticmethod
    def _wait(futures: list[ray.ObjectRef]):
        unfinished = futures
        while unfinished:
            finished, unfinished = ray.wait(unfinished)
            ray.get(finished)

    @staticmethod
    def _gather(futures: list[ray.ObjectRef]) -> list[Any]:
        unfinished = futures
        results = [None] * len(futures)
        while unfinished:
            finished, unfinished = ray.wait(unfinished)
            for f in finished:
                results[futures.index(f)] = ray.get(f)
        return results

    def train_topology(  # noqa: D102
        self,
        results: dict[IslandID | MainlandID, Any],
        mainlands: list[MainlandID] | MainlandID | None = None,
    ) -> dict[str, Any]:
        if mainlands is None:
            mainlands = list(self.mainland)
        if not isinstance(mainlands, list):
            mainlands = [mainlands]
        util = torch.stack([self.stack_conspecific_data(results[ml]) for ml in sorted(mainlands)])
        return self.topology.optimize_softmax(util, mainlands)

    def is_alive(self) -> bool:  # type: ignore[override]
        """Return whether the actor is alive.

        Makes sure all the island servers and population server are alive and ready.

        Returns
        -------
        bool
            If the actor is alive
        """
        unfinished = [island_server.is_alive.remote() for island_server in self.island_servers.values()]  # type: ignore[attr-defined]
        unfinished += [self.population_server.is_alive.remote()]  # type: ignore[attr-defined]
        while unfinished:
            _, unfinished = ray.wait(unfinished)
        return True

    def migrate(self, *args, **kwargs):  # noqa: D102
        pass
