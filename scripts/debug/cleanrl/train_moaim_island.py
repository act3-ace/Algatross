import sys
import ray

from collections import defaultdict
from pathlib import Path

from algatross.utils.parsers.yaml_loader import load_config
from algatross.utils.types import ConstructorData, ConstructorDataDict, IslandID, MainlandID, NumpyRandomSeed, OptimizationTypeEnum
from algatross.algorithms.genetic.mo_aim.islands.ray_islands import IslandServer
from algatross.algorithms.genetic.mo_aim.population import PopulationServer
from algatross.utils.random import get_generators
from ray.util.queue import Queue
from ray.util import ActorPool

import torch
import numpy as np
import pandas as pd

from algatross.algorithms.genetic.mo_aim.topology import MOAIMTopology
from algatross.experiments.functional import pre_init, init_constructors
from algatross.algorithms.genetic.mo_aim.archipelago.functional import init_archipelago, train_topology


def append_and_save(df, new_data, save_path):
    new_df = pd.DataFrame(new_data)
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_pickle(save_path)
    return df


def test_island_iteration():
    config_file = "config/simple_tag/test_algatross.yml"  # noqa: PLR2004
    print(f"Using configuration: {config_file}")

    # parameter
    num_islands = 12
    num_mainlands = 0
    island_iters = 25
    mainland_iters = 1
    epochs = 100
    n_workers = 20
    save_freq = 25

    save_folder = Path("scripts/debug/cleanrl/data_out-iter=25-islands=12/")
    if not save_folder.exists():
        save_folder.mkdir(parents=True)

    df = pd.DataFrame()
    df_path = save_folder / "df.pkl"

    config = load_config(config_file)
    seed = config["seed"]
    _numpy_generator, _torch_generator = get_generators(seed=seed, seed_global=True)
    context, _, n_workers = pre_init(config, n_workers)

    with context:
        # archipelago init
        islands_constructors, mainlands_constructors = init_constructors(config, num_islands, num_mainlands)

        arch_size = len(islands_constructors) + len(mainlands_constructors)
        n_workers = arch_size if n_workers is None else min(arch_size, n_workers)

        ## create island servers
        _namespace = ray.get_runtime_context().namespace
        archipelago = config["archipelago_constructor"].config
        island_servers = ActorPool(
            [
                IslandServer.options(name=f"Island Server {idx}", namespace=_namespace, max_concurrency=1).remote(  # type: ignore[attr-defined]
                    result_queue=None,
                    env_constructor_data=archipelago["env_constructor_data"],
                    seed=seed,
                )
                for idx in range(n_workers)
            ],
        )

        ## create island objects
        _island, _mainland, _archipelago = init_archipelago(islands_constructors, mainlands_constructors)

        island_populations = {idx: isl["population"] for idx, isl in enumerate(islands_constructors)}
        mainland_populations = {idx: ml["population"] for idx, ml in enumerate(mainlands_constructors, len(island_populations))}

        ## database server
        population_server = PopulationServer.options(name="Population Server", namespace=_namespace).remote(  # type: ignore[attr-defined]
            island_populations=island_populations,
            mainland_populations=mainland_populations,
            seed=seed,
            result_queue=None,
        )

        ##############
        # evolve
        current_epoch = dict.fromkeys(_archipelago, 0)
        remaining_this_epoch = []

        def evolve_island(actor, value):
            return actor.evolve.remote(*value)

        def get_next_population(island_id: IslandID) -> ray.ObjectRef:
            return population_server.get_population_to_evolve.remote(island_id)

        ## submit a batch of results to gather from the workers and get the result generator
        tasks = [
            (
                isl,
                next_idx,
                get_next_population(next_idx),
                island_iters if isl in set(_island.values()) else mainland_iters,
                current_epoch[next_idx],
            )
            for next_idx, isl in _archipelago.items()
        ]

        data = []

        for island, island_id, new_pop, train_result in island_servers.map_unordered(evolve_island, tasks):
            print("island id", island_id)
            _archipelago[island_id] = island
            if current_epoch[island_id] < epochs:
                # add the island back into the queue
                remaining_this_epoch.append((island_id, island))

            if island_id in _island:
                _island[island_id] = island

            result = ray.get(population_server.set_population.remote(island_id, new_pop, current_epoch[island_id], problem=island.problem))  # type: ignore[arg-type, attr-defined]

            db_dict = train_result[f"island/{island_id}"]["algorithm"]
            db_dict.update(
                {
                    "epoch": current_epoch[island_id],
                    "index": f"island/{island_id}/epoch{current_epoch[island_id]}",
                    "island": island_id,
                }
            )
            data.append(db_dict)

            if len(data) % save_freq == 0:
                df = append_and_save(df, data, df_path)
                data = []

            for k, v in train_result[f"island/{island_id}"]["algorithm"].items():
                if "loss" in k or "fitness" in k:
                    print(k, v[0])

            current_epoch[island_id] += 1

            if min(current_epoch.values()) < epochs and remaining_this_epoch:
                # submit another task to the actor pool
                next_idx, isl = remaining_this_epoch.pop(_numpy_generator.integers(len(remaining_this_epoch)))
                n = island_iters if isl in set(_island.values()) else mainland_iters
                island_servers.submit(evolve_island, (isl, next_idx, get_next_population(next_idx), n, current_epoch[next_idx]))

        # save leftovers
        if len(data):
            df = append_and_save(df, data, df_path)
            data = []


if __name__ == "__main__":
    test_island_iteration()
