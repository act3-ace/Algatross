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
import random

from algatross.algorithms.genetic.mo_aim.topology import MOAIMTopology
from algatross.utils.io import save_checkpoint, load_checkpoint
from algatross.experiments.functional import pre_init, init_constructors
from algatross.algorithms.genetic.mo_aim.archipelago.functional import init_archipelago, train_topology


def append_and_save(df, new_data, save_path):
    new_df = pd.DataFrame(new_data)
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_pickle(save_path)
    return df


def test_mainland_iteration():
    # parameter
    num_islands = 1
    num_mainlands = 1
    island_iters = 1
    mainland_iters = 1
    epochs = 3
    n_workers = 1
    save_freq = 1
    ckpt_interval_epochs = 1
    softmax_reset_interval = 5

    # checkpoint_folder = Path(f"scripts/debug/cleanrl/debug_data_out-iter=1-mainlands=1-islands=1/epoch-2")
    checkpoint_folder = None
    if checkpoint_folder:
        config_file = checkpoint_folder / "test_algatross.yml"
        from_state = load_checkpoint(checkpoint_folder)
        data_direction = "data_in-out"
    else:
        config_file = "config/simple_tag/test_algatross.yml"  # noqa: PLR2004
        from_state = {}
        data_direction = "data_out"

    print(f"Using configuration: {config_file}")

    save_folder = Path(f"scripts/debug/cleanrl/debug_{data_direction}-iter={island_iters}-mainlands={num_mainlands}-islands={num_islands}/")
    if not save_folder.exists():
        save_folder.mkdir(parents=True)

    df = pd.DataFrame()
    df_path = save_folder / "df.pkl"

    config = load_config(config_file)
    seed = config["seed"]

    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    # the mother of all random generators
    _numpy_generator, _torch_generator = get_generators(seed=seed)
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
        _island, _mainland, _archipelago = init_archipelago(
            islands_constructors,
            mainlands_constructors,
            islands_state=from_state.get("islands", None),
            mainlands_state=from_state.get("mainlands", None)
        )

        island_populations = {idx: isl["population"] for idx, isl in enumerate(islands_constructors)}
        mainland_populations = {idx: ml["population"] for idx, ml in enumerate(mainlands_constructors, len(island_populations))}

        ## database server
        population_server = PopulationServer.options(name="Population Server", namespace=_namespace).remote(  # type: ignore[attr-defined]
            island_populations=island_populations,
            mainland_populations=mainland_populations,
            seed=seed,
            result_queue=None,
        )
        if checkpoint_folder:
            population_server.set_state.remote(from_state.get("population_server", {}))


        ## special step for mainland. It should happen after init_archipelago
        if checkpoint_folder:
            topology = from_state["topology"]
            sub_epochs = from_state["epoch"]
            print(f"\tResume from epoch {sub_epochs}")
        else:
            topology = MOAIMTopology()
            sub_epochs = 0

        topology.set_archipelago(list(_island), list(_mainland))
        ray.get(population_server.initialize_mainlands.remote(topology.get_softmax_fn()))

        # checkpoint before training
        save_checkpoint(
            storage_path=save_folder / f"epoch-{sub_epochs}",
            config_file=config_file,
            population_server=population_server,
            epoch=sub_epochs,
            topology=topology,
            islands=[ray.get(isl_ref) for _, isl_ref in _island.items()],  # includes UDA
            mainlands=[ray.get(isl_ref) for _, isl_ref in _mainland.items()],  # include UDA
        )

        while sub_epochs < epochs:
            ##############
            # evolve
            current_epoch = dict.fromkeys(_archipelago, 0)
            remaining_this_epoch = []
            conspecific_utility = {}

            # torch.manual_seed(seed + sub_epochs)
            # torch.cuda.manual_seed(seed + sub_epochs)
            # np.random.seed(seed + sub_epochs)
            # random.seed(seed + sub_epochs)
            print("=" * 25)
            print("=" * 12 + f"\tepoch {sub_epochs} -> epoch {sub_epochs + ckpt_interval_epochs}")
            print("=" * 25)

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
                if current_epoch[island_id] < ckpt_interval_epochs:
                    # add the island back into the queue
                    remaining_this_epoch.append((island_id, island))

                if island_id in _island:
                    uri_prefix = "island"
                    _island[island_id] = island

                elif island_id in _mainland:
                    uri_prefix = "mainland"
                    _mainland[island_id] = island
                    conspecific_utility[island_id] = train_result[f"mainland/{island_id}"]["algorithm"]["conspecific_utility"]

                    if softmax_reset_interval and current_epoch[island_id] % softmax_reset_interval == 0:
                        topology.reset_softmax_for(island_id)

                    print(
                        {
                            f"archipelago/0/mainland/{island_id}": {
                                "epoch": current_epoch[island_id],
                                **train_topology(
                                    topology, conspecific_utility, island_id, _island)
                            }
                        }
                    )
                    ray.get(population_server.set_topology.remote(topology.get_softmax_fn()))  # type: ignore[attr-defined]



                result = ray.get(population_server.set_population.remote(island_id, new_pop, current_epoch[island_id]))  # type: ignore[arg-type, attr-defined]

                db_dict = train_result[f"{uri_prefix}/{island_id}"]["algorithm"]
                db_dict.update({
                    "epoch": current_epoch[island_id] + sub_epochs,  # offset
                    "index": f"{uri_prefix}/{island_id}/epoch{current_epoch[island_id]}",
                    "island": island_id,
                })
                data.append(db_dict)

                if len(data) % save_freq == 0:
                    df = append_and_save(df, data, df_path)
                    data = []

                for k, v in train_result[f"{uri_prefix}/{island_id}"]["algorithm"].items():
                    if "loss" in k or "fitness" in k:
                        print(k, v[0])

                current_epoch[island_id] += 1

                if min(current_epoch.values()) < ckpt_interval_epochs and remaining_this_epoch:
                    # submit another task to the actor pool
                    next_idx, isl = remaining_this_epoch.pop(_numpy_generator.integers(len(remaining_this_epoch)))
                    n = island_iters if isl in set(_island.values()) else mainland_iters
                    island_servers.submit(evolve_island, (isl, next_idx, get_next_population(next_idx), n, current_epoch[next_idx]))

            # save leftovers
            if len(data):
                df = append_and_save(df, data, df_path)
                data = []

            sub_epochs += ckpt_interval_epochs

            save_checkpoint(
                storage_path=save_folder / f"epoch-{sub_epochs}",
                config_file=config_file,
                population_server=population_server,
                epoch=sub_epochs,
                topology=topology,
                islands=[ray.get(isl_ref) for _, isl_ref in _island.items()],  # includes UDA
                mainlands=[ray.get(isl_ref) for _, isl_ref in _mainland.items()],  # include UDA
            )


if __name__ == "__main__":
    test_mainland_iteration()
