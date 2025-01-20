"""Tests for the determinism of an experiment."""

import dataclasses
import itertools

from typing import TYPE_CHECKING

import numpy as np

import ray

from treelib import Tree

from algatross.experiments.ray_experiment import RayExperiment
from algatross.utils.parsers.yaml_loader import load_config
from algatross.utils.random import get_generator_entropy, get_generator_integer

if TYPE_CHECKING:
    from algatross.algorithms.genetic.mo_aim.islands.base import RemoteUDI
    from algatross.algorithms.genetic.mo_aim.population import MOAIMPopulation
    from algatross.algorithms.genetic.mo_aim.problem import UDP


@dataclasses.dataclass
class MOAIMClassRandomState:  # noqa: D101
    generator: np.random.Generator

    @property
    def entropy_str(self):  # noqa: D102
        entropy = 0 if self.generator is None else get_generator_entropy(self.generator)
        return f"{entropy}"

    @property
    def integer_str(self):  # noqa: D102
        integer = 0 if self.generator is None else get_generator_integer(self.generator)
        return f"{integer}"

    def __str__(self):  # noqa: D105
        return f"entropy={self.entropy_str}"


class TestDeterminismExperiment(RayExperiment):
    """An experiment for testing deterministic attributes of MO-AIM classes.

    When writing the test config file:
    - archieplago_constructor.config.warmup_generations MUST be 0
    - archieplago_constructor.config.warmup_iterations MUST be 0

    Parameters
    ----------
        RayExperiment (_type_): _description_
    """

    def __init__(self, config_file: str, test_dir: str, _: dict | None = None):
        config = load_config(config_file)
        config["log_dir"] = test_dir
        super().__init__(config_file, config=config)

    def run_experiment(self) -> Tree:
        """
        Get the relevant states of numpy generators to conduct the test.

        Returns
        -------
        Tree
            python Tree object representing the random states
        """
        archipelago = self.archipelago
        island_ids = archipelago.island
        mainland_ids = archipelago.mainland
        island_servers = archipelago.island_servers

        entropy_tree = Tree()
        entropy_tree.create_node(
            tag="archipelago",
            identifier="archipelago",
            data=MOAIMClassRandomState(
                generator=archipelago._numpy_generator,  # noqa: SLF001
            ),
        )

        entropy_tree.create_node(
            tag="population_server",
            identifier="population_server",
            parent="archipelago",
            data=MOAIMClassRandomState(generator=ray.get(archipelago.population_server.get_random_generators.remote())[0]),  # type: ignore[attr-defined]
        )

        for idx in itertools.chain(island_ids, mainland_ids):
            island_type = "island" if idx in island_ids else "mainland"

            problem: UDP = ray.get(island_servers[idx].get_problem.remote())  # type: ignore[assignment, attr-defined]
            runner = problem.runner_handle  # type: ignore[attr-defined]
            island: RemoteUDI = ray.get(island_servers[idx].get_island.remote())  # type: ignore[assignment, attr-defined]
            algorithm = island.algorithm
            population: MOAIMPopulation = ray.get(archipelago.population_server.get_archipelago_populations.remote())[idx]  # type: ignore[attr-defined]

            server_name = f"{island_type}_server_{idx}"
            try:
                server_g = island_servers[idx]._numpy_generator  # noqa: SLF001
            except AttributeError:
                server_g = None

            entropy_tree.create_node(
                tag=server_name,
                identifier=server_name,
                parent="archipelago",
                data=MOAIMClassRandomState(generator=server_g),
            )

            island_name = f"{island_type}_{idx}"
            try:
                island_g = island._numpy_generator  # noqa: SLF001
            except AttributeError:
                island_g = None

            entropy_tree.create_node(
                tag=island_name,
                identifier=island_name,
                parent=server_name,
                data=MOAIMClassRandomState(generator=island_g),
            )

            udp_name = f"udp_{idx}"
            entropy_tree.create_node(
                tag=udp_name,
                identifier=udp_name,
                parent=island_name,
                data=MOAIMClassRandomState(
                    generator=problem._numpy_generator,  # noqa: SLF001
                ),
            )
            runner_name = f"runner_{idx}"
            entropy_tree.create_node(
                tag=runner_name,
                identifier=runner_name,
                parent=udp_name,
                data=MOAIMClassRandomState(
                    generator=runner._numpy_generator,  # noqa: SLF001
                ),
            )

            algorithm_name = f"algorithm_{idx}"
            entropy_tree.create_node(
                tag=algorithm_name,
                identifier=algorithm_name,
                parent=island_name,
                data=MOAIMClassRandomState(
                    generator=algorithm._numpy_generator,  # noqa: SLF001
                ),
            )

            population_name = f"population_{idx}"
            entropy_tree.create_node(
                tag=population_name,
                identifier=population_name,
                parent=server_name,
                data=MOAIMClassRandomState(
                    generator=population._numpy_generator,  # noqa: SLF001
                ),
            )

            agents_list_name = f"agents_{idx}"
            entropy_tree.create_node(
                tag=agents_list_name,
                identifier=agents_list_name,
                parent=server_name,
                data=MOAIMClassRandomState(generator=None),
            )
            for agent_id, agent in ray.get(island_servers[idx].get_agents.remote()).items():  # type: ignore[attr-defined]
                agent_name = f"{server_name}/{agent_id}"
                entropy_tree.create_node(
                    tag=agent_name,
                    identifier=agent_name,
                    parent=agents_list_name,
                    data=MOAIMClassRandomState(generator=agent.np_random),
                )

        self.result_listener.stop()
        self.context.__exit__()

        return entropy_tree
