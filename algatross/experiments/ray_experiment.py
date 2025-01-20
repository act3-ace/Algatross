"""Module for ray experiments."""

import logging
import logging.handlers

from collections import defaultdict
from multiprocessing import Queue
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import ray

from ray.tune import register_env, registry

from gymnasium.spaces import Space

from algatross.algorithms.genetic.mo_aim.archipelago.ray_archipelago import RayMOAIMArchipelago
from algatross.utils.debugging import log_debugging_ports
from algatross.utils.io import resolve_checkpoint_ancestors, resolve_checkpoint_epoch
from algatross.utils.loggers.console_logger import LogConsole, LogConsoleHandler
from algatross.utils.loggers.constants import DEBUGGING_PORTS_HANDLER, DEBUGGING_PORTS_LOGGING_LEVEL, MESSAGES_HANDLER
from algatross.utils.loggers.formatters import JSONLogStringFormatter
from algatross.utils.loggers.json_logger import JSONLogger
from algatross.utils.loggers.logger_container import LogHandlerContainer
from algatross.utils.loggers.shared_buffer import SharedBuffer
from algatross.utils.loggers.tensorboard import TensorboardLogger
from algatross.utils.merge_dicts import merge_dicts
from algatross.utils.parsers.yaml_loader import load_config
from algatross.utils.types import ConstructorData, IslandID, MainlandID, NumpyRandomSeed

if TYPE_CHECKING:
    from ray.rllib import MultiAgentEnv

    from algatross.utils.loggers.base_loggers import BaseHandler, BaseLogger


class RayExperiment:
    """
    An experiment class for training ray archipelagos.

    Parameters
    ----------
    config_file : str
        The file to load containing the configuration
    config : dict | None, optional
        The additional configuration to merge with the data in the config file, default is :data:`python:None`
    """

    default_loggers: list[type] = [JSONLogger, TensorboardLogger, LogConsole]  # noqa: RUF012
    """The default loggers to use by this experiment if None are specified."""
    context: ray._private.worker.RayContext
    """The ray context."""
    config_parents: list[Path]
    """The paths to parent configuration files."""
    config_patches: list[Path]
    """The paths to patch configuration files."""
    config_path: str
    """The path to the configuration file."""
    extra_evolve_args: dict
    """Extra keyword arguments to supply to the evolve step."""
    storage_path: Path
    """The path to the storage location for this experiments results."""
    result_queue: Queue
    """The queue for experiment results."""
    result_listener: logging.handlers.QueueListener
    """The listener for extracting results from the result queue."""
    epochs: int
    """The number of epochs to evolve the archipelago."""
    island_iterations: int
    """The number of iterations to evolve the islands per epoch."""
    mainland_iterations: int
    """The number of iterations to evolve the mainlands per epoch."""
    ckpt_interval_epochs: int
    """The number of epochs between which a checkpoint should be logged."""
    island_constructors: dict[IslandID, ConstructorData]
    """A dictionary of constructors for each island."""
    mainland_constructors: dict[MainlandID, ConstructorData]
    """A dictionary of constructors for each mainland."""
    archipelago: RayMOAIMArchipelago
    """The archipelago object."""
    logger: logging.Logger
    """The logger for this experiment."""

    def __init__(self, config_file: str, config: dict | None = None):
        self.config_parents: list[Path] = []
        self.config_patches: list[Path] = []
        self.config_path: str = config_file

        if config is None:
            config = load_config(config_file)

        log_level = config.get("log_level", logging.ERROR)
        self.extra_evolve_args = config.get("extra_evolve_args", {})

        ray_config = config.get("ray_config", {})
        ray_config["log_to_driver"] = ray_config.get("log_to_driver") or False
        worker_process_setup_hook = ray_config.get("runtime_env", {}).get("worker_process_setup_hook", None)

        if worker_process_setup_hook is None:
            # propagate logger setting to workers
            def worker_process_setup_hook():
                logger = logging.getLogger("ray")
                logger.setLevel(log_level)

        if config.get("debug"):

            def _custom_debug_hook():
                worker_process_setup_hook()
                log_debugging_ports()

            ray_config["runtime_env"] = ray_config.get("runtime_env", {}) | {"worker_process_setup_hook": _custom_debug_hook}

        else:
            ray_config["runtime_env"] = ray_config.get("runtime_env", {}) | {"worker_process_setup_hook": worker_process_setup_hook}

        self.context = ray.init(**ray_config).__enter__()

        worker_process_setup_hook()  # configure driver
        self.logger = logging.getLogger("ray")
        self.logger.info(f"\t Logging level set to {self.logger.level}!")

        self.setup_loggers(config=config, config_file=config_file, parallel_backend="ray")

        checkpoint_folder = None
        if config.get("checkpoint_folder", None):
            checkpoint_folder = Path(config["checkpoint_folder"])
            resume_epoch = -1 if config.get("resume_epoch", None) is None else config["resume_epoch"]
            checkpoint_folder, resume_epoch = resolve_checkpoint_epoch(checkpoint_folder, resume_epoch)
            config["resume_epoch"] = resume_epoch

            if not checkpoint_folder.exists():
                self.logger.warning("\tOption ``checkpoint_folder'' is set in config but no checkpoint found, ignoring!")
                checkpoint_folder = None
            else:
                self.config_parents, self.config_patches, config = resolve_checkpoint_ancestors(checkpoint_folder, loaded_config=config)

        config["from_checkpoint_folder"] = checkpoint_folder
        self.setup_actors(config=config)

    @staticmethod
    def get_constructors(
        island_configs: dict,
        env_spaces: dict[str, dict[str, Space]],
        storage_path: Path,
        seed: NumpyRandomSeed | None = None,
    ) -> dict:
        """
        Get the constructors for the islands.

        Parameters
        ----------
        island_configs : dict
            Dictionary of constructor data for the islands.
        env_spaces : dict[str, dict[str, Space]]
            The spaces from the environment
        storage_path : Path
            The path to store the experiment results
        seed : NumpyRandomSeed | None, optional
            The seed value to set for all random seeds, :data:`python:None`

        Returns
        -------
        dict
            Dictionary of island, problem, algorithm, and population constructors.
        """
        constructors = defaultdict(list)
        for isl_data in island_configs:
            space = env_spaces[isl_data["island_constructor"].config.get("env_name")]
            isl_data["problem_constructor"].config.update(space)
            if seed:
                isl_data["problem_constructor"].config["seed"] = isl_data["problem_constructor"].config.get("seed") or seed
                isl_data["algorithm_constructor"].config["seed"] = isl_data["algorithm_constructor"].config.get("seed") or seed
                isl_data["population_constructor"].config["seed"] = isl_data["population_constructor"].config.get("seed") or seed
            isl_data["population_constructor"].config["storage_path"] = storage_path
            constructors["constructors"].append(isl_data["island_constructor"])
            constructors["problem_constructors"].append(isl_data["problem_constructor"])
            constructors["algorithm_constructors"].append(isl_data["algorithm_constructor"])
            constructors["population_constructors"].append(isl_data["population_constructor"])
        return constructors

    def run_experiment(self):
        """Carry out one training aeon.

        A training aeon is a the completion of all training epochs. Each training epoch is the completion of all training iterations.
        """
        sub_epochs: int = self.archipelago.initial_epoch
        # initial ckpt
        self.archipelago.save_checkpoint(
            self.storage_path / f"epoch-{sub_epochs}",
            epoch=sub_epochs,
            config_path=self.config_path,
            config_patches=self.config_patches,
            config_parents=self.config_parents,
        )

        while sub_epochs < self.epochs:
            should_continue = True
            evolve_args = {
                "island_iterations": self.island_iterations,
                "mainland_iterations": self.mainland_iterations,
                "epochs": self.ckpt_interval_epochs,
                "storage_path": self.storage_path,
                **self.extra_evolve_args,
            }
            while should_continue:
                for result in self.archipelago.evolve(**evolve_args):
                    should_continue &= result

                    if not should_continue:
                        break

            sub_epochs += self.ckpt_interval_epochs

            # time to checkpoint
            self.archipelago.save_checkpoint(
                self.storage_path / f"epoch-{sub_epochs}",
                epoch=sub_epochs,
                config_path=self.config_path,
                config_patches=self.config_patches,
                config_parents=self.config_parents,
            )

        self.result_listener.stop()
        self.context.__exit__()

    def render_episodes(self, max_episodes: int = 5, render_islands: bool = False, render_mainlands: bool = False, **kwargs):
        """
        Render the episodes.

        Parameters
        ----------
        max_episodes : int, optional
            The maximum number of episodes to render, by default 5
        render_islands : bool, optional
            Whether to render the islands, by default :data:`python:False`
        render_mainlands : bool, optional
            Whether to render the mainlands, by default :data:`python:False`
        `**kwargs`
            Additional keyword arguments.
        """
        kwargs.setdefault("num_island_elites", 1)
        kwargs.setdefault("elites_per_island_team", 1)
        kwargs.setdefault("num_mainland_teams", 1)
        with self.context:
            storage_path = self.storage_path / "visualize" / f"epoch-{self.archipelago._initial_epoch}"  # noqa: SLF001
            storage_path.mkdir(parents=True, exist_ok=True)
            kwargs["storage_path"] = storage_path
            self.archipelago.render_episodes(
                max_episodes=max_episodes,
                render_islands=render_islands,
                render_mainlands=render_mainlands,
                **kwargs,
            )

    def setup_loggers(self, config: dict[str, Any], **kwargs):
        """Set up the loggers, queue, and listeners.

        Parameters
        ----------
        config : dict[str, Any]
            The experiment config dict.
        `**kwargs`
            Additional keyword arguments.
        """
        log_dir = config.get("log_dir") or "."
        experiment_name = config.get("experiment_name") or str(uuid4())[-8:]
        storage_path = Path(log_dir) / experiment_name
        self.logger.info(f"Storage path for this experiment is: {storage_path}")
        self.storage_path = storage_path

        extra_loggers: list[ConstructorData] = config.get("extra_loggers", [])
        for logger_constructor in extra_loggers:
            logger_constructor.config = merge_dicts(logger_constructor.config, kwargs)

        self.result_queue: Queue = Queue()

        result_loggers: list[logging.Logger | BaseLogger] = [
            *[
                JSONLogger(storage_path=self.storage_path / obj_cls, filter_regex=f"^{obj_cls}", **kwargs)
                for obj_cls in ["archipelago", "island", "mainland"]
            ],
            *[
                TensorboardLogger(storage_path=self.storage_path / obj_cls, filter_regex=f"^{obj_cls}", **kwargs)
                for obj_cls in ["archipelago", "island", "mainland"]
            ],
            *[logger_constructor.construct(storage_path=self.storage_path) for logger_constructor in extra_loggers],
        ]
        log_handlers: list[BaseHandler] = []
        result_handlers: list[BaseHandler] = [
            LogHandlerContainer(
                result_loggers,
                buffer=SharedBuffer(),
                level=logging.DEBUG if config.get("debug") else logging.INFO,
                strict_level=False,
            ),
        ]

        if config.get("rich_console", True):
            log_console = LogConsole(
                log_file=(self.storage_path / "results.json").resolve(),
                log_dir=self.storage_path,
                experiment_name=experiment_name,
                debug=config.get("debug"),
                dashboard_url=self.context.dashboard_url,
                session_dir=self.context.address_info["session_dir"],
                **kwargs,
            )

            port_handler = LogConsoleHandler(
                log_console.update_debug_ports,
                level=DEBUGGING_PORTS_LOGGING_LEVEL,
                name=DEBUGGING_PORTS_HANDLER,
                strict_level=True,
                formatter=JSONLogStringFormatter(),
            )
            message_handler = LogConsoleHandler(
                log_console.update_messages,
                level=logging.DEBUG if config.get("debug") else logging.INFO,
                name=MESSAGES_HANDLER,
                strict_level=False,
            )

            logging.getLogger("ray").addHandler(port_handler)
            logging.getLogger("ray").addHandler(message_handler)

            result_loggers.append(log_console)
            log_handlers.extend([port_handler, message_handler])

        self.result_listener = logging.handlers.QueueListener(self.result_queue, *result_handlers)
        self.result_listener.start()

    def setup_actors(self, config: dict[str, Any], **kwargs):
        """Set up the remote actors and wait for the archipelago to be ready.

        Parameters
        ----------
        config : dict[str, Any]
            The experiment config
        `**kwargs`
            Additional keyword arguments.
        """
        self.epochs = config["epochs"]
        self.island_iterations = config["island_iterations"]
        self.mainland_iterations = config["mainland_iterations"]
        self.ckpt_interval_epochs = config["ckpt_interval_epochs"]

        spaces = self.register_envs(config, **kwargs)

        self.island_constructors = self.get_constructors(
            config["islands"],
            env_spaces=spaces,
            storage_path=self.storage_path,
            seed=config["seed"],
        )
        self.mainland_constructors = self.get_constructors(
            config["mainlands"],
            env_spaces=spaces,
            storage_path=self.storage_path,
            seed=config["seed"],
        )

        archipelago = config["archipelago_constructor"]
        archipelago.config["topology"] = config["topology_constructor"].construct()
        archipelago.config["seed"] = archipelago.config.get("seed") or config["seed"]
        archipelago.config["island_constructors"] = self.island_constructors
        archipelago.config["mainland_constructors"] = self.mainland_constructors
        archipelago.config["ray_context"] = self.context
        archipelago.config["debug"] = config.get("debug")
        archipelago.config["result_queue"] = self.result_queue
        archipelago.config["from_checkpoint_folder"] = config["from_checkpoint_folder"]
        archipelago.config["storage_path"] = self.storage_path
        archipelago.config.update(self.extra_evolve_args)  # passed to islands init in kwargs

        self.archipelago: RayMOAIMArchipelago = archipelago.constructor(**archipelago.config)

        arch_ready = False
        while not arch_ready:
            arch_ready = self.archipelago.is_alive()

    @staticmethod
    def register_envs(config: dict[str, Any], get_spaces: bool = True, **kwargs) -> dict[str, dict[str, Space]]:
        """Register the environments and maybe get their spaces.

        Parameters
        ----------
        config : dict[str, Any]
            The experiment config containing an environment map
        get_spaces : bool, optional
            Whether to return the spaces for each environment in the map, :data:`python:True`
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        dict[str, dict[str, Space]]
            A dictionary of environment names mapping to a dict of their ``obs_space`` and ``act_space``
        """
        spaces = {}
        for env_name, env_constructor in config["environment_map"].items():
            register_env(env_name, env_constructor)
            if get_spaces:
                spaces[env_name] = RayExperiment.get_env_spaces(env_name, **kwargs)
        return spaces

    @staticmethod
    def get_env_spaces(env_name: str, close: bool = True, **kwargs) -> dict[str, Space]:
        """Get the observation and action spaces for the environment.

        Parameters
        ----------
        env_name : str
            Environment name in the global registry
        close : bool, optional
            Close the environment when done, :data:`python:True`
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        dict[str, Space]
            A dictionary with the ``obs_space`` and ``act_space``
        """
        env: MultiAgentEnv = registry._global_registry.get(registry.ENV_CREATOR, env_name)()  # noqa: SLF001
        spaces = {"obs_space": env.observation_space, "act_space": env.action_space}
        if close:
            env.close()
        return spaces
