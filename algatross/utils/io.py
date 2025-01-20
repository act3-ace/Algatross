"""Input/output utilites."""

import logging
import re
import shutil

from collections import deque
from collections.abc import Callable, MutableMapping
from itertools import chain
from pathlib import Path
from reprlib import repr as repr_
from sys import getsizeof
from typing import Any

import ray

from ray import cloudpickle

from torch import nn

from algatross.actors.ray_actor import RayActor
from algatross.utils.merge_dicts import apply_xc_config
from algatross.utils.parsers.yaml_loader import load_config


def save_cloudpickle(o: Any, file_path: str | Path):  # noqa: ANN401
    """
    Save a file with cloudpickle.

    Parameters
    ----------
    o : Any
        The object to pickle.
    file_path : str | Path
        The path at which to pickle the object.
    """
    with Path(file_path).open("wb+") as f:
        cloudpickle.dump(o, f)


def load_cloudpickle(file_path: str | Path) -> Any:  # noqa: ANN401
    """Load a cloudpickled file.

    Parameters
    ----------
    file_path : str | Path
        The path to load.

    Returns
    -------
    Any
        The loaded object.
    """
    with Path(file_path).open("rb+") as f:
        return cloudpickle.load(f)


def save_checkpoint(
    storage_path: Path | str,
    config_path: Path | str,
    population_server: RayActor,
    epoch: int,
    topology: nn.Module,
    islands: list[RayActor],
    mainlands: list[RayActor],
    config_patches: list[Path | str] | None = None,
    config_parents: list[Path | str] | None = None,
):
    """
    Save a checkpoint to a path.

    Parameters
    ----------
    storage_path : Path | str
        The path to store the checkpoint.
    config_path : Path | str
        The path to the configuration file.
    population_server : algatross.actors.ray_actor.RayActor
        The population server actor.
    epoch : int
        The current epoch.
    topology : torch.nn.Module
        The archipelago topology.
    islands : list[algatross.actors.ray_actor.RayActor]
        The list of island actors.
    mainlands : list[algatross.actors.ray_actor.RayActor]
        The list of mainland actors.
    config_patches : list[Path | str] | None, optional
        The patches to apply to the config.
    config_parents : list[Path | str] | None, optional
        The parent configuration files.
    """
    config_patches = config_patches or []
    config_parents = config_parents or []
    logger = logging.getLogger("ray")

    storage_path = Path(storage_path)
    if not storage_path.exists():
        storage_path.mkdir(parents=True)

    config_path = Path(config_path)
    shutil.copy2(config_path, storage_path / config_path.name)
    # copy parents to known file pattern
    parent_counter = 0
    for parent in config_parents:
        shutil.copy2(parent, storage_path / f"parent-{parent_counter}.yml")
        parent_counter += 1

    # copy current (new parent) to a known filename to make loading easier
    shutil.copy2(storage_path / config_path.name, storage_path / f"parent-{parent_counter}.yml")
    for ix, patch in enumerate(config_patches):
        shutil.copy2(patch, storage_path / f"patch-{ix}.yml")

    # save population server as folder hierarchy
    logger.debug("Checkpointing population server params and populations")
    server_state: dict = ray.get(population_server.get_state.remote())  # type: ignore[attr-defined, assignment]

    server_folder = storage_path / "population_server"
    server_folder.mkdir(parents=True, exist_ok=True)

    for island_type in ["mainland", "island"]:
        for island_id, pop_state in server_state[f"{island_type}_populations"].items():
            island_folder = server_folder / f"{island_type}-{island_id}"
            island_folder.mkdir(parents=True, exist_ok=True)
            save_cloudpickle(pop_state, island_folder / "population.pkl")

    for k, v in server_state.items():
        if k in {"island_populations", "mainland_populations"}:
            continue

        save_cloudpickle(v, server_folder / f"{k}.pkl")

    logger.info(f"Checkpointing {len(islands)} island(s) @ {storage_path.as_posix()}/islands.pkl")
    save_cloudpickle([ray.get(isl.get_state.remote()) for isl in islands], storage_path / "islands.pkl")  # type: ignore[attr-defined]

    logger.info(f"Checkpointing {len(mainlands)} mainland(s) @ {storage_path.as_posix()}/mainlands.pkl")
    save_cloudpickle([ray.get(mainland.get_state.remote()) for mainland in mainlands], storage_path / "mainlands.pkl")  # type: ignore[attr-defined]

    archipelago_state = {"epoch": epoch, "topology": topology}
    for k, v in archipelago_state.items():
        logger.debug(f"Checkpointing {k} @ {storage_path.as_posix()}")
        save_cloudpickle(v, storage_path / f"{k}.pkl")


def load_checkpoint(storage_path: str | Path) -> dict:
    """Load a checkpoint from a storage path.

    Parameters
    ----------
    storage_path : str | Path
        The checkpoint path to load.

    Returns
    -------
    dict
        Loaded checkpoint content.
    """
    logger = logging.getLogger("ray")
    if isinstance(storage_path, str):
        storage_path = Path(storage_path)

    rx = re.compile(r"-(\d+)")
    archipelago_state = {}
    logger.info(f"Load checkpoint folder from {storage_path.as_posix()}")

    for file in storage_path.glob("*.pkl"):
        logger.debug(f"Found data file {file}")
        archipelago_state[file.stem] = load_cloudpickle(file)

    archipelago_state["population_server"] = {}
    server_folder = storage_path / "population_server"

    logger.debug("Collecting island populations...")
    island_populations = {}
    for island_folder in sorted(server_folder.glob("island-*")):
        assert island_folder.is_dir(), f"Found weird island file that should be a folder: {island_folder.as_posix()}"  # noqa: S101
        island_id = int(rx.findall(island_folder.name)[0])
        island_populations[island_id] = load_cloudpickle(island_folder / "population.pkl")

    archipelago_state["population_server"]["island_populations"] = island_populations

    logger.debug("Collecting mainland populations...")
    mainland_populations = {}
    for mainland_folder in sorted(server_folder.glob("mainland-*")):
        assert mainland_folder.is_dir(), f"Found weird mainland file that should be a folder: {mainland_folder.as_posix()}"  # noqa: S101
        mainland_id = int(rx.findall(mainland_folder.name)[0])
        mainland_populations[mainland_id] = load_cloudpickle(mainland_folder / "population.pkl")

    archipelago_state["population_server"]["mainland_populations"] = mainland_populations

    logger.debug("Collecting population server parameter...")
    for file in server_folder.glob("*.pkl"):
        archipelago_state["population_server"][file.stem] = load_cloudpickle(file)

    return archipelago_state


def resolve_checkpoint_epoch(checkpoint_folder: Path, epoch: int = -1) -> tuple[Path, int]:  # noqa: D103
    rx = re.compile(r"(\d+)")

    def folder_epoch_getter(folder: Path) -> int:
        res = rx.findall(folder.stem)
        if len(res) == 0:
            msg = f"Checkpoint epoch integers not found in {checkpoint_folder}"
            raise ValueError(msg)
        return int(res[0])

    if epoch == -1:
        candidates = sorted([node for node in checkpoint_folder.glob("epoch-*") if node.is_dir()], key=folder_epoch_getter)
        assert candidates, f"No folders with the pattern ``epoch-*'' were found in {checkpoint_folder.as_posix()}!"  # noqa: S101
        checkpoint_folder = candidates[-1]
        epoch = folder_epoch_getter(candidates[-1])
    else:
        checkpoint_folder /= f"epoch-{epoch}"

    return checkpoint_folder, epoch


def resolve_checkpoint_ancestors(  # noqa: D103
    checkpoint_folder: Path,
    loaded_config: dict | None = None,
) -> tuple[list[Path], list[Path], dict]:
    loaded_config = loaded_config or {}
    config_parents = sorted(checkpoint_folder.glob("parent-*"))

    # ancestral key priority (decreasing): current >> parent >> grandparent >> ...
    for parent in reversed(config_parents):
        from_config = load_config(str(parent))
        # only update the keys not in current config (ancestral update)
        loaded_config.update({k: v for k, v in from_config.items() if k not in loaded_config})

    config_patches = list(checkpoint_folder.glob("patch-*"))
    # apply experiment_cascade patch if available
    for patch in config_patches:
        loaded_config = apply_xc_config(loaded_config, patch)  # type: ignore[arg-type]

    return config_parents, config_patches, loaded_config


def increment_filepath_maybe(filepath: str | Path) -> str:  # noqa: D103
    filepath = Path(filepath)
    stem = filepath.stem
    ext = filepath.suffix  # includes leading .
    ix = 0
    candidate_path = filepath.parent / f"{stem}-{ix}{ext}"
    while candidate_path.exists():
        ix += 1
        candidate_path = filepath.parent / f"{stem}-{ix}{ext}"

    return candidate_path.as_posix()


# https://code.activestate.com/recipes/577504-compute-memory-footprint-of-an-object-and-its-cont/
def total_size(o: Any, handlers: MutableMapping[type, Callable] | None = None, logger: logging.Logger | None = None) -> int:  # noqa: ANN401
    """
    Return the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

    .. code:: python

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    Parameters
    ----------
    o : Any
        The object to size.
    handlers : MutableMapping[type, Callable] | None, optional
        The dict of logging handlers, :data:`python:None`.
    logger : logging.Logger | None, optional
        The logger, :data:`python:None`.

    Returns
    -------
    int
        The size of the object.
    """
    handlers = handlers or {}
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter, list: iter, deque: iter, dict: dict_handler, set: iter, frozenset: iter}
    all_handlers.update(handlers)  # type: ignore[arg-type] # user handlers take precedence
    seen = set()  # track which object id's have already been seen
    default_size = getsizeof(0)  # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:  # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if logger is not None:
            logger.debug(f"{s}, {type(o)}, {repr_(o)}")

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)
