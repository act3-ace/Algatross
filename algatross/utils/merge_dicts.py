"""A module of dict merge methods taken from RLlib since their implementations are planned to be deprecated."""

import itertools

from collections.abc import Callable, Sequence
from copy import deepcopy
from inspect import signature
from typing import Any

import numpy as np

import tree


def merge_dicts(left: dict, right: dict) -> dict:
    """
    Deep merge ``right`` into ``left``.

    Creates a deep copy of left and recursively merges the values from right into left so that dict values are updated
    rather than overwritten.

    Parameters
    ----------
    left : dict
        The base dictionary to merge
    right : dict
        The incoming dictionary to merge

    Returns
    -------
    dict
        ``left`` recursively deep updated with the values in ``right``
    """
    merged = deepcopy(left)
    deepmerge(merged, right)
    return merged


def deepmerge(left: dict, right: dict):
    """
    Recursively merge ``right`` into ``left`` in-place.

    If a key exists in both dictionaries and its value is a dictionary the recursion continues. Thus
    the ``left`` dictionary is updated by the ``right`` without erasing the values of nested dictionaries.

    Parameters
    ----------
    left : dict
        The base dictionary to be merged into.
    right : dict
        The dictionary to merge into ``left``.
    """
    for key, value in right.items():
        if isinstance(value, dict) and key in left:
            deepmerge(left[key], value)
        else:
            left[key] = value


def list_to_stack(list_of_dicts: list[dict], axis: int | Sequence[int] = 0) -> dict:
    """Convert a list of dictionaries into a dictionary of stacked numpy arrays.

    Parameters
    ----------
    list_of_dicts : list[dict]
        A list of dictionaries whose keys are to be stacked into arrays.
    axis : int | Sequence[int], optional
        The axis along which to stack, default is 0.

    Returns
    -------
    dict
        A dictionary of stacked values from each dict.
    """
    keys = set()
    flat = [{"/".join(str(p) for p in path[1:]): info} for path, info in tree.flatten_with_path(list_of_dicts)]
    for k in flat:
        keys.update(set(k))
    return {k: np.stack([d[k] for d in flat if k in d], axis=axis) for k in keys}  # type: ignore[arg-type]


def get_struct(nested: dict) -> dict:
    """Get the structure of the nested dictionary.

    Non-dictionary values are mapped to None.

    Parameters
    ----------
    nested : dict
        A nested dict structure

    Returns
    -------
    dict
        The structure of the dictionary.
    """
    return tree.traverse(lambda x: None if isinstance(x, dict) else tree.MAP_TO_NONE, nested)


def flatten_dicts(nested: dict) -> dict:
    """Flatten a nested dictionary.

    Parameters
    ----------
    nested : dict
        A nested dictionary

    Returns
    -------
    dict
        The flattened dictionary.
    """
    flat = tree.flatten_with_path_up_to(get_struct(nested), nested)
    return {"/".join(str(p) for p in path): info for path, info in flat}


def apply_xc_config(root_config: dict, child_config: dict) -> dict:
    """Apply the experiment cascade child_config onto the experiment root config.

    Parameters
    ----------
    root_config : dict
        The base configuration
    child_config : dict
        The child configuration to apply

    Returns
    -------
    dict
        The updated base configuration
    """
    for island_config in root_config["islands"]:
        # only island has these
        island_config["problem_constructor"].config["trainer_constructor_data"].config["train_config"]["num_sgd_iter"] = child_config[
            "num_sgd_iters"
        ]
        island_config["problem_constructor"].config["trainer_constructor_data"].config["train_config"]["batch_size"] = child_config[
            "batch_size"
        ]
        island_config["problem_constructor"].config["trainer_constructor_data"].config["train_config"]["sgd_minibatch_size"] = child_config[
            "sgd_minibatch_size"
        ]
        print("=" * 20)
        print("\ttrainer_constructor_data")
        print(island_config["problem_constructor"].config["trainer_constructor_data"])

        if "sigma" in child_config:
            island_config["population_constructor"].config["random_emitter_config"]["sigma"] = child_config["sigma"]
            island_config["population_constructor"].config["emitter_config"]["sigma"] = child_config["sigma"]

        if "novelty_threshold" in child_config:
            island_config["population_constructor"].config["archive_config"]["novelty_threshold"] = child_config["novelty_threshold"]
            island_config["population_constructor"].config["result_archive_config"]["novelty_threshold"] = child_config["novelty_threshold"]
            print("=" * 20)
            print("\tpopulation_constructor")
            print(island_config["population_constructor"].config)

    for island_config in itertools.chain(root_config["islands"], root_config["mainlands"]):
        # universal param for island and mainland
        island_config["problem_constructor"].config["trainer_constructor_data"].config["rollout_config"]["batch_size"] = child_config[
            "batch_size"
        ]
        island_config["algorithm_constructor"].config["training_iterations"] = child_config[
            "uda_training_iterations"
        ]  # iterations inside fitness()

    root_config["archipelago_constructor"].config["warmup_generations"] = child_config.get("warmup_generations", 10)
    root_config["archipelago_constructor"].config["warmup_iterations"] = child_config.get("warmup_iterations", 1)
    root_config["island_iterations"] = child_config["island_iterations"]  # iterations inside run_evolve
    root_config["mainland_iterations"] = child_config["mainland_iterations"]
    root_config["epochs"] = child_config["epochs"]
    root_config["seed"] = child_config["seed"]

    return root_config


def filter_keys(datacls: Callable, **kwargs) -> dict[str, Any]:
    """Filter the keyword arguments based on the signature of the object.

    Parameters
    ----------
    datacls : Callable
        The object whose signature should be used to filter the ``kwargs``

    Returns
    -------
    dict[str, Any]
        The filtered keyword dictionary

    Raises
    ------
    ValueError
        The error raised by :func:`~inspect.signature` if the object has a signature, otherwise the ``kwargs`` are
        returned unchanged
    """
    filtered = {}
    try:
        for key in signature(datacls).parameters:
            if key in kwargs:
                filtered[key] = kwargs[key]
    except ValueError as err:
        if "no signature found" in err.args[0]:
            return kwargs
        raise
    else:
        return filtered
