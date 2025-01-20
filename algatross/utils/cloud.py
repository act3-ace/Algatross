"""Cloud storage handling utilities."""

from typing import Any

import ray


def indicated_ray_get(obj: Any) -> tuple[Any, bool]:  # noqa: ANN401
    """Return an un-rayed object, indicate if it was ref or not.

    Parameters
    ----------
    obj : Any
        The object to get from the object store

    Returns
    -------
    tuple[Any, bool]
        The object and an indicator of whether or not the object was retrieved from the object store
    """
    if isinstance(obj, ray.ObjectRef):
        return ray.get(obj), True
    return obj, False


def indicated_ray_put(obj: Any, indicator: bool) -> ray.ObjectRef | Any:  # noqa: ANN401
    """Ray.put if we have an indicator.

    Parameters
    ----------
    obj : Any
        The object to put
    indicator : bool
        Whether to put the object into the object store

    Returns
    -------
    ray.ObjectRef | Any
        The object or a reference to the object in the object store.
    """
    if indicator:
        return ray.put(obj)
    return obj
