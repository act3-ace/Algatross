"""A module of custom exceptions."""

from collections.abc import Sequence


class MissingParallelizationBackendError(ImportError):
    """
    An error raised whenever no parallelization backends were able to be imported.

    Parameters
    ----------
    `*args`
        Positional arguments.
    name : str | None, optional
        The module name.
    path : str | None, optional
        The path to the module.
    """

    def __init__(self, *args: object, name: str | None = None, path: str | None = None) -> None:
        msg = "At least one parallelization backend (ray, dask) must be available"
        super().__init__(msg, *args, name=name, path=path)


class MissingTensorBackendError(ImportError):
    """
    An error raised whenever no tensor backends were able to be imported.

    Parameters
    ----------
    `*args`
        Positional arguments.
    name : str | None, optional
        The module name.
    path : str | None, optional
        The path to the module.
    """

    def __init__(self, *args: object, name: str | None = None, path: str | None = None) -> None:
        msg = "At least one tensor backend (torch, tensorflow) must be available"
        super().__init__(msg, *args, name=name, path=path)


class InitNotCalledError(RuntimeError):
    """
    Error raised whenever the :python:`__init__` has not been called.

    Parameters
    ----------
    obj : object
        The object which has not had its init called
    `*args`
        Additional positional arguments.
    """

    def __init__(self, obj: object, *args: Sequence) -> None:
        msg = f"`__init__` method of {obj} has not been called"
        super().__init__(msg, *args)
