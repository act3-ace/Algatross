from algatross.utils.exceptions import MissingParallelizationBackendError

from .base import RemoteMOAIMArchipelago

try:
    from .ray_archipelago import RayMOAIMArchipelago
except ModuleNotFoundError as err:
    if "No module named 'ray" in err.args[0]:
        RayMOAIMArchipelago = None  # type: ignore[misc]
    else:
        raise

if RayMOAIMArchipelago is None:
    raise MissingParallelizationBackendError(name="archipelago", path="algatross.algorithsm.genetic.mo_aim.archipelago")

__all__ = ["RayMOAIMArchipelago", "RemoteMOAIMArchipelago"]
