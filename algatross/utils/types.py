"""A module of basic type definitions for use throughout MO-MARL."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any, Literal, NamedTuple, TypeAlias, overload
from uuid import UUID, uuid4

import numpy as np

from numpy.random import BitGenerator, Generator, SeedSequence

from ray.rllib.policy.sample_batch import SampleBatch

from torch import nn

from algatross.utils.merge_dicts import deepmerge

_ArrayLikeInt: TypeAlias = int | np.integer[Any] | Sequence[int | np.integer[Any]] | Sequence[Sequence[Any]]

MainlandID: TypeAlias = int
"""An identifier for an agent in an environment."""
IslandID: TypeAlias = int
"""The id of a island in the archipelago."""
InhabitantID: TypeAlias = str
"""The id of an inhabitant on an island."""
TeamID: TypeAlias = int
"""The id of a team on an island."""
TeammateID: TypeAlias = int
"""The index of a team member on a team."""
AgentID: TypeAlias = str | int
"""An identifier for an agent in an environment."""
PlatformID: TypeAlias = str | int
"""The id of a platform in an environment."""


NumpyRandomSeed: TypeAlias = int | _ArrayLikeInt | SeedSequence | BitGenerator | Generator
"""A valid argument seed to :mod:`numpy.random` classes."""
IslandSpec: TypeAlias = tuple[Literal["island"], IslandID] | tuple[Literal["mainland"], MainlandID]
"""A specification carrying an island type and the id of the island in the archipelago."""
IslandTypeStr: TypeAlias = Literal["island", "mainland"]
"""A string representing the type of the island."""

ConvLayerType: TypeAlias = nn.Conv1d | nn.Conv2d | nn.Conv3d
ConvTransposeLayerType: TypeAlias = nn.ConvTranspose1d | nn.ConvTranspose2d | nn.ConvTranspose3d

Conv1dLayerType: TypeAlias = (
    nn.Conv1d
    | nn.ConvTranspose1d
    | nn.MaxPool1d
    | nn.AdaptiveMaxPool1d
    | nn.AvgPool1d
    | nn.AdaptiveAvgPool1d
    | nn.BatchNorm1d
    | nn.InstanceNorm1d
    | nn.ConstantPad1d
    # | nn.CircularPad1d
    # | nn.ZeroPad1d
    | nn.ReflectionPad1d
    | nn.ReplicationPad1d
)
Conv2dLayerType: TypeAlias = (
    nn.Conv2d
    | nn.ConvTranspose2d
    | nn.MaxPool2d
    | nn.AdaptiveMaxPool2d
    | nn.AvgPool2d
    | nn.AdaptiveAvgPool2d
    | nn.FractionalMaxPool2d
    | nn.BatchNorm2d
    | nn.InstanceNorm2d
    | nn.ConstantPad2d
    # | nn.CircularPad2d
    | nn.ZeroPad2d
    | nn.ReflectionPad2d
    | nn.ReplicationPad2d
)

Conv3dLayerType: TypeAlias = (
    nn.Conv3d
    | nn.ConvTranspose3d
    | nn.MaxPool3d
    | nn.AdaptiveMaxPool3d
    | nn.AvgPool3d
    | nn.AdaptiveAvgPool3d
    | nn.FractionalMaxPool3d
    | nn.BatchNorm3d
    | nn.InstanceNorm3d
    | nn.ConstantPad3d
    # | nn.CircularPad3d
    # | nn.ZeroPad3d
    | nn.ReflectionPad3d
    | nn.ReplicationPad3d
)

SimpleMaxPoolLayerType: TypeAlias = nn.MaxPool1d | nn.MaxPool2d | nn.MaxPool3d
AdaptiveMaxPoolLayerType: TypeAlias = nn.AdaptiveMaxPool1d | nn.AdaptiveMaxPool2d | nn.AdaptiveMaxPool3d
FractionalMaxPoolLayerType: TypeAlias = nn.FractionalMaxPool2d | nn.FractionalMaxPool3d
MaxPoolLayerType: TypeAlias = SimpleMaxPoolLayerType | AdaptiveMaxPoolLayerType | FractionalMaxPoolLayerType

AdaptiveAvgPoolLayerType: TypeAlias = nn.AdaptiveAvgPool1d | nn.AdaptiveAvgPool2d | nn.AdaptiveAvgPool3d
SimpleAvgPoolLayerType: TypeAlias = nn.AvgPool1d | nn.AvgPool2d | nn.AvgPool3d
AvgPoolLayerType: TypeAlias = SimpleAvgPoolLayerType | AdaptiveAvgPoolLayerType

InstanceNormLayerType: TypeAlias = nn.InstanceNorm1d | nn.InstanceNorm2d | nn.InstanceNorm3d
BatchNormLayerType: TypeAlias = nn.BatchNorm1d | nn.BatchNorm2d | nn.BatchNorm3d
NormLayerType: TypeAlias = BatchNormLayerType | InstanceNormLayerType

ConstantPadLayerType: TypeAlias = nn.ConstantPad1d | nn.ConstantPad2d | nn.ConstantPad3d
ReflectionPadLayerType: TypeAlias = nn.ReflectionPad1d | nn.ReflectionPad2d | nn.ReflectionPad3d
ReplicationPadLayerType: TypeAlias = nn.ReplicationPad1d | nn.ReplicationPad2d | nn.ReplicationPad3d
PadLayerType: TypeAlias = ConstantPadLayerType | ReflectionPadLayerType | ReplicationPadLayerType

CNNLayerType: TypeAlias = Conv1dLayerType | Conv2dLayerType | Conv3dLayerType


class OptimizationTypeEnum(Enum):
    """OptimizationTypeEnum enumeration class for optimization types."""

    MAX = "max"
    """Represents a maximization problem."""
    MIN = "min"
    """Represents a minimzation problem."""


class RolloutData(NamedTuple):
    """RolloutData a named tuple to ensure we have certain data."""

    rollout: np.ndarray
    """The rollout data for the decision variable."""
    objective: float | np.ndarray
    """The fitness of the decision variable."""
    solution: np.ndarray
    """The genome (decision variable) itself."""


class IslandSample(NamedTuple):
    """IslandSample a named tuple to ensure we have the appropriate information about how many migrants to sample from the given island."""

    island: IslandID
    """The ID of the island from which to sample migrants."""
    migrants: int
    """The number of migrants to sample from the islant."""


class TeamFront(NamedTuple):
    """A named tuple containing the team ID as well as the front it's in."""

    front_index: int
    """The Pareto front containing this team."""
    team_id: TeamID
    """The ID of the team."""


@dataclass
class ConstructorData:
    """Tuple containing a constructor and configuration to pass to the constructor."""

    constructor: Callable
    """The constructor which returns an object when passed the configuration."""
    config: dict[str, Any] = field(default_factory=dict)
    """The configuration to pass to the constructor."""

    def construct(self, **kwargs) -> Any:  # noqa: ANN401
        """Call the constructor on the stored config.

        Parameters
        ----------
        `**kwargs`
            Keyword arguments

        Returns
        -------
        Any
            The constructed object.
        """
        config = {**self.config}
        if kwargs:
            deepmerge(config, kwargs)
        return self.constructor(**config)

    def __call__(self, **kwargs):  # noqa: D102
        return self.construct(**kwargs)


ConstructorDataDict: TypeAlias = dict[
    Literal["constructors", "problem_constructors", "algorithm_constructors", "population_constructors"],
    Sequence[ConstructorData],
]
"""A dictionary of constructor data for each component needed for MO-MARL."""


@dataclass
class MOAIMIslandInhabitant:
    """Dataclass for containing identifying data for inhabitants of a MO-AIM Island."""

    name: AgentID
    """Name of the agent in the environment."""
    team_id: TeamID
    """ID of the team to which this member belongs."""
    inhabitant_id: InhabitantID
    """ID in the archive where is member is stored."""
    island_id: IslandID
    """ID of the island where this inhabitant normally resides."""
    current_island_id: IslandID | MainlandID
    """ID of the island the inhabitant is currently on."""
    genome: np.ndarray
    """Genome of the member (model parameters)."""
    conspecific_utility_dict: dict[str, float]
    """Conspecific utility (fitness)."""
    db_hash: UUID = field(default_factory=uuid4, init=False, kw_only=True)
    """Unique hash indicating the ID of the inhabitant in a database."""

    def __hash__(self) -> int:
        """Generate a hash from the ``db_hash`` property.

        Returns
        -------
        int
            The hash value.
        """
        return hash(str(self.db_hash))

    @overload
    def __eq__(self, value: dict, /) -> bool: ...

    @overload
    def __eq__(self, value: list, /) -> bool: ...

    @overload
    def __eq__(self, value: object, /) -> bool: ...

    def __eq__(self, value, /):
        """
        Compare whether two Inhabitants are equal.

        Any object which defines a mapping is allowed for comparison. Inhabitants are considered equal if all attributes are the
        same except the ``db_hash``, which is allowed to differ.

        Parameters
        ----------
        value : dict | list | object
            The other object being compared for equality.

        Returns
        -------
        bool
            Whether the two values are equal.
        """
        if id(self) == id(value):
            # they're the same object
            return True

        eq = True
        if isinstance(value, dict):
            for f in fields(self):
                if f.name != "db_hash":
                    eq &= f.name in value and np.equal(getattr(self, f.name), value[f.name]).all()
                if not eq:
                    # we can stop here
                    break
        elif isinstance(value, list):
            for f in fields(self):
                if f.name != "db_hash":
                    try:
                        idx = value.index((f.name, getattr(self, f.name)))
                    except ValueError:
                        eq = False
                        # we can stop here
                        break
                    else:
                        eq &= np.equal(getattr(self, f.name), value[idx][1]).all()
                if not eq:
                    # we can stop here
                    break
        else:
            for f in fields(self):
                if f.name != "db_hash":
                    eq &= hasattr(value, f.name) and np.equal(getattr(self, f.name), getattr(value, f.name)).all()
                if not eq:
                    # we can stop here
                    break
        return eq

    def __setattr__(self, name: str, value: Any, /) -> None:  # noqa: ANN401
        """
        Set the attribute given by ``name`` to ``value`` and updates the hash if necessary.

        This function makes sure that any time an attribute is modified the hash is updated so we can be sure that
        Whether two objects are different even if __eq__ returns True.

        Parameters
        ----------
        name : str
            The name of the attribute to set.
        value : Any
            The new value for the attribute.
        """
        if name != "db_hash" and (not hasattr(self, name) or np.any(getattr(self, name, None) != value)):
            self.db_hash = uuid4()
        super().__setattr__(name, value)


MigrantData: TypeAlias = tuple[MOAIMIslandInhabitant | np.ndarray | None, RolloutData | list[SampleBatch] | None]
"""An inhabitant coupled with its rollout data."""
MigrantFlock: TypeAlias = list[MigrantData]
"""A flock of migrants, i.e. a group of :class:`~algatross.utils.types.MigrantData`."""
MigrantQueue: TypeAlias = dict[IslandID | MainlandID, MigrantFlock]
"""A dictionary of migrants waiting to come to the island."""

TeamDict: TypeAlias = dict[TeamID, dict[TeammateID, MOAIMIslandInhabitant]]
"""A mapping from team ids to the team itself."""
