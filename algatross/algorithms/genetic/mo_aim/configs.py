"""Module of configuration dataclasses."""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from ribs.archives import ArchiveBase, CVTArchive
from ribs.emitters import EmitterBase, GaussianEmitter

from algatross.configs.pyribs.archive_config import ArchiveConfig
from algatross.configs.pyribs.emitter_config import EmitterConfig
from algatross.quality_diversity.emitters.random import RandomEmitter
from algatross.utils.types import AgentID, NumpyRandomSeed


@dataclass
class MOAIMIslandUDAConfig:
    """Config dataclass for MO-AIM Island UDAs."""

    conspecific_data_keys: list[str]
    training_iterations: int
    seed: NumpyRandomSeed | None = None


# TODO: tournament, crossover, selection configs


@dataclass
class MOAIMIslandPopulationConfig:
    """Config dataclass for MO-AIM Island populations."""

    storage_path: Path
    solution_dim: dict[AgentID, int] | None = None

    archive_base_class: type[ArchiveBase] = CVTArchive
    archive_config: dict | ArchiveConfig = field(default_factory=dict)

    use_result_archive: bool = False
    result_archive_base_class: type[ArchiveBase] = CVTArchive
    result_archive_config: dict | ArchiveConfig = field(default_factory=dict)

    emitter_base_class: type[EmitterBase] = GaussianEmitter
    emitter_config: dict | EmitterConfig = field(default_factory=dict)

    random_emitter_base_class: type[EmitterBase] = RandomEmitter
    random_emitter_config: dict | EmitterConfig = field(default_factory=dict)

    max_samples_per_migrant: int = -1

    seed: NumpyRandomSeed | None = None

    encoder_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = torch.nn.functional.mse_loss

    team_size: int = 5
    qd_samples: int = 5
    qd_experience_buffer_size: int = 1_000_000


@dataclass
class MOAIMRLPopulationConfig:
    """Config dataclass for MO-AIM classic RL populations."""

    storage_path: Path
    seed: NumpyRandomSeed | None = None
    team_size: int = 5


@dataclass
class MOAIMRLUDAConfig:
    """Config dataclass for MO-AIM Island UDAs."""

    conspecific_data_keys: list[str]
    training_iterations: int
    seed: NumpyRandomSeed | None = None


@dataclass
class MOAIMMainlandUDAConfig:
    """Config dataclass for MO-AIM Mainland UDAs."""

    conspecific_utility_keys: list[str]
    conspecific_utility_objectives: list[str]
    n_elites: int | None = None
    seed: NumpyRandomSeed | None = None
    mutation_noise: float = 1.0


@dataclass
class MOAIMMainlandPopulationConfig:
    """Config dataclass of MO-AIM mainland populations."""

    # mutation_operator
    # crossover_operator

    storage_path: Path
    team_size: int = 5
    n_teams: int = 5

    # proportion of teams which survive as elites
    elite_proportion: float = 0.2
    seed: NumpyRandomSeed | None = None


@dataclass
class MetadataContainer:
    """Generic dataclass for pymoo archive metadata."""

    metadata_type: str
    metadata: Any


@dataclass
class BehaviorClassificationConfig:
    """Configuration settings for behavior classification algorithms."""

    trajectory_length: int = 100
    num_samples: int = 10
    reduction_method: str | Callable = "mean"
