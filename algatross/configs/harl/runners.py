"""Config dataclasses for HARL Runners."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class HARLRunnerConfig:
    """Default configuration for HARL Runners."""

    # rollout
    gamma: float = 0.9
    episode_length: int = 200
    action_aggregation: str = "prod"
    n_rollout_threads: int = 20

    # device
    cuda: bool = True
    cuda_deterministic: bool = True
    torch_threads: int = 4

    # RNN
    state_type: Literal["EP", "FP"] = "EP"

    # flags
    use_valuenorm: bool = True
    use_linear_lr_decay: bool = False
    use_proper_time_limits: bool = True
