"""Base configuration dataclasses for HARL agents."""

from dataclasses import dataclass, field

from algatross.configs.harl.algorithms import HAA2CAlgorithmConfig, HAPPOAlgorithmConfig, HARLAlgorithmConfig, HATRPOAlgorithmConfig
from algatross.configs.harl.models import HARLModelConfig, HARLOnPolicyModelConfig
from algatross.configs.harl.runners import HARLRunnerConfig


@dataclass
class HARLAgentConfig:
    """Base configuration for HARL agents."""

    model_config: HARLModelConfig = field(default_factory=HARLModelConfig)
    algorithm_config: HARLAlgorithmConfig = field(default_factory=HARLAlgorithmConfig)
    runner_config: HARLRunnerConfig = field(default_factory=HARLRunnerConfig)


@dataclass
class HAPPOAgentConfig(HARLAgentConfig):
    """Base configuration for HAPPO agents."""

    model_config: HARLOnPolicyModelConfig = field(default_factory=HARLOnPolicyModelConfig)
    algorithm_config: HAPPOAlgorithmConfig = field(default_factory=HAPPOAlgorithmConfig)


@dataclass
class HAA2CAgentConfig(HARLAgentConfig):
    """Base configuration for HAA2C agents."""

    model_config: HARLOnPolicyModelConfig = field(default_factory=HARLOnPolicyModelConfig)
    algorithm_config: HAA2CAlgorithmConfig = field(default_factory=HAA2CAlgorithmConfig)


@dataclass
class HATRPOAgentConfig(HARLAgentConfig):
    """Base configuration for HATRPO agents."""

    model_config: HARLOnPolicyModelConfig = field(default_factory=HARLOnPolicyModelConfig)
    algorithm_config: HATRPOAlgorithmConfig = field(default_factory=HATRPOAlgorithmConfig)
