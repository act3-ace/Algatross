"""Base configuration dataclasses for HARL algorithms."""

from dataclasses import dataclass


@dataclass
class HARLAlgorithmConfig:
    """Base configuration for all HARL algorithms."""

    gamma: float = 0.95
    max_grad_norm: float = 10.0

    share_param: bool = False
    fixed_order: bool = False

    use_max_grad_norm: bool = True


@dataclass
class OnPolicyHARLAlgorithmConfig(HARLAlgorithmConfig):
    """Base configuration for On-Policy HARL algorithms."""

    # number of epochs for critic update
    critic_epoch: int = 5
    # whether to use clipped value loss
    use_clipped_value_loss: bool = True
    # clip parameter
    clip_param: float = 0.2
    # number of mini-batches per epoch for actor update
    actor_num_mini_batch: int = 1
    # number of mini-batches per epoch for critic update
    critic_num_mini_batch: int = 1
    # coefficient for entropy term in actor loss
    entropy_coef: float = 0.01
    # coefficient for value loss
    value_loss_coef: float = 1

    # whether to use Generalized Advantage Estimation (GAE)
    use_gae: bool = True
    # GAE lambda
    gae_lambda: float = 0.95
    # whether to use huber loss
    use_huber_loss: bool = True
    # whether to use policy active masks
    use_policy_active_masks: bool = True
    # huber delta
    huber_delta: float = 10.0
    # method of aggregating the probability of multi-dimensional actions, choose from prod, mean
    action_aggregation: str = "prod"


@dataclass
class HAA2CAlgorithmConfig(OnPolicyHARLAlgorithmConfig):
    """Base configuration for HAA2C algorithms."""

    # A2C
    a2c_epoch: int = 5


@dataclass
class HAPPOAlgorithmConfig(OnPolicyHARLAlgorithmConfig):
    """Base configuration for HAPPO algorithms."""

    # ppo
    ppo_epoch: int = 5

    # target kl divergence in ppo update
    target_kl: float = 0.01
    # coefficient for KL loss
    kl_loss_coef: float = 0.2


@dataclass
class HATRPOAlgorithmConfig(OnPolicyHARLAlgorithmConfig):
    """Base configuration for HATRPO algorithms."""

    # TRPO
    # kl threshold in HATRPO update
    kl_threshold: float = 0.01
    # line search steps
    ls_step: int = 10
    # accept ratio in HATRPO update
    accept_ratio: float = 0.5
    # backtracking coefficient in line search
    backtrack_coeff: float = 0.8
