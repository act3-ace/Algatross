"""Dataclasses for HARL Models."""

from dataclasses import dataclass, field


@dataclass
class HARLModelConfig:
    """Default configuration for HARL models."""

    # network parameters
    hidden_sizes: list[int] = field(default_factory=lambda: [128, 128])
    """Hidden sizes for mlp module in the network."""

    activation_func: str = "relu"
    """Activation function,.

    Choose from ``sigmoid``, ``tanh``, ``relu``, ``leaky_relu``, ``selu``
    """

    use_feature_normalization: bool = True
    """Whether to use feature normalization."""

    initialization_method: str = r"orthogonal_"
    """Initialization method for network parameters, choose from ``xavier_uniform_``, ``orthogonal_``, ..."""

    lr: float = 5e-4
    """Actor learning rate, ineffective for hatrpo."""

    critic_lr: float = 5e-4
    """Critic learning rate."""

    opti_eps: float = 1e-8
    """Eps in Adam."""


@dataclass
class HARLOnPolicyModelConfig(HARLModelConfig):
    """Default configuration for HARL models."""

    gain: float = 0.01
    """Gain of the output layer of the network."""

    # recurrent parameters
    use_naive_recurrent_policy: bool = False
    """Whether to use rnn policy (data is not chunked for training)."""

    use_recurrent_policy: bool = False
    """Whether to use rnn policy (data is chunked for training)."""

    recurrent_n: int = 1
    """Number of recurrent layers."""

    data_chunk_length: int = 10
    """# length of data chunk; only useful when use_recurrent_policy is True; episode_length has to be a multiple of data_chunk_length."""

    # optimizer parameters
    weight_decay: float = 0.0
    """Weight_decay in Adam."""

    # parameters of diagonal Gaussian distribution
    std_x_coef: float = 1.0
    """Standard deviation x-coefficient."""

    std_y_coef: float = 0.5
    """Standard deviation y-coefficient."""
