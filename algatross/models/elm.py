import functools  # noqa: D100
import math

from collections.abc import Callable
from typing import Literal

import torch

from torch import nn
from torch.nn import functional as F  # noqa: N812

from algatross.extreme_learning_machines.utils.init import ELM_INITIALIZER_MAP
from algatross.extreme_learning_machines.utils.ops import torch_calc_regularized_mlgelm_weights


class MLGELMEncoderBlock(nn.Module):
    """
    A single GELM encoder block for a multi-layer GELM.

    Parameters
    ----------
    in_features : int
        Number of encoder input features.
    out_features : int
        Number of encoder output features.
    bias : bool, optional
        Whether to include bias in the layers, default is :data:`python:True`.
    initializer : Literal["normal", "orthogonal", "uniform", "kaiming_normal", "kaiming_uniform"]
        The type of initializer to use for network parameters, default is "orthogonal".
    initializer_config : dict | None, optional
        Keyword arguments to pass to the initializer, default is :data:`python:None`.
    activation : nn.Module | type[nn.Module] | Callable[[torch.Tensor], torch.Tensor] | None, optional
        Activation function to use with the network, default is :data:`python:None`.
    activation_config : dict | None, optional
        The config dictionary to pass to the activation function constructor, default is :data:`python:None`.
    classification_activation : nn.Module | type[nn.Module] | Callable[[torch.Tensor], torch.Tensor] | None, optional
        Activation function to use with the network when used as a classifier, default is :data:`python:None`.
    classification_activation_config : dict | None, optional
        The config dictionary to pass to the activation function constructor when used as a classifier, default is :data:`python:None`.
    elm_learning_config : dict | None, optional
        The config dictionary to pass to the learning function, default is :data:`python:None`.
    device : str | torch.device | None, optional
        The device to create the network on, default is :data:`python:None`.
    dtype : torch.dtype | None, optional
        The datatype to use for the encoder, default is :data:`python:None`.
    """

    in_features: int
    """The input feature size for the network."""
    encoding_dim: int
    """The output feature size for the network."""
    initializer_fn: Callable
    """The initializer function for network layer weights."""
    initializer_str: str
    """The string representing the name of the initializer."""
    activation: nn.Module | type[nn.Module] | Callable[[torch.Tensor], torch.Tensor]
    """The activation function to use with the network layers."""
    activation_config: dict
    """The config to pass to the activation function constructor."""
    classification_activation: "Callable[[torch.Tensor], torch.Tensor]"
    """The activation function to use with the network layers when used as a classifier."""
    classification_activation_config: dict
    """The config to pass to the classification activation function constructor."""
    elm_learning_config: dict
    """The configuration dictionary to pass to the learning function."""
    decoder_weight: nn.Parameter
    """The decoder weights."""
    encoder_weight: nn.Parameter
    """The encoder weights."""
    encoder_bias: nn.Parameter
    """The encoder bias."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        initializer: Literal["normal", "orthogonal", "uniform", "kaiming_normal", "kaiming_uniform"] = "orthogonal",
        initializer_config: dict | None = None,
        activation: nn.Module | type[nn.Module] | Callable[[torch.Tensor], torch.Tensor] | None = None,
        activation_config: dict | None = None,
        classification_activation: nn.Module | type[nn.Module] | Callable[[torch.Tensor], torch.Tensor] | None = None,
        classification_activation_config: dict | None = None,
        elm_learning_config: dict | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.encoding_dim = out_features
        initializer_config = initializer_config or {}
        self.initializer_fn = functools.partial(ELM_INITIALIZER_MAP[initializer], **initializer_config)
        self.initializer_str = initializer

        self.activation_config = activation_config or {}
        self.classification_activation_config = classification_activation_config or {}
        self.elm_learning_config = elm_learning_config or {}
        self._previous_hidden_preds: torch.Tensor | None = None
        self._a_inv: torch.Tensor | None = None

        activation = nn.Softmax() if activation is None else activation
        classification_activation = nn.Sigmoid() if classification_activation is None else classification_activation

        if isinstance(activation, nn.Module):
            self.activation: Callable[[torch.Tensor], torch.Tensor] = activation
        elif isinstance(activation, type):
            self.activation = activation(**self.activation_config)

        if isinstance(classification_activation, nn.Module):
            self.classification_activation: Callable[[torch.Tensor], torch.Tensor] = classification_activation
        elif isinstance(classification_activation, type):
            self.classification_activation = classification_activation(**self.activation_config)

        # Gradients should not flow through ELM weights & biases
        self.encoder_weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs), requires_grad=False)  # type: ignore[call-overload]
        self.decoder_weight = nn.Parameter(torch.zeros((in_features, out_features), **factory_kwargs), requires_grad=False)  # type: ignore[call-overload]
        if bias:
            self.encoder_bias = nn.Parameter(torch.empty(out_features, **factory_kwargs), requires_grad=False)  # type: ignore[call-overload]
        else:
            self.register_parameter("encoder_bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Re-initialize the network parameters."""
        # TODO: override this with a better init strategy for ELMs
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        self.initializer_fn(self.encoder_weight)

        if self.encoder_bias is not None:
            if self.initializer_str == "orthogonal":
                self.encoder_bias.unsqueeze_(0)
                self.initializer_fn(self.encoder_bias)
                self.encoder_bias.squeeze_()
            else:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)  # noqa: SLF001
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.encoder_bias, -bound, bound)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the layer.

        Parameters
        ----------
        inputs : torch.Tensor
            The input to make the prediction

        Returns
        -------
        torch.Tensor
            The forward prediction
        """
        # return F.linear(self.activation(F.linear(inputs, self.encoder_weight, self.encoder_bias)), self.decoder_weight.mT)  # noqa: ERA001
        return F.linear(inputs, self.decoder_weight.mT)

    def forward_classification(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the layer using the activation for classification.

        Parameters
        ----------
        inputs : torch.Tensor
            The input to make the prediction

        Returns
        -------
        torch.Tensor
            The forward prediction
        """
        return self.classification_activation(self(inputs))

    def decode(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Decode the encoded input.

        Parameters
        ----------
        inputs : torch.Tensor
            The input to decode

        Returns
        -------
        torch.Tensor
            The decoded input
        """
        return F.linear(inputs, self.decoder_weight)

    def extra_repr(self) -> str:
        """Get the string representation of the network.

        Returns
        -------
        str
            The string representation of the encoder
        """
        return (
            f"in_features={self.in_features}, encoding_dim={self.encoding_dim}, bias={self.encoder_bias is not None}, "
            f"activation={self.activation}, classification_activation={self.classification_activation}"
        )

    def learn_weights(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Learn the output weights of the ELM using ``targets``.

        Parameters
        ----------
        targets : torch.Tensor
            The learning targets

        Returns
        -------
        torch.Tensor
            The reconstruction loss
        """
        hidden_outs = self.activation(F.linear(targets, self.encoder_weight, self.encoder_bias))
        new_weights, self._a_inv = torch_calc_regularized_mlgelm_weights(
            hidden_outs=hidden_outs,  # type: ignore[arg-type]
            targets=targets,
            current_elm_weights=self.decoder_weight.data,
            previous_hidden_outs=self._previous_hidden_preds,
            a_inv=self._a_inv,
            **self.elm_learning_config,
        )
        self.decoder_weight.data.copy_(new_weights.transpose(1, 0))
        self._previous_hidden_preds = hidden_outs.clone()
        return F.mse_loss(F.linear(hidden_outs, self.decoder_weight), targets, reduction="mean")  # type: ignore[arg-type]


class MLGELMEncoder(nn.Sequential):
    """
    Multilayer Genralized ELM encoder.

    Parameters
    ----------
    encoder_depth : int
        The number of hidden layers for the encoder.
    embedding_dim : int
        The number of output nodes for the encoder.
    bias : bool, optional
        Whether to include a bias variable for the hidden layers, default is :data:`python:True`.
    initializer : Literal["normal", "orthogonal", "uniform", "kaiming_normal", "kaiming_uniform"]
        The initializer to use for the layers, default is :python:`"orthogonal"`
    initializer_config : dict | None, optional
        The config dictionary to pass to the initializaer methods, default is :data:`python:None`.
    activation : nn.Module | type[nn.Module] | Callable[[torch.Tensor], torch.Tensor] | None, optional
        The activation to use with the layers, default is :data:`python:None`.
    activation_config : dict | None, optional
        The config dictionary to use when setting up the activation functions, default is :data:`python:None`.
    classification_activation : nn.Module | type[nn.Module] | Callable[[torch.Tensor], torch.Tensor] | None, optional
        The activation to use for classification, default is :data:`python:None`.
    classification_activation_config : dict | None, optional
        The config dictionary to use for the classification activation, default is :data:`python:None`.
    device : str | torch.device | None, optional
        The device to place the network onto, default is :data:`python:None`.
    dtype : torch.dtype | None, optional
        The dtype to use for the network, default is :data:`python:None`.
    `**kwargs`
        The keyword arguments to use when calling the learning methods.
    """

    elm_learning_config: dict
    """The keyword arguments passed to the learning functions."""
    elm_block_init_kwargs: dict
    """The keyword arguments to use when initializing sub-blocks of ELMs."""
    final_embedding_dim: int
    """The output size of the network."""
    encoder_depth: int
    """The number of hidden layers of the network."""
    in_features: int | None
    """The number of input nodes to the network."""

    def __init__(
        self,
        encoder_depth: int,
        embedding_dim: int,
        bias: bool = True,
        initializer: Literal["normal", "orthogonal", "uniform", "kaiming_normal", "kaiming_uniform"] = "orthogonal",
        initializer_config: dict | None = None,
        activation: nn.Module | type[nn.Module] | Callable[[torch.Tensor], torch.Tensor] | None = None,
        activation_config: dict | None = None,
        classification_activation: nn.Module | type[nn.Module] | Callable[[torch.Tensor], torch.Tensor] | None = None,
        classification_activation_config: dict | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: dict,
    ):
        super().__init__()
        self.elm_learning_config = kwargs
        self.elm_block_init_kwargs = {
            "elm_learning_config": {**self.elm_learning_config},
            "initializer": initializer,
            "initializer_config": initializer_config,
            "activation": activation,
            "activation_config": activation_config,
            "classification_activation": classification_activation,
            "classification_activation_config": classification_activation_config,
            "device": device,
            "dtype": dtype,
            "bias": bias,
        }
        self.final_embedding_dim = embedding_dim
        self.encoder_depth = encoder_depth
        self.in_features: int | None = None

        self.to(device=device, dtype=dtype)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # noqa: D102
        encoding_targets = inputs.view(inputs.shape[0], -1)

        if self.in_features is None:
            self._init_encoder_blocks(encoding_targets.shape[1])
            self.learn_weights(encoding_targets)

        for mod in self.children():
            encoding_targets = mod(encoding_targets)

        return encoding_targets

    def learn_weights(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Learn the weights of the ELM encoder.

        Parameters
        ----------
        targets : torch.Tensor
            Encoding targets

        Returns
        -------
        torch.Tensor
            The encoding loss.
        """
        learning_targets = targets.view(targets.shape[0], -1)
        if self.in_features is None:
            self._init_encoder_blocks(learning_targets.shape[1])
        losses = []
        for mod in self.children():
            losses.append(mod.learn_weights(learning_targets))
            learning_targets = mod(learning_targets)
        return torch.stack(losses)

    def _init_encoder_blocks(self, in_features: int):
        self.in_features = in_features
        slope = (self.final_embedding_dim - self.in_features) / self.encoder_depth
        prev_lyr = in_features
        for mod_idx in range(self.encoder_depth):
            lyr_out = in_features + math.ceil((mod_idx + 1) * slope)
            mod = MLGELMEncoderBlock(prev_lyr, lyr_out, **self.elm_block_init_kwargs)
            self.add_module(f"mlgelm_encoder_{mod_idx}", mod)
            prev_lyr = lyr_out
