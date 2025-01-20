"""Linear layers to use in ELMs."""

import functools
import math

from collections.abc import Callable
from typing import Literal

import torch

from torch import nn
from torch.nn import functional as F  # noqa: N812

from algatross.extreme_learning_machines.utils.init import ELM_INITIALIZER_MAP
from algatross.extreme_learning_machines.utils.ops import torch_calc_regularized_elm_weights


class ELMHiddenLinear(nn.Module):
    """
    ELMHiddenLinear a hidden ELM linear layer.

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
    device : str | torch.device | None, optional
        The device to create the network on, default is :data:`python:None`.
    dtype : torch.dtype | None, optional
        The datatype to use for the encoder, default is :data:`python:None`.
    """

    in_features: int
    """The input feature size for the network."""
    out_features: int
    """The output feature size for the network."""
    initializer_fn: Callable
    """The initializer function for network layer weights."""
    initializer_str: str
    """The string representing the name of the initializer."""
    weight: nn.Parameter
    """The network weights."""
    bias: nn.Parameter
    """The network bias."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        initializer: Literal["normal", "orthogonal", "uniform", "kaiming_normal", "kaiming_uniform"] = "orthogonal",
        initializer_config: dict | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        initializer_config = initializer_config or {}
        self.initializer_fn = functools.partial(ELM_INITIALIZER_MAP[initializer], **initializer_config)
        self.initializer_str = initializer

        # Gradients should not flow through ELM weights & biases
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs), requires_grad=False)  # type: ignore[call-overload]
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs), requires_grad=False)  # type: ignore[call-overload]
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Re-initialize the network parameters."""
        # TODO: override this with a better init strategy for ELMs
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        self.initializer_fn(self.weight)

        if self.bias is not None:
            if self.initializer_str == "orthogonal":
                self.bias.unsqueeze_(0)
                self.initializer_fn(self.bias)
                self.bias.squeeze_()
            else:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)  # noqa: SLF001
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        """
        Forward pass through the layer.

        Parameters
        ----------
        input : torch.Tensor
            The input to make the prediction

        Returns
        -------
        torch.Tensor
            The forward prediction
        """
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        """
        Get the string representation of the network.

        Returns
        -------
        str
            The string representation of the encoder
        """
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class ELMLinear(nn.Module):
    """
    ELMLinear output layer for an ELM network.

    Parameters
    ----------
    in_features : int
        Number of encoder input features.
    out_features : int
        Number of encoder output features.
    elm_learning_config : dict | None, optional
        The config dictionary to pass to the learning function, default is :data:`python:None`.
    device : str | torch.device | None, optional
        The device to create the network on, default is :data:`python:None`.
    dtype : torch.dtype | None, optional
        The datatype to use for the encoder, default is :data:`python:None`.
    """

    in_features: int
    """The input feature size for the network."""
    out_features: int
    """The output feature size for the network."""
    elm_learning_config: dict
    """The configuration dictionary to pass to the learning function."""
    weight: nn.Parameter
    """The network weights."""

    __constants__ = ["in_features", "out_features"]  # noqa: RUF012

    def __init__(
        self,
        in_features: int,
        out_features: int,
        elm_learning_config: dict | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.elm_learning_config = elm_learning_config or {}
        self.in_features = in_features
        self.out_features = out_features
        # Gradients should not flow through ELM output weights
        self.weight = nn.Parameter(torch.zeros((out_features, in_features), **factory_kwargs), requires_grad=False)  # type: ignore[call-overload]
        self._a_inv: torch.Tensor | None = None

    def set_parameters(self, weights: torch.Tensor) -> None:
        """
        Set the parameters given by weights.

        Parameters
        ----------
        weights : torch.Tensor
            The new weight parameters
        """
        self.weight.data.copy_(weights.data.detach())

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        """
        Forward pass through the layer.

        Parameters
        ----------
        input : torch.Tensor
            The input to make the prediction

        Returns
        -------
        torch.Tensor
            The forward prediction
        """
        return F.linear(input, self.weight)

    def extra_repr(self) -> str:
        """
        Get the string representation of the network.

        Returns
        -------
        str
            The string representation of the encoder
        """
        return f"in_features={self.in_features}, out_features={self.out_features}"

    def learn_weights(self, preds: torch.Tensor, targets: torch.Tensor, weight_matrix: torch.Tensor | None = None) -> None:
        """
        Learn the output weights of the ELM using ``targets``.

        Parameters
        ----------
        preds : torch.Tensor
            The predictions to use for learning, :data:`python:None`. If None then the value stored in the last hidden layers
            ``_last_output`` buffer is used.
        targets : torch.Tensor
            The learning targets
        weight_matrix : torch.Tensor | None, optional
            The weighting matrix to pass to the learning function, :data:`python:None`
        """
        preds = self._last_output if preds is None else preds
        new_weights, self._a_inv = torch_calc_regularized_elm_weights(
            hidden_outs=preds,
            targets=targets,
            current_elm_weights=self.weight.data,
            a_inv=self._a_inv,
            weight_matrix=weight_matrix,
            **self.elm_learning_config,
        )
        self.set_parameters(new_weights.transpose(1, 0))
