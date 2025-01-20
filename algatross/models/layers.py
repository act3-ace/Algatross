"""Neural network model layers."""

import torch

from torch import nn


class BiasLayer(nn.Module):
    """
    BiasLayer a free-floating bias layer which adds a state-independant log std dev to a network prediction for each network output.

    Parameters
    ----------
    num_bias_vars : int
        The number of variables representing the distribution bias to add to the output. In other words, the number of times the
        bias should be repeated before concatenating to the input
    `*args`
        Additional positional arguments.
    `**kwargs`
        Additional keyword arguments.
    """

    def __init__(self, num_bias_vars: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.log_std = nn.Parameter(torch.as_tensor([0.0] * num_bias_vars))
        self.register_parameter("log_std", self.log_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward append a log std to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor, usually the mean action predictions output by an actor network.

        Returns
        -------
        torch.Tensor
            The input tensor with a single log_std duplicated for each value in ``x``. This log std does not depend on the state in any way.
        """
        if x.ndim == 2:  # noqa: PLR2004
            return torch.cat([x, self.log_std.unsqueeze(0).repeat([len(x), 1])], dim=1)
        return torch.cat([x, self.log_std])[None]
