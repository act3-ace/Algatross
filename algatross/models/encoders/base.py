"""Base classes for encoders."""

from abc import ABC, abstractmethod

import torch

from torch import nn


class BaseEncoder(nn.Module, ABC):
    """
    A base class for encoders.

    Parameters
    ----------
    optimizer_class : type[torch.optim.Optimizer]
        The class of optimizer to use with this encoder
    optimizer_kwargs : dict | None, optional
        The keyword arguments to pass to the optimizers constructor, default is :data:`python:None`.
    `*args`
        Additional positional arguments.
    sample_input : torch.Tensor | None, optional
        A sample input to use for inferring properties, default is :data:`python:None`
    `**kwargs`
        Additional keyword arguments.
    """

    def __init__(
        self,
        optimizer_class: type[torch.optim.Optimizer],
        optimizer_kwargs: dict | None = None,
        *args,
        sample_input: torch.Tensor | None = None,  # noqa: ARG002
        **kwargs,
    ):
        super().__init__()
        self._optimizer_class = optimizer_class
        self._optimizer_kwargs = optimizer_kwargs or {}
        self._optimizer: torch.optim.Optimizer | None = None

    def reset_optimizer(self):
        """Re-initialize the optimizer."""
        self._optimizer = self._optimizer_class(self.parameters(), **self._optimizer_kwargs)

    @property
    def optimizer(self) -> torch.optim.Optimizer | None:  # noqa: D102
        if self._optimizer is None:
            self.reset_optimizer()
        return self._optimizer

    @optimizer.setter
    def optimizer(self, other: torch.optim.Optimizer):
        if not isinstance(other, torch.optim.Optimizer):
            msg = "Trying to set optimizer as a non-Optimizer class. Optimizers must subclass from `torch.optim.Optimizer`"
            raise TypeError(msg)
        self._optimizer = other

    @abstractmethod
    def encode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Encode get the encoding for the input.

        Parameters
        ----------
        x : torch.Tensor
            The input to encode
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        torch.Tensor
            The encoded input

        Raises
        ------
        NotImplementedError
            If subclasses don't override this method
        """
        msg = "Subclasses must override this method"
        raise NotImplementedError(msg)

    @abstractmethod
    def decode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Encode get the decoded input.

        Parameters
        ----------
        x : torch.Tensor
            The input to decode
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        torch.Tensor
            The decoded input
        """

    def _infer_kwargs_from_sample_input(self, sample_input: torch.Tensor, **kwargs):  # noqa: PLR6301
        """Infer keyword arguments to __init__ from the sample input and modify ``kwargs`` in place.

        Parameters
        ----------
        sample_input : torch.Tensor
            A sample tensor including the batch dimension

        Raises
        ------
        NotImplementedError
            If this is not overriden by a base class
        """
        msg = "Sub classes must override this method"
        raise NotImplementedError(msg)

    @abstractmethod
    def loss(self, data: torch.Tensor, targets: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Calculate the encoder loss from the data.

        Parameters
        ----------
        data : torch.Tensor
            The input data to be encoded and used for training.
        targets : torch.Tensor
            The encoding targets
        `*args`
            Additional positional arguments.
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        torch.Tensor
            The encoding loss
        """

    def fit(self, data: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Train the encoder on the data.

        Parameters
        ----------
        data : torch.Tensor
            The data to use for training the encoder.
        `*args`
            Additional positional arguments.
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        torch.Tensor
            The encoding loss
        """
        encoder_out = self(data)
        encoder_loss = self.loss(encoder_out, data, *args, **kwargs)
        self.optimizer.zero_grad(True)
        encoder_loss.backward()
        self.optimizer.step()
        return encoder_loss
