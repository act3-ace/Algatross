"""Principal Component Analysis based encoder(s)."""

from collections.abc import Sequence

import numpy as np

import torch

from torch import Tensor

from sklearn.decomposition import IncrementalPCA

from algatross.models.encoders.base import BaseEncoder


class PCAEncoder(BaseEncoder):
    """An encoder based on PCA.

    Uses scikit-learn's IncrementalPCA to learn an embedding.

    Parameters
    ----------
    n_components : int | None, optional
        The number of components to reduce the input to, default is :data:`python:None`.
    channels_in : int | None, optional
        The number of input channels, default is :data:`python:None`.
    sequence_length : int | None, optional
        The sequence (trajectory) length, default is :data:`python:None`.
    sample_input : Tensor | None, optional
        The tensor to use as a sample input for inferred dimensions, default is :data:`python:None`.
    `*args`
        Additional positional arguments.
    `**kwargs`
        Additional keyword arguments.
    """

    def __init__(
        self,
        n_components: int | None = None,
        channels_in: int | None = None,
        sequence_length: int | None = None,
        *args,
        sample_input: Tensor | None = None,
        **kwargs,
    ):
        if "optimizer_class" in kwargs:
            kwargs.pop("optimizer_class")
        if "optimizer_kwargs" in kwargs:
            kwargs.pop("optimizer_kwargs")

        if sample_input is not None:
            kwargs = self._infer_kwargs_from_sample_input(sample_input, **kwargs)
            channels_in = kwargs.get("channels_in", channels_in)
            sequence_length = kwargs.get("sequence_length", sequence_length)

        super().__init__(None, None, *args, sample_input=sample_input, **kwargs)
        self.encoder = IncrementalPCA(n_components=n_components)
        self.n_components = n_components
        self._channels_in = channels_in
        self._sequence_length = sequence_length

    def encode(self, x: Tensor | np.ndarray, **kwargs) -> Tensor:  # noqa: D102
        if isinstance(x, Tensor):
            x = x.detach()
        x, original_shape, reshape = self.flatten_extra_dims(x, **kwargs)
        x = self.encoder.transform(x)
        x = x.reshape((*original_shape[: -kwargs.get("sample_dim", 2)], -1, self.n_components)) if reshape else x
        if isinstance(x, Tensor):
            return x
        return torch.from_numpy(x)

    def decode(self, x: Tensor | np.ndarray, **kwargs) -> Tensor:  # noqa: D102
        if isinstance(x, Tensor):
            x = x.detach()
        x, original_shape, reshape = self.flatten_extra_dims(x, **kwargs)
        x = self.encoder.inverse_transform(x)
        x = x.reshape((*original_shape[: -kwargs.get("sample_dim", 2)], self._sequence_length, self._channels_in)) if reshape else x
        if isinstance(x, Tensor):
            return x
        return torch.from_numpy(x)

    def forward(self, x: Tensor | np.ndarray, **kwargs) -> Tensor:  # noqa: D102
        if isinstance(x, Tensor):
            x = x.detach()
        x, original_shape, reshape = self.flatten_extra_dims(x, **kwargs)
        x = self.decode(self.encode(x))
        return x.reshape(original_shape) if reshape else x

    def loss(self, data: Tensor, targets: Tensor, *args, **kwargs) -> Tensor:  # noqa: D102
        data, _, reshape = self.flatten_extra_dims(data, **kwargs)  # type: ignore[assignment]
        if reshape:
            targets = targets.reshape((-1, data.shape[-1]))
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        if isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets)
        return torch.nn.functional.mse_loss(data, targets)

    def fit(self, data: Tensor, *args, **kwargs) -> Tensor:  # noqa: D102
        if isinstance(data, Tensor):
            data = data.detach()
        data, _, _ = self.flatten_extra_dims(data, **kwargs)  # type: ignore[assignment]
        if data.shape[0] < self.n_components:
            return torch.zeros((1,), dtype=data.dtype)
        self.encoder.partial_fit(data)
        encoder_out = self(data)
        return self.loss(encoder_out, data)

    def _infer_kwargs_from_sample_input(self, sample_input: torch.Tensor, **kwargs):  # noqa: PLR6301, ANN202
        kwargs["channels_in"] = kwargs.get("channels_in", sample_input.shape[-1])
        kwargs["sequence_length"] = kwargs.get("sequence_length", sample_input.shape[-2])
        return kwargs

    @staticmethod
    def flatten_extra_dims(
        data: np.ndarray | torch.Tensor,
        sample_dim: int = 2,
        **kwargs,
    ) -> tuple[np.ndarray | torch.Tensor, Sequence[int], bool]:
        """Flatten the data so it is 2D, keeping the last ``sample_dim`` in the second dimension.

        Parameters
        ----------
        data : np.ndarray | torch.Tensor
            The data to (maybe) reshape
        sample_dim : int, optional
            The dimensionality of a single data sample, by default 2
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        np.ndarray | torch.Tensor
            The data (maybe) reshaped to be 2D
        Sequence[int]
            The original shape
        bool
            Flag indicating whether or not reshaping was done.
        """
        reshape = data.ndim > 2  # noqa: PLR2004
        original_shape = data.shape
        if reshape:
            data = data.reshape((-1, np.prod(data.shape[-sample_dim:])))  # type: ignore[arg-type]
        return data, original_shape, reshape
