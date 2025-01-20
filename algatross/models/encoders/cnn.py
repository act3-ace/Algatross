"""Collection of neural network models to be used as components of the evolutionary process."""

from collections.abc import Sequence
from typing import Literal

import numpy as np

import torch

from torch import nn

from algatross.models.encoders.base import BaseEncoder
from algatross.utils.types import CNNLayerType


def _cast_output(*output: tuple[int, ...], cast: list | None = None) -> tuple[int, int] | tuple[int, int, int] | tuple[int, int, int, int]:
    """
    _cast_output casts the output to a flat tuple of ints recursively.

    Parameters
    ----------
    *output : tuple[int, ...]
        The output to cast to a ints
    cast : list | None, optional
        The container of already-int-casted inputs, :data:`python:None`

    Returns
    -------
    tuple[int, int] | tuple[int, int, int] | tuple[int, int, int, int]
        The flattened and int-casted inputs
    """
    cast = [] if cast is None else cast
    for o in output:
        if len(o_arr := np.atleast_1d(o)) == 1:
            cast.append(int(o_arr[0]))
        else:
            cast.extend(_cast_output(o, cast=cast))
    return tuple(cast)


def _get_padding(layer: nn.Module, name: Literal["padding", "output_padding"], n_dim: int) -> np.ndarray | None:
    """
    Get the padding from the layer and does any reshaping so that a single value for each layer dimension is returns.

    If the padding is an int or singular value then it is doubled. Otherwise the size of the padding spec must be divisible by
    two (two padding sides per each input dimension).

    After determining the padding for each dimension these values are summed so there is a single value per dimension. This can
    then be used to calculate the dimensions of the output of CNN layers.

    Parameters
    ----------
    layer : nn.Module
        The layer containing the padding attribute
    name : Literal["padding", "output_padding"]
        The name of the padding attribute to return.
    n_dim : int
        Dimensionality of the CNN (1d, 2d, 3d).

    Returns
    -------
    np.ndarray | None
        The padding for each dimension of the CNN input.

    Raises
    ------
    RuntimeError
        _description_
    """
    if hasattr(layer, name):
        padding = np.asarray(getattr(layer, name)).flatten()
        if len(padding) == 1:
            padding = padding.repeat(2 * n_dim)
        if len(padding) % 2:
            msg = f"{name} length must be divisible by 2, got {padding}"
            raise RuntimeError(msg)
        padding = np.pad(padding, pad_width=(0, len(padding) - 2 * n_dim)).reshape((n_dim, 2))
        return padding.sum(axis=-1)
    return None


def _maybe_get_attr_array(obj: object, name: str) -> np.ndarray | None:
    """
    _maybe_get_attr_array get an attribute array from the object if it has the given named attribute.

    Parameters
    ----------
    obj : object
        The object mabe containing the attribute
    name : str
        The name of the attribute to retrieve.

    Returns
    -------
    np.ndarray | None
        The value of the attribute mapped into a 1D array, returns None if the object doesn't have the attribute.
    """
    return np.atleast_1d(np.asarray(getattr(obj, name))).flatten() if hasattr(obj, name) else None


def calc_cnn_out(layer: CNNLayerType, in_shape: Sequence[int]) -> tuple[int, int] | tuple[int, int, int] | tuple[int, int, int, int]:
    """
    Calc_cnn_out calculates the output shape of the CNN layer.

    Parameters
    ----------
    layer : CNNLayerType
        The layer for which we are trying to calculate the output shape.
    in_shape : Sequence[int]
        The input shape to the layer

    Returns
    -------
    tuple[int, int] | tuple[int, int, int] | tuple[int, int, int, int]
        The output shape:
            [C, L] for 1D CNN layers
            [C, H, W] for 2D CNN layers
            [C, D, H, W] for 3D CNN layers

    Raises
    ------
    TypeError
        If the layer output calculation can't be determined for the given layer type.
    """
    in_channels = np.asarray(in_shape[0])
    window = np.atleast_1d(np.asarray([*in_shape[1:]])).flatten()
    n_dim = len(in_shape) - 1

    out_channels = _maybe_get_attr_array(layer, "out_channels")
    stride = _maybe_get_attr_array(layer, "stride")
    kernel = _maybe_get_attr_array(layer, "kernel_size")
    dilation = _maybe_get_attr_array(layer, "dilation")
    output_size = _maybe_get_attr_array(layer, "output_size")

    if kernel is not None and len(kernel) == 1:
        kernel = kernel.repeat(n_dim)
    if dilation is not None and len(dilation) == 1:
        dilation = dilation.repeat(n_dim)
    if stride is not None and len(stride) == 1:
        stride = stride.repeat(n_dim)

    padding = _get_padding(layer, "padding", n_dim)
    output_padding = _get_padding(layer, "output_padding", n_dim)

    if isinstance(layer, nn.Conv1d | nn.MaxPool1d | nn.Conv2d | nn.MaxPool2d | nn.Conv3d | nn.MaxPool3d):
        pad_term = padding
        kd_term = -(kernel if dilation is None else dilation * (kernel - 1)) - 1
        return _cast_output(
            out_channels,  # type: ignore[arg-type]
            *np.floor((window + pad_term + kd_term) / stride + 1),
        )
    if isinstance(layer, nn.AvgPool1d | nn.AvgPool2d | nn.AvgPool3d):
        pad_term = padding
        kd_term = -kernel
        return _cast_output(in_channels, *np.floor((window + pad_term + kd_term) / stride + 1))  # type: ignore[arg-type]
    if isinstance(layer, nn.MaxUnpool1d | nn.MaxUnpool2d | nn.MaxUnpool3d | nn.ConvTranspose1d | nn.ConvTranspose2d | nn.ConvTranspose3d):
        pad_term = -padding + (0 if output_padding is None else output_padding + 1)
        kd_term = kernel if dilation is None else (kernel - 1) * dilation
        return _cast_output(in_channels if out_channels is None else out_channels, *((window - 1) * stride + pad_term + kd_term))  # type: ignore[arg-type]
    if isinstance(
        layer,
        nn.AdaptiveAvgPool3d
        | nn.AdaptiveMaxPool3d
        | nn.AdaptiveAvgPool3d
        | nn.AdaptiveMaxPool3d
        | nn.AdaptiveAvgPool3d
        | nn.AdaptiveMaxPool3d,
    ):
        return _cast_output(in_channels, *output_size)  # type: ignore[arg-type]
    if isinstance(layer, nn.BatchNorm1d | nn.InstanceNorm1d | nn.BatchNorm2d | nn.InstanceNorm2d | nn.BatchNorm3d | nn.InstanceNorm3d):
        return _cast_output(in_channels, *window)  # type: ignore[arg-type]
    if isinstance(
        layer,
        nn.ConstantPad1d
        | nn.ReflectionPad1d
        | nn.ReplicationPad1d
        | nn.ConstantPad2d
        | nn.ZeroPad2d
        | nn.ReflectionPad2d
        | nn.ReplicationPad2d
        | nn.ConstantPad3d
        | nn.ReflectionPad3d
        | nn.ReplicationPad3d,
    ):
        return _cast_output(in_channels, *(padding + window))  # type: ignore[arg-type]
    msg = f"Could not determine an output shape for layer of type {type(layer)}"
    raise TypeError(msg)


class CNN1DEncoder(BaseEncoder):
    """
    CNN1DEncoder a 1D CNN Encoder/Decoder architecture.

    Parameters
    ----------
    channels_in : int | None, optional
        The input channel width, default is :data:`python:None`,
    sequence_length : int | None, optional
        The sequence length, default is :data:`python:None`.
    embedding_dim : int, optional
        The dimensionality of the embedding default is 20
    cnn_layers_per_stack : int, optional
        The number of CNN layers per stack of intermediate modules, default is 3.
    cnn_stacks : int, optional
        The total number of CNN stacks, default is 3
    optimizer_class : type[torch.optim.Optimizer], optional
        The class of the optimizer to use to train the encoder, default is :class:`~torch.optim.Adam`
    optimizer_kwargs : dict | None, optional
        The keyword arguments to pass to the optimizer constructor.
    `*args`
        Additional positional arguments.
    sample_input : torch.Tensor | None, optional
        Sample input to use to infer some of the configuration values, default is :data:`python:None`.
    `**kwargs`
        Additional keyword arguments
    """

    embedding_dim: int
    """The dimensionality of the encoding."""

    def __init__(
        self,
        channels_in: int | None = None,
        sequence_length: int | None = None,
        embedding_dim: int = 20,
        cnn_layers_per_stack: int = 3,
        cnn_stacks: int = 3,
        optimizer_class: type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: dict | None = None,
        *args,
        sample_input: torch.Tensor | None = None,
        **kwargs,
    ):
        super().__init__(optimizer_class=optimizer_class, optimizer_kwargs=optimizer_kwargs)
        if sample_input is not None:
            kwargs = self._infer_kwargs_from_sample_input(sample_input, **kwargs)
            channels_in = kwargs.get("channels_in", channels_in)
            sequence_length = kwargs.get("sequence_length", sequence_length)
        self.embedding_dim = embedding_dim

        cnn_module, cnn_module_rev, info = self._build_cnn_layers(
            channels_in=channels_in,
            sequence_length=sequence_length,
            cnn_layers_per_stack=cnn_layers_per_stack,
            cnn_stacks=cnn_stacks,
        )

        channels_in, channels_out, sequence_length = (info[key] for key in ["channels_in", "channels_out", "sequence_length"])

        self.linear_input_shape = channels_in * sequence_length
        self.cnn_output_shape = channels_out

        linear_module, linear_module_rev = self._build_linear_layers(channels_in=channels_in, sequence_length=sequence_length)

        self.encoder = nn.ModuleList([cnn_module, linear_module]).double()
        self.decoder = nn.ModuleList([linear_module_rev, cnn_module_rev]).double()

        # Register hooks to convert to double if we were given something else
        self.encoder[0].register_forward_pre_hook(_convert_input_dtype)
        self.decoder[0].register_forward_pre_hook(_convert_input_dtype)
        self.encoder[0].register_forward_hook(_convert_output_dtype)
        self.decoder[0].register_forward_hook(_convert_output_dtype)

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
        """
        out = self.encoder[0](x)
        return self.encoder[1](out)

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
        out = self.decoder[0](x)
        return self.decoder[1](out)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward conduct a forward pass through the encoded and then the decoder.

        Parameters
        ----------
        x : torch.Tensor
            The input to be reconstructed.
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        torch.Tensor
            The reconstructed input after being encoded.
        """
        if x.ndim > 3:  # noqa: PLR2004
            batch_shape = x.shape[:2]
            data_shape = x.shape[2:]
        else:
            batch_shape = x.shape[:1]
            data_shape = x.shape[1:]
        return self.decode(self.encode(x.reshape(-1, *data_shape))).reshape(*batch_shape, -1)

    def _build_cnn_layers(  # noqa: PLR6301
        self,
        channels_in: int,
        sequence_length: int,
        cnn_layers_per_stack: int,
        cnn_stacks: int,
    ) -> tuple[nn.Module, nn.Module, dict[str, int]]:
        channels_out = channels_in

        decay_scale = cnn_layers_per_stack * cnn_stacks

        def decay_channels_in(c_in, i=0, j=0):
            return max(int(np.ceil(c_in * np.exp(((i * j) / decay_scale - 1) / np.sqrt(decay_scale)) - 1)), 1)

        # Final decoder layers try to infer higher dimensions and make a forward CNN prediction on the imagined representation
        reversed_channels = int(np.ceil(channels_in * channels_in / decay_channels_in(channels_in)))
        final_decoder_layer = [
            nn.ConvTranspose1d(channels_in, reversed_channels, np.clip(channels_in, 1, 5, dtype=np.int32), padding=0),
            nn.Dropout1d(0.05),
            nn.Tanh(),
            nn.Conv1d(reversed_channels, channels_in, np.clip(channels_in, 1, 5, dtype=np.int32), padding=0),
        ]

        cnn_modules = []
        cnn_modules_rev = [nn.Sequential(*final_decoder_layer)]
        for _j in range(3):
            c_layers = []
            c_layers_rev = []
            for _i in range(3):
                channels_in = channels_out
                channels_out = decay_channels_in(channels_in, i=_i, j=_j)
                kernel = np.clip(channels_out, 1, 5, dtype=np.int32)

                layer = nn.Conv1d(channels_in, channels_out, kernel, padding=2)
                layer_rev = nn.ConvTranspose1d(channels_out, channels_in, kernel, padding=2)

                nn.init.kaiming_normal_(layer.weight, nonlinearity="leaky_relu", a=0.01)
                nn.init.kaiming_normal_(layer_rev.weight, nonlinearity="leaky_relu", a=0.01)

                l_list = [layer, nn.Dropout1d(0.05), nn.LeakyReLU()]
                l_list_rev = [nn.LeakyReLU(), nn.Dropout1d(0.05), layer_rev]

                channels_in, sequence_length = calc_cnn_out(layer, [channels_in, sequence_length])  # type: ignore[misc]

                c_layers.extend(l_list)
                c_layers_rev.extend(l_list_rev)
            c_layers.extend([nn.AvgPool1d(3, stride=1, padding=1)])
            channels_in, sequence_length = calc_cnn_out(c_layers[-1], [channels_in, sequence_length])  # type: ignore[misc, arg-type]
            cnn_modules.append(nn.Sequential(*c_layers))
            cnn_modules_rev.append(nn.Sequential(*c_layers_rev[::-1]))
        return (
            nn.Sequential(*cnn_modules),
            nn.Sequential(*cnn_modules_rev[::-1]),
            {"channels_in": channels_in, "channels_out": channels_out, "sequence_length": sequence_length},
        )

    def _build_linear_layers(self, channels_in: int, sequence_length: int) -> tuple[nn.Module, nn.Module]:
        layer_in = self.linear_input_shape
        layer_out = max(1, int(0.75 * layer_in))
        layers: list[nn.Module] = [nn.Flatten()]
        layers_rev: list[nn.Module] = [nn.Unflatten(1, (channels_in, sequence_length))]
        while layer_in > self.embedding_dim:
            layer_out = max(self.embedding_dim, int(0.75 * layer_in))
            layer = nn.Linear(layer_in, layer_out)
            layer_rev = nn.Linear(layer_out, layer_in)

            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.constant_(layer.bias, torch.rand(1).item())
            nn.init.kaiming_normal_(layer_rev.weight, nonlinearity="relu")
            nn.init.constant_(layer_rev.bias, torch.rand(1).item())

            layers.extend([layer, nn.Dropout(0.05), nn.ReLU() if max(1, int(0.75 * layer_in)) > self.embedding_dim else nn.Tanh()])
            layers_rev.extend([nn.ReLU(), nn.Dropout(0.05), layer_rev])
            layer_in = layer_out
            layer_out = max(1, int(0.75 * layer_in))
        return nn.Sequential(*layers), nn.Sequential(*layers_rev[::-1])

    def _infer_kwargs_from_sample_input(self, sample_input: torch.Tensor, **kwargs) -> dict:  # noqa: PLR6301
        kwargs["channels_in"] = kwargs.get("channels_in", sample_input.shape[-2])
        kwargs["sequence_length"] = kwargs.get("sequence_length", sample_input.shape[-1])
        return kwargs

    def loss(self, data: torch.Tensor, targets: torch.Tensor, *args, **kwargs) -> torch.Tensor:  # noqa: PLR6301, D102
        return torch.nn.functional.mse_loss(data, targets)


def _convert_input_dtype(module: nn.Module, args: Sequence) -> Sequence:
    """
    _convert_input_dtype forward pre-hook to convert the input tensor to the correct dtype.

    Parameters
    ----------
    module : nn.Module
        The module being forward-called
    args : Sequence
        The arguments passed to the modules ``forward`` call.

    Returns
    -------
    Sequence
        The arguments cast to the correct dtype
    """
    module.__original_dtype = args[0].dtype  # noqa: SLF001
    if args[0].dtype != torch.double:
        module.__convert_dtype = True  # type: ignore[assignment] # noqa: SLF001
        args[0] = args[0].double()  # type: ignore[index]
    else:
        module.__convert_dtype = False  # type: ignore[assignment] # noqa: SLF001
    return args


def _convert_output_dtype(module: nn.Module, args: Sequence[torch.Tensor], output: torch.Tensor) -> torch.Tensor:  # noqa: ARG001
    """
    Complementary hook to _convert_input_dtype which converts the network output back into the original dtype.

    Parameters
    ----------
    module : nn.Module
        The module being forward-called.
    args : Sequence[torch.Tensor]
        The inputs to the modules ``forward`` method.
    output : torch.Tensor
        The output of the modules ``forward`` method.

    Returns
    -------
    torch.Tensor
        The output of the module cast to the original dtype.
    """
    if hasattr(module, "__convert_dtype") and module.__convert_dtype:  # noqa: SLF001
        output = output.to(dtype=module.__original_dtype)  # noqa: SLF001
    return output
