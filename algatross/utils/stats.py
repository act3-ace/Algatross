"""Statistial functionality utils."""

from collections.abc import Callable
from math import pi, sqrt
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

import torch

from scipy.stats import norm

from algatross.utils.merge_dicts import flatten_dicts

if TYPE_CHECKING:
    from types import ModuleType


def calc_grad_norm(optimizer: torch.optim.Optimizer) -> torch.Tensor:
    """
    Calculate the norm of the gradient for the parameters controlled by this optimizer.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer containing the parameters.

    Returns
    -------
    torch.Tensor
        The global norm of the gradient.
    """
    global_norm: torch.Tensor = 0.0  # type: ignore[assignment]
    with torch.no_grad():
        for group in optimizer.param_groups:
            for params in group["params"]:
                if params is not None:
                    global_norm += params.norm()
    return global_norm


def torch_percentile(
    a: torch.Tensor,
    q: torch.Tensor,
    dim: torch.Tensor | None = None,
    keepdim: bool = False,
    *,
    interpolation: Literal["nearest", "linear", "lower", "higher", "midpoint"] = "linear",
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Get the percentiles of a dataset.

    A pytorch equivalent of NumPys :func:`~numpy.percentile` function.

    Parameters
    ----------
    a : torch.Tensor
        The tensor containing the data points.
    q : torch.Tensor
        The percentiles to return.
    dim : torch.Tensor | None, optional
        The dimension along which to take the percentile, :data:`python:None`.
    keepdim : bool, optional
        Whether to keep the original dimensions, :data:`python:False`.
    interpolation : Literal["nearest", "linear", "lower", "higher", "midpoint"], optional
        The type of interpolation to use, by default "linear".
    out : torch.Tensor | None, optional
        The tensor to store the outputs, :data:`python:None`.

    Returns
    -------
    torch.Tensor
        The desired percentiles from ``a``.
    """
    if dim is None:
        results = _flat_percentile(a=a.view(-1), q=q, keepdim=keepdim, interpolation=interpolation)
    else:
        results = []  # type: ignore[assignment]
        permuted = list(range(len(a.shape)))
        permuted.pop(dim)
        for row in a.permute(dim, *permuted):  # type: ignore[call-overload]
            result = _flat_percentile(a=row, q=q, keepdim=keepdim, interpolation=interpolation)
            results.append(result)  # type: ignore[attr-defined]
        results = torch.stack(results, dim=dim)  # type: ignore[arg-type]
        if keepdim:
            results.unsqueeze_(dim)  # type: ignore[arg-type]
    if out is not None:
        out.data.copy_(results)
    return results


def summarize(results: dict, custom_metrics: dict[str, Callable[..., np.ndarray]] | None = None, **kwargs) -> dict[str, Any]:
    """Summarize the values in ``results``.

    Parameters
    ----------
    results : dict
        A dictionary of numeric values or array-like values to summarize.
    custom_metrics : dict[str, Callable[..., np.ndarray]], optional
        Additional metrics to report, defaults to :data:`python:None`.
    `**kwargs`
        Keyword arguments.

    Returns
    -------
    dict[str, Any]
        The summary of ``results``.
    """
    result = {}
    for k, v in flatten_dicts(results).items():
        if isinstance(v, str) or (isinstance(v, np.ndarray) and v.dtype == np.uint8):
            result[k] = v
            continue
        v_arr = np.atleast_1d(v)
        if np.prod(v_arr.shape) > 1:
            result[f"{k}_mean"] = v_arr.mean(axis=0)
            result[f"{k}_max"] = v_arr.max(axis=0)
            result[f"{k}_min"] = v_arr.min(axis=0)
            if custom_metrics:
                for metric, metric_fn in custom_metrics.items():
                    result[f"{k}_{metric}"] = metric_fn(v_arr)
        else:
            result[k] = v_arr.item()
    return result


def _flat_percentile(
    a: torch.Tensor,
    q: torch.Tensor,
    keepdim: bool = False,
    *,
    interpolation: Literal["nearest", "linear", "lower", "higher", "midpoint"] = "linear",
) -> torch.Tensor:
    factory_kwargs = {"device": a.device, "dtype": a.dtype}
    a_flat_sorted = a.view(-1).sort().values
    a_numel = a.numel()
    if interpolation == "nearest":
        ks = [1 + round(float(qq / 100.0) * (a_numel - 1)) for qq in q]
        results = torch.tensor([a_flat_sorted[k].item() for k in ks], **factory_kwargs)  # type: ignore[arg-type]
    else:
        lowers = [1 + int(float(qq / 100.0) * (a_numel - 1)) for qq in q]
        uppers = [lower + 1 for lower in lowers]
        l_vals = [a_flat_sorted[lower].item() for lower in lowers]
        u_vals = [a_flat_sorted[upper].item() for upper in uppers]
        if interpolation == "linear":
            l_pcts = [lower / a_numel for lower in lowers]
            u_pcts = [upper / a_numel for upper in uppers]
            results = torch.tensor(
                [
                    l_val + (u_val - l_val) * (qq - l_pct) / (u_pct - l_pct)
                    for qq, l_pct, u_pct, l_val, u_val in zip(q, l_pcts, u_pcts, l_vals, u_vals, strict=True)
                ],
                **factory_kwargs,  # type: ignore[arg-type]
            )
        if interpolation == "midpoint":
            results = torch.tensor(
                [l_val + (u_val - l_val) * 0.5 for l_val, u_val in zip(l_vals, u_vals, strict=True)],
                **factory_kwargs,  # type: ignore[arg-type]
            )
        if interpolation == "lower":
            results = torch.tensor(list(l_vals), **factory_kwargs)  # type: ignore[arg-type]
        if interpolation == "upper":
            results = torch.tensor(list(u_vals), **factory_kwargs)
    if keepdim:
        for _ in range(a.ndim - 1):
            results.unsqueeze_()  # type: ignore[call-arg]
    return results


def interquantile_range(
    data: torch.Tensor | np.ndarray,
    range_begin: float = 1.0 / 14.0,
    range_end: float = 13.0 / 14.0,
) -> torch.Tensor | np.ndarray:
    r"""
    Estimate the interquantile range.

    Generates the range between the given quantiles using:

    .. math::

        IQR = invNorm(range\_end) - invNorm(range\_begin)

    Parameters
    ----------
    data : torch.Tensor | np.ndarray
        The sample data.
    range_begin : float, optional
        The lower quantile, by default 1/14.
    range_end : float, optional
        The upper quantile, by default 13/14.

    Returns
    -------
    torch.Tensor | np.ndarray
        The distance between the two quantiles.
    """
    factory_kwargs = {"dtype": data.dtype}
    if isinstance(data, np.ndarray):
        percentile_fn = np.percentile
        factory = np.array
    else:
        percentile_fn = torch_percentile  # type: ignore[assignment]
        factory_kwargs["device"] = data.device
        factory = torch.tensor  # type: ignore[assignment]
    symmetric = range_begin == 1 - range_end
    iqr = (2.0 * norm().ppf(range_end) if symmetric else norm().ppf(range_end) - norm().ppf(range_begin)).item()
    q_a, q_b = percentile_fn(data, factory([100 * range_begin, 100 * range_end], **factory_kwargs))  # type: ignore[arg-type]
    return (q_b - q_a) / iqr


def mean_difference(data: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    r"""
    Estimate the mean difference of a distribution of ``n_samples`` samples having ``std`` standard deviation.

    Von Andraes method from David H.A., “Early sample measures of variability, Statistical Science”, Vol.13, No.4, pp.368-377, 1998:

    .. math::

        std(G) = \frac{2\sigma}{\left [ n(n-1)\pi \right ]^{1/2}} \cdot \left [ n \left ( \frac{1}{3}\pi + 2\sqrt{3} - 4 \right ) + \left ( 6 - 4\sqrt{3} + \frac{1}{3} \pi \right ) \right ] ^ {1/2}

    Parameters
    ----------
    data : torch.Tensor | np.ndarray
        The sample data.

    Returns
    -------
    torch.Tensor | np.ndarray
        The estimated mean deviation of the samples.
    """  # noqa: E501
    factory_kwargs = {"dtype": data.dtype}
    if isinstance(data, np.ndarray):
        factory = np.array
    else:
        factory_kwargs["device"] = data.device
        factory = torch.tensor  # type: ignore[assignment]
    n_samples = data.shape[0]
    x_1 = 2 * data.std() / factory(np.sqrt(n_samples * (n_samples - 1) * np.pi), **factory_kwargs)  # type: ignore[arg-type]
    x_2 = n_samples * factory(np.pi / 3 + 2 * np.sqrt(3) - 4, **factory_kwargs)  # type: ignore[arg-type]
    x_3 = factory(6 - 4 * np.sqrt(3) + np.pi / 3, **factory_kwargs)  # type: ignore[arg-type]
    return x_1 * (x_2 + x_3).sqrt()


def mean_absolute_deviation(data: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    r"""
    Estimate the mean absolute deviation of ``n_samples`` samples having ``std`` standard deviation.

    From David H.A., “Early sample measures of variability, Statistical Science”, Vol.13, No.4, pp.368-377, 1998:

    .. math::

        E(D) = \sigma \sqrt{\frac{n-1}{n}}\cdot\sqrt{\frac{2}{\pi}}

    Parameters
    ----------
    data : torch.Tensor | np.ndarray
        The sample data.

    Returns
    -------
    torch.Tensor | np.ndarray
        The estimated mean deviation of the samples.
    """
    factory_kwargs = {"dtype": data.dtype}
    if isinstance(data, np.ndarray):
        return data.std() * np.array(sqrt((data.shape[0] - 1) / data.shape[0]) * sqrt(2.0 / pi), **factory_kwargs)  # type: ignore[arg-type]
    factory_kwargs["device"] = data.device
    return torch.tensor(torch_mean_absolute_deviation(data), **factory_kwargs)  # type: ignore[arg-type]


def torch_mean_absolute_deviation(data: torch.Tensor) -> torch.Tensor:
    r"""
    Estimate the mean absolute deviation of ``n_samples`` samples having ``std`` standard deviation.

    From David H.A., “Early sample measures of variability, Statistical Science”, Vol.13, No.4, pp.368-377, 1998:

    .. math::

        E(D) = \sigma \sqrt{\frac{n-1}{n}}\cdot\sqrt{\frac{2}{\pi}}

    Parameters
    ----------
    data : torch.Tensor
        The sample data.

    Returns
    -------
    torch.Tensor
        The estimated mean deviation of the samples.
    """
    return data.std() * sqrt((data.shape[0] - 1) / data.shape[0]) * sqrt(2.0 / pi)


VAREST_METHODS: dict[str, Callable[[torch.Tensor | np.ndarray], torch.Tensor | np.ndarray]] = {
    "iqr": interquantile_range,
    "mean_difference": mean_difference,
    "mean_absolute_deviation": mean_absolute_deviation,
}


def linear_weight_interpolation(s_hat: torch.Tensor | np.ndarray, c_1: float = 2.5, c_2: float = 3.0) -> torch.Tensor | np.ndarray:
    """
    Interpolate ``s_hat`` between ``c_1``, ``c_2`` linearly.

    Parameters
    ----------
    s_hat : torch.Tensor | np.ndarray
        The estimate for standard deviation or variance.
    c_1 : float, optional
        The interpolation lower bound, by default 2.5.
    c_2 : float, optional
        The interpolation upper bound, by default 3.0.

    Returns
    -------
    torch.Tensor | np.ndarray
        ``s_hat`` interpolated between ``c_1`` and ``c_2``.
    """
    factory_kwargs = {"dtype": s_hat.dtype}
    if isinstance(s_hat, torch.Tensor):
        backend_module: ModuleType = torch
        factory_kwargs["device"] = s_hat.device
        factory = torch.tensor
    elif isinstance(s_hat, np.ndarray):
        backend_module = np
        factory = np.array  # type: ignore[assignment]
    return backend_module.where(
        s_hat <= c_1,
        factory(1, **factory_kwargs),  # type: ignore[arg-type]
        backend_module.where(
            backend_module.logical_and(c_1 < s_hat, s_hat < c_2),
            (c_2 - s_hat) / (c_2 - c_1),
            factory(0.0, **factory_kwargs),  # type: ignore[arg-type]
        ),
    )


def tanh_weight_interpolation(
    s_hat: torch.Tensor | np.ndarray,
    c: Literal["3.0", "4.0", "5.0", "6.0"] = "6.0",
    k: Literal["4.0", "4.5", "5.0"] = "4.5",
) -> torch.Tensor | np.ndarray:
    """
    Smoothly interpolate s_hat using the ``tanh`` formulation of a redescending M-estimator.

    Parameters
    ----------
    s_hat : torch.Tensor | np.ndarray
        The std estimation.
    c : Literal["3.0", "4.0", "5.0", "6.0"], optional
        The c coefficient for the tanh estimator, by default 6.0.
    k : Literal["4.0", "4.5", "5.0"], optional
        The k coefficient for the tanh estimator, by default 4.5.

    Returns
    -------
    torch.Tensor | np.ndarray
        The value of :python:`s_hat` smoothly interpolated using tanh.
    """
    factory_kwargs = {"dtype": s_hat.dtype}
    if isinstance(s_hat, torch.Tensor):
        backend_module: ModuleType = torch
        factory_kwargs["device"] = s_hat.device
        factory = torch.tensor
    elif isinstance(s_hat, np.ndarray):
        backend_module = np
        factory = np.array  # type: ignore[assignment]
    c_str = f"{c:0.1f}"
    k_str = f"{k:0.1f}"
    c_float = float(c_str)
    k_float = float(k_str)
    a = TANH_INTERPOLATION_TABLE[c_str][k_str]["A"]
    b = TANH_INTERPOLATION_TABLE[c_str][k_str]["B"]
    p = TANH_INTERPOLATION_TABLE[c_str][k_str]["p"]
    tanh_factor = np.sqrt(a * (k_float - 1)) * np.tanh(0.5 * np.sqrt((k_float - 1) * b * b / a) * (c_float - s_hat))
    return backend_module.where(
        s_hat <= p,
        factory(1, **factory_kwargs),  # type: ignore[arg-type]
        backend_module.where(
            backend_module.logical_and(p < s_hat, s_hat < c),
            tanh_factor,
            factory(0.0, **factory_kwargs),  # type: ignore[arg-type]
        ),
    )


def step_weight_interpolation(s_hat: torch.Tensor | np.ndarray, c_1: float = 2.5) -> torch.Tensor | np.ndarray:
    """
    Interpolate :python:`s_hat` estimate using a step funciton.

    Parameters
    ----------
    s_hat : torch.Tensor | np.ndarray
        The std estimation.
    c_1 : float, optional
        The cutoff for :python:`s_hat`, by default 2.5.

    Returns
    -------
    torch.Tensor | np.ndarray
        The interpolated weights.
    """
    factory_kwargs = {"dtype": s_hat.dtype}
    if isinstance(s_hat, torch.Tensor):
        backend_module: ModuleType = torch
        factory_kwargs["device"] = s_hat.device
        factory = torch.tensor
    elif isinstance(s_hat, np.ndarray):
        backend_module = np
        factory = np.array  # type: ignore[assignment]
    return backend_module.where(
        s_hat <= c_1,
        factory(1, **factory_kwargs),  # type: ignore[arg-type]
        factory(0.0, **factory_kwargs),  # type: ignore[arg-type]
    )


TANH_INTERPOLATION_TABLE = {
    "3.0": {
        "4.0": {"A": 0.493810, "B": 0.628945, "p": 1.096215},
        "4.5": {"A": 0.604251, "B": 0.713572, "p": 1.304307},
        "5.0": {"A": 0.680593, "B": 0.769313, "p": 1.470089},
    },
    "4.0": {
        "4.0": {"A": 0.725616, "B": 0.824330, "p": 1.435830},
        "4.5": {"A": 0.804598, "B": 0.877210, "p": 1.634416},
        "5.0": {"A": 0.857044, "B": 0.911135, "p": 1.803134},
    },
    "5.0": {
        "4.0": {"A": 0.782111, "B": 0.867433, "p": 1.523457},
        "4.5": {"A": 0.849105, "B": 0.910228, "p": 1.715952},
        "5.0": {"A": 0.893243, "B": 0.937508, "p": 1.882458},
    },
    "6.0": {
        "4.0": {"A": 0.793552, "B": 0.875862, "p": 1.541383},
        "4.5": {"A": 0.857058, "B": 0.915911, "p": 1.730683},
        "5.0": {"A": 0.899024, "B": 0.941556, "p": 1.895246},
    },
}

LMS_WEIGHT_INTERPOLATION_METHODS: dict[str, Callable[[torch.Tensor | np.ndarray], torch.Tensor | np.ndarray]] = {
    "linear": linear_weight_interpolation,
    "step": step_weight_interpolation,
    "tanh": tanh_weight_interpolation,
}
