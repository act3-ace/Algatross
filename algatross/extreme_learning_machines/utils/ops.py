"""Operations for working with extreme learning machines."""

import math

from cmath import sqrt as c_sqrt
from math import sqrt as r_sqrt
from typing import Literal

import numpy as np

import torch

from torch import nn

from algatross.utils.stats import LMS_WEIGHT_INTERPOLATION_METHODS, VAREST_METHODS


def calc_regularized_elm_weights(
    hidden_outs: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
    current_elm_weights: torch.Tensor | np.ndarray,
    risk_gamma: float = 1.0,
    a_inv: torch.Tensor | np.ndarray | None = None,
    previous_hidden_outs: torch.Tensor | None = None,
    weighted: bool = True,
    incremental: bool = True,
    varest_method: Literal["iqr", "mean_difference", "mean_absolute_deviation"] = "iqr",
    varest_config: dict | None = None,
    weighting_interpolation_method: Literal["linear", "tanh", "step"] = "linear",
    weighting_interpolation_config: dict | None = None,
    *,
    weight_matrix: torch.Tensor | np.ndarray | None = None,
) -> tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray]:
    """
    Calculate the ELM output layer weights using the regularized (potentially incremental) update algorithm.

    Parameters
    ----------
    hidden_outs : torch.Tensor | np.ndarray
        Current hidden layer output
    targets : torch.Tensor | np.ndarray
        Learning targets
    current_elm_weights : torch.Tensor | np.ndarray
        Current output layer weights
    risk_gamma : float, optional
        Empirical vs. structural risk trade-off parameter, by default 1.0
    a_inv : torch.Tensor | np.ndarray | None, optional
        Inverse matrix from a previous call, :data:`python:None`
    previous_hidden_outs : torch.Tensor | None, optional
        Previous hidden layer output
    weighted : bool, optional
        Whether to calculate the sample weight matrix, :data:`python:True`
    incremental : bool, optional
        Whether to use the incremental update algorithm, :data:`python:True`
    varest_method : Literal["iqr", "mean_difference", "mean_absolute_deviation"], optional
        Variance estimation method, by default "iqr"
    varest_config : dict | None, optional
        Variance estimation method config, :data:`python:None`
    weighting_interpolation_method : Literal["linear", "tanh", "step"], optional
        Weight matrix interpolation method, by default "linear"
    weighting_interpolation_config : dict | None, optional
        Weight matrix interpolation method config, :data:`python:None`
    weight_matrix : torch.Tensor | np.ndarray | None, optional
        Weight matrix if you want to supply your own instead of having one calculated, :data:`python:None`

    Returns
    -------
    torch.Tensor | np.ndarray
        The updated regularized ELM output layer weights
    torch.Tensor | np.ndarray
        The updated ``a_inv`` matrix
    """
    if isinstance(hidden_outs, torch.Tensor):
        return torch_calc_regularized_elm_weights(
            hidden_outs=hidden_outs,
            targets=targets,  # type: ignore[arg-type]
            current_elm_weights=current_elm_weights,  # type: ignore[arg-type]
            risk_gamma=risk_gamma,
            a_inv=a_inv,  # type: ignore[arg-type]
            previous_hidden_outs=previous_hidden_outs,
            weighted=weighted,
            weight_matrix=weight_matrix,  # type: ignore[arg-type]
            incremental=incremental,
            varest_method=varest_method,
            varest_config=varest_config,
            weighting_interpolation_method=weighting_interpolation_method,
            weighting_interpolation_config=weighting_interpolation_config,
        )

    first = a_inv is None
    factory_kwargs = {"dtype": hidden_outs.dtype}

    batch_size = hidden_outs.shape[0]
    hidden_nodes = hidden_outs.shape[-1]

    if weighted:
        weight_matrix = (
            calc_weight_matrix(
                hidden_outs,
                targets,
                risk_gamma=risk_gamma,
                varest_method=varest_method,
                varest_config=varest_config,
                weighting_interpolation_method=weighting_interpolation_method,
                weighting_interpolation_config=weighting_interpolation_config,
            )
            if weight_matrix is None
            else weight_matrix
        )
    else:
        weight_matrix = np.eye(batch_size, **factory_kwargs)

    htw = hidden_outs.transpose(1, 0).conj() @ weight_matrix

    if a_inv is None:
        if batch_size > hidden_nodes:
            a_inv = imqrginv(risk_gamma * np.eye(hidden_nodes, **factory_kwargs) + htw @ hidden_outs)
        else:
            d_t = risk_gamma * np.eye(batch_size, **factory_kwargs) + hidden_outs @ htw
            a_inv = imqrginv(d_t)

    b = a_inv @ htw if batch_size > hidden_nodes else htw @ a_inv

    if first or not incremental:
        return b @ targets, a_inv

    if batch_size > hidden_nodes:
        k_matrix = (
            np.eye(hidden_nodes, **factory_kwargs) - b @ imqrginv(hidden_outs @ b + torch.eye(batch_size, **factory_kwargs)) @ hidden_outs
        )
        beta = k_matrix @ (current_elm_weights.mT + b @ targets)  # type: ignore[union-attr]
        a_inv = k_matrix @ a_inv
    else:
        d_t = risk_gamma * torch.eye(hidden_outs.shape[0], device=hidden_outs.device, dtype=hidden_outs.dtype) + hidden_outs @ htw  # type: ignore[attr-defined, assignment]
        c_t = imqrginv(d_t - hidden_outs @ previous_hidden_outs.mH @ a_inv @ previous_hidden_outs @ hidden_outs.mH @ weight_matrix)  # type: ignore[attr-defined]
        beta = current_elm_weights.mT + (  # type: ignore[union-attr]
            previous_hidden_outs.mH @ a_inv @ previous_hidden_outs
            - torch.eye(hidden_outs.shape[-1], device=hidden_outs.device, dtype=hidden_outs.dtype)  # type: ignore[attr-defined]
        ) @ hidden_outs.mH @ weight_matrix @ c_t @ (  # type: ignore[attr-defined]
            hidden_outs @ current_elm_weights.mT - targets  # type: ignore[union-attr]
        )

        a_01 = a_inv @ previous_hidden_outs @ hidden_outs.mH  # type: ignore[attr-defined]
        a_10 = hidden_outs @ previous_hidden_outs.mH @ a_inv
        a_inv += a_01 @ c_t @ a_10

    return beta, a_inv


def torch_calc_regularized_elm_weights(
    hidden_outs: torch.Tensor,
    targets: torch.Tensor,
    current_elm_weights: torch.Tensor,
    risk_gamma: float = 1.0,
    a_inv: torch.Tensor | None = None,
    previous_hidden_outs: torch.Tensor | None = None,
    weighted: bool = True,
    incremental: bool = True,
    varest_method: Literal["iqr", "mean_difference", "mean_absolute_deviation"] = "iqr",
    varest_config: dict | None = None,
    weighting_interpolation_method: Literal["linear", "tanh", "step"] = "linear",
    weighting_interpolation_config: dict | None = None,
    *,
    weight_matrix: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the ELM output layer weights using the regularized (potentially incremental) update algorithm.

    Parameters
    ----------
    hidden_outs : torch.Tensor
        Hidden layer output.
    targets : torch.Tensor
        Learning targets.
    current_elm_weights : torch.Tensor
        Current output layer weights.
    risk_gamma : float, optional
        Empirical vs. structural risk trade-off parameter, by default 1.0.
    a_inv : torch.Tensor | None, optional
        Inverse matrix from a previous call, :data:`python:None`.
    previous_hidden_outs : torch.Tensor | None, optional
        The previous outputs from the hidden layers.
    weighted : bool, optional
        Whether to calculate the sample weight matrix, :data:`python:True`.
    incremental : bool, optional
        Whether to use the incremental update algorithm, :data:`python:True`.
    varest_method : Literal["iqr", "mean_difference", "mean_absolute_deviation"], optional
        Variance estimation method, by default "iqr".
    varest_config : dict | None, optional
        Variance estimation method config, :data:`python:None`.
    weighting_interpolation_method : Literal["linear", "tanh", "step"], optional
        Weight matrix interpolation method, by default "linear".
    weighting_interpolation_config : dict | None, optional
        Weight matrix interpolation method config, :data:`python:None`.
    weight_matrix : torch.Tensor | None, optional
        Weight matrix if you want to supply your own instead of having one calculated, :data:`python:None`.

    Returns
    -------
    torch.Tensor
        The updated regularized ELM output layer weights.
    torch.Tensor
        The updated ``a_inv`` matrix.
    """
    first = a_inv is None or previous_hidden_outs is None
    factory_kwargs = {"dtype": hidden_outs.dtype, "device": hidden_outs.device}

    batch_size = hidden_outs.shape[0]
    hidden_nodes = hidden_outs.shape[-1]

    if weighted:
        weight_matrix = (
            torch_calc_weight_matrix(
                hidden_outs,
                targets,
                risk_gamma=risk_gamma,
                varest_method=varest_method,
                varest_config=varest_config,
                weighting_interpolation_method=weighting_interpolation_method,
                weighting_interpolation_config=weighting_interpolation_config,
            )
            if weight_matrix is None
            else weight_matrix
        )
    else:
        weight_matrix = torch.eye(batch_size, **factory_kwargs)  # type: ignore[call-overload]

    htw = hidden_outs.mH @ weight_matrix

    if a_inv is None:
        if batch_size > hidden_nodes:
            a_inv = torch_imqrginv(risk_gamma * torch.eye(hidden_nodes, **factory_kwargs) + htw @ hidden_outs)  # type: ignore[call-overload]
        else:
            d_t = risk_gamma * torch.eye(batch_size, **factory_kwargs) + hidden_outs @ htw  # type: ignore[call-overload]
            a_inv = torch_imqrginv(d_t)

    b = a_inv @ htw if batch_size > hidden_nodes else htw @ a_inv

    if first or not incremental:
        return b @ targets, a_inv

    if batch_size > hidden_nodes:
        k_matrix = fused_calc_k_matrix(torch.eye(hidden_nodes, **factory_kwargs), b, hidden_outs, torch.eye(batch_size, **factory_kwargs))  # type: ignore[call-overload]
        beta = fused_calc_beta_matrix(k_matrix, current_elm_weights.mT, b, targets)
        a_inv = k_matrix @ a_inv
    else:
        d_t = risk_gamma * torch.eye(batch_size, **factory_kwargs) + hidden_outs @ htw  # type: ignore[call-overload]
        c_t = fused_small_batch_incremental_c(
            hidden_outs=hidden_outs,
            previous_hidden_outs=previous_hidden_outs,
            a_inv=a_inv,
            weight_matrix=weight_matrix,
            risk_gamma=risk_gamma,
        )
        beta = fused_small_batch_incremental_beta(
            hidden_outs=hidden_outs,
            targets=targets,
            current_elm_weights=current_elm_weights,
            previous_hidden_outs=previous_hidden_outs,
            a_inv=a_inv,
            weight_matrix=weight_matrix,
            c_current=c_t,
        )

        a_01 = a_inv @ previous_hidden_outs @ hidden_outs.mH
        a_10 = hidden_outs @ previous_hidden_outs.mH @ a_inv
        a_inv += a_01 @ c_t @ a_10

    return beta, a_inv


def calc_weight_matrix(
    hidden_outs: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
    risk_gamma: float = 1.0,
    varest_method: Literal["iqr", "mean_difference", "mean_absolute_deviation"] = "iqr",
    varest_config: dict | None = None,
    weighting_interpolation_method: Literal["linear", "tanh", "step"] = "linear",
    weighting_interpolation_config: dict | None = None,
    *,
    risk_eye: torch.Tensor | np.ndarray | None = None,
) -> torch.Tensor | np.ndarray:
    """
    Calculate the weighting matrix for regularized ELM using torch backend.

    Parameters
    ----------
    hidden_outs : torch.Tensor | np.ndarray
        Hidden layer outputs.
    targets : torch.Tensor | np.ndarray
        Target values.
    risk_gamma : float, optional
        Empirical vs. structural risk control variable, by default 1.0
    varest_method : Literal["iqr", "mean_difference", "mean_absolute_deviation"], optional
        Varience estimation method, by default "iqr".
    varest_config : dict | None, optional
        Varience estimateion method config, :data:`python:None`.
    weighting_interpolation_method : Literal["linear", "tanh", "step"], optional
        Weight interpolation method, by default "linear".
    weighting_interpolation_config : dict | None, optional
        Weight interpolation method config, :data:`python:None`.
    risk_eye : torch.Tensor | np.ndarray | None, optional
        Identity matrix * risk_gama, :data:`python:None`.

    Returns
    -------
    torch.Tensor | np.ndarray
        Diagonal weight matrix for the samples.
    """
    varest_config = varest_config or {}
    weighting_interpolation_config = weighting_interpolation_config or {}
    factory_kwargs = {"dtype": hidden_outs.dtype}
    if isinstance(hidden_outs, torch.Tensor):
        return torch_calc_weight_matrix(
            hidden_outs=hidden_outs,
            targets=targets,  # type: ignore[arg-type]
            risk_gamma=risk_gamma,
            varest_method=varest_method,
            varest_config=varest_config,
            weighting_interpolation_method=weighting_interpolation_method,
            weighting_interpolation_config=weighting_interpolation_config,
            risk_eye=risk_eye,  # type: ignore[arg-type]
        )

    batch_size = hidden_outs.shape[0]
    hidden_nodes = hidden_outs.shape[-1]

    risk_eye = risk_gamma * np.eye(hidden_nodes, **factory_kwargs) if risk_eye is None else risk_eye  # type: ignore[arg-type]

    if batch_size > hidden_nodes:
        # beta <- pinv(I + HtH) x Ht
        a_inv = imqrginv(risk_eye + hidden_outs.transpose(1, 0).conj() @ hidden_outs)
        mul_0 = a_inv
        mul_1 = hidden_outs.transpose(1, 0).conj()
    else:
        # beta <- Ht x pinv(I + HHt)
        a_inv = imqrginv(risk_eye + hidden_outs @ hidden_outs.transpose(1, 0).conj())
        mul_0 = hidden_outs.transpose(1, 0).conj()
        mul_1 = a_inv  # type: ignore[assignment]

    beta = mul_0 @ mul_1 @ targets
    epsilon = (hidden_outs @ beta - targets).sum(-1, keepdims=True)
    s_hat = VAREST_METHODS[varest_method](epsilon, **varest_config)

    nu = LMS_WEIGHT_INTERPOLATION_METHODS[weighting_interpolation_method](
        np.linalg.norm(epsilon / s_hat, axis=-1),
        **weighting_interpolation_config,
    )

    return np.power(np.diag(nu), 2.0)


def torch_calc_weight_matrix(
    hidden_outs: torch.Tensor,
    targets: torch.Tensor,
    risk_gamma: float = 1.0,
    varest_method: Literal["iqr", "mean_difference", "mean_absolute_deviation"] = "iqr",
    varest_config: dict | None = None,
    weighting_interpolation_method: Literal["linear", "tanh", "step"] = "linear",
    weighting_interpolation_config: dict | None = None,
    *,
    risk_eye: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Calculate the weighting matrix for regularized ELM using torch backend.

    Parameters
    ----------
    hidden_outs : torch.Tensor
        Hidden layer outputs.
    targets : torch.Tensor
        Target values.
    risk_gamma : float, optional
        Empirical vs. structural risk control variable, by default 1.0.
    varest_method : Literal["iqr", "mean_difference", "mean_absolute_deviation"], optional
        Varience estimation method, by default "iqr".
    varest_config : dict | None, optional
        Varience estimateion method config, :data:`python:None`.
    weighting_interpolation_method : Literal["linear", "tanh", "step"], optional
        Weight interpolation method, by default "linear".
    weighting_interpolation_config : dict | None, optional
        Weight interpolation method config, :data:`python:None`.
    risk_eye : torch.Tensor | None, optional
        Identity matrix * risk_gama, :data:`python:None`.

    Returns
    -------
    torch.Tensor
        Diagonal weight matrix for the samples.
    """
    varest_config = varest_config or {}
    weighting_interpolation_config = weighting_interpolation_config or {}
    factory_kwargs = {"dtype": hidden_outs.dtype, "device": hidden_outs.device}

    batch_size = hidden_outs.shape[0]
    hidden_nodes = hidden_outs.shape[-1]

    risk_eye = risk_gamma * torch.eye(hidden_nodes, **factory_kwargs) if risk_eye is None else risk_eye  # type: ignore[call-overload]

    if batch_size > hidden_nodes:
        # beta <- pinv(I + HtH) x Ht
        a_inv = torch_imqrginv(risk_eye + hidden_outs.mH @ hidden_outs)
        mul_0 = a_inv
        mul_1 = hidden_outs.mH
    else:
        # beta <- Ht x pinv(I + HHt)
        a_inv = torch_imqrginv(risk_eye + hidden_outs @ hidden_outs.mH)
        mul_0 = hidden_outs.mH
        mul_1 = a_inv

    beta = mul_0 @ mul_1 @ targets
    epsilon = (hidden_outs @ beta - targets).sum(-1, keepdim=True)
    s_hat = VAREST_METHODS[varest_method](epsilon, **varest_config)

    nu = LMS_WEIGHT_INTERPOLATION_METHODS[weighting_interpolation_method](
        torch.linalg.norm(epsilon / s_hat, dim=-1),
        **weighting_interpolation_config,
    )
    return torch.diag(nu).pow(2.0)  # type: ignore[arg-type]


def elm_forward_hidden_out_hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor):  # noqa: A002, ARG001
    """
    Store the output of the last hidden layer as a forward hook.

    Parameters
    ----------
    module : nn.Module
        The module.
    input : torch.Tensor
        The input tensor.
    output : torch.Tensor
        The output tensor.
    """
    module._last_output = output.detach().clone()  # noqa: SLF001


def geninv(a: torch.Tensor | np.ndarray, epsilon: float = 1e-9) -> torch.Tensor | np.ndarray:
    """
    Calculate the Moore_Penrose generalized inverse using Cholesky factorization.

    Outlined in [Fast Computation of Moore-Penrose Inverse Matrices](http://arxiv.org/abs/0804.4809)

    Parameters
    ----------
    a : torch.Tensor | np.ndarray
        The matrix to be inverted.
    epsilon : float, optional
        A small positive number for numerical stability, by default :python:`1e-9`.

    Returns
    -------
    torch.Tensor | np.ndarray
        The pseudoinverse of  ``a``.
    """
    if isinstance(a, torch.Tensor):
        return torch_geninv(a, epsilon=epsilon)
    sqrt = c_sqrt if np.iscomplexobj(a) else r_sqrt
    m, n = a.shape

    if m < n:
        transpose = True
        g = a @ a.transpose(1, 0).conj()
        n = m
    else:
        transpose = False
        g = a.transpose(1, 0).conj() @ a

    diag_g = np.diag(g)
    tol = diag_g[diag_g > 0].min() * epsilon

    chol = np.zeros_like(g)
    r = -1
    for k in range(n):
        r += 1
        chol[k:n, [r]] = g[k:n, [k]] - chol[k:n, :r] @ chol[[k], :r].transpose(1, 0).conj() if r != 0 else g[k:n, [k]]
        if chol[k, r] > tol:
            chol[k, r] = sqrt(chol[k, r])
            if k < n:
                chol[k + 1 : n, [r]] /= chol[[k], [r]]
        else:
            r -= 1
    chol = chol[:, : r + 1]
    m = np.linalg.inv(chol.transpose(1, 0).conj() @ chol)  # type: ignore[assignment]
    return (
        a.transpose(1, 0).conj() @ chol @ m @ m @ chol.transpose(1, 0).conj()
        if transpose
        else chol @ m @ m @ chol.transpose(1, 0).conj() @ a.transpose(1, 0).conj()
    )


def torch_geninv(g: torch.Tensor, epsilon: float = 1e-9) -> torch.Tensor:
    """
    Calculate the Moore_Penrose generalized inverse using Cholesky factorization in pytorch.

    Outlined in [Fast Computation of Moore-Penrose Inverse Matrices](http://arxiv.org/abs/0804.4809)

    Parameters
    ----------
    g : torch.Tensor
        The tensor to invert.
    epsilon : float, optional
        A small positive value for numerical stability, by default :python:`1e-9`

    Returns
    -------
    torch.Tensor
        The pseudoinverse of ``g``.
    """
    m, n = g.shape
    if m < n:
        transpose = True
        a = g.mm(g.mH)
        n = m
    else:
        transpose = False
        a = g.mH.mm(g)

    tol = a.diag().where(a.diag() > 0, 0).min() * epsilon

    chol = torch.zeros_like(a)
    r = -1
    for k in range(n):
        r += 1
        chol[k:n, [r]] = a[k:n, [k]] - chol[k:n, :r] @ chol[[k], :r].mH if r != 0 else a[k:n, [k]]
        if chol[k, r] > tol:
            chol[k, r] = chol[k, r].sqrt()
            if k < n:
                chol[k + 1 : n, [r]] /= chol[[k], [r]]
        else:
            r -= 1
    chol = chol[:, : r + 1]
    m = (chol.mH @ chol).inverse()  # type: ignore[assignment]
    if transpose:
        return fused_geninv_transpose(g, m, chol)
    return fused_geninv(g, m, chol)


@torch.jit.script
def fused_geninv(g: torch.Tensor, m: torch.Tensor, chol: torch.Tensor) -> torch.Tensor:
    """
    Run the ``geninv`` algorithm with a fused kernel.

    Parameters
    ----------
    g : torch.Tensor
        A pseudo-invertible matrix.
    m : torch.Tensor
        The squared Cholesky factorization matrix of ``g``.
    chol : torch.Tensor
        The Cholesky factorization matrix of ``g``.

    Returns
    -------
    torch.Tensor
        The pseudoinverse of ``g``.
    """
    return chol @ m @ m @ chol.mH @ g.mH


@torch.jit.script
def fused_geninv_transpose(g: torch.Tensor, m: torch.Tensor, chol: torch.Tensor) -> torch.Tensor:
    """
    Run the ``geninv`` algorithm with a fused kernel for the transposed case.

    Parameters
    ----------
    g : torch.Tensor
        Pseudo-invertible matrix.
    m : torch.Tensor
        The squared Cholesky factorization matrix of ``g``.
    chol : torch.Tensor
        The Cholesky factorization matrix of ``g``.

    Returns
    -------
    torch.Tensor
        The pseudoinverse of ``g``.
    """
    return g.mH.mm(chol).mm(m).mm(m).mm(chol.mH)


def imqrginv(a: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """
    Improved QR-factorization of the Moore-Penrose generalized inverse.

    Outlined in [Improved Qrginv Algorithm for Computing Moore-Penrose Inverse Matrices](https://www.hindawi.com/journals/isrn/2014/641706/)

    Parameters
    ----------
    a : torch.Tensor | np.ndarray
        The matrix to invert.

    Returns
    -------
    torch.Tensor | np.ndarray
        The pseudoinverse of  ``a``.
    """
    if isinstance(a, torch.Tensor):
        return torch_imqrginv(a)
    q, r = np.linalg.qr(a, mode="reduced")
    return r.transpose(1, 0).conj() @ np.linalg.inv(r @ r.transpose(1, 0).conj()) @ q.transpose(1, 0).conj()


@torch.jit.script
def fused_imqrginv(q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """
    Improved QR-factorization of the Moore-Penrose generalized inverse with a fused kernel.

    Outlined in [Improved Qrginv Algorithm for Computing Moore-Penrose Inverse Matrices](https://www.hindawi.com/journals/isrn/2014/641706/)

    Parameters
    ----------
    q : torch.Tensor
        The Q factorization matrix.
    r : torch.Tensor
        The R factorization matrix.

    Returns
    -------
    torch.Tensor
        The inverse of the matrix having the given QR factorization.
    """
    return r.mH.mm(r.mm(r.mH).inverse().mm(q.mH))


def torch_imqrginv(a: torch.Tensor) -> torch.Tensor:
    """
    Improved QR-factorization of the Moore-Penrose generalized inverse for pytorch.

    Outlined in [Improved Qrginv Algorithm for Computing Moore-Penrose Inverse Matrices](https://www.hindawi.com/journals/isrn/2014/641706/)

    Parameters
    ----------
    a : torch.Tensor
        The matrix to invert.

    Returns
    -------
    torch.Tensor
        The pseudoinverse of  ``a``.
    """
    q, r = torch.linalg.qr(a, mode="reduced")
    return fused_imqrginv(q, r)


@torch.jit.script
def fused_calc_k_matrix(eye: torch.Tensor, b: torch.Tensor, hidden_outs: torch.Tensor, batch_eye: torch.Tensor) -> torch.Tensor:
    """
    Calculate the K-matrix for incremental ELM with a fused kernel.

    Parameters
    ----------
    eye : torch.Tensor
        A ``hidden_nodes x hidden_nodes`` identity matrix.
    b : torch.Tensor
        The b matrix, differs depending on relationship between batch size and hidden node size.
    hidden_outs : torch.Tensor
        The hidden layer outputs.
    batch_eye : torch.Tensor
        The batch_size x batch_size identity matrix.

    Returns
    -------
    torch.Tensor
        K matrix for incremental ELM.
    """
    return eye - b @ torch_imqrginv(hidden_outs @ b + batch_eye) @ hidden_outs


@torch.jit.script
def fused_calc_beta_matrix(
    k_matrix: torch.Tensor,
    current_elm_weights: torch.Tensor,
    b: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Compute an incremental update to beta (ELM output weights) using the ``k_matrix`` with a fused kernel.

    Parameters
    ----------
    k_matrix : torch.Tensor
        The k_matrix.
    current_elm_weights : torch.Tensor
        The current ELM output weights.
    b : torch.Tensor
        The b matrix, meaning changes depending on the relationship between batch size and hidden layer size.
    targets : torch.Tensor
        The learning targets.

    Returns
    -------
    torch.Tensor
        The updated ELM output weights.
    """
    return k_matrix @ (current_elm_weights + b @ targets)


@torch.jit.script
def fused_small_batch_incremental_c(
    hidden_outs: torch.Tensor,
    previous_hidden_outs: torch.Tensor,
    a_inv: torch.Tensor,
    weight_matrix: torch.Tensor,
    risk_gamma: float,
) -> torch.Tensor:
    """
    Calculate the C matrix for the Minimum Norm incremental ELM algorithm.

    From [An incremental extreme learning machine for online sequential learning problems](https://www.sciencedirect.com/science/article/pii/S0925231213010059)

    Parameters
    ----------
    hidden_outs : torch.Tensor
        The hidden layer outputs.
    previous_hidden_outs : torch.Tensor
        The previous output from the hidden layer.
    a_inv : torch.Tensor
        The inverse matrix.
    weight_matrix : torch.Tensor
        The risk weight matrix which trades off empirical risk with structural risk,
    risk_gamma : float
        The empirical vs. structural risk control parameter.

    Returns
    -------
    torch.Tensor
        The C matrix for MN-IELM.
    """
    d_t = (
        risk_gamma * torch.eye(hidden_outs.shape[0], device=hidden_outs.device, dtype=hidden_outs.dtype)
        + hidden_outs @ hidden_outs.mH @ weight_matrix
    )
    return torch_imqrginv(d_t - hidden_outs @ previous_hidden_outs.mH @ a_inv @ previous_hidden_outs @ hidden_outs.mH @ weight_matrix)


@torch.jit.script
def fused_small_batch_incremental_c_laplacian(
    hidden_outs: torch.Tensor,
    previous_hidden_outs: torch.Tensor,
    a_inv: torch.Tensor,
    weight_matrix: torch.Tensor,
    laplacian: torch.Tensor,
    risk_gamma: float,
) -> torch.Tensor:
    """
    Calculate the C matrix for the Minimum Norm incremental ELM algorithm.

    From [An incremental extreme learning machine for online sequential learning problems](https://www.sciencedirect.com/science/article/pii/S0925231213010059)

    Parameters
    ----------
    hidden_outs : torch.Tensor
        The hidden layer outputs.
    previous_hidden_outs : torch.Tensor
        The previous output from the hidden layer.
    a_inv : torch.Tensor
        The inverse matrix.
    weight_matrix : torch.Tensor
        The risk weight matrix.
    laplacian : torch.Tensor
        The graph laplacian matrix.
    risk_gamma : float
        The empirical vs. structural risk control parameter.

    Returns
    -------
    torch.Tensor
        The C matrix for MN-IELM.
    """
    d_t = (
        risk_gamma * torch.eye(hidden_outs.shape[0], device=hidden_outs.device, dtype=hidden_outs.dtype)
        + hidden_outs @ hidden_outs.mH @ weight_matrix
        + hidden_outs @ hidden_outs.mH @ laplacian
    )
    return torch_imqrginv(d_t - hidden_outs @ previous_hidden_outs.mH @ a_inv @ previous_hidden_outs @ hidden_outs.mH @ weight_matrix)


@torch.jit.script
def fused_small_batch_incremental_beta(
    hidden_outs: torch.Tensor,
    targets: torch.Tensor,
    current_elm_weights: torch.Tensor,
    previous_hidden_outs: torch.Tensor,
    a_inv: torch.Tensor,
    weight_matrix: torch.Tensor,
    c_current: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate beta for the Minimum Norm Incremental ELM algorithm.

    From [An incremental extreme learning machine for online sequential learning problems](https://www.sciencedirect.com/science/article/pii/S0925231213010059)

    Parameters
    ----------
    hidden_outs : torch.Tensor
        The current hidden layer outputs.
    targets : torch.Tensor
        The training targets.
    current_elm_weights : torch.Tensor
        The current ELM output layer weights.
    previous_hidden_outs : torch.Tensor
        The previous hidden layer outputs.
    a_inv : torch.Tensor
        The inverse matrix.
    weight_matrix : torch.Tensor
        The weight matrix.
    c_current : torch.Tensor
        The current C matrix

    Returns
    -------
    torch.Tensor
        Updated ELM output weights.
    """
    return current_elm_weights.mT + (
        previous_hidden_outs.mH @ a_inv @ previous_hidden_outs
        - torch.eye(hidden_outs.shape[-1], device=hidden_outs.device, dtype=hidden_outs.dtype)
    ) @ hidden_outs.mH @ weight_matrix @ c_current @ (hidden_outs @ current_elm_weights.mT - targets)


def pinv(a: torch.Tensor | np.ndarray, method: Literal["native", "imqrginv", "geninv"] = "imqrginv") -> torch.Tensor | np.ndarray:
    """
    Compute the Moore-Penrose generalized inverse of  ``a`` .

    Parameters
    ----------
    a : torch.Tensor | np.ndarray
        The matrix to invert.
    method : Literal["native", "imqrginv", "geninv"], optional
        The algorithm to use when calculating the pseudo-inverse, by default "imqrginv".

        - ``native`` uses the ``pinv`` function of the array library of ``a`` which is equivalent to using SVD
        - ``svd`` is the same as ``native``
        - ``imqrginv`` uses the improved QR generalized inverse algorithm
        - ``geninv`` uses the fast generalized inverse algorithm

    Returns
    -------
    torch.Tensor | np.ndarray
        The pseudo-inverse of ``a``.

    Raises
    ------
    ValueError
        If an invalid method is chosen.
    """
    if method in {"native", "svd"}:
        backend_module = np if isinstance(a, np.ndarray) else torch
        return backend_module.linalg.pinv(a)
    if method == "imqrginv":
        return imqrginv(a)
    if method == "geninv":
        return geninv(a)
    msg = f"Invalid `method` argument ({method}). Expected 'native', 'svd', 'imqrginv', 'geninv'"
    raise ValueError(msg)


@torch.jit.script
def fused_calc_part_laplacian(stacked: torch.Tensor) -> torch.Tensor:
    """
    Calculate the component of the laplacian.

    Parameters
    ----------
    stacked : torch.Tensor
        The stacked input tensors.

    Returns
    -------
    torch.Tensor
        The partial laplacian of the graph represented by ``stacked``.
    """
    laplacian = stacked.pow(2.0).sum(dim=-1)
    return (laplacian / (2 * laplacian.var())).neg().exp()


def torch_calc_regularized_mlgelm_weights(  # noqa: PLR0913
    hidden_outs: torch.Tensor,
    targets: torch.Tensor,
    current_elm_weights: torch.Tensor,
    risk_gamma: float = 1.0,
    weight_kappa: float = 1.0,
    laplacian_lambda: float = 1.0,
    k_neighbors: float = 5,
    a_inv: torch.Tensor | None = None,
    previous_hidden_outs: torch.Tensor | None = None,
    weighted: bool = True,
    incremental: bool = True,
    varest_method: Literal["iqr", "mean_difference", "mean_absolute_deviation"] = "iqr",
    varest_config: dict | None = None,
    weighting_interpolation_method: Literal["linear", "tanh", "step"] = "linear",
    weighting_interpolation_config: dict | None = None,
    *,
    weight_matrix: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Calculate the ELM output layer weights using the regularized (potentially incremental) update algorithm.

    Parameters
    ----------
    hidden_outs : torch.Tensor
        The hidden layer output.
    targets : torch.Tensor
        The learning targets.
    current_elm_weights : torch.Tensor
        The current output layer weights.
    risk_gamma : float, optional
        The empirical vs. structural risk trade-off parameter, by default 1.0.
    weight_kappa : float, optional
        The coefficient value :math:`\kappa` default is 1.0.
    laplacian_lambda : float, optional
        The coefficient value :math:`\lambda` for the laplacian, default is 1.0.
    k_neighbors : float, optional
        The number of k-neighbors for regularization.
    a_inv : torch.Tensor | None, optional
        The inverse matrix from a previous call, :data:`python:None`.
    previous_hidden_outs : torch.Tensor | None, optional
        The previous outputs from the hidden layers, default is :data:`python:None`.
    weighted : bool, optional
        Whether to calculate the sample weight matrix, :data:`python:True`.
    incremental : bool, optional
        Whether to use the incremental update algorithm, :data:`python:True`.
    varest_method : Literal["iqr", "mean_difference", "mean_absolute_deviation"], optional
        Variance estimation method, by default "iqr".
    varest_config : dict | None, optional
        Variance estimation method config, :data:`python:None`.
    weighting_interpolation_method : Literal["linear", "tanh", "step"], optional
        Weight matrix interpolation method, by default "linear".
    weighting_interpolation_config : dict | None, optional
        Weight matrix interpolation method config, :data:`python:None`.
    weight_matrix : torch.Tensor | None, optional
        Weight matrix if you want to supply your own instead of having one calculated, :data:`python:None`.

    Returns
    -------
    torch.Tensor
        The updated regularized ELM output layer weights.
    torch.Tensor
        The updated ``a_inv`` matrix.
    """
    first = a_inv is None or previous_hidden_outs is None
    factory_kwargs = {"dtype": hidden_outs.dtype, "device": hidden_outs.device}

    batch_size = hidden_outs.shape[0]
    hidden_nodes = hidden_outs.shape[-1]

    if weighted:
        weight_matrix = (
            torch_calc_weight_matrix(
                hidden_outs,
                targets,
                risk_gamma=risk_gamma,
                varest_method=varest_method,
                varest_config=varest_config,
                weighting_interpolation_method=weighting_interpolation_method,
                weighting_interpolation_config=weighting_interpolation_config,
            )
            if weight_matrix is None
            else weight_matrix
        )
    else:
        weight_matrix = torch.eye(batch_size, **factory_kwargs)  # type: ignore[call-overload]
    weight_matrix.mul_(weight_kappa)

    if 0 < k_neighbors < 1:
        k_neighbors = math.ceil(k_neighbors * batch_size)

    laplacian = torch.stack([(targets - x).pow(2.0).sum(dim=-1) for x in targets])
    laplacian = (laplacian / (2 * laplacian.var())).neg().exp()
    laplacian[
        ~torch.zeros(laplacian.shape, dtype=torch.bool, device=laplacian.device).scatter_(
            -1,
            laplacian.topk(k_neighbors + 1, dim=-1).indices,  # type: ignore[arg-type]
            1,
        )
    ] = 0
    laplacian = laplacian_lambda * (laplacian.sum(dim=-1).diag() - laplacian)

    if a_inv is None and batch_size > hidden_nodes:
        a_inv = torch.eye(hidden_nodes, **factory_kwargs).addmm(  # type: ignore[call-overload]
            beta=risk_gamma,
            alpha=1,
            mat1=hidden_outs.mH @ weight_matrix,
            mat2=hidden_outs,
        )
        a_inv.addmm_(beta=1, alpha=1, mat1=hidden_outs.mH @ laplacian, mat2=hidden_outs)
        a_inv = torch_imqrginv(a_inv)
    elif a_inv is None:
        hht = hidden_outs @ hidden_outs.mH
        a_inv = torch.eye(batch_size, **factory_kwargs).addmm(beta=risk_gamma, alpha=1, mat1=hht, mat2=weight_matrix)  # type: ignore[call-overload]
        a_inv.addmm_(beta=1, alpha=1, mat1=hht, mat2=laplacian)
        del hht
        a_inv = torch_imqrginv(a_inv)

    b = (a_inv @ hidden_outs.mH if batch_size > hidden_nodes else hidden_outs.mH @ a_inv) @ weight_matrix

    if first or not incremental:
        return b @ targets, a_inv

    if batch_size > hidden_nodes:
        k_matrix = fused_calc_k_matrix(torch.eye(hidden_nodes, **factory_kwargs), b, hidden_outs, torch.eye(batch_size, **factory_kwargs))  # type: ignore[call-overload]
        beta = fused_calc_beta_matrix(k_matrix, current_elm_weights.mT, b, targets)
        a_inv.addmm_(mat1=k_matrix, mat2=a_inv, beta=0, alpha=1)
    else:
        c_t = fused_small_batch_incremental_c_laplacian(
            hidden_outs=hidden_outs,
            previous_hidden_outs=previous_hidden_outs,
            a_inv=a_inv,
            weight_matrix=weight_matrix,
            laplacian=laplacian,
            risk_gamma=risk_gamma,
        )
        beta = fused_small_batch_incremental_beta(
            hidden_outs=hidden_outs,
            targets=targets,
            current_elm_weights=current_elm_weights,
            previous_hidden_outs=previous_hidden_outs,
            a_inv=a_inv,
            weight_matrix=weight_matrix,
            c_current=c_t,
        )

        a_inv.addmm_(
            mat1=a_inv @ previous_hidden_outs @ hidden_outs.mH @ c_t,
            mat2=hidden_outs @ previous_hidden_outs.mH @ a_inv,
            beta=1,
            alpha=1,
        )

    return beta, a_inv
