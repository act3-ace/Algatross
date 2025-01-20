"""Initialization functionality for ELMs."""

from collections.abc import Callable

from torch import nn

ELM_INITIALIZER_MAP: dict[str, Callable] = {
    "normal": nn.init.normal_,
    "orthogonal": nn.init.orthogonal_,
    "uniform": nn.init.uniform_,
    "kaiming_uniform": nn.init.kaiming_uniform_,
    "kaiming_normal": nn.init.kaiming_normal_,
    "xavier_uniform": nn.init.xavier_uniform_,
    "xavier_normal": nn.init.xavier_normal_,
}
