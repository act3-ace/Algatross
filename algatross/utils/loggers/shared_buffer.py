"""Contains shared buffers for use with logging."""

from collections import defaultdict
from collections.abc import Callable
from typing import Any

import numpy as np

from algatross.utils.merge_dicts import flatten_dicts


class SharedBuffer:
    """
    SharedBuffer provides a results buffer to share between loggers.

    Parameters
    ----------
    metrics : list[str] | None
        The metrics to select from the defaults to calculate on the logged values, default is :data:`python:None`.
    custom_metrics : dict[str, Callable] | None
        The custom metrics to include along with the default metrics.
    `**kwargs`
        Additional keyword arguments.
    """

    default_metrics: dict[str, Callable] = {"min": np.min, "max": np.max, "mean": np.mean}  # noqa: RUF012
    """The default metrics of the class which can be selected to call on logs."""
    metrics: dict[str, Callable]
    """
    The actual metrics which will be called on the logs.

    These are selected from the default metrics and any custom metrics supplied in the constructor.
    """

    def __init__(self, metrics: list[str] | None = None, custom_metrics: dict[str, Callable] | None = None, **kwargs):
        self.metrics = {**self.default_metrics} if metrics is None else {k: self.default_metrics[k] for k in metrics}

        if custom_metrics is not None:
            self.metrics.update(custom_metrics)

        self._buffer: dict[str, list] = defaultdict(list)

    def add(self, value: dict[str, Any]):
        """
        Add data to the buffer.

        Parameters
        ----------
        value : dict[str, Any]
            The value to be stored in the buffer
        """
        infos = flatten_dicts(value)
        for path, info in infos.items():
            if isinstance(info, str):
                continue
            self._buffer[path].append(np.asanyarray(info))

    def flush(self) -> dict[str, np.ndarray]:
        """Flush the buffer to the logging destination.

        Returns
        -------
        dict[str, np.ndarray]
            The flushed contents of the buffer
        """
        flushed = {}
        for k, v in self._buffer.items():
            if len(v) > 1:
                stacked = np.stack(v)
                for metric, metric_fn in self.metrics.items():
                    flushed[f"{k}_{metric}"] = metric_fn(stacked, axis=0)
            else:
                flushed[k] = v[0]
        for k, v in flushed.items():
            flushed[k] = v.item() if isinstance(v, np.ndarray) and np.prod(v.shape) == 1 else v  # type: ignore[attr-defined]
        self._buffer.clear()
        return flushed
