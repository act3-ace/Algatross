"""Timing module."""

import json
import logging
import time

from contextlib import ContextDecorator
from pathlib import Path
from typing import ClassVar

import numpy as np

from ray.util.queue import Queue

import torch

from algatross.utils.loggers.constants import TIMER_LOGGER
from algatross.utils.loggers.encoders import SafeFallbackEncoder
from algatross.utils.loggers.remote_logger import RemoteQueueHandler

TIMER_METRICS = {"sum", "mean"}

logger = logging.getLogger(TIMER_LOGGER)


class TimerError(Exception):
    """A custom exception used to report errors in use of timer class."""


class Timer(ContextDecorator):
    """
    A timer class to measure the time of a code block.

    Parameters
    ----------
    name : str
        The name for the timer.
    metric : str | None, optional
        The metric this timer tracks (``sum``, ``mean``), default is :data:`python:None` which is the same as ``sum``.
    info : str | None, optional
        Additional info for this timer, default is :data:`python:None`.
    queue : Queue | None, optional
        A queue for this timer default is :data:`python:None`.
    log_file : str | Path | None, optional
        The path to a log file for this timer, default is :data:`python:None`.
    """

    disabled: bool = False
    """Whether this timer is disabled."""
    timers: ClassVar[dict[str, list[float]]] = {}
    """The timers container in this :python:`Timer`."""
    metrics: ClassVar[dict[str, str]] = {}
    """The metrics this :python:`Timer` tracks."""
    info: ClassVar[dict[str, str]] = {}
    """Extra info carried by this timer."""
    logger: logging.Logger
    """The logger for this timer."""

    _start_time: float | None = None

    def __init__(
        self,
        name: str,
        metric: str | None = None,
        info: str | None = None,
        queue: Queue | None = None,
        log_file: str | Path | None = None,
    ) -> None:
        if queue is not None:
            logger.addHandler(RemoteQueueHandler(queue))

        if log_file:
            logger.addHandler(logging.StreamHandler(Path(log_file).resolve().open("a")))

        if metric not in TIMER_METRICS:
            logger.warning(
                f"Provided metric {metric} not in available metrics {TIMER_METRICS}. Setting to 'sum'",
                extra={name: {"info": info or ""}, **self.info},
            )
            metric = "sum"

        self.name = name
        if not Timer.disabled and self.name is not None and self.name not in self.timers:
            self.timers.setdefault(self.name, [])
            self.metrics[self.name] = metric
            if info:
                self.info[self.name] = info

    def start(self) -> None:
        """Start a new timer.

        Raises
        ------
        TimerError
            If :meth:`start` has already been called
        """
        if self._start_time is not None:
            msg = "timer is running. Use .stop() to stop it"
            raise TimerError(msg)

        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop the timer, and report the elapsed time.

        Returns
        -------
        float
            The elapsed time since :meth:`start` was called.

        Raises
        ------
        TimerError
            If :meth:`start` has not been called.
        """
        if self._start_time is None:
            msg = "timer is not running. Use .start() to start it"
            raise TimerError(msg)

        # Calculate elapsed time
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None

        # Report elapsed time
        if self.name:
            self.timers[self.name].append(elapsed_time)

        return elapsed_time

    @classmethod
    def to(cls, device: str | torch.device = "cpu") -> None:
        """
        Create a new timer on a different device.

        Parameters
        ----------
        device : str | torch.device, optional
            The device to which the timer will be sent, default is :python:`"cpu"`.
        """
        if cls.timers:
            for k, v in cls.timers.items():
                if isinstance(v, torch.Tensor):
                    cls.timers[k] = v.to(device)

    @classmethod
    def reset(cls) -> None:
        """Reset all timers."""
        for timer in cls.timers.values():
            timer.clear()
        cls._start_time = None

    @classmethod
    def compute(cls) -> dict[str, torch.Tensor]:
        """Reduce the timers to a single value.

        Returns
        -------
        dict[str, torch.Tensor]
            The reduced metrics.
        """
        return {k: getattr(np, cls.metrics[k])(v) for k, v in cls.timers.items()}

    @classmethod
    def log(cls):
        """Reduce the timers to a single value."""
        logger.info(json.dumps(cls.timers, cls=SafeFallbackEncoder), extra=cls.info)

    def __enter__(self) -> "Timer":
        """Start a new timer as a context manager.

        Returns
        -------
        Timer
            The timer itself.
        """
        if not Timer.disabled:
            self.start()
        return self

    def __exit__(self, *exc_info):
        """
        Stop the context manager timer.

        Parameters
        ----------
        *exc_info : list
            Exception info
        """
        if not Timer.disabled:
            self.stop()
