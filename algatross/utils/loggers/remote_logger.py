"""Asyncronous loggers."""

import logging
import logging.handlers

from ray.util.queue import Queue


class RemoteQueueHandler(logging.handlers.QueueHandler):
    """A QueueHandler using a ray.util.Queue from ray.

    Makes sure to set the log level for the logger as well.

    Parameters
    ----------
    queue : Queue
        The logging queue to handle.
    formatter : logging.Formatter | None, optional
        The formatter to use when logging, default is :data:`python:None`
    """

    queue: Queue
    """The remote queue handled by this handler."""
    formatter: logging.Formatter
    """The formatter this handler uses to format log messages."""

    def __init__(self, queue: Queue, formatter: logging.Formatter | None = None) -> None:
        super().__init__(queue)

        if formatter:
            self.setFormatter(formatter)
