"""Containers which hold loggers."""

import logging
import sys

from typing import Any, SupportsIndex

from algatross.utils.loggers.base_loggers import BaseHandler, BaseLogger
from algatross.utils.loggers.shared_buffer import SharedBuffer


class LoggerContainer:
    """
    LoggerContainer stores results in a shared buffer and calls logging methogs for each of the loggrs in the container.

    Parameters
    ----------
    loggers : list[BaseLogger]
        The list of loggers to initialize into this container.
    buffer : SharedBuffer
        The log messsage buffer these loggers should share.
    """

    def __init__(self, loggers: list[BaseLogger], buffer: SharedBuffer):
        self.loggers = loggers
        self.buffer = buffer

    def log(self, value: dict[str, Any]):
        """
        Log the value to the shared buffer.

        Parameters
        ----------
        value : dict[str, Any]
            The value to be logged.
        """
        self.buffer.add(value)

    def dump(self):
        """Dump the results stored in the buffer to each of the loggers."""
        results = self.buffer.flush()
        for logger in self.loggers:
            logger.dump(results)

    def close(self):
        """Close any open IO streams."""
        for logger in self.loggers:
            logger.close()


class LogHandlerContainer(BaseHandler):
    """Handler for a container of loggers.

    Parameters
    ----------
    loggers : list[logging.Logger | BaseLogger]
        The loggers to initialize into this container.
    buffer : SharedBuffer
        The log message buffer the loggers will share.
    level : int | str, optional
        The logging level for filtering messages, default is 0,
    name : str | None
        The name of this handler, default is :data:`python:None`.
    strict_level : bool
        Whether to strictly obey logging level, default is :data:`python:True`.
    """

    def __init__(
        self,
        loggers: list[logging.Logger | BaseLogger],
        buffer: SharedBuffer,
        level: int | str = 0,
        name: str | None = None,
        strict_level: bool = True,
    ) -> None:
        super().__init__(level=level, name=name, strict_level=strict_level)
        self.loggers = loggers
        self.buffer = buffer

    def append(self, logger: logging.Logger | BaseLogger):
        """Append a logger to the list of loggers.

        Parameters
        ----------
        logger : logging.Logger | BaseLogger
            The logger to add

        Raises
        ------
        TypeError
            If ``logger`` is not a :class:`~algatross.utils.loggers.base_loggers.BaseLogger`
        """
        if not isinstance(logger, logging.Logger | BaseLogger):
            msg = f"`LogHandlerContainer.append` requires a {logging.Logger} or {BaseLogger} instance, instead got {type(logger)}"
            raise TypeError(msg)
        self.loggers.append(logger)

    def extend(self, loggers: list[logging.Logger | BaseLogger]):
        """Extend the loggers with a list of loggers.

        Parameters
        ----------
        loggers : list[logging.Logger | BaseLogger]
            A list of loggers to add

        Raises
        ------
        TypeError
            If any of the loggers in the list are not a :class:`~algatross.utils.loggers.base_loggers.BaseLogger`
        """
        if not all(isinstance(logger, logging.Logger | BaseLogger) for logger in loggers):
            types = [type(logger) for logger in loggers]
            msg = f"`LogHandlerContainer.extend` requires a list of {logging.Logger} or {BaseLogger} instances, instead got {types}"
            raise TypeError(msg)
        self.loggers.extend(loggers)

    def pop(self, index: int = -1) -> logging.Logger | BaseLogger:
        """Remove and return the logger at the given index.

        Parameters
        ----------
        index : int, optional
            The index of the logger to remove, by default -1

        Returns
        -------
        logging.Logger | BaseLogger
            The logger at the index
        """
        return self.loggers.pop(index)

    def remove(self, logger: logging.Logger | BaseLogger):
        """Remove the logger from the list.

        Parameters
        ----------
        logger : logging.Logger | BaseLogger
            The logger to remove
        """
        self.loggers.remove(logger)

    def index(self, logger: logging.Logger | BaseLogger, start: SupportsIndex = 0, stop: SupportsIndex = sys.maxsize) -> int:
        """Get the index of the logger.

        Parameters
        ----------
        logger : logging.Logger | BaseLogger
            The logger to find.
        start : SupportsIndex, optional
            The start position of the list, by default 0
        stop : SupportsIndex, optional
            The end position of the list, by default sys.maxsize

        Returns
        -------
        int
            The index to retrieve
        """
        return self.loggers.index(logger, start, stop)

    def __contains__(self, logger: logging.Logger | BaseLogger) -> bool:  # noqa: D105
        return logger in self.loggers

    def _emit(self, record: logging.LogRecord):
        self.buffer.add(record.msg)  # type: ignore[arg-type]
        self.flush()

    def flush(self):
        """Dump the results stored in the buffer to each of the loggers."""
        results = self.buffer.flush()
        for logger in self.loggers:
            if hasattr(logger, "dump"):
                logger.dump(results)

    def close(self):
        """Close any open IO streams."""
        super().close()
        for logger in self.loggers:
            if hasattr(logger, "close"):
                logger.close()
