"""Base classes to use with logging."""

import logging
import operator
import re

from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any


class BaseLogger(ABC):
    """BaseLogger defines the abstract methods and properties for all loggers."""

    destination: Any

    @abstractmethod
    def dump(self, result: dict[str, Any]):
        """
        Log the results to the logging destination.

        Parameters
        ----------
        result : dict[str, Any]
            _description_
        """

    @abstractmethod
    def close(self):
        """Close any open filestreams or sessions."""


class FileLogger(BaseLogger):
    """
    FileLogger defines the attributes and methods for loggers which write to filestreams.

    Parameters
    ----------
    storage_path : str | Path | None, optional
        The path to the storage folder for log data, default is :data:`python:None`.
    log_filename : str | None, optional
        The name of the log file to use for log output, default is :data:`python:None`.
    filter_regex : str | None, optional
        The regex to use when filtering messages from the logger.
    `**kwargs`
        Additional keyword arguments.
    """

    storage_path: Path
    """The path to the storage folder for log data."""
    log_filename: str
    """The name of the log file to use for log output, default is :data:`python:None`."""
    default_filename: str = "results.txt"
    """The default value for the filename if one is not provided."""
    filter_regex: str | None = None
    """The regex to use when filtering messages from the logger."""

    def __init__(self, storage_path: str | Path | None, log_filename: str | None = None, filter_regex: str | None = None, **kwargs):
        self.storage_path = Path("." if storage_path is None else storage_path).absolute()
        if not self.storage_path.exists():
            self.storage_path.mkdir(parents=True)
        self.log_file = self.storage_path / (log_filename or self.default_filename)  # type: ignore[assignment]
        self.filter_regex = filter_regex

    def dump(self, result: dict[str, Any]):
        """
        Maybe log the result depending on whether there is a regex defined.

        Parameters
        ----------
        result : dict[str, Any]
            The results to dump to the log
        """
        if self.filter_regex is not None:
            result = {res: val for res, val in result.items() if re.match(self.filter_regex, res)}
        if result:
            self._dump(result)

    def _dump(self, result: dict[str, Any]):
        raise NotImplementedError

    @property
    def log_file(self) -> IO:
        """The output filestream where results will be written.

        Returns
        -------
        IO
            The output filestream
        """
        return self.destination

    @log_file.setter
    def log_file(self, path: Path):
        self.destination = path.open(mode="at", encoding="UTF-8")

    def close(self):  # noqa: D102
        self.log_file.close()


class BaseHandler(logging.Handler):
    """
    A base logging handler for filtering records to a specific logger.

    Parameters
    ----------
    level : int | str, optional
        The logging level for =this handler, default is 0.
    name : str | None, optional
        The name for this handler.
    strict_level : bool, optional
        Whether to strictly obey the logging level, default is :data:`python:True`.
    """

    _compare_level: Callable

    def __init__(self, level: int | str = 0, name: str | None = None, strict_level: bool = True) -> None:
        super().__init__(level)
        if name:
            self.name = name
        self.strict_level = strict_level

    @property
    def compare_level(self) -> Callable:
        """
        Compare the handler level to the record level.

        Returns
        -------
        bool
            Compare the handler level to the record level.
        """
        return self._compare_level

    @property
    def strict_level(self) -> bool:
        """
        Whether to use a strict equality when comparing the record level to the handler level.

        Set the comparison operator accordingly.

        Returns
        -------
        bool
            Whether to use a strict equality when comparing the record level to the handler level.
        """
        return self._strict_level

    @strict_level.setter
    def strict_level(self, other: bool):
        self._strict_level = other
        if self._strict_level:
            self._compare_level = operator.eq
        else:
            self._compare_level = operator.ge

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D102
        if self.compare_level(record.levelno, self.level):
            try:
                self._emit(record)
            except RecursionError:  # See issue 36272 (logging)
                raise
            except Exception:  # noqa: BLE001
                self.handleError(record)

    def _emit(self, record: logging.LogRecord):  # noqa: PLR6301
        """Emit the record to the log.

        Must be overridden by subclasses. This implements the specific logic for handling the record.

        Parameters
        ----------
        record : logging.LogRecord
            The log record.

        Raises
        ------
        NotImplementedError
            If the method is not overriden.
        """
        msg = "emit must be implemented by BaseHandler subclasses"
        raise NotImplementedError(msg)
