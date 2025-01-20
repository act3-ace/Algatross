"""Base classes for ray actors."""

import logging

import ray

from ray.util.queue import Queue

from algatross.utils.loggers.constants import RESULTS_LOGGER, RESULTS_LOGGING_LEVEL
from algatross.utils.loggers.formatters import MappingLogFormatter
from algatross.utils.loggers.remote_logger import RemoteQueueHandler

logger = logging.getLogger("ray")


class RayActor:
    """
    Set up logging and stores the ray context.

    Parameters
    ----------
    log_queue : Queue | None, optional
        The queue to use for logging, default is :data:`python:None`
    result_queue : Queue | None, optional
        The queue to use for results, default is :data:`python:None`
    `**kwargs`
        Additional keyword arguments.
    """

    debug_host: str
    """The host for connecting to a debugging port."""

    extra_log_info: dict
    """Extra information to snd to the logger."""

    _results_logger: logging.Logger
    _actor_name = ""

    def __init__(self, log_queue: Queue | None = None, result_queue: Queue | None = None, **kwargs) -> None:
        self._context = ray.get_runtime_context()
        self._namespace = self._context.namespace
        self._actor_name = self._context.get_actor_name()
        self._child_actor_options = {"namespace": self._namespace, "max_concurrency": 1}

        if result_queue is not None:
            logging.getLogger(RESULTS_LOGGER).addHandler(RemoteQueueHandler(result_queue, formatter=MappingLogFormatter()))

        if log_queue is not None:
            logger.addHandler(RemoteQueueHandler(log_queue))

        self.debug_host, self.debug_port = None, None
        self._relog_ports = False

        self.extra_log_info = {
            "actor": str(self),
            "actor_name": self._context.get_actor_name(),
            "task_id": self._context.get_task_id(),
            "actor_id": self._context.get_actor_id(),
            "job_id": self._context.get_job_id(),
        }

    def log(self, msg: object, logger_name: str | None = None, level: int = logging.INFO):
        """Log a message to the desired logger after (maybe) serializing.

        Parameters
        ----------
        msg : object
            The message to log
        logger_name : str | None, optional
            The name of the logger to log the message to. Default is "ray".
        level : int, optional
            The logging level, by default logging.INFO
        """
        if self._relog_ports and self.debug_host:
            actor_name = self._context.get_actor_name()
            if actor_name:
                self._relog_ports = False
        lgr = logger if logger_name is None else logging.getLogger(logger_name)
        lgr.log(level=level, msg=msg, extra=self.extra_log_info)

    def log_result(self, result: dict, logger_name: str = RESULTS_LOGGER, level: int = RESULTS_LOGGING_LEVEL):
        """Log a result to the results logger.

        This is shorthand for :python:`self.log(msg=result, logger_name=RESULTS_LOGGER, level=logging.INFO)`

        Parameters
        ----------
        result : dict
            The result to log
        logger_name : str, optional
            The name of the results logger, by default RESULTS_LOGGER
        level : int, optional
            Log level, by default RESULTS_LOGGING_LEVEL
        """
        self.log(msg=result, logger_name=logger_name, level=level)

    @staticmethod
    def is_alive() -> bool:
        """Return whether the actor is alive.

        Essentially a useful no-op for blocking until the actor is ready

        Returns
        -------
        bool
            If the actor is alive and ready
        """
        return True

    def __repr__(self) -> str:  # noqa: D105
        return f"{self._actor_name} <{self.__class__.__name__}>" if self._actor_name else self.__class__.__name__
