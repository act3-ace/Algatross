"""Debugging utilites."""

import json
import logging
import logging.handlers

from collections.abc import Callable
from typing import Any

import ray

from ray.rllib.utils.typing import AgentID

from algatross.utils.loggers.constants import DEBUGGING_PORTS_LOGGER, DEBUGGING_PORTS_LOGGING_LEVEL

_algatross_debug_ports: dict[str, int] | None = None

logger = logging.getLogger("ray")
debug_port_logger = logging.getLogger(DEBUGGING_PORTS_LOGGER)
debug_port_logger.setLevel(logging.INFO)


def get_debugging_ports() -> dict:
    """Get a dictionary of ports to use for debugging.

    Returns
    -------
    dict
        A mapping from port to the actor/process using the port.
    """
    global _algatross_debug_ports  # noqa: PLW0603
    if _algatross_debug_ports is None:
        _algatross_debug_ports = {}
    return _algatross_debug_ports


def log_debugging_ports(wait_for_connect: bool = False) -> tuple[str, int, str | None]:
    """Log the ports used to attach a debugger to a remote process for debugging.

    Parameters
    ----------
    wait_for_connect : bool, optional
        Whether to block the thread until a debugger is attached, :data:`python:False`.

    Returns
    -------
    tuple[str, int, str | None]
        The hostname string for the debugger, the port number for the debugger, the name of the ray actor which is linked to the host and
        port for debugging.
    """
    context: ray.runtime_context.RuntimeContext = ray.get_runtime_context()

    if context is not None:
        import os  # noqa: PLC0415

        os.environ["RAY_ADDRESS"] = context.gcs_address

    return set_trace(wait_for_connect=wait_for_connect)


def set_trace(wait_for_connect: bool = False) -> tuple[str, int, str | None]:
    """Interrupt the flow of the program and drop into the Ray debugger.

    Can be used within a Ray task or actor.

    Parameters
    ----------
    wait_for_connect : bool, optional
        Whether to halt the program to wait for a connection on the port, defaults to false.

    Returns
    -------
    tuple[str, int, str | None]
        The hostname string for the debugger, the port number for the debugger, and the name of the actor.
    """
    from ray.util.debugpy import _ensure_debugger_port_open_thread_safe, _try_import_debugpy  # noqa: PLC0415, PLC2701
    from ray.util.state import get_worker  # noqa: PLC0415

    debugpy = _try_import_debugpy()
    if not debugpy:
        return ("", 0, None)

    _ensure_debugger_port_open_thread_safe()

    context = ray.get_runtime_context()
    worker = get_worker(context.get_worker_id())
    actor_name = context.get_actor_name()
    actor = context.get_actor_id() or context.get_worker_id() or context.get_job_id()
    actor = f"Name={actor_name} Id={actor} (pid={worker.pid})"
    logger.log(level=DEBUGGING_PORTS_LOGGING_LEVEL, msg=json.dumps({"port": f"{worker.ip}:{worker.debugger_port}", "actor": actor}))

    if wait_for_connect:
        with ray._private.worker.global_worker.worker_paused_by_debugger():  # noqa: SLF001
            logger.info("Waiting for debugger to attach...")
            debugpy.wait_for_client()

        logger.info("Debugger client is connected")
    else:
        logger.info("Continuing without attached debugger")
    return worker.ip, worker.debugger_port, actor_name


def log_agent_params(agent_map: dict[AgentID, Any], log_fn: Callable):
    """
    Log the agent parameters.

    Parameters
    ----------
    agent_map : dict[AgentID, Any]
        Map of agent ids to agents.
    log_fn : Callable
        The function to call to log the info.
    """
    k = next(iter(agent_map.keys()))
    log_fn(agent_map[k].flat_parameters.numpy()[:5])
