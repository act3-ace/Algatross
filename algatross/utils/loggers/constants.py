"""Logging constants."""

import logging

TIMER_LOGGER = "algatross_timer_logger"
RESULTS_LOGGER = "algatross_results_logger"
DEBUGGING_PORTS_LOGGER = "algatross_debugging_ports"
MESSAGES_LOGGER = "ray"

TIMER_HANDLER = "algatross_timer_handler"
RESULTS_HANDLER = "algatross_results_handler"
DEBUGGING_PORTS_HANDLER = "algatross_debugging_ports_handler"
MESSAGES_HANDLER = "algatross_messages_handler"

TIMER_LOGGING_LEVEL = logging.INFO + 1
RESULTS_LOGGING_LEVEL = logging.INFO + 2
DEBUGGING_PORTS_LOGGING_LEVEL = logging.DEBUG + 1
MESSAGES_LOGGING_LEVEL = logging.INFO

LOGGER_FORMATS = {RESULTS_LOGGER: "json"}

LEVEL_COLORS = {logging.INFO: "green", logging.DEBUG: "magenta", logging.WARNING: "yellow", logging.ERROR: "red"}
LEVEL_COLORS[TIMER_LOGGING_LEVEL] = LEVEL_COLORS[logging.INFO]
LEVEL_COLORS[RESULTS_LOGGING_LEVEL] = LEVEL_COLORS[logging.INFO]
LEVEL_COLORS[DEBUGGING_PORTS_LOGGING_LEVEL] = LEVEL_COLORS[logging.DEBUG]
