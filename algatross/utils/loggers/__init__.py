import logging

from .constants import DEBUGGING_PORTS_LOGGING_LEVEL, RESULTS_LOGGER, RESULTS_LOGGING_LEVEL, TIMER_LOGGING_LEVEL

logging.getLogger(RESULTS_LOGGER).setLevel(logging.INFO)
logging.addLevelName(TIMER_LOGGING_LEVEL, "TIMER")
logging.addLevelName(RESULTS_LOGGING_LEVEL, "RESULT")
logging.addLevelName(DEBUGGING_PORTS_LOGGING_LEVEL, "DEBUG PORT")
