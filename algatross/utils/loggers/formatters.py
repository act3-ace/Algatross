"""Logging formatters."""

import json
import logging

from algatross.utils.loggers.constants import LOGGER_FORMATS
from algatross.utils.loggers.encoders import JSONLogRecordEncoder


class JSONLogStringFormatter(logging.Formatter):
    """JSONLogStringFormatter class for loading json objects from strings."""

    def formatMessage(self, record: logging.LogRecord) -> str:  # noqa: D102, N802
        return json.loads(super().formatMessage(record))


class MappingLogFormatter(logging.Formatter):
    """A class which formats a log record according to the logger format mapping."""

    def format(self, record: logging.LogRecord) -> dict:  # type: ignore[override] # noqa: D102, PLR6301
        record_format = LOGGER_FORMATS.get(record.name)
        if record_format == "json":
            record = JSONLogRecordEncoder.encode_message(record)
        return record.msg  # type: ignore[return-value]
