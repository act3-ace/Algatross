"""Loggers which output JSON format."""

import json

from typing import Any

from algatross.utils.loggers.base_loggers import FileLogger
from algatross.utils.loggers.encoders import SafeFallbackEncoder


class JSONLogger(FileLogger):
    """JSONLogger logs results in JSON format to a ``.json`` file."""

    default_filename: str = "results.json"

    def _dump(self, result: dict[str, Any]):
        json.dump(result, self.log_file, cls=SafeFallbackEncoder)
        self.log_file.write("\n")
        self.log_file.flush()
