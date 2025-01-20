"""Loggers which output in tensorboard format."""

import warnings

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ray.tune.logger.tensorboardx import VALID_SUMMARY_TYPES

from tensorboardX import SummaryWriter

from algatross.utils.loggers.base_loggers import FileLogger


class TensorboardLogger(FileLogger):
    """
    TensorboardLogger logs results in tensorboard event file format.

    Parameters
    ----------
    storage_path : str | Path | None, optional
        The path to the storage folder for tensorboard data, default is :data:`python:None`.
    log_filename : str | None, optional
        The name of the log file to use for tensorboard output, default is :data:`python:None`.
    filter_regex : str | None, optional
        The regex to use when filtering messages from the logger.
    `**kwargs`
        Keyword arguments.
    """

    def __init__(self, storage_path: str | Path | None, log_filename: str | None = None, filter_regex: str | None = None, **kwargs):
        self.storage_path = storage_path if storage_path is None else Path(storage_path).absolute()
        self.log_file = log_filename  # type: ignore[assignment]
        self.filter_regex = filter_regex

    @property  # type: ignore[override]
    def log_file(self) -> Path:  # noqa: D102
        return self.destination

    @log_file.setter
    def log_file(self, path: str | None):
        self.summary_writer = SummaryWriter(self.storage_path, filename_suffix=path or "", flush_secs=30)
        self.destination = Path(self.summary_writer.file_writer.event_writer._ev_writer._file_name)  # noqa: SLF001

    def close(self):  # noqa: D102
        self.summary_writer.close()

    def _dump(self, result: dict[str, Any]):
        epochs: dict[str, dict[str, Any]] = defaultdict(dict)
        for attr, value in result.items():
            if "epoch" in attr:
                item_type, item_key = attr.split("/")[:2]
                if item_key not in epochs[item_type] or int(value) > epochs[item_type][item_key]:
                    epochs[item_type][item_key] = int(value)

        for attr, value in result.items():
            item_type, item_key = attr.split("/")[:2]
            if isinstance(value, tuple(VALID_SUMMARY_TYPES)) and not np.isnan(value):
                self.summary_writer.add_scalar(attr, value, global_step=epochs[item_type][item_key])
            elif (isinstance(value, list) and len(value) > 0) or (isinstance(value, np.ndarray) and value.size > 0):
                # Must be a single image.
                if isinstance(value, np.ndarray) and value.ndim == 3:  # noqa: PLR2004
                    self.summary_writer.add_image(attr, value, global_step=epochs[item_type][item_key])
                    continue

                # Must be a batch of images.
                if isinstance(value, np.ndarray) and value.ndim == 4:  # noqa: PLR2004
                    self.summary_writer.add_images(attr, value, global_step=epochs[item_type][item_key])
                    continue

                # Must be video
                if isinstance(value, np.ndarray) and value.ndim == 5:  # noqa: PLR2004
                    self.summary_writer.add_video(attr, value, global_step=epochs[item_type][item_key], fps=20)
                    continue

                try:
                    self.summary_writer.add_histogram(attr, value, global_step=epochs[item_type][item_key])
                # In case TensorboardX still doesn't think it's a valid value
                # (e.g. `[[]]`), warn and move on.
                except (ValueError, TypeError):
                    warnings.warn(f"You are trying to log an invalid value ({attr}={value}) via {type(self).__name__}!", stacklevel=1)
        self.summary_writer.flush()
