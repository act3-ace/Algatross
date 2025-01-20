"""Logging encoders."""

import json
import logging
import numbers

from abc import ABC, abstractmethod
from collections.abc import Sequence
from json import JSONEncoder

import numpy as np

import torch

import tree

from algatross.utils.merge_dicts import get_struct

defaultRecordFactory = logging.getLogRecordFactory()  # noqa: N816


class LogRecordEncoder(ABC):
    """Base class for encoding log record messages."""

    @classmethod
    def encode_message(cls, record: logging.LogRecord) -> logging.LogRecord:
        """Format the log records messasge.

        Parameters
        ----------
        record : logging.LogRecord
            The logging record

        Returns
        -------
        logging.LogRecord
            The logging record with an updated message
        """
        record.msg = cls.encode(record.msg)
        return record

    @classmethod
    @abstractmethod
    def encode(cls, obj: object) -> object:
        """Encode the object into a record message.

        Parameters
        ----------
        obj : object
            The object to encode

        Returns
        -------
        object
            The encoded object
        """


class SafeFallbackEncoder(json.JSONEncoder):
    """
    SafeFallbackEncoder copy-pasted from Ray internals.

    Parameters
    ----------
    nan_str : str
        String to replace nans with, default is "null"
    include_tensors : bool
        Whether to include tensors in the JSON output, default is :data:`python:True`
    `**kwargs`
        Additional keyword arguments.
    """

    def __init__(self, nan_str: str = "null", include_tensors: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.nan_str = nan_str
        self.include_tensors = include_tensors

    def default(self, value):  # noqa: D102, PLR0911
        try:
            if (type(value).__module__ in {np.__name__, torch.__name__}) and isinstance(value, np.ndarray | torch.Tensor):
                return value.tolist() if self.include_tensors else []

            if isinstance(value, np.bool_):
                return bool(value)

            if isinstance(value, Sequence):
                return list(value)

            if np.isnan(value):
                return self.nan_str

            if issubclass(type(value), numbers.Integral):
                return int(value)

            if issubclass(type(value), numbers.Number):
                return float(value)

            return super().default(value)
        except Exception:  # noqa: BLE001
            return str(value)  # give up, just stringify it (ok for logs)


class JSONLogRecordEncoder(LogRecordEncoder):
    """A class which encodes a log record as a JSON object."""

    encoder: JSONEncoder = SafeFallbackEncoder()

    @classmethod
    def encode(cls, obj: object) -> object:  # noqa: D102
        return tree.map_structure_up_to(get_struct(obj), cls.encoder.default, obj)  # type: ignore[arg-type]
