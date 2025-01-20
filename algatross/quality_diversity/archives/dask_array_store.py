"""Provides ArrayStore using Dask for array management.."""

import itertools
import json
import numbers

from collections import UserDict
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from types import TracebackType
from typing import Any, Literal

import pandas as pd

import numpy as np

import dask.array as da

from filelock import FileLock
from numpy_groupies import aggregate_nb as aggregate
from ribs._utils import readonly
from ribs.archives._array_store import (
    ArrayStore as _ArrayStore,
    ArrayStoreIterator,
    Update,
)

from algatross.utils.loggers.encoders import SafeFallbackEncoder


class ArrayPlaceHolder(AbstractContextManager):
    def __init__(self, shape, dtype, storage_path):
        self.storage_path = Path(storage_path).absolute()
        self.shape = shape
        self.dtype = dtype

    def __repr__(self) -> str:
        return f"ArrayPlaceHolder(shape={self.shape}, dtype={self.dtype}, path={self.storage_path})"

    def __enter__(self, *args, **kwargs) -> da.Array:
        """Return the dask array loaded from a stack of NumPy files.

        Parameters
        ----------
        `*args`
            Additional positional arguments.
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        da.Array
            The loaded dask array
        """
        return self._get_dask_array(*args, **kwargs)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        raise exc_value

    def __call__(self, *args, **kwargs) -> da.Array:
        """Return the dask array loaded from a stack of NumPy files.

        Parameters
        ----------
        `*args`
            Additional positional arguments.
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        da.Array
            The loaded dask array
        """
        return self._get_dask_array(*args, **kwargs)

    def _get_dask_array(self, *args, **kwargs) -> da.Array:
        return da.from_npy_stack(self.storage_path, mmap_mode="r")


class DaskDict(UserDict):
    """
    A dictionary for storing np.ndarray items as dask arrays.

    Parameters
    ----------
    arg_dict : dict | None, optional
        The dictionary to initialize with, default is :data:`python:None`.
    storage_path : str | Path | None, optional
        The path to spill the contents to, default is :data:`python:None`.
    dask_array_kwargs : dict | None, optional
        The keyword arguments to pass to dask arrays.
    `**kwargs`
        Additional keyword arguments.
    """

    storage_path: Path

    def __init__(
        self,
        arg_dict: dict | None = None,
        /,
        storage_path: str | Path | None = None,
        dask_array_kwargs: dict | None = None,
        **kwargs,
    ):
        self.storage_path = Path(storage_path or ".").absolute()
        self.dask_array_kwargs = dask_array_kwargs or {}

        if not self.storage_path.exists():
            self.storage_path.mkdir(parents=True)

        super().__init__(arg_dict, **kwargs)

    def __getitem__(self, key: Any) -> Any:  # noqa: ANN401
        with FileLock(str(self.storage_path / f"{key}.lock")).acquire():
            if isinstance(self.data[key], ArrayPlaceHolder):
                return self.data[key]()

            with (self.storage_path / f"{key}.json").open("r") as fp:
                loaded = json.load(fp)
            stored = super().__getitem__(key)
            # make sure the stored value matches the one we hold.
            if loaded != stored:
                with (self.storage_path / f"{key}.json").open("w") as fp:
                    json.dump(stored, fp, cls=SafeFallbackEncoder)
        return stored

    def __setitem__(self, key: Any, value: Any) -> None:  # noqa: ANN401
        if not isinstance(value, da.Array) and isinstance(value, np.ndarray):
            # if we get numpy array object convert to dask
            if np.prod(value.shape) == 0:
                value = np.full(np.maximum(value.shape, 1), fill_value=0, dtype=value.dtype)
            value = da.from_array(value, **self.dask_array_kwargs)

        if isinstance(value, da.Array):
            if not (self.storage_path / key).exists():
                (self.storage_path / key).mkdir(parents=True)
            with FileLock(str(self.storage_path / f"{key}.lock")).acquire():
                da.to_npy_stack(str(self.storage_path / key), value, axis=0)
        else:
            with FileLock(str(self.storage_path / f"{key}.lock")).acquire(), (self.storage_path / f"{key}.json").open("w") as fp:
                json.dump(value, fp, cls=SafeFallbackEncoder)

        if isinstance(value, np.ndarray | da.Array):
            value = ArrayPlaceHolder(shape=value.shape, dtype=value.dtype, storage_path=self.storage_path / key)

        # store the value normally
        super().__setitem__(key, value)

    def __or__(self, other):
        copied = DaskDict(self.items(), storage_path=self.storage_path, dask_array_kwargs=self.dask_array_kwargs)
        copied |= other
        return copied

    def __ior__(self, other):
        self.update(other)
        return self

    def update(self, other):
        super().update(other)
        for key, val in self.data.items():
            if not isinstance(val, ArrayPlaceHolder):
                self[key] = val


class ArrayStore(_ArrayStore):
    def __init__(self, field_desc, capacity, storage_path, dask_array_kwargs):
        self.storage_path = Path(storage_path)
        self.dask_array_kwargs = dask_array_kwargs or {}
        if not self.storage_path.exists():
            self.storage_path.mkdir(parents=True)

        self._props = DaskDict(
            {
                "capacity": capacity,
                "occupied": np.zeros(capacity, dtype=bool),
                "n_occupied": 0,
                "occupied_list": np.empty(capacity, dtype=np.int32),
                "updates": np.array([0, 0]),
            },
            storage_path=self.storage_path / "props",
            dask_array_kwargs=dask_array_kwargs,
        )

        self._fields = DaskDict(storage_path=self.storage_path / "fields", dask_array_kwargs=dask_array_kwargs)
        for name, (field_shape, dtype) in field_desc.items():
            if name == "index":
                msg = f"`{name}` is a reserved field name."
                raise ValueError(msg)
            if not name.isidentifier():
                msg = f"Field names must be valid identifiers: `{name}`"
                raise ValueError(msg)

            if isinstance(field_shape, numbers.Integral):
                field_shape = (field_shape,)  # noqa: PLW2901

            array_shape = (capacity, *tuple(field_shape))
            self._fields[name] = np.empty(array_shape, dtype)

    def __len__(self) -> int:
        return self._props["n_occupied"]

    def __iter__(self) -> ArrayStoreIterator:
        return ArrayStoreIterator(self)

    @property
    def capacity(self) -> int:
        return self._props["capacity"]

    @property
    def occupied(self) -> np.ndarray:
        with FileLock(self.storage_path / "props" / "occupied.lock").acquire():
            return readonly(da.from_npy_stack(str(self.storage_path / "props" / "occupied"), mmap_mode="r").compute())

    @property
    def occupied_list(self) -> np.ndarray:
        with FileLock(self.storage_path / "props" / "occupied_list.lock").acquire():
            return readonly(
                da.from_npy_stack(str(self.storage_path / "props" / "occupied_list"), mmap_mode="r")[: self._props["n_occupied"]].compute(),
            )

    def retrieve(  # noqa: PLR0912
        self,
        indices: np.ndarray | da.Array,
        fields: str | Sequence[str] | None = None,
        return_type: Literal["dict", "tuple", "pandas"] = "dict",
    ) -> tuple[np.ndarray, dict[str, np.ndarray] | tuple[np.ndarray] | pd.DataFrame]:
        single_field = isinstance(fields, str)

        if isinstance(indices, da.Array):
            indices = indices.compute()

        indices = np.asarray(indices, dtype=np.int32)

        occupied: da.Array = (
            da.stack([self._props["occupied"][idx] for idx in indices]) if indices.ndim > 1 else self._props["occupied"][indices]
        )  # Induces copy.

        if single_field:
            data: dict[str, np.ndarray] | Sequence[np.ndarray] | pd.DataFrame = None
        elif return_type in {"dict", "pandas"}:
            data = {}
        elif return_type == "tuple":
            data = []
        else:
            msg = f"Invalid return_type {return_type}."
            raise ValueError(msg)

        if single_field:
            fields = [fields]  # type: ignore[list-item]
        elif fields is None:
            fields = itertools.chain(self._fields, ["index"])  # type: ignore[assignment]

        for name in fields:
            # Collect array data.
            #
            # Note that fancy indexing with indices already creates a copy, so
            # only `indices` needs to be copied explicitly.
            if name == "index":
                arr = indices.copy()
            elif name in self._fields:
                arr = (
                    da.stack([self._fields[name][idx] for idx in indices]) if indices.ndim > 1 else self._fields[name][indices]
                )  # Induces copy.
            else:
                msg = f"`{name}` is not a field in this ArrayStore."
                raise ValueError(msg)

            # Accumulate data into the return type.
            if single_field:
                data = arr
            elif return_type == "dict" and isinstance(data, dict):
                data[name] = arr
            elif return_type == "tuple" and isinstance(data, list):
                data.append(arr)
            elif return_type == "pandas" and isinstance(data, dict):
                if len(arr.shape) == 1:  # Scalar entries.
                    data[name] = arr
                elif len(arr.shape) == 2:  # noqa: PLR2004 # 1D array entries.
                    for i in range(arr.shape[1]):
                        data[f"{name}_{i}"] = arr[:, i]
                else:
                    msg = f"Field `{name}` has shape {arr.shape[1:]} -- cannot convert fields with shape >1D to Pandas"
                    raise ValueError(msg)

        occupied, data, *_ = da.compute(occupied, data)

        if single_field:
            data = np.require(data, requirements=["W", "O"])
        else:
            for name, arr in enumerate(data) if return_type == "tuple" else data.items():  # type: ignore[assignment]
                data[name] = np.require(arr, requirements=["W", "O"])
        occupied = np.require(occupied, requirements=["W", "O"])  # type: ignore[assignment]

        # Postprocess return data.
        if return_type == "tuple":
            data = tuple(data)
        elif return_type == "pandas":
            # Data above are already copied, so no need to copy again.
            data = pd.DataFrame.from_dict(data)

        return occupied, data  # type: ignore[return-value]

    def add(
        self,
        indices: np.ndarray | da.Array,
        new_data: dict[str, np.ndarray | da.Array],
        extra_args: dict[str, np.ndarray | da.Array],
        transforms: Sequence[Callable],
    ) -> dict:
        updates = self._props["updates"]
        updates[Update.ADD] += 1
        self._props["updates"] = updates

        add_info: dict = {}

        for transform in transforms:
            occupied, cur_data = self.retrieve(indices)
            indices, new_data, add_info = transform(indices, new_data, add_info, extra_args, occupied, cur_data)

        # Shortcut when there is nothing to add to the store.
        if len(indices) == 0:
            return add_info

        # Verify that the array shapes match the indices.
        for name, arr in new_data.items():
            if len(arr) != len(indices):
                msg = (
                    f"In `new_data`, the array for `{name}` has length {len(arr)} but should be the same length as indices ({len(indices)})"
                )
                raise ValueError(msg)

        # Verify that new_data ends up with the correct fields after the
        # transforms.
        if new_data.keys() != self._fields.keys():
            msg = (
                f"`new_data` had keys {new_data.keys()} but should have the "
                f"same keys as this ArrayStore, i.e., {self._fields.keys()}. "
                "You may be seeing this error if your archive has "
                "extra_fields but the fields were not passed into "
                "archive.add() or scheduler.tell()."
            )
            raise ValueError(msg)

        # Update occupancy data.
        unique_indices = np.where(aggregate(indices, 1, func="len") != 0)[0]
        cur_occupied = self._props["occupied"][unique_indices].compute()
        new_indices = unique_indices[~cur_occupied]
        n_occupied = self._props["n_occupied"]
        new_occupied, new_occupied_list = self._props["occupied"], self._props["occupied_list"]
        new_occupied[new_indices] = True
        new_occupied_list[n_occupied : n_occupied + len(new_indices)] = new_indices

        self._props["occupied"] = new_occupied
        self._props["occupied_list"] = new_occupied_list
        self._props["n_occupied"] = n_occupied + len(new_indices)

        # Insert into the ArrayStore. Note that we do not assume indices are
        # unique. Hence, when updating occupancy data above, we computed the
        # unique indices. In contrast, here we let NumPy's default behavior
        # handle duplicate indices.
        for name, arr in da.compute(*self._fields.items()):
            arr[indices] = new_data[name]
            self._fields[name] = arr

        return add_info

    def clear(self):
        updates = self._props["updates"]
        updates[Update.CLEAR] += 1
        self._props["updates"] = updates
        self._props["n_occupied"] = 0  # Effectively clears occupied_list too.
        self._props["occupied"] = da.full_like(self._props["occupied"], False)

    def resize(self, capacity):
        if capacity <= self._props["capacity"]:
            msg = f"New capacity ({capacity}) must be greater than current capacity ({self._props['capacity']}."
            raise ValueError(msg)

        cur_capacity = self._props["capacity"]
        self._props["capacity"] = capacity

        cur_occupied = self._props["occupied"]
        new_occupied = np.zeros(capacity, dtype=bool)
        new_occupied[:cur_capacity] = cur_occupied
        self._props["occupied"] = new_occupied

        cur_occupied_list = self._props["occupied_list"]
        new_occupied_list = np.empty(capacity, dtype=np.int32)
        new_occupied_list[:cur_capacity] = cur_occupied_list
        self._props["occupied_list"] = new_occupied_list

        for name, cur_arr in self._fields.items():
            new_shape = (capacity, *cur_arr.shape[1:])
            new_field = np.empty(new_shape, cur_arr.dtype)
            new_field[:cur_capacity] = cur_arr
            self._fields[name] = new_field

    def as_raw_dict(self) -> dict:
        d = {}
        for prefix, attr in [("props", self._props), ("fields", self._fields)]:
            for name, val in attr.items():
                if isinstance(val, da.Array):
                    val = da.from_npy_stack(str(self.storage_path / prefix / name), mmap_mode="r").compute()  # noqa: PLW2901
                if isinstance(val, np.ndarray):
                    val = readonly(val.view())  # noqa: PLW2901
                d[f"{prefix}.{name}"] = val
        return d

    @staticmethod
    def from_raw_dict(d: dict, storage_path: str | Path, dask_array_kwargs: dict | None = None) -> "ArrayStore":
        store = ArrayStore({}, 0, storage_path=storage_path, dask_array_kwargs=dask_array_kwargs)  # Create an empty store.
        storage_path = Path(storage_path)

        props = DaskDict(
            {name.removeprefix("props."): arr for name, arr in d.items() if name.startswith("props.")},
            storage_path=storage_path / "props",
            dask_array_kwargs=dask_array_kwargs,
        )
        if props.keys() != store._props.keys():
            msg = f"Expected props to have keys {store._props.keys()} but only found {props.keys()}"
            raise ValueError(msg)

        fields = DaskDict(
            {name.removeprefix("fields."): arr for name, arr in d.items() if name.startswith("fields.")},
            storage_path=storage_path / "fields",
            dask_array_kwargs=dask_array_kwargs,
        )

        store._props = props
        store._fields = fields

        return store
