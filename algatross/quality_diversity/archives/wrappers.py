"""Module of archive wrappers."""

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from ribs.archives import ArchiveBase


class FileArchiveWrapper:
    """
    A wrapper for storing archive to disk instead of memory.

    Parameters
    ----------
    archive : ArchiveBase
        The archive to wrap.
    name : str
        The name to store the archive under.
    storage_path : Path
        The path to store the archive under.
    `**kwargs`
        Additional keyword arguments.
    """

    unwrapped: ArchiveBase
    """The unwrapped archive."""
    archive_name: str
    """The name under which the archive is stored."""
    archive_path: Path
    """The path under which the archive is stored, including the
    :attr:`~algatross.quality_diversity.archives.wrappers.FileArchiveWrapper.archive_name`.
    """

    def __init__(self, archive: ArchiveBase, name: str, storage_path: Path, **kwargs):
        if not (storage_path / "archives" / name).exists():
            (storage_path / "archives" / name).mkdir(parents=True)
        self.unwrapped = archive
        self.archive_name = name
        self.archive_path = storage_path / "archives" / name
        self.load()

    def load(self):
        """Load the fields and property arrays as memory maps (disk)."""
        for obj in (self.unwrapped._store._fields, self.unwrapped._store._props):  # noqa: SLF001
            for field, arr in obj.items():
                if not (self.archive_path / f"{field}.npy").exists() or not isinstance(arr, np.memmap):
                    np.save(self.archive_path / field, arr, allow_pickle=False)
                obj[field] = np.load(self.archive_path / f"{field}.npy", mmap_mode="r+", allow_pickle=False)

    def __dir__(self) -> Iterable[str]:  # noqa: D105
        return ["unwrapped", "archive_name", "archive_path", "load"]

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401, D105
        if name not in dir(self):
            return getattr(self.unwrapped, name)
        return self.__dict__.get(name)

    def __setattr__(self, name: str, value: Any) -> None:  # noqa: ANN401, D105
        if name not in dir(self):
            setattr(self.unwrapped, name, value)
        self.__dict__[name] = value

    def __delattr__(self, name: str) -> None:  # noqa: D105
        if name not in dir(self):
            delattr(self.unwrapped, name)
        self.__dict__.pop(name, None)

    def __len__(self) -> int:  # noqa: D105
        return len(self.unwrapped)

    def __iter__(self) -> Iterator:  # noqa: D105
        return iter(self.unwrapped)
