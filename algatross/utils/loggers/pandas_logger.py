"""Logger for output results to pandas dataframe."""

from pathlib import Path
from typing import IO

from algatross.utils.loggers.base_loggers import FileLogger


class PandasLogger(FileLogger):
    """
    PandasLogger defines the attributes and methods for loggers which write to pickled pandas dataframe.

    Parameters
    ----------
    storage_path : str | Path | None, optional
        The storage path to use for logging pandas data, default is :data:`python:None`.
    log_filename : str | None, optional
        The filename to use when logging data, default is :data:`python:None`.
    `**kwargs`
        Additional keyword arguments.
    """

    default_filename: str = "results.pkl"
    """The default filename this logger will use if none is given at instantiation."""

    def __init__(self, storage_path: str | Path | None = None, log_filename: str | None = None, **kwargs):
        self.storage_path = Path("." if storage_path is None else storage_path).absolute()
        if not self.storage_path.exists():
            self.storage_path.mkdir(parents=True)
        self.log_file = self.storage_path / (log_filename or self.default_filename)  # type: ignore[assignment]

    @property
    def log_file(self) -> IO:
        """The output filestream where results will be written.

        Returns
        -------
        IO
            The output filestream
        """
        return self.destination

    @log_file.setter
    def log_file(self, path: Path):
        self.destination = path.open(mode="at", encoding="UTF-8")

    def close(self):  # noqa: D102
        self.log_file.close()
