"""Plotting utilities."""

import json
import re

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import pandas as pd

import numpy as np

import pytz

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

rx = re.compile(r"/(\d+)/")  # regex for island id
rt = re.compile(r"^(\w+)\/")  # regex for island type


DEFAULT_SIZE_GUIDANCE: dict[str, int] = {
    "compressedHistograms": 1,
    "images": 1,
    "scalars": 0,  # 0 means load all
    "histograms": 0,
}
"""Default guidance to pass to a tensorboard ``EventAccumulator``."""


@dataclass
class IslandMetaData:
    """Metadata for an island."""

    island_id: int
    """The ID of the island in the archipelago."""
    island_type: str
    """The type of island (island/mainland)."""


def metric_path_to_island_meta(s: str) -> IslandMetaData:
    """
    Convert a logged metric to island metadata.

    Parameters
    ----------
    s : str
        The logged metric path.

    Returns
    -------
    IslandMetaData
        The metadata for the island.
    """
    island_id = rx.findall(s)[0]  # use first match
    island_type = rt.findall(s)[0]

    return IslandMetaData(island_id=island_id, island_type=island_type)


"""
cf https://github.com/theRealSuperMario/supermariopy/blob/master/scripts/tflogs2pandas.py
modified to create correct indexing behavior
modified to add histogram post processing
"""


def tflog2pandas(path: str | Path, mode: str = "scalar") -> pd.DataFrame:
    """Convert single tensorflow log file to pandas DataFrame.

    Parameters
    ----------
    path : str | Path
        The path to tensorflow log file
    mode : str, optional
        The conversion mode, default is :python:`scalar`

    Returns
    -------
    pd.DataFrame
        The converted dataframe

    Raises
    ------
    ValueError
        If an invalid ``mode`` is specified.
    """
    if mode == "scalar":
        runlog_data = pd.DataFrame({"metric": [], "value": [], "step": [], "index": []})
    elif mode == "histogram":
        runlog_data = pd.DataFrame({"metric": [], "height": [], "limits": [], "step": [], "index": []})
    else:
        msg = f"Bad mode given: {mode}. Must be either 'scalar' or 'histogram'"
        raise ValueError(msg)

    try:
        event_acc = EventAccumulator(path, DEFAULT_SIZE_GUIDANCE)
        event_acc.Reload()

        if mode == "scalar":
            tags = event_acc.Tags()["scalars"]
            global_increment = 0
            for tag in tags:
                event_list = event_acc.Scalars(tag)
                values = [x.value for x in event_list]
                step = [x.step for x in event_list]
                index = np.arange(len(step)) + global_increment
                r = {"metric": [tag] * len(step), "value": values, "step": step, "index": index}
                r = pd.DataFrame(r, index=index)
                runlog_data = pd.concat([runlog_data, r])
                global_increment += len(step)

        elif mode == "histogram":
            tags = event_acc.Tags()["histograms"]
            global_increment = 0
            for tag in tags:
                event_list = event_acc.Histograms(tag)

                def push_zeroth_limit(limits):
                    # make compatible with pyplot stairs()
                    dx = abs(limits[1] - limits[0])
                    return [limits[0] - dx, *limits]

                limits = [x.histogram_value.bucket_limit for x in event_list]
                limits = list(map(push_zeroth_limit, limits))
                heights = [x.histogram_value.bucket for x in event_list]

                step = [x.step for x in event_list]
                index = np.arange(len(step)) + global_increment
                r = {"metric": [tag] * len(step), "heights": heights, "edges": limits, "step": step, "index": index}
                r = pd.DataFrame(r, index=index)
                runlog_data = pd.concat([runlog_data, r])
                global_increment += len(step)

    # Dirty catch of DataLossError
    except Exception:  # noqa: BLE001
        import traceback  # noqa: PLC0415

        print(f"Event file possibly corrupt: {path}")
        traceback.print_exc()

    return runlog_data


def get_time_stamp(tz: str = "US/Eastern") -> str:
    """
    Get current string-like timestamp.

    Parameters
    ----------
    tz : str, optional
        The timezone for the timestamp, by default "US/Eastern"

    Returns
    -------
    str
        The timestamp as a formatted string.
    """
    utc_now = pytz.utc.localize(datetime.now(timezone.utc))
    return f"{utc_now.astimezone(pytz.timezone(tz)).strftime('%m%d%y-%H%M%S')}"


def eventfile_to_dataframe(file_path: str | Path, out_format: str = "table", mode: str = "scalar") -> pd.DataFrame:
    """
    Convert a tensorboard event file to a dataframe.

    Parameters
    ----------
    file_path : str | Path
        The path of the tensorboard event file to convert.
    out_format : str, optional
        The output out_format, either :python:`"table"` or :python:`"tensorboard"`, by default :python:`"table"`
    mode : str, optional
        The conversion mode, either :python:`"scalar"` or :python:`"histogram"`, by default :python:`"scalar"`

    Returns
    -------
    pd.DataFrame
        The event file converted to a pandas :class:`~pandas.DataFrame`

    Raises
    ------
    NotImplementedError
        If :python:`out_format == "tensorboard"`
    ValueError
        If an invalid ``mode`` is given.
    """
    if mode not in {"scalar", "histogram"}:
        # Error early
        msg = f"Bad mode given: {mode}. Must be either 'scalar' or 'histogram'"
        raise ValueError(msg)

    frame = tflog2pandas(file_path, mode=mode).astype({"index": int, "step": int})

    if out_format == "tensorboard":
        # |- metric   value   step -|
        # |  abc/0/b  0.000     2  -|
        # |  abc/0/c  0.000     1  -|
        msg = "tensorboard out_format not compatible with this script."
        raise NotImplementedError(msg)
    if out_format == "table":
        # |- k1  k2  k3  epoch -|
        # |  v1  v2  v3    0   -|
        # |  v2  v3  v4    1   -|
        metrics = frame["metric"].unique()
        island_meta_data = metric_path_to_island_meta(metrics[0])

        _island_type = island_meta_data.island_type
        _island_id = island_meta_data.island_id

        keep_metric_map = {metric: 0 for metric in metrics if "epoch" in metric}
        keep_metric_map |= {metric: 1 for metric in metrics if "epoch" not in metric}

        out_df = pd.DataFrame()
        _ix = 0
        for epoch, sdf in frame.groupby("step"):
            data = defaultdict(list)
            for _, row in sdf.iterrows():
                if not keep_metric_map[row["metric"]]:
                    continue

                if mode == "scalar":
                    data[row["metric"] + "/value"].append(row["value"])
                elif mode == "histogram":
                    data[row["metric"] + "/edges"].append(row["edges"])
                    data[row["metric"] + "/heights"].append(row["heights"])

            new_df = pd.DataFrame.from_dict(data)
            new_df["epoch"] = epoch
            new_df["step"] = epoch
            out_df = pd.concat([out_df, new_df])

        return out_df

    msg = f"Invalid out_format: {out_format}"
    raise ValueError(msg)


def resultjson_to_dataframe(file_path: Path, out_format: str = "table") -> pd.DataFrame:
    """
    Convert a JSON file of results to a dataframe.

    Parameters
    ----------
    file_path : Path
        The path of the JSON file.
    out_format : str, optional
        The output out_format. Either :python:`"table"` or :python:`"tensorboard"`, by default "table"

    Returns
    -------
    pd.DataFrame
        The JSON file loaded as a :class:`~pandas.DataFrame`

    Raises
    ------
    ValueError
        If an invalid ``out_format`` is given.
    """
    if out_format == "table":
        with file_path.open("r") as f:
            data = defaultdict(list)
            # json1
            # json2
            # ...
            # jsonN
            jl = f.readlines()
            for ix, j in enumerate(jl):
                results = json.loads(j)
                any_metric = next(iter(results.keys()))
                meta_data = metric_path_to_island_meta(any_metric)
                # |- k1  k2  k3 -|
                # |  v1  v2  v3 -|
                for k, v in results.items():
                    if "epoch" in k:
                        data["epoch"].append(v)
                    else:
                        data[k].append(v)

                data["index"].append(f"{ix}")
                data["island_type"].append(meta_data.island_type)
                data["island_id"].append(meta_data.island_id)

        frame = pd.DataFrame(data=data)

    elif out_format == "tensorboard":
        with file_path.open("r") as f:
            data = defaultdict(list)
            # json1
            # json2
            # ...
            # jsonN
            jl = f.readlines()
            ix = 0
            for j in jl:
                results = json.loads(j)
                any_metric = next(iter(results.keys()))
                meta_data = metric_path_to_island_meta(any_metric)
                prefix = f"{meta_data.island_type}/{meta_data.island_id}"

                # |- metric   value   step -|
                # |  abc/0/b  0.000     2  -|
                for k, v in results.items():
                    if isinstance(v, list) and len(v) == 0:
                        continue
                    if k == f"{prefix}/epoch":
                        continue
                    data["metric"].append(k)
                    data["step"].append(results[f"{prefix}epoch"])
                    data["value"].append(v)
                    data["index"].append(ix)
                    data["island_type"].append(meta_data.island_type)
                    data["island_id"].append(meta_data.island_id)
                    ix += 1

            frame = pd.DataFrame(data=data)
    else:
        msg = f"Invalid out_format: {out_format}"
        raise ValueError(msg)

    return postprocess_dataframe(frame)


def postprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the content of a dataframe.

    This will set the index to :python:`"index"`, and cast the type of the columns:

    - :python:`"island_id"` &rarr; :class:`python:int`
    - :python:`"epoch"` &rarr; :class:`python:int`

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to postprocess.

    Returns
    -------
    pd.DataFrame
        The postprocessed dataframe.
    """
    return df.set_index("index").astype({"island_id": int, "epoch": int})


def plot_metric_key(
    f: mpl.figure.Figure,  # noqa: ARG001
    ax: mpl.axes.Axes,
    df: pd.DataFrame,
    island_id: int,
    key: str,
    printable: str | None = None,
    reduction: str = "none",
):
    """
    Plot the specific metric housed under ``key``.

    Parameters
    ----------
    f : mpl.figure.Figure
        The figure for rendering (unused)
    ax : mpl.axes.Axes
        The axis object to render the plot on.
    df : pd.DataFrame
        The dataframe containing the data to plot.
    island_id : int
        The ID of the island.
    key : str
        The metric to plot
    printable : str | None, optional
        The label to apply, by default None
    reduction : str, optional
        Reduction method to use, by default "none"
    """
    x = []
    y = []
    y_err = []
    for _name, row in df[df["island_id"] == island_id].sort_values(by="epoch").iterrows():
        epoch = int(row["epoch"])
        data = row[key]
        if isinstance(data, float):
            x.append(epoch)
            y.append(data)
        else:
            data = np.asarray(data)
            if reduction == "mean":
                x.append(epoch)
                data_mean = data.mean()
                data_std = data.std()
                y.append(data_mean)
                y_err.append(data_std)
            else:
                num_steps = np.prod(data.shape)
                x.extend(np.linspace(start=0, stop=1 - (1.0 / num_steps), num=num_steps) + epoch)
                data = data.reshape(np.prod(data.shape))
                y.extend(data)
                y_err.extend(np.zeros_like(data))

    y = np.asarray(y)  # type: ignore[assignment]
    y_err = np.asarray(y_err)  # type: ignore[assignment]

    ax.plot(x, y, label=printable if printable is not None else key)

    if len(y_err):
        ax.fill_between(x, y - y_err, y + y_err, alpha=0.2)  # type: ignore[operator]

    if True in np.isnan(y):
        ax.set_title(f"Island {int(island_id)} (NaN detected)", c="r")
    else:
        ax.set_title(f"Island {int(island_id)}")

    ax.set_xlabel("epochs")
    ax.legend()
