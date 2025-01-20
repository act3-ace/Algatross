import numpy as np
import torch
import re
import pandas as pd
import pytz
import argparse
import re
import json
import itertools
import logging

from time import sleep
from datetime import datetime
from matplotlib import pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from algatross.utils.parsers.yaml_loader import load_config

logging.basicConfig()

MIN_TENSORBOARD_FILE_SIZE_B = 100
MIN_RESULTSJSON_FILE_SIZE_B = 100
rx = re.compile(r"/(\d+)/")  # regex for island id
rt = re.compile(r"^(\w+)\/")  # regex for island type


@dataclass
class IslandMetaData:
    island_id: int
    island_type: str


def metric_path_to_island_meta(s: str) -> IslandMetaData:
    ## s: island/0/aaaaa/bbb/ccccc
    island_id = rx.findall(s)[0]  # use first match
    island_type = rt.findall(s)[0]

    return IslandMetaData(
        island_id=island_id,
        island_type=island_type,
    )


"""
cf https://github.com/theRealSuperMario/supermariopy/blob/master/scripts/tflogs2pandas.py
modified to create correct indexing behavior
"""
def tflog2pandas(path: str) -> pd.DataFrame:
    """Convert single tensorflow log file to pandas DataFrame.

    Parameters
    ----------
    path : str
        path to tensorflow log file

    Returns
    -------
    pd.DataFrame
        converted dataframe
    """
    DEFAULT_SIZE_GUIDANCE = {
        "compressedHistograms": 1,
        "images": 1,
        "scalars": 0,  # 0 means load all
        "histograms": 1,
    }
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": [], "index": []})

    try:
        event_acc = EventAccumulator(path, DEFAULT_SIZE_GUIDANCE)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        global_increment = 0
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            index = np.arange(len(step)) + global_increment
            r = {"metric": [tag] * len(step), "value": values, "step": step, "index": index}
            r = pd.DataFrame(r, index=index)
            runlog_data = pd.concat([runlog_data, r])
            global_increment += len(step)
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()

    return runlog_data


def get_time_stamp(timezone: str = "US/Eastern") -> str:
    """Get current string-like timestamp."""
    utc_now = pytz.utc.localize(datetime.utcnow())
    tz = pytz.timezone(timezone)
    utc_as_tz = utc_now.astimezone(tz)
    return f"{utc_as_tz.strftime('%m%d%y-%H%M%S')}"


def eventfile_to_dataframe(file_path: Path, format="table") -> pd.DataFrame:
    df = tflog2pandas(file_path).astype({
        'index': int,
        'step': int,
    })
    if format == "tensorboard":
        # |- metric   value   step -|
        # |  abc/0/b  0.000     2  -|
        # |  abc/0/c  0.000     1  -|
        raise NotImplementedError("tensorboard format not compatible with this script.")
    elif format == "table":
        # |- k1  k2  k3  epoch -|
        # |  v1  v2  v3    0   -|
        # |  v2  v3  v4    1   -|
        metrics = df['metric'].unique()
        island_meta_data = metric_path_to_island_meta(metrics[0])

        island_type = island_meta_data.island_type
        island_id = island_meta_data.island_id

        keep_metric_map = {
            metric: 0 for metric in metrics if 'epoch' in metric
        }
        keep_metric_map |= {
            metric: 1 for metric in metrics if 'epoch' not in metric
        }

        out_df = pd.DataFrame()
        ix = 0
        for epoch, sdf in df.groupby("step"):
            data = defaultdict(list)
            for _, row in sdf.iterows():
                if not keep_metric_map[row['metric']]:
                    continue

                data[row['metric']].append(row['value'])

            new_df = pd.DataFrame.from_dict(data)
            new_df['epoch'] = epoch
            new_df['step'] = epoch
            out_df = pd.concat([out_df, new_df])
        
        df = out_df
            
    else:
        raise ValueError(f"Invalid format: {format}")

    df = postprocess_dataframe(df)
    return df


def resultjson_to_dataframe(file_path: Path, format="table") -> pd.DataFrame:
    if format == "table":
        with file_path.open("r") as f:
            data = defaultdict(list)
            # json1
            # json2
            # ...
            # jsonN
            jl = f.readlines()
            for ix, j in enumerate(jl):
                results = json.loads(j)
                any_metric = list(results.keys())[0]
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

        df = pd.DataFrame(data=data)

    elif format == "tensorboard":
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
                any_metric = list(results.keys())[0]
                meta_data = metric_path_to_island_meta(any_metric)
                prefix = f"{meta_data.island_type}/{meta_data.island_id}"
                
                # |- metric   value   step -|
                # |  abc/0/b  0.000     2  -|
                for k, v in results.items():
                    if isinstance(v, list) and len(v) == 0: continue
                    if k == f'{prefix}/epoch': continue
                    data['metric'].append(k)
                    data['step'].append(results[f'{prefix}epoch'])
                    data['value'].append(v)
                    data['index'].append(ix)
                    data['island_type'].append(meta_data.island_type)
                    data['island_id'].append(meta_data.island_id)
                    ix += 1

            df = pd.DataFrame(data=data)
    else:
        raise ValueError(f"Invalid format: {format}")

    df = postprocess_dataframe(df)
    return df


def postprocess_dataframe(df):
    df = df.set_index("index")

    return df.astype({
            'island_id': int,
            'epoch': int
    })


def plot_metric_key(f, ax, df, island_id, key, printable=None, reduction='none'):
    x = []
    y = []
    y_err = []
    for name, row in df[df['island_id'] == island_id].sort_values(by='epoch').iterrows():
        epoch = int(row['epoch'])
        data = row[key]
        if isinstance(data, float):
            x.append(epoch)
            y.append(data)
        else:
            # shape: ( island_iter, uda_iter * sgd_iter )
            data = np.asarray(data)
            if reduction == 'mean':
                x.append(epoch)
                data_mean = data.mean()
                data_std = data.std()
                y.append(data_mean)
                y_err.append(data_std)
            else:
                num_steps = np.prod(data.shape)
                x.extend( np.linspace(start=0, stop=1-(1./num_steps), num=num_steps) + epoch )
                data = data.reshape(np.prod(data.shape))
                y.extend(data)
                y_err.extend(np.zeros_like(data))

    y = np.asarray(y)
    y_err = np.asarray(y_err)

    ax.plot(x, y, label=printable if printable is not None else key)

    if len(y_err):
        ax.fill_between(x, y-y_err, y+y_err, alpha=0.2)

    if True in np.isnan(y):
        ax.set_title(f"Island {int(island_id)} (NaN detected)", c='r')
    else:
        ax.set_title(f"Island {int(island_id)}")

    ax.set_xlabel('epochs')
    ax.legend()


if __name__ == "__main__":
    _opt = argparse.ArgumentParser()
    _opt.add_argument('folders', nargs="+", help="List of folders to plot.")
    _opt.add_argument('--xc', action='store_true', help="Load from experiment cascade configs.")
    _opt.add_argument('--xc_island_types', nargs="+", help="Island types to consider for experiment cascade", default=["island", "mainland"])
    # _opt.add_argument('search_terms',  help="")
    _opt.add_argument('--priority', default="resultsjson", help="Prioritize resultsjson or tensorboard eventfile.")
    opt = _opt.parse_args()

    ts = get_time_stamp()
    logger = logging.getLogger(f"plot://{ts}")
    logger.setLevel(logging.DEBUG)

    out_roots = []
    if opt.xc:
        for config_root in opt.folders:
            config_files = itertools.chain(Path(config_root).glob("*.yml"), Path(config_root).glob("*.yaml"))
            for xc_file in config_files:
                xc = load_config(xc_file)
                experiment_name = f"search{xc.get('tag', '')}"
                out_roots.extend([
                    (Path(xc['save_root']) / experiment_name / island_type).as_posix()
                    for island_type in opt.xc_island_types
                ])  
    else:
        out_roots = opt.folders

    for out_root in out_roots:
        logger.info("\t" + out_root)
        out_root = Path(out_root)

        if opt.priority == "resultsjson":
            results_path = out_root / "results.json"
            if not results_path.exists():
                raise RuntimeError(f"No file found at {results_path.as_posix()}")
            if results_path.stat().st_size < MIN_RESULTSJSON_FILE_SIZE_B:
                logger.warning(f"File for {out_root} wasn't big enough! Skipping...")
                sleep(1)
                continue
            
            df = resultjson_to_dataframe(results_path)
        else:
            candidates = list(out_root.glob("events.out.tfevents*"))
            if len(candidates) == 0:
                raise RuntimeError(f"No files matching ` events.out.tfevents* ' found at {out_root.as_posix()}")
            candidates = [c for c in candidates if c.stat().st_size >= MIN_TENSORBOARD_FILE_SIZE_B]
            if len(candidates) == 0:
                logger.warning(f"No file big enough! Looked for >= {MIN_TENSORBOARD_FILE_SIZE_B} bytes")
                sleep(1)
                continue

            results_path = list(candidates)[0].as_posix()
            df = eventfile_to_dataframe(results_path)

        figures_dir = Path(out_root) / "figures" / ts
        if not figures_dir.exists():
            figures_dir.mkdir(parents=True)


        ## TODO: redo this so its 1 island per row or something
        ## All islands on a single plot for each metric
        max_plt_cols = 7
        island_ids = df['island_id'].unique()
        keep_keys = []

        logger.debug('=' * 25)
        search_terms = [
            'loss',
            "reward",
            # "tag_score",
        ]

        for key in df.keys():
            for term in search_terms:
                if term in key:
                    if key not in keep_keys:
                        keep_keys.append(key)


        logger.debug(f"\ttotal count {len(keep_keys)}")
        count = len(keep_keys) * len(island_ids)
        logger.debug(f"\ttotal count {count}")
        num_rows = max(1, count // max_plt_cols) + bool(count % max_plt_cols)
        logger.debug(f"\tnum rows {num_rows}")
        num_cols = max_plt_cols
        logger.debug(f"\tnum cols {num_cols}")
        logger.debug(f"\tkept {keep_keys}")

        f, ax = plt.subplots(ncols=num_cols, nrows=num_rows)
        f.set_size_inches(num_cols * 6, num_rows * 7)
        col = 0
        row = 0
        counter = 0

        for ix, key in enumerate(keep_keys):
            for island_id, idf in df.groupby('island_id'):
                if num_rows > 1:
                    axes_obj = ax[row, col]
                else:
                    axes_obj = ax[col]

                display_key = "/".join(key.split("/")[-2:])
                plot_metric_key(f, axes_obj, idf, island_id, key, printable=display_key)
                col = (col + 1) % num_cols
                if counter != 0 and col == 0:
                    row += 1
                
                counter += 1
        
        fig_path = figures_dir / f"allisland-{'-'.join(search_terms)}.pdf"
        f.savefig(fig_path)
        logger.info(f"\t>> Saved to {fig_path}")
        plt.close()

        ## a plot per metric
        max_plt_cols = 7

        ## TODO: as args
        search_terms = [
            # "tag_score",
            # "minimum_ally_speed",
            # "minimum_adversary_speed",
            # "closest_ally_distance",
            # "closest_adversary_distance",
            # "closest_landmark_distance",
            # "boundary_penalty",
            "loss",
            "reward",
        ]

        for term in search_terms:
            logger.debug('=' * 25)
            logger.debug(f'\t\t{term}')
            keep_keys = []

            for key in df.keys():
                if term in key:
                    if key not in keep_keys:
                        keep_keys.append(key)


            logger.debug(f"\ttotal count {count}")
            count = len(keep_keys) * len(island_ids)
            logger.debug(f"\ttotal count {count}")
            num_rows = max(1, count // max_plt_cols) + bool(count % max_plt_cols)
            logger.debug(f"\tnum rows {num_rows}")
            logger.debug(f"\tleftover {count % max_plt_cols}")
            num_cols = max_plt_cols
            logger.debug(f"\tnum cols {num_cols}")
            logger.debug(f"\tkept {keep_keys}")

            f, ax = plt.subplots(ncols=num_cols, nrows=num_rows)
            f.set_size_inches(num_cols * 6, num_rows * 7)
            col = 0
            row = 0
            counter = 0

            for ix, key in enumerate(keep_keys):

                for island_id, idf in df.groupby('island_id'):

                    if num_rows > 1:
                        axes_obj = ax[row, col]
                    else:
                        axes_obj = ax[col]

                    display_key = "/".join(key.split("/")[-3:])
                    plot_metric_key(f, axes_obj, idf, island_id, key, printable=display_key)
                    col = (col + 1) % num_cols
                    if counter != 0 and col == 0:
                        row += 1
                    
                    counter += 1
            
            fig_path = figures_dir / f"allisland-{term}.pdf"
            f.savefig(fig_path)
            logger.info(f"\t>> Saved to {fig_path}")
            plt.close()

    logger.info("Done!")
