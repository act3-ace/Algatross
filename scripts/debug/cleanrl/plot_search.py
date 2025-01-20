import numpy as np
import torch
import re
import pandas as pd
import pytz
import argparse

from datetime import datetime
from matplotlib import pyplot as plt
from pathlib import Path


def get_time_stamp(timezone: str = "US/Eastern") -> str:
    """Get current string-like timestamp."""
    utc_now = pytz.utc.localize(datetime.utcnow())
    tz = pytz.timezone(timezone)
    utc_as_tz = utc_now.astimezone(tz)
    return f"{utc_as_tz.strftime('%m%d%y-%H%M%S')}"


def plot_col(f, ax, df, island_id, key, unique, out_dir='figures'):
    # f, ax = plt.subplots(1, 1)
    x = []
    y = []
    y_err = []
    for name, row in df[df['island_id'] == island_id].sort_values(by='epoch').iterrows():
        epoch = int(row['epoch'])
        x.append(epoch)
        data = row[key]
        if isinstance(data, float):
            y.append(data)
        else:
            data_mean = data.mean()
            data_std = data.std()
            y.append(data_mean)
            y_err.append(data_std)

    y = np.asarray(y)
    y_err = np.asarray(y_err)

    ax.plot(x, y, label=unique)

    if len(y_err):
        ax.fill_between(x, y-y_err, y+y_err, alpha=0.2)

    if True in np.isnan(y):
        ax.set_title(f"Island {int(island_id)} (NaN detected)", c='r')
    else:
        ax.set_title(f"Island {int(island_id)}")

    ax.set_xlabel('epochs')
    ax.legend()
    # plt.show()
    # plt.close()


def df_from_path(from_path):
    df = pd.read_pickle(from_path)
    df = df.set_index("index")

    rx = re.compile(r"/(\d+)/")
    for name, row in df.iterrows():
        island_id = rx.findall(name)[0]
        df.at[name, 'island_id'] = int(island_id)

    df = df.astype({
        'island_id': int,
        'epoch': int
    })
    return df


if __name__ == "__main__":
    _opt = argparse.ArgumentParser()
    _opt.add_argument('root_folder', help="List of folders to plot.")
    _opt.add_argument('out_root', help="Where to save")
    opt = _opt.parse_args()

    ts = get_time_stamp()
    
    root_folder = Path(opt.root_folder)
    out_root = Path(opt.out_root)

    figures_dir = out_root / ts / "figures"
    if not figures_dir.exists():
        figures_dir.mkdir(parents=True)

    run_ids = root_folder.glob("*")

    keep_keys = None
    # search_terms = [
    #     "tag_score",
    #     "minimum_ally_speed",
    #     "minimum_adversary_speed",
    #     "closest_ally_distance",
    #     "closest_adversary_distance",
    #     "closest_landmark_distance",
    #     "boundary_penalty",
    # ]
    search_terms = [
        # 'fitness',
        "evolve/info/team/0/agent/agent_0/training/total_loss",
    ]
    term_to_f = {term: (None, None) for term in search_terms}

    for ix, run_id in enumerate(run_ids):
        id_name = run_id.name
        from_path = run_id / "search_data_out" / "df.pkl"
        df = df_from_path(from_path)

        island_ids = df['island_id'].unique()

        ## plot per fitness metric
        max_plt_cols = 7

        for term in search_terms:
            print('=' * 25)
            print(f'\t{term}')
            count = 0

            if keep_keys is None:
                for key in df.keys():
                    if term in key:
                        if key not in keep_keys:
                            count += 1
                            keep_keys.append(key)


            print("total count", count)
            count *= len(island_ids)
            print("total count", count)
            num_rows = max(1, count // max_plt_cols)
            leftover = count % max_plt_cols
            print("num rows", num_rows)
            print("leftover", leftover)
            num_cols = max_plt_cols + bool(leftover)
            print("num cols", num_cols)
            print("kept", keep_keys)

            if term_to_f[term][0] is None:
                f, ax = plt.subplots(ncols=num_cols, nrows=num_rows)
                f.set_size_inches(num_cols * 6, num_rows * 7)
                term_to_f[term] = (f, ax)
            
            col = 0
            row = 0
            counter = 0

            f, ax = term_to_f[term]

            for ix, key in enumerate(keep_keys):

                for island_id, idf in df[df['island_id'] == 0].groupby('island_id'):

                    if num_rows > 1:
                        axes_obj = ax[row, col]
                    else:
                        axes_obj = ax[col]

                    plot_col(f, axes_obj, idf, island_id, key, key + f" {id_name}")
                    col = (col + 1) % num_cols
                    if counter != 0 and col == 0:
                        row += 1
                    
                    counter += 1
                
            f.savefig(figures_dir / f"allisland-{term}.pdf")

        if ix >= 5:
            break

    plt.close()

    print("Done!")