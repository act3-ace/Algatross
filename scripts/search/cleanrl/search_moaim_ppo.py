import sys
import ray
import argparse
import shutil
import traceback
import os

from collections import defaultdict
from pathlib import Path

from algatross.utils.parsers.yaml_loader import load_config
from algatross.utils.merge_dicts import apply_xc_config
from algatross.experiments.ray_experiment import RayExperiment

import torch
import numpy as np
import pandas as pd
import random



if __name__ == "__main__":
    _opt = argparse.ArgumentParser()
    _opt.add_argument('base_config', help="The base mo-marl config")
    _opt.add_argument('child_config', help="A child config created with experiment cascade")
    opt = _opt.parse_args()

    print(f"Using configuration: {opt.base_config}")
    config = load_config(opt.base_config)

    xc = load_config(opt.child_config)
    config = apply_xc_config(config, xc)

    # RayExperiment will follow this pattern for save_folder
    save_folder = Path(xc["save_root"]) / f"search{xc.get('tag', '')}"
    if not save_folder.exists():
        save_folder.mkdir(parents=True)

    config["log_dir"] = xc["save_root"]
    config["experiment_name"] = f"search{xc.get('tag')}"

    shutil.copy2(opt.child_config, save_folder / "extras_config.yml")

    try:
        experiment = RayExperiment(config_file=opt.base_config, config=config)
        experiment.run_experiment()
    except Exception:
        print(traceback.format_exc())
        with open(save_folder / "error.txt", "w") as f:
            f.write(traceback.format_exc())
