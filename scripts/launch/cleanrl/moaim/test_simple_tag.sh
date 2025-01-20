#!/bin/bash

set -e

OMP_NUM_THREADS=12 RAY_COLOR_PREFIX=1 uv run \
    python scripts/training/cleanrl/train_moaim_ppo.py config/simple_tag/test_algatross.yml
