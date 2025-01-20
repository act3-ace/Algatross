#!/bin/bash

set -e

OMP_NUM_THREADS=1 RAY_COLOR_PREFIX=1 uv run \
    python test/test_moaim_determinism.py config/simple_tag/test_det_algatross.yml
