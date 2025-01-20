#!/bin/bash

set -e

# XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform
OMP_NUM_THREADS=1 RAY_COLOR_PREFIX=1 uv run \
    python scripts/training/cleanrl/train_moaim_ppo.py config/ma_inspection/inspection/test_algatross.yml
