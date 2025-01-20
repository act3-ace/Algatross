#!/bin/bash -e

# rm -rf mo-marl
# conda create -n marl_env -y python=3.10
# eval "$(command conda 'shell.bash' 'hook' 2>/dev/null)"
# conda run -n marl_env pip install poetry
# conda run -n marl_env poetry install --with=dev,test,lint
# conda run -n marl_env poetry install  # normal
# conda run -n marl_env poetry install --with=dev,test,lint,docs --all-extras

# uv sync will install dev,test,lint since they are default groups
uv sync --frozen --all-extras
# deprecated:
# conda run -n marl_env pip install 'setuptools<70' numpy==1.25.0
