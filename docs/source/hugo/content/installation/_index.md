---
title: Installation
---


Prerequisites:

* [uv](https://docs.astral.sh/uv/)
* [swig](https://www.swig.org/index.html)
* Python3 header files

Simply run

```bash
sudo apt-install swig python3-dev
git clone https://git.act3-ace.com/stalwart/ascension/mo-marl.git
cd mo-marl
uv sync --frozen --all-extras
```

This will create the virtual environment at `.venv`, then just prefix your commands with `uv run` or `uv pip`, alternatively alias the following:

```bash
alias python="uv run --frozen python"
alias pip="uv pip"
```
