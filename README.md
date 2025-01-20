# Algatross

A multi-objective multi-agent reinforcement learning library centered around asymmetric coevolution in PyTorch

Features:

* Total algorithm customization from .yaml files.
* Total integration with [Ray](https://www.ray.io/) library for highly scalable training and inference.
* Fast and stable distributed implementation of Multi-objective Asymmetric Island Model (MO-AIM) proposed by [Dixit and Tumer (2023)](https://doi.org/10.1145/3583131.3590524).
* Integration with Heterogeneous-Agent Reinforcement Learning (HARL) proposed by [Zhong et al. (2024)](http://arxiv.org/abs/2304.09870).
* Baseline implementations for PPO in RLlib and CleanRL.
* Distributed plotting and visualization for supported environments.
* Native filesystem-based multi-agent checkpoints. Navigate archipelago checkpoints entirely in a file browser.

## Installation

Prerequisites:

* swig (e.g., `apt install swig` on debian variants)
* Python3 header files (e.g,. `apt install python3-dev`)
* [uv](https://docs.astral.sh/uv/)

Simply run

```bash
./startup.sh
```

This will create the virtual environment at `.venv`, then just prefix your commands with `uv run` or `uv pip`, alternatively alias the following:

```bash
alias python="uv run --frozen python"
alias pip="uv pip"
```

## Usage

Some common workloads may be found in `scripts/launch`.

### MWE

To check that everything is working with the simple MPE simple tag environment, you can run:

```bash
./scripts/launch/cleanrl/moaim/test_simple_tag.sh
```

This launcher will invoke the python script `scripts/training/cleanrl/train_moaim_ppo.py` and use an example .yaml configuration for simple tag at `config/simple_tag/test_algatross.yml`. An experiment run will be created at `experiments/`, which you can then reference as a checkpoint to resume later or generate visualizations.

### Load checkpoint

You can load in a checkpoint by creating a configuration file with the key `checkpoint_folder` pointing to the experiment created in the training run (e.g., `experiments/c53d9dd7`). You may also override some other configuration settings, such as number of epochs and number of iterations. As an example, you would modify `checkpoint_folder` in `config/simple_tag/test_algatross_ckpt.yml` with the experiment you created in the MWE, then simply run

```bash
./scripts/launch/cleanrl/moaim/test_simple_tag_ckpt.sh
```

You may also resume from a specific epoch by setting `resume_epoch` in the .yaml configuration file. By default it will automatically grab the latest epoch.

### Visualization

You can visualize your trained multi-agent team on each island with the island visualization script:

```bash
python scripts/visualize/cleanrl/viz_moaim_island.py experiment_config [num_episodes]
```

The only requirement is that your experiment configuration contains an entry for `checkpoint_folder`.

<!-- ## Documentation

The documentation for MO-MARL is organized as follows:

- **[Quick Start Guide](docs/quick-start-guide.md)**: provides documentation of prerequisites, downloading, installing, and configuring MO-MARL.
- **[User Guide](docs/user-guide.md)**: provides a conceptual overview of MO-MARL by explaining key concepts. This doc also helps users understand the benefits, usage, and best practices for working with MO-MARL. -->

## Supported Environments

| Environment        | Supported | Tested    | Special Install Instructions |
|--------------------|-----------|-----------|------------------------------|
| MPE Simple Tag     | Yes       | Yes       | None!                        |
| MPE Simple Spread  | Yes       | Yes       | None!                        |
| SMACv2             | Yes       | No | None!                        |
| STARS MAInspection | Yes       | Yes       | None!           |

<!-- * [1] `uv sync --all-extras -->

## Architecture

### MO-AIM

Our MO-AIM implementation follows this logical layout:

<!-- markdownlint-disable MD040-->
```
RayExperiment - algatross/experiments/ray_experiment.py
├─ Main entrypoint (e.g., run_experiment)
├─ Environment registration (from RLlib)
├─ Experiment configuration (a dictionary)
│
└─ [RayActor] MOAIMRayArchipelago - algatross/algorithms/genetic/mo_aim/archipelago/ray_archipelago.py
    ├─ Entrypoint for archipelago evolution (evolve())
    │
    ├─ [RayActor] PopulationServer - algatross/algorithms/genetic/mo_aim/population.py
    │   ├─ MOAIMIslandPopulation for island/mainland 0
    │   ├─ ...
    │   └─ MOAIMIslandPopulation for island/mainland n
    │       ├─ PyRibs Emitters
    │       │   └─ Mutation and elites logic
    │       ├─ PyRibs Archive
    │       ├─ Quality Diversity (QD) logic
    │       └─ Rollout buffers
    │
    └─ Dictionary of islands
        ├─ [RayActor] IslandServer for island/mainland 0 - algatross/algorithms/genetic/mo_aim/islands/ray_islands.py
        ├─ ...
        └─ [RayActor] IslandServer for island/mainland n ...
            ├─ Environment (MultiAgentEnv from RLlib)
            └─ Island (UDI)
                ├─ Evolution entrypoint!
                ├─ Algorithm (UDA)
                │   └─ Team evolution logic
                └─ Problem (UDP)
                    ├─ Fitness scorer
                    └─ Environment Runner (Tape machine) - algatross/environments/runners.py

```
<!-- markdownlint-enable MD040-->

where [RayActor] denotes a new spawned actor in the ray cluster, which is logically treated as a forked thread leveraging the ray cluster network for IPC.

## ACE Hub Environment

Launch this library in an ACE Hub environment with this link:

**[MO-MARL ACE Hub Environment][ace-hub-url]**

> Alternative: **[VPN-only link][ace-hub-url-vpn]**

### Development Environments

Launch the latest development version of this library in an ACE Hub environment with this link:

**[MO-MARL Development Environment][ace-hub-url-dev]**

> Alternative: **[VPN-only link][ace-hub-url-vpn-dev]**

## How to Contribute

* **[Developer Guide](docs/developer-guide.md)**: detailed guide for contributing to the MO-MARL repository.

## Support

* Message @tgresavage or @wgarcia on Mattermost.
* **[Troubleshooting FAQ](docs/troubleshooting-faq.md)**: consult list of frequently asked questions and their answers.
* **Mattermost channel(<!-- replace this with a URL and make link active -->)**: create a post in the MO-MARL channel for assistance.
* **Create a GitLab issue by email(<!-- replace this with a URL and make link active -->)**

[ace-hub-url]: <https://hub.ace.act3.ai/environments/0?replicas=1&image=reg.git.act3-ace.com/stalwart/ascension/mo-marl:latest&hubName=mo-marl&proxyType=normal&resources\[cpu\]=1&resources\[memory\]=1Gi&shm=64Mi>
[ace-hub-url-vpn]: <https://hub.lion.act3-ace.ai/environments/0?replicas=1&image=reg.git.act3-ace.com/stalwart/ascension/mo-marl/cicd:latest&hubName=mo-marl&proxyType=normal&resources\[cpu\]=1&resources\[memory\]=1Gi&shm=64Mi>
[ace-hub-url-dev]: <https://hub.ace.act3.ai/environments/0?replicas=1&image=reg.git.act3-ace.com/stalwart/ascension/mo-marl:latest&hubName=mo-marl&proxyType=normal&resources\[cpu\]=1&resources\[memory\]=1Gi&shm=64Mi>
[ace-hub-url-vpn-dev]: <https://hub.lion.act3-ace.ai/environments/0?replicas=1&image=reg.git.act3-ace.com/stalwart/ascension/mo-marl/cicd:latest&hubName=mo-marl&proxyType=normal&resources\[cpu\]=1&resources\[memory\]=1Gi&shm=64Mi>

### Troubleshooting

1. ```bash
    Authorization error accessing https://git.act3-ace.com/api/v4/projects/1287/packages/pypi/simple/corl/
    ```

    Try running the ACT3 login script (or `act3-pt login`), or try refreshing the access token in your `~/.netrc` file.
