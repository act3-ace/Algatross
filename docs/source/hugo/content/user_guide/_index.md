---
title: User Guide
---

{{< toctree >}}

## Architecture

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

## Minimum Working Example

To check that everything is working with the simple MPE simple tag environment, you can run:

```bash
./scripts/launch/cleanrl/moaim/test_simple_tag.sh
```

This launcher will invoke the python script `scripts/training/cleanrl/train_moaim_ppo.py` and use an example .yaml configuration for simple tag at `config/simple_tag/test_algatross.yml`. An experiment run will be created at `experiments/`, which you can then reference as a checkpoint to resume later or generate visualizations.

## Load checkpoint

You can load in a checkpoint by creating a configuration file with the key `checkpoint_folder` pointing to the experiment created in the training run (e.g., `experiments/c53d9dd7`). You may also override some other configuration settings, such as number of epochs and number of iterations. As an example, you would modify `checkpoint_folder` in `config/simple_tag/test_algatross_ckpt.yml` with the experiment you created in the MWE, then simply run

```bash
./scripts/launch/cleanrl/moaim/test_simple_tag_ckpt.sh
```

You may also resume from a specific epoch by setting `resume_epoch` in the .yaml configuration file. By default it will automatically grab the latest epoch.

## Visualization

You can visualize your trained multi-agent team on each island with the island visualization script:

```bash
python scripts/visualize/cleanrl/viz_moaim_island.py experiment_config [num_episodes]
```

The only requirement is that your experiment configuration contains an entry for `checkpoint_folder`.
