import argparse
from pathlib import Path
from algatross.experiments.ray_experiment import RayExperiment


DEFAULT_CONFIG = {
    "checkpoint_folder": "",
    "debug": False,
    "log_level": 20,
    "experiment_name": "visualize_moaim",
    "log_dir": "",
    "rich_console": False,
    "ray_config": {"log_to_driver": True},
}


if __name__ == "__main__":
    _opt = argparse.ArgumentParser()
    # main args
    _opt.add_argument('config_or_ckpt', help="Configuration file to resume from, e.g., config/simple_spread/test_algatross_ckpt.yml, or ckpt directory.")
    _opt.add_argument('--num_episodes', help="Number of episodes to render for each island/mainland, for each team.", default=5, type=int)
    _opt.add_argument('--num_island_elites', help="Total number of elites to grab per island.", default=1, type=int)
    _opt.add_argument('--elites_per_island_team', help="How many elites to greedy grab per island team. Remaining allies use random genomes.", default=1, type=int)
    _opt.add_argument('--num_mainland_teams', help="How many of top mainland non-greedy teams to render.", default=1, type=int)
    # filters, default behavior is render nothing and error
    _opt.add_argument('--islands', help="Render islands.", action='store_true')
    _opt.add_argument('--mainlands', help="Render mainlands.", action='store_true')
    opt = _opt.parse_args()

    if (opt.islands | opt.mainlands) == False:
        msg = f"You must select something to render with --islands, --mainlands, or both."
        raise ValueError(msg)

    config_or_ckpt = Path(opt.config_or_ckpt)
    if config_or_ckpt.is_dir():
        jit_config = DEFAULT_CONFIG.copy()
        jit_config["checkpoint_folder"] = config_or_ckpt.as_posix()
        jit_config["log_dir"] = config_or_ckpt.as_posix()  # will end up as config_or_ckpt / visualize_moaim
        experiment_args = {"config": jit_config, "config_file": None}
    else:
        experiment_args = {"config_file": config_or_ckpt.as_posix()}

    print(f"Visualize configuration or ckpt: {opt.config_or_ckpt}")
    experiment = RayExperiment(**experiment_args)
    experiment.render_episodes(
        max_episodes=opt.num_episodes,
        elites_per_island_team=opt.elites_per_island_team,
        num_island_elites=opt.num_island_elites,
        num_mainland_teams=opt.num_mainland_teams,
        render_islands=opt.islands,
        render_mainlands=opt.mainlands,
    )
