# Problem
# Algo
# Population
# Island
# Archipelago


# from algatross.algorithms.genetic.mo_aim.archipelago import DaskMOAIMArchipelago, RayMOAIMArchipelago

# from algatross.algorithms.genetic.mo_aim.islands import RayMOAIMIslandUDI, RayMOAIMMainlandUDI
import sys

from algatross.experiments.ray_experiment import RayExperiment

if __name__ == "__main__":
    config_file = "config/simple_tag/test_algatross.yml" if len(sys.argv) < 2 else sys.argv[1]  # noqa: PLR2004
    print(f"Running configuration: {config_file}")
    try:
        experiment = RayExperiment(config_file=config_file)
        experiment.run_experiment()
    except KeyboardInterrupt:
        print("\n\nEarly exit.\n\n")
