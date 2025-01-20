import sys

from pathlib import Path
from uuid import uuid4

from algatross.experiments.test_det_experiment import TestDeterminismExperiment
from algatross.utils.random import seed_global

if __name__ == "__main__":
    config_file = "config/simple_tag/test_algatross.yml" if len(sys.argv) < 2 else sys.argv[1]  # noqa: PLR2004
    print(f"Running configuration: {config_file}")
    seed_global(2629859447)

    experiment = TestDeterminismExperiment(config_file=config_file, test_dir="experiment_tests")
    entropy_tree = experiment.run_experiment()
    print("Generator entropies as they were spawned:")
    print(entropy_tree.show(data_property="entropy_str", idhidden=False, stdout=False))

    print("Generator integers:")
    print(entropy_tree.show(data_property="integer_str", idhidden=False, stdout=False))
