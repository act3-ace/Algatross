"""Tests for memory consumption."""

from typing import TYPE_CHECKING

import memray

from algatross.experiments.ray_experiment import RayExperiment
from algatross.utils.parsers.yaml_loader import load_config

if TYPE_CHECKING:
    from zipp import Path


class TestMemoryExperiment(RayExperiment):
    """An experiment for testing deterministic attributes of MO-AIM classes.

    When writing the test config file:
    - archieplago_constructor.config.warmup_generations MUST be 0
    - archieplago_constructor.config.warmup_iterations MUST be 0

    Parameters
    ----------
        RayExperiment (_type_): _description_
    """

    def __init__(self, config_file: str, test_dir: str, _: dict | None = None):
        config = load_config(config_file)
        config["log_dir"] = test_dir
        super().__init__(config_file, config=config)

    def run_experiment(self):  # noqa: D102
        memray_dir = self.storage_path / "memray"
        memray_dir.mkdir(parents=True, exist_ok=True)

        tracker_path = memray_dir / "run_experiment.bin"
        with memray.Tracker(tracker_path):
            _ = super().run_experiment()
            return tracker_path

    def render_episodes(  # noqa: D102
        self,
        max_episodes: int = 5,
        render_islands: bool = False,
        render_mainlands: bool = False,
        **kwargs,
    ) -> "Path":
        memray_dir = self.storage_path / "memray"
        memray_dir.mkdir(parents=True, exist_ok=True)

        tracker_path = memray_dir / "render_episodes.bin"
        print(f"\t Writing tracker to {tracker_path}")
        with memray.Tracker(tracker_path):
            _ = super().render_episodes(
                max_episodes=max_episodes,
                render_islands=render_islands,
                render_mainlands=render_mainlands,
                **kwargs,
            )
            return tracker_path
