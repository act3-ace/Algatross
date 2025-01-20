from pathlib import Path
from tempfile import NamedTemporaryFile

import ray

from ray import cloudpickle

from algatross.algorithms.genetic.mo_aim.archipelago.ray_archipelago import RayMOAIMArchipelago, logger


class DebugVizArchipelago(RayMOAIMArchipelago):
    def render_episodes(self, storage_path: str | Path, max_episodes: int = 5):  # type: ignore[override]
        if isinstance(storage_path, str):
            storage_path = Path(storage_path)
        unfinished = [
            island_server.evolve.remote(  # type: ignore[attr-defined]
                n=1,
                train=False,
                visualize=True,
                max_episodes=max_episodes,
                training_iterations=1,
                rollout_config={"batch_mode": "complete_episodes"},
            )
            for island_id, island_server in self._island.items()
        ]
        while unfinished:
            finished, unfinished = ray.wait(unfinished)
            for island_id, rollout_result in ray.get(finished):
                assert island_id in self._island, "Bad island!"  # noqa: S101
                iter_rendered_episodes_map = rollout_result[f"island/{island_id}"]["iter_rendered_episode_map"]

                team_to_episodes = iter_rendered_episodes_map[0]
                # First make sure runner.visualize_step() outputs the observation data we expect to see each frame.
                # Now save the obs data so we can open it in a jupyter notebook, where the viz code will be incubated.
                # breakpoint()  # noqa: ERA001

                with NamedTemporaryFile("wb+", delete=False, suffix=".pkl") as tf:
                    cloudpickle.dump(team_to_episodes, tf)
                    logger.info(f"Saved debug data to {tf.name}")

                # for team in team_to_episodes:
                # episodes = team_to_episodes[team]  # noqa: ERA001
                # self.log(
                #     make_movies(
                #         episodes,
                #         storage_path / f"island-{island_id}" / f"team-{team}",  # noqa: ERA001
                #         get_frame_functional=mpe_frame_func,  # noqa: ERA001
                #         fps=15,  # noqa: ERA001
                #     ),
                # ) #
