"""Algorithm classes for working with classic RL algorithms."""

import logging

from copy import deepcopy
from typing import Any

import numpy as np

import ray

from algatross.algorithms.genetic.mo_aim.algorithm import UDA
from algatross.algorithms.genetic.mo_aim.classic_rl.population import MOAIMRLPopulation
from algatross.algorithms.genetic.mo_aim.classic_rl.problem import MOAIMRLUDP
from algatross.algorithms.genetic.mo_aim.configs import MOAIMRLUDAConfig
from algatross.environments.runners import ManagedEnvironmentsContext
from algatross.utils.merge_dicts import filter_keys, list_to_stack
from algatross.utils.random import resolve_seed
from algatross.utils.types import AgentID, MOAIMIslandInhabitant, NumpyRandomSeed, RolloutData, TeamID


class MOAIMRLUDA(UDA):
    """
    User-defined algorithm for evolution on MO-AIM Islands.

    Parameters
    ----------
    seed : NumpyRandomSeed | None, optional
        The random seed to use for the algorithm.
    `**kwargs`
        Additional keyword arguments.
    """

    config: MOAIMRLUDAConfig
    """The config dataclass used for the island."""
    rollout_buffer: list[RolloutData]
    """The rollout data for agents of this algorithm."""

    def __init__(self, seed: NumpyRandomSeed | None = None, **kwargs) -> None:
        config_dict = deepcopy(kwargs)
        config_dict["seed"] = config_dict.get("seed") or seed
        self.config = MOAIMRLUDAConfig(**filter_keys(MOAIMRLUDAConfig, **config_dict))
        self._fitness: np.ndarray | None = None
        self._numpy_generator = resolve_seed(seed)  # type: ignore[arg-type]
        self.rollout_buffer = []

    def set_state(self, state: dict):  # noqa: D102
        for key, value in state.items():
            if key not in {"problem", "_problem"}:
                setattr(self, key, value)

    def get_state(self) -> dict:
        """Get the algorithm state.

        Returns
        -------
        dict
            The algorithm state.
        """
        state = deepcopy({k: v for k, v in self.__dict__.items() if k not in {"problem", "_problem"}})
        # state["problem"] = self.problem.get_state()  # noqa: ERA001
        return state  # noqa: RET504

    def evolve_candidates(
        self,
        team: dict[AgentID, MOAIMIslandInhabitant],
        envs: list,
        *,
        problem: MOAIMRLUDP,
        **kwargs,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Generate a rollout for a single candidate and add to the populations rollout buffer.

        Parameters
        ----------
        team : dict[AgentID, MOAIMIslandInhabitant]
            The team of agents to evolve
        envs : list
            The list of environments to use for rollouts
        problem : MOAIMRLUDP,
            The problem this algorithm is solving
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        tuple[np.ndarray, dict[str, Any]]
            A dict of info for this evolution.
        """
        kwargs.setdefault("training_iterations", self.config.training_iterations)

        f, infos = problem.fitness(team, envs=envs, **kwargs)
        fit_team = infos.pop("teams", {})
        self._fitness = f.mean()

        # Team state was recorded in the rollout_buffer, but message it back for future iterations of this island run.
        infos["teams"] = fit_team
        return self._fitness, infos

    @property
    def fitness(self):  # noqa: D102
        if self._fitness is None:
            msg = "Must call `evolve` first"
            raise RuntimeError(msg)
        return self._fitness

    def on_evolve_begin(  # type: ignore[override] # noqa: D102
        self,
        pop: MOAIMRLPopulation,
        *,
        problem: MOAIMRLUDP,
        **kwargs,
    ) -> tuple[MOAIMRLPopulation, dict[str, Any]]:
        pop, results = super().on_evolve_begin(pop, problem=problem, **kwargs)  # type: ignore[assignment]
        results |= pop.add_migrants_to_buffer(problem=problem, **kwargs)
        results["teams"] = {}

        return pop, results

    def on_evolve_step(  # type: ignore[override] # noqa: D102
        self,
        pop: MOAIMRLPopulation,
        teams: dict | None = None,
        *,
        problem: MOAIMRLUDP,
        **kwargs,
    ) -> tuple[MOAIMRLPopulation, dict[str, Any]]:
        pop, results = super().on_evolve_step(pop, problem=problem, **kwargs)  # type: ignore[assignment]
        remote = kwargs.get("remote", False)

        # refresh=False : init teams once per aeon
        # refresh=True  : query emitter every island iteration for new teams
        training_teams = pop.get_teams_for_training(problem=problem, **kwargs)
        opposing_teams = pop.get_teams_for_competition(problem=problem, **kwargs)

        training_teams = ray.get(training_teams) if remote else training_teams  # type: ignore[arg-type]
        opposing_teams = ray.get(opposing_teams) if remote else opposing_teams  # type: ignore[call-overload]

        teams = training_teams  # type: ignore[assignment]
        if len(opposing_teams):
            for k in teams:
                teams[k] |= opposing_teams.get(k, {})
        results["teams"] = teams
        return pop, results

    def on_evolve_end(  # type: ignore[override]
        self,
        pop: MOAIMRLPopulation,
        teams: dict | None = None,
        *,
        problem: MOAIMRLUDP,
        **kwargs,
    ) -> tuple[MOAIMRLPopulation, dict[str, Any]]:
        """Run operations at the end of a training epoch.

        Add migrants to the populations migrant buffer and update the population archives.

        Parameters
        ----------
        pop : MOAIMRLPopulation
            The population being evolved.
        teams : dict[AgentID, MOAIMIslandInhabitant]
            The team of agents to evolve
        problem : MOAIMRLUDP,
            The problem this algorithm is solving
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        tuple[MOAIMRLPopulation, dict[str, Any]]
            Any extra info returned by the callback.
        """
        teams = {} if teams is None else teams
        pop, results = super().on_evolve_end(pop, problem=problem, **kwargs)  # type: ignore[assignment]
        # Below is technically not necessary unless refresh=False for ally teams during on_evolve_step

        results["teams"] = teams

        return pop, results

    def evolve(  # type: ignore[override]
        self,
        pop: MOAIMRLPopulation,
        teams: dict,
        envs: list,
        *,
        problem: MOAIMRLUDP,
        **kwargs,
    ) -> tuple[MOAIMRLPopulation, dict[str, Any]]:
        """
        Conduct one single evolution step on the population.

        Parameters
        ----------
        pop : MOAIMRLPopulation
            The population to evolve
        teams : dict
            The team of agents to evolve
        envs : list
            The list of environments to use for rollouts
        problem : MOAIMRLUDP,
            The problem this algorithm is solving
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        MOAIMRLPopulation
            The evolved population
        dict[str, Any]
            Additional results for this step.

        Raises
        ------
        ValueError
            More than one team was given
        """
        results: dict[str, Any] = {}
        team_to_rendered_episodes: dict[int, Any] = {}
        if len(teams) > 1:
            msg = f"{self.__class__} does not accept more than one team for training at a time. Got {len(teams)} teams."
            raise ValueError(msg)

        for team_id, team in teams.items():
            fitness, infos = self.evolve_candidates(team, problem=problem, envs=envs, **kwargs)
            teams[team_id] = infos.pop("teams", {})
            team_to_rendered_episodes[team_id] = infos.pop("rendered_episodes", [])
            results[f"fitness/team/{team_id}"] = fitness
            results[f"info/team/{team_id}"] = infos

        # just hard-set the params and fitness
        first_genome = None
        first_name = ""
        for name, teammate in teams[TeamID(0)].items():
            if name in problem.training_agents:
                if first_genome is None:
                    first_genome, first_name = teammate.genome, name
                    continue
                if (teammate.genome == first_genome).all():
                    msg = (
                        "Found mismatch between trained genomes. Expected all trained genomes to be identical "
                        f"but the genome of {name} doesn't match {first_name}"
                    )
                    raise ValueError(msg)
        for agent_id in problem.training_agents:
            first_genome = teams[TeamID(0)][agent_id].genome
            break
        pop._current_flat_params = first_genome  # noqa: SLF001
        pop._current_best_fitness = self.fitness  # noqa: SLF001

        return pop, {"teams": teams, "evolve": results, "team_to_rendered_episodes": team_to_rendered_episodes}

    def run_evolve(  # type: ignore[override]
        self,
        pop: MOAIMRLPopulation,
        iterations: int,
        *,
        problem: MOAIMRLUDP,
        envs: list,
        **kwargs,
    ) -> tuple[MOAIMRLPopulation, dict[str, Any]]:
        """Run the evolutionary step on the population for a certain number of epochs.

        Parameters
        ----------
        pop : MOAIMRLPopulation
            The population being evolved
        iterations : int
            The number of iterations to run the evolution.
        problem : MOAIMRLUDP,
            The problem this algorithm is solving
        envs : list
            The list of environments to use for rollouts
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        MOAIMRLPopulation
            The evolved population
        dict[str, Any]
            Any extra info returned by the evolution step
        """
        # import memray  # noqa: ERA001
        # from algatross.utils.io import increment_filepath_maybe  # noqa: ERA001
        # tracker_dir = kwargs["storage_path"] / "memray"  # noqa: ERA001
        # tracker_dir.mkdir(parents=True, exist_ok=True)  # noqa: ERA001
        # tracker_path = increment_filepath_maybe(tracker_dir / "run_evolve.bin")  # noqa: ERA001
        # with memray.Tracker(tracker_path):
        pop, results = self.on_evolve_begin(pop, problem=problem, envs=envs, **kwargs)
        teams = results.pop("teams", {})
        iter_results: list[dict] = []
        iter_rendered_episode_map: list[dict] = []
        logger = logging.getLogger("ray")

        for it in range(iterations):
            logger.info(f"IslandUDA run_evolve step {it}")

            with ManagedEnvironmentsContext(env_list=envs) as mec_envs:
                pop, step_msg = self.on_evolve_step(pop, problem=problem, teams=teams, envs=mec_envs, **kwargs)
                teams = step_msg.pop("teams", {})

                pop, evolve_msg = self.evolve(pop, problem=problem, teams=teams, envs=mec_envs, **kwargs)
                teams = evolve_msg.pop("teams", {})
                iter_rendered_episode_map.append(evolve_msg.pop("team_to_rendered_episodes", {}))
                iter_results.extend((step_msg, evolve_msg))

        results |= list_to_stack(iter_results)
        logger.info(f"Finished evolving for {iterations} steps, time to update my archive.")

        # update archive in pop
        pop, end_results = self.on_evolve_end(pop, problem=problem, teams=teams, envs=envs, **kwargs)
        teams = end_results.pop("teams", {})
        results |= end_results

        # done updating pop.archive so delete
        del teams

        return pop, {"algorithm": results, "iter_rendered_episode_map": iter_rendered_episode_map}

    def run_evaluate(  # noqa: PLR6301, D102
        self,
        pop: MOAIMRLPopulation,  # type: ignore[override]
        envs: list,
        problem: MOAIMRLUDP,  # type: ignore[override]
        **kwargs,
    ) -> dict[str, Any]:
        _logger = logging.getLogger("ray")

        # always uses pop._current_flat_params for all allies
        eval_teams = pop.get_teams_for_training(problem=problem, **kwargs)
        opposing_teams = pop.get_teams_for_competition(refresh=False, problem=problem, **kwargs)

        if isinstance(eval_teams, ray.ObjectRef):
            eval_teams = ray.get(eval_teams)
        if isinstance(opposing_teams, ray.ObjectRef):
            opposing_teams = ray.get(opposing_teams)

        teams = eval_teams
        if len(opposing_teams):
            for k in teams:
                teams[k] |= opposing_teams.get(k, {})

        results: dict[str, Any] = {}
        team_to_rendered_episodes: dict[int, Any] = {}

        with ManagedEnvironmentsContext(env_list=envs) as mec_envs:
            for team_id, team in teams.items():
                fitness, infos = problem.fitness(team, envs=mec_envs, **kwargs)
                _ = infos.pop("teams", {})

                team_to_rendered_episodes[team_id] = infos.pop("rendered_episodes", [])
                results[f"fitness/team/{team_id}"] = fitness
                results[f"info/team/{team_id}"] = infos

        return {"algorithm": results, "iter_rendered_episode_map": [team_to_rendered_episodes]}

    def get_trained_buffers(self) -> list[RolloutData]:
        """Will always empty because we force-update the agent data.

        Returns
        -------
        list[RolloutData]
            The rollout buffers for trained agents
        """
        return self.rollout_buffer
