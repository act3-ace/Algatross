"""Module containing the core MO-AIM algorithms."""

import logging

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from copy import deepcopy
from typing import Any, Literal, overload
from uuid import uuid4

import numpy as np

from pymoo.core.individual import Individual
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.cython.non_dominated_sorting import fast_best_order_sort
from pymoo.operators.survival.rank_and_crowding.metrics import get_crowding_function

import ray

from ray.rllib.policy.sample_batch import SampleBatch

import tree

from algatross.algorithms.genetic.mo_aim.configs import MOAIMIslandUDAConfig, MOAIMMainlandUDAConfig
from algatross.algorithms.genetic.mo_aim.population import MOAIMIslandPopulation, MOAIMMainlandPopulation, MOAIMPopulation
from algatross.algorithms.genetic.mo_aim.problem import UDP, MOAIMIslandUDP, MOAIMMainlandUDP
from algatross.algorithms.genetic.mo_aim.pymoo.crossover import SimulatedBinaryCrossover
from algatross.algorithms.genetic.mo_aim.pymoo.misc import binary_tournament
from algatross.algorithms.genetic.mo_aim.pymoo.nds import NonDominatedSorting
from algatross.algorithms.genetic.mo_aim.pymoo.survival import RankAndCrowding
from algatross.algorithms.genetic.mo_aim.pymoo.tournament import TournamentSelection
from algatross.environments.runners import ManagedEnvironmentsContext
from algatross.utils.merge_dicts import filter_keys, list_to_stack, merge_dicts
from algatross.utils.random import resolve_seed
from algatross.utils.types import (
    AgentID,
    InhabitantID,
    IslandID,
    MOAIMIslandInhabitant,
    NumpyRandomSeed,
    RolloutData,
    TeamDict,
    TeamFront,
    TeamID,
)

MOAIM_ISLAND_ALGORITHM_CONFIG_DEFAULTS = {"conspecific_data_keys": ["actions"]}


class UDA(ABC):
    """
    ABC for user-defined algorithms.

    Users will have to define :python:`__init__` and :meth:`evolve`

    Parameters
    ----------
    config : Any
        The config for this UDA
    seed : NumpyRandomSeed | None, optional
        The seed for randomness, default is :data:`python:None`.
    `*args`
        Positional arguments.
    `**kwargs`
        Keyword arguments.
    """

    config: Any
    _numpy_generator: np.random.Generator

    @abstractmethod
    def __init__(self, config: Any, seed: NumpyRandomSeed | None = None, *args, **kwargs):  # noqa: ANN401
        pass

    @overload
    def on_evolve_begin(
        self,
        pop: MOAIMIslandPopulation,
        *,
        problem: MOAIMIslandUDP,
        **kwargs,
    ) -> tuple[MOAIMIslandPopulation, dict[str, Any]]: ...
    @overload
    def on_evolve_begin(
        self,
        pop: MOAIMMainlandPopulation,
        *,
        problem: MOAIMMainlandUDP,
        **kwargs,
    ) -> tuple[MOAIMMainlandPopulation, dict[str, Any]]: ...
    @overload
    def on_evolve_begin(self, pop: MOAIMPopulation, *, problem: UDP, **kwargs) -> tuple[MOAIMPopulation, dict[str, Any]]: ...
    def on_evolve_begin(self, pop, *, problem, **kwargs):  # noqa: PLR6301, ARG002
        """
        On_evolve_begin callback to run before the current training epoch begins.

        Parameters
        ----------
        pop : MOAIMIslandPopulation | MOAIMMainlandPopulation
            The population being evolved
        problem : MOAIMIslandUDP | MOAIMMainlandUDP
            The problem to use for evaluating fitness.
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        dict[str, Any]
            Extra info returned by the callback
        """
        return pop, {}

    @overload
    def on_evolve_step(
        self,
        pop: MOAIMIslandPopulation,
        *,
        problem: MOAIMIslandUDP,
        **kwargs,
    ) -> tuple[MOAIMIslandPopulation, dict[str, Any]]: ...
    @overload
    def on_evolve_step(
        self,
        pop: MOAIMMainlandPopulation,
        *,
        problem: MOAIMMainlandUDP,
        **kwargs,
    ) -> tuple[MOAIMMainlandPopulation, dict[str, Any]]: ...
    @overload
    def on_evolve_step(self, pop: MOAIMPopulation, *, problem: UDP, **kwargs) -> tuple[MOAIMPopulation, dict[str, Any]]: ...
    def on_evolve_step(self, pop, *, problem, **kwargs):  # noqa: PLR6301, ARG002
        """
        On_evolve_step callback to run each iteration of the epoch.

        Parameters
        ----------
        pop : MOAIMIslandPopulation | MOAIMMainlandPopulation
            The population being evolved
        problem : MOAIMIslandUDP | MOAIMMainlandUDP
            The problem to use for evaluating fitness.
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        dict[str, Any]
            Extra info returned by the callback
        """
        return pop, {}

    @overload
    def on_evolve_end(
        self,
        pop: MOAIMIslandPopulation,
        *,
        problem: MOAIMIslandUDP,
        **kwargs,
    ) -> tuple[MOAIMIslandPopulation, dict[str, Any]]: ...
    @overload
    def on_evolve_end(
        self,
        pop: MOAIMMainlandPopulation,
        *,
        problem: MOAIMMainlandUDP,
        **kwargs,
    ) -> tuple[MOAIMMainlandPopulation, dict[str, Any]]: ...
    @overload
    def on_evolve_end(self, pop: MOAIMPopulation, *, problem: UDP, **kwargs) -> tuple[MOAIMPopulation, dict[str, Any]]: ...
    def on_evolve_end(self, pop, *, problem, **kwargs):  # noqa: PLR6301, ARG002
        """
        On_evolve_end callback to run at the end of the current epoch.

        Parameters
        ----------
        pop : MOAIMIslandPopulation | MOAIMMainlandPopulation
            The population being evolved
        problem : MOAIMIslandUDP | MOAIMMainlandUDP
            The problem to use for evaluating fitness.
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        dict[str, Any]
            Extra info returned by the callback
        """
        return pop, {}

    @overload
    def evolve(
        self,
        pop: MOAIMIslandPopulation,
        *,
        problem: MOAIMIslandUDP,
        **kwargs,
    ) -> tuple[MOAIMIslandPopulation, dict[str, Any]]: ...

    @overload
    def evolve(
        self,
        pop: MOAIMMainlandPopulation,
        *,
        problem: MOAIMMainlandUDP,
        **kwargs,
    ) -> tuple[MOAIMMainlandPopulation, dict[str, Any]]: ...

    @overload
    def evolve(self, pop: MOAIMPopulation, *, problem: UDP, **kwargs) -> tuple[MOAIMPopulation, dict[str, Any]]: ...

    @abstractmethod
    def evolve(self, pop, *, problem, **kwargs):
        """
        Conduct one single evolution step on the population.

        Parameters
        ----------
        pop : MOAIMIslandPopulation | MOAIMMainlandPopulation
            The population to evolve
        problem : MOAIMIslandUDP | MOAIMMainlandUDP
            The problem to use for evaluating fitness.
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        tuple[MOAIMIslandPopulation | MOAIMMainlandPopulation, dict[str, Any]]
            The evolved population
        dict[str, Any]
            Additional results for this step.
        """

    @overload
    def run_evolve(
        self,
        pop: MOAIMMainlandPopulation,
        iterations: int,
        *,
        problem: MOAIMMainlandUDP,
        **kwargs,
    ) -> tuple[MOAIMMainlandPopulation, dict[str, Any]]: ...

    @overload
    def run_evolve(
        self,
        pop: MOAIMIslandPopulation,
        iterations: int,
        *,
        problem: MOAIMIslandUDP,
        **kwargs,
    ) -> tuple[MOAIMIslandPopulation, dict[str, Any]]: ...

    @overload
    def run_evolve(
        self,
        pop: MOAIMPopulation,
        iterations: int,
        *,
        problem: UDP,
        **kwargs,
    ) -> tuple[MOAIMPopulation, dict[str, Any]]: ...

    def run_evolve(self, pop, iterations, *, problem, **kwargs):
        """Run the evolutionary step on the population for a certain number of epochs.

        Parameters
        ----------
        pop : MOAIMIslandPopulation | MOAIMMainlandPopulation
            The population being evolved
        iterations : int
            The number of iterations to run the evolution.
        problem : MOAIMIslandUDP | MOAIMMainlandUDP
            The problem to use for evaluating fitness.
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        MOAIMIslandPopulation | MOAIMMainlandPopulation
            The evolved population
        dict[str, Any]
            Any extra info returned by the evolution step
        """
        results = {}
        pop, callback_results = self.on_evolve_begin(pop, problem=problem, **kwargs)
        results |= callback_results
        iter_results: list[dict] = []
        for _i in range(iterations):
            pop, callback_results = self.on_evolve_step(pop, problem=problem, **kwargs)
            iter_results.append(callback_results)
            pop, callback_results = self.evolve(pop, problem=problem, **kwargs)
            iter_results.append({"evolve": callback_results.pop("evolve", {})})
        results |= list_to_stack(iter_results)
        pop, callback_results = self.on_evolve_end(pop, problem=problem, **kwargs)
        results |= callback_results
        return pop, {"algorithm": results}

    @abstractmethod
    def run_evaluate(self, pop: MOAIMPopulation, envs: list, problem: UDP, **kwargs) -> dict[str, Any]:
        """
        Run evaluation on the population.

        Parameters
        ----------
        pop : MOAIMIslandPopulation | MOAIMMainlandPopulation
            The population to evaluate.
        envs : list
            The list of environments to use for evaluation.
        problem : UDP
            The problem this population is solving
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        dict[str, Any]]
            The info gathered by evaluation callbacks.
        """

    @property
    @abstractmethod
    def fitness(self) -> np.ndarray:
        """
        Fetch the output of the last call to the fitness function of the UDP used by the algorithm.

        When overriding this method the sign of the fitness should be modified so that the values can be optimized
        as a minimization problem. For example, if the fitness function of the problem is to be maximized, as is
        typically the case in reinforcement learning, this method should return a value which is the fitness multiplied
        by -1. If the objective is to minimize the fitness values then this function can return the values unchanged.

        Returns
        -------
        np.ndarray
            The fitness value(s) from the last call to the problems fitness
        """

    def get_state(self) -> dict:
        """Get the algorithm state.

        Returns
        -------
        dict
            The algorithm state.
        """
        return self.__dict__

    def set_state(self, state: dict):
        """
        Set the algorithm state.

        Parameters
        ----------
        state : dict
            The state to set for this UDA
        """
        self.__dict__ |= state


class MOAIMIslandUDA(UDA):
    """
    User-defined algorithm for evolution on MO-AIM Islands.

    Parameters
    ----------
    seed : NumpyRandomSeed | None, optional
        The seed for randomness, default is :data:`python:None`
    `**kwargs`
        Additional keyword arguments.
    """

    config: MOAIMIslandUDAConfig
    """The configuration for this island."""
    rollout_buffer: list[RolloutData]
    """The rollout buffer for this island."""

    def __init__(self, seed: NumpyRandomSeed | None = None, **kwargs) -> None:
        config_dict = merge_dicts(MOAIM_ISLAND_ALGORITHM_CONFIG_DEFAULTS, kwargs)
        config_dict["seed"] = config_dict.get("seed") or seed
        self.config = MOAIMIslandUDAConfig(**filter_keys(MOAIMIslandUDAConfig, **config_dict))
        self.rollout_buffer = []
        self._fitness: np.ndarray | None = None
        # self._numpy_generator = get_generators(seed=self.config.seed, seed_global=False)[0]  # noqa: ERA001
        self._numpy_generator = resolve_seed(seed)  # type: ignore[arg-type]

    def set_state(self, state: dict):  # noqa: D102
        for key, value in state.items():
            if key not in {"problem", "_problem"}:
                setattr(self, key, value)

    def get_state(self) -> dict[str, Any]:
        """
        Get the algorithm state.

        Returns
        -------
        dict[str, Any]
            The algorithm state dictionary.
        """
        state = deepcopy({k: v for k, v in self.__dict__.items() if k not in {"problem", "_problem"}})
        # state["problem"] = self.problem.get_state()  # noqa: ERA001
        return state  # noqa: RET504

    def evolve_candidates(
        self,
        team: dict[AgentID, MOAIMIslandInhabitant],
        envs: list,
        *,
        problem: MOAIMIslandUDP,
        **kwargs,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Generate a rollout for a single candidate and add to the populations rollout buffer.

        Parameters
        ----------
        team : dict[AgentID, MOAIMIslandInhabitant]
            The team of agents to evolve.
        envs : list
            The environments to use when gathering rollouts
        problem : MOAIMIslandUDP
            The problem to use for evaluating fitness.
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        tuple[np.ndarray, dict[str, Any]]
            The fitness of the team and extra infos gathered by the problem
        """
        kwargs.setdefault("training_iterations", self.config.training_iterations)

        f, infos = problem.fitness(team, envs=envs, **kwargs)
        fit_team = infos.pop("teams", {})
        self._fitness = f.mean()

        # trained_weights = {agent_id: inhabitant.genome for agent_id, inhabitant in team}  # noqa: ERA001

        # Add team weights to rollout_buffer. It will later be consumed during on_evolve_end to update population archive.
        for idx, training_agent in enumerate(problem.training_agents):
            cdata = [
                np.concatenate([np.atleast_1d(rollout[key]).reshape(len(rollout), -1) for key in problem.conspecific_data_keys], axis=-1)
                for rollout in problem.rollout_buffers[training_agent]
                if len(rollout) >= problem.behavior_classification_config.trajectory_length
            ]
            # zero-pad the beginning if slice_len < 0 (slicing from end)
            # otherwise zero-pad end (slicing from beginning):
            conspecific_data = np.stack([
                cd[-problem.behavior_classification_config.trajectory_length :]
                for cd in cdata[-problem.behavior_classification_config.num_samples :]
            ])

            self.rollout_buffer.append(RolloutData(conspecific_data, f[idx], fit_team[training_agent].genome))

        # Team state was recorded in the rollout_buffer, but message it back for future iterations of this island run.
        infos["teams"] = fit_team
        return self._fitness, infos

    @property
    def fitness(self):  # noqa: D102
        if self._fitness is None:
            msg = "Must call `evolve` first"
            raise RuntimeError(msg)
        return self._fitness

    def get_trained_buffers(self) -> list[RolloutData]:
        """
        Return the buffers of trained agents.

        Returns
        -------
        list[RolloutData]
            The buffers of rollout data of trained agents
        """
        return self.rollout_buffer

    def on_evolve_begin(  # type: ignore[override] # noqa: D102
        self,
        pop: MOAIMIslandPopulation,
        *,
        problem: MOAIMIslandUDP,
        **kwargs,
    ) -> tuple[MOAIMIslandPopulation, dict[str, Any]]:
        pop, results = super().on_evolve_begin(pop, problem=problem, **kwargs)
        results |= pop.add_migrants_to_buffer(update_archives=False, problem=problem, **kwargs)
        results["teams"] = {}

        return pop, results

    def on_evolve_step(  # type: ignore[override] # noqa: D102
        self,
        pop: MOAIMIslandPopulation,
        teams: dict | None = None,
        *,
        problem: MOAIMIslandUDP,
        **kwargs,
    ) -> tuple[MOAIMIslandPopulation, dict[str, Any]]:
        pop, results = super().on_evolve_step(pop, problem=problem, **kwargs)
        remote = kwargs.get("remote", False)

        # refresh=False : init teams once per aeon
        # refresh=True  : query emitter every island iteration for new teams
        training_teams = pop.get_teams_for_training(refresh=True, problem=problem, **kwargs)
        opposing_teams = pop.get_teams_for_competition(refresh=False, problem=problem, **kwargs)

        training_teams = ray.get(training_teams) if remote else training_teams  # type: ignore[call-overload]
        opposing_teams = ray.get(opposing_teams) if remote else opposing_teams  # type: ignore[call-overload]

        teams = training_teams
        # TODO: If the setting for team_size doesn't match the size of training_agents in config, get a weird bug where teams is empty.
        if len(teams) == 0:
            msg = "The population gave no teams! Check island's population.config.team_size or problem.config.training_agents parameter."
            raise ValueError(msg)

        if len(opposing_teams):
            for k in teams:
                teams[k] |= opposing_teams.get(k, {})
        results["teams"] = teams
        return pop, results

    def on_evolve_end(  # type: ignore[override]
        self,
        pop: MOAIMIslandPopulation,
        teams: dict | None = None,
        *,
        problem: MOAIMIslandUDP,
        **kwargs,
    ) -> tuple[MOAIMIslandPopulation, dict[str, Any]]:
        """
        Run operations at the end of a training epoch.

        Add migrants to the populations migrant buffer and update the population archives.

        Parameters
        ----------
        pop : MOAIMIslandPopulation
            The population being evolved.
        teams : dict, optional
            The evolved teams, default is :data:`python:None`
        problem : MOAIMIslandUDP
            The problem to use for evaluating fitness.
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        tuple[MOAIMIslandPopulation, dict[str, Any]]
            Any extra info returned by the callback.
        """
        teams = {} if teams is None else teams
        pop, results = super().on_evolve_end(pop, problem=problem, **kwargs)
        results |= pop.add_migrants_to_buffer(
            migrants={None: [(None, t) for t in self.get_trained_buffers()]},
            update_archives=True,
            problem=problem,
            **kwargs,
        )
        self.rollout_buffer.clear()
        # Below is technically not necessary unless refresh=False for ally teams during on_evolve_step
        _ = pop.step_training_teams_cache(teams, problem=problem, **kwargs)

        results["teams"] = teams

        return pop, results

    def evolve(  # type: ignore[override]
        self,
        pop: MOAIMIslandPopulation,
        teams: dict,
        envs: list,
        *,
        problem: MOAIMIslandUDP,
        **kwargs,
    ) -> tuple[MOAIMIslandPopulation, dict[str, Any]]:
        """
        Conduct one single evolution step on the population.

        Parameters
        ----------
        pop : MOAIMIslandPopulation
            The population to evolve
        teams : dict, optional
            The evolved teams, default is :data:`python:None`
        envs : list
            The environments to use for evolution
        problem : MOAIMIslandUDP
            The problem being worked on by this population
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        tuple[MOAIMIslandPopulation, dict[str, Any]]
            Additional results for this step.
        """
        results: dict[str, Any] = {}
        team_to_rendered_episodes: dict[int, Any] = {}
        for team_id, team in teams.items():
            fitness, infos = self.evolve_candidates(team, problem=problem, envs=envs, **kwargs)
            teams[team_id] = infos.pop("teams", {})
            team_to_rendered_episodes[team_id] = infos.pop("rendered_episodes", [])
            results[f"fitness/team/{team_id}"] = fitness
            results[f"info/team/{team_id}"] = infos

        return pop, {"teams": teams, "evolve": results, "team_to_rendered_episodes": team_to_rendered_episodes}

    def run_evolve(  # type: ignore[override]
        self,
        pop: MOAIMIslandPopulation,
        iterations: int,
        *,
        problem: MOAIMIslandUDP,
        envs: list,
        **kwargs,
    ) -> tuple[MOAIMIslandPopulation, dict[str, Any]]:
        """Run the evolutionary step on the population for a certain number of epochs.

        Parameters
        ----------
        pop : MOAIMIslandPopulation
            The population being evolved
        iterations : int
            The number of iterations to run the evolution.
        problem : MOAIMIslandUDP
            The problem to use for evaluating fitness.
        envs : list
            The environments to use when gathering rollouts
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        tuple[MOAIMIslandPopulation, dict[str, Any]]
            The evolved population and any extra info returned by the evolution step
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

    def run_evaluate(  # type: ignore[override] # noqa: PLR6301
        self,
        pop: MOAIMIslandPopulation,
        envs: list,
        problem: MOAIMIslandUDP,
        num_island_elites: int = 1,
        elites_per_island_team: int = 1,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Run evaluation on the population.

        Parameters
        ----------
        pop : MOAIMIslandPopulation
            The population to evaluate.
        envs : list
            The list of environments to use for evaluation.
        problem : MOAIMIslandUDP
            The problem to evaluate this population on
        num_island_elites : int = 1
            The total number of elites on the island.
        elites_per_island_team : int = 1
            The number of elites per team.
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        dict[str, Any]]
            The info gathered by evaluation callbacks.
        """
        _logger = logging.getLogger("ray")
        remote = kwargs.get("remote", False)

        # This will greedily assign elite_per_team on every team until num_island_elites is exhausted. Remaining allies are random.
        # Elites are sorted based on the archive objective. Ideal team composition is decided on mainlands.
        eval_teams = pop.get_teams_for_evaluation(
            problem=problem,
            num_island_elites=num_island_elites,
            elites_per_island_team=elites_per_island_team,
            **kwargs,
        )
        # same as training
        # TODO: better competition logic
        opposing_teams = pop.get_teams_for_competition(refresh=False, problem=problem, **kwargs)

        eval_teams = ray.get(eval_teams) if remote else eval_teams  # type: ignore[call-overload]
        opposing_teams = ray.get(opposing_teams) if remote else opposing_teams  # type: ignore[call-overload]

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


class MOAIMMainlandUDA(UDA):
    """
    MOAIMMainlandUDA is the user-defined algorithm for evolution on the MO-AIM mainlands.

    Algorithm assumes a minimization problem so fitness functions should return a value where smaller values are better.

    Parameters
    ----------
    seed : NumpyRandomSeed | None, optional
        The seed for randomness, default is :data:`python:None`
    `**kwargs`
        Additional keyword arguments.
    """

    config: MOAIMMainlandUDAConfig
    """The configuration for this mainland."""
    crowding_func: Callable
    """The crowding function for this algorithm."""
    nds: NonDominatedSorting
    """The non-dominated sorting class to use for this algorithm."""
    survival: RankAndCrowding
    """The survival operator for this mainland."""
    tournament: TournamentSelection
    """The tournament selection to use for this mainland."""
    crossover: SimulatedBinaryCrossover
    """The crossover to use for this mainland."""
    opposing_teams: dict[TeamID, list[MOAIMIslandInhabitant]]
    """The opposing teams to use when evaluating in a competitive setting."""
    tournament_type: Literal["comp_by_dom_and_crowding", "comp_by_rank_and_crowding"] = "comp_by_dom_and_crowding"
    """The type of tournament for this mainland."""

    _rollout_buffers: dict[TeamID, dict[AgentID, list[SampleBatch]]]
    _cumulative_conspecific_utility: dict[IslandID, np.ndarray]

    def __init__(self, seed: NumpyRandomSeed | None = None, **kwargs):
        config_dict = {**kwargs}
        config_dict["seed"] = config_dict.get("seed") or seed
        self.config = MOAIMMainlandUDAConfig(**filter_keys(MOAIMMainlandUDAConfig, **config_dict))
        self._conspecific_utility_dict: dict[IslandID, dict] = {}

        # self._numpy_generator = get_generators(seed=self.config.seed, seed_global=False)[0]  # noqa: ERA001
        self._numpy_generator = resolve_seed(seed)  # type: ignore[arg-type]

        crowding_func_ = get_crowding_function("cd")
        self._rollout_buffers = {}
        self._fitness = None

        self.crowding_func = crowding_func_
        self.nds = NonDominatedSorting(method=fast_best_order_sort)
        self.survival = RankAndCrowding(generator=self._numpy_generator)
        self.tournament = TournamentSelection(func_comp=binary_tournament, generator=self._numpy_generator)
        self.crossover = SimulatedBinaryCrossover(eta=15, prob=0.9, n_offsprings=1, generator=self._numpy_generator)
        self._cumulative_conspecific_utility: dict[IslandID, np.ndarray] = {}
        self.opposing_teams: dict[TeamID, list[MOAIMIslandInhabitant]] = {}

    def set_state(self, state: dict):
        """
        Get the algorithm state.

        Parameters
        ----------
        state : dict
            The algorithm state to set
        """
        for key, value in state.items():
            if key not in {"problem", "_problem"}:
                setattr(self, key, value)

    def get_state(self) -> dict[str, Any]:
        """Get the algorithm state.

        Returns
        -------
        dict[str, Any]
            The algorithms state dictionary.
        """
        return deepcopy({k: v for k, v in self.__dict__.items() if k not in {"problem", "_problem"}})

    def get_and_reset_cumulative_conspecific_utility(self) -> dict[str, np.ndarray]:
        """
        Calculate the cumulative conspecific utility for each islands on this mainland.

        Sums the conspecific utility data from the accumulator and resets the accumulator.

        Returns
        -------
        dict[str, np.ndarray]
            A dictionary which maps the distinct IslandIDs contained on the mainland to the accumulated conspecific utility
            of the population of each of those islands.
        `**kwargs`
            Additional keyword arguments.
        """
        ccu = {isl: v.sum(axis=-1) for isl, v in self._cumulative_conspecific_utility.items()}
        self._cumulative_conspecific_utility.clear()
        return {"/".join(map(str, k)): v for k, v in tree.flatten_with_path(ccu)}

    def _do_selection(
        self,
        pop: MOAIMMainlandPopulation,
        teams: TeamDict,
        fitness: dict[TeamID, np.ndarray],
        *,
        problem: MOAIMMainlandUDP,
        **kwargs,
    ) -> tuple[np.ndarray, Sequence[TeamFront], Sequence[TeamFront], dict[str, Any]]:
        """
        Do selection using NDS and binary tournament.

        Parameters
        ----------
        pop : MOAIMMainlandPopulation
            The population being evolved
        teams : TeamDict
            The teams from the population we are evolving
        fitness : dict[TeamID, np.ndarray]
            The fitness of the population
        problem : MOAIMMainlandUDP
            The problem to use for evaluating fitness.
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        np.ndarray
            The indices of non-elites to use for mating
        Sequence[TeamFront]
            The elite teams
        Sequence[TeamFront]
            The non-elite teams
        dict[str, Any]
            Result info to log

        Raises
        ------
        RuntimeError
            The same individuals were found on multiple teams
        """
        results = {}
        mates = tree.flatten(teams.values())
        if len(mates) != len({ind.inhabitant_id for ind in mates}):
            counts: dict[str, int] = {
                ind.inhabitant_id: sum(1 for ind_id in [indv.inhabitant_id for indv in mates] if ind_id == ind.inhabitant_id)
                for ind in mates
            }
            counts = {ind_id: cnt for ind_id, cnt in counts.items() if cnt > 1}
            msg = f"Found individuals belonging to multiple teams: {counts}."
            raise RuntimeError(msg)

        moo_pop = Population.create(*[Individual(X=np.array([team_id]), F=fitness[team_id]) for team_id in teams])
        n_elites = pop.n_elites
        n_non_elites = len(teams) - pop.n_elites

        # sort the teams
        ndf: list[np.ndarray] = self.nds.do(moo_pop.get("F").astype(float, copy=False))  # type: ignore[assignment]
        moo_pop = self.survival._do(moo_pop, ndf, n_survive=len(teams))  # noqa: SLF001
        s_subs = self.tournament.do(None, moo_pop[n_elites:], n_non_elites, 1, algorithm=self).get("X").squeeze()

        elite_indices: list[TeamFront] = []
        non_elite_indices: list[TeamFront] = []

        results["fronts/total"] = len(ndf)

        # calculate the crowding distances for each front
        for front_idx, front in enumerate(ndf):
            results[f"fronts/{front_idx}/count"] = len(front)
            front_fitness = -np.stack([moo_pop[team].get("F") for team in front], axis=-1)
            for metric, f in zip(problem.fitness_metric_keys, front_fitness, strict=True):
                results[f"fronts/{front_idx}/fitness/{metric}_mean"] = f.mean(axis=-1)
                results[f"fronts/{front_idx}/fitness/{metric}_max"] = f.max(axis=-1)
                results[f"fronts/{front_idx}/fitness/{metric}_min"] = f.min(axis=-1)

            # for team_id in front:
            for team in front:
                team_id = TeamID(moo_pop[team].get("X")[0])
                team_front = TeamFront(front_index=front_idx, team_id=team_id)
                # add the solution as an elite based on crowding distance
                if len(elite_indices) < n_elites:
                    elite_indices.append(team_front)
                else:
                    non_elite_indices.append(team_front)

        elite_fitness = -np.stack([moo_pop[team_front.team_id].get("F") for team_front in elite_indices], axis=-1)
        non_elite_fitness = -np.stack([moo_pop[team_front.team_id].get("F") for team_front in non_elite_indices], axis=-1)
        elite_fitness_mean = elite_fitness.mean(axis=-1)
        elite_fitness_max = elite_fitness.max(axis=-1)
        elite_fitness_min = elite_fitness.min(axis=-1)
        non_elite_fitness_mean = non_elite_fitness.mean(axis=-1)
        non_elite_fitness_max = non_elite_fitness.max(axis=-1)
        non_elite_fitness_min = non_elite_fitness.min(axis=-1)
        for metric_idx, metric in enumerate(problem.fitness_metric_keys):
            results[f"fitness/elites/{metric}_mean"] = elite_fitness_mean[metric_idx]
            results[f"fitness/elites/{metric}_max"] = elite_fitness_max[metric_idx]
            results[f"fitness/elites/{metric}_min"] = elite_fitness_min[metric_idx]
            results[f"fitness/non_elites/{metric}_mean"] = non_elite_fitness_mean[metric_idx]
            results[f"fitness/non_elites/{metric}_max"] = non_elite_fitness_max[metric_idx]
            results[f"fitness/non_elites/{metric}_min"] = non_elite_fitness_min[metric_idx]

        return s_subs, elite_indices, non_elite_indices, results

    def _do_mating(
        self,
        teams: TeamDict,
        elite_indices: Sequence[TeamFront],
        non_elite_indices: Sequence[TeamFront],
        mating_non_elites: "Sequence[TeamID]",
        *,
        problem: MOAIMMainlandUDP,
        **kwargs,
    ) -> tuple[list[MOAIMIslandInhabitant], list[MOAIMIslandInhabitant], dict[InhabitantID, MOAIMIslandInhabitant], dict[str, Any]]:
        """
        Do mating between elites and non-elites using crossover and mutation.

        Parameters
        ----------
        teams : TeamDict
            The population we are evolving.
        elite_indices : Sequence[TeamID]
            The IDs of elite teams to which the indices (IDs) in ``mating_elites`` refer.
        non_elite_indices : Sequence[TeamID]
            The IDs of non-elite teams to which the indices (IDs) in ``mating_non_elites`` refer.
        mating_non_elites : Sequence[Sequence[TeamID]]
            The IDs of non-elite teams selected for each mating couple
        problem : MOAIMMainlandUDP
            The problem to use for evaluating fitness.
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        elites : list[MOAIMIslandInhabitant]
            The island inhabitants belonging to the elite teams
        non_elites : list[MOAIMIslandInhabitant]
            The island inhabitants belonging to the non-elite teams
        free_agents : dict[InhabitantID, MOAIMIslandInhabitant]
            The newly added free-agents which will replace the non-elites
        results : dict[str, Any]
            A dictionary of results from this step

        Raises
        ------
        RuntimeError
            If agents were found belonging to multiple teams
        """
        # construct a dictionary mapping island IDs to elite inhabitants so we can quickly
        # choose a valid policy for crossover from the elites
        results: dict[str, Any] = {}
        e_island_inhabitants: dict[IslandID, list[MOAIMIslandInhabitant]] = defaultdict(list)
        e_island_inhabitant_probs: dict[IslandID, list[float]] = defaultdict(list)

        # Construct the list of elites and non-elites as well as an iterator for the non-elites
        # so that we can replace them in the population.
        elites: list[MOAIMIslandInhabitant] = []
        non_elites: list[MOAIMIslandInhabitant] = []
        for team_front in elite_indices:
            elites.extend(teams[team_front.team_id].values())
            for inhabitant in teams[team_front.team_id].values():
                e_island_inhabitants[inhabitant.island_id].append(inhabitant)
                e_island_inhabitant_probs[inhabitant.island_id].append(float(team_front.front_index))

        for team_front in non_elite_indices:
            non_elites.extend(teams[team_front.team_id].values())

        # set the probability of choosing each elite as a softmax of the front it's in
        for island_id, probs in e_island_inhabitant_probs.items():
            probs = np.exp(-np.asarray(probs))  # noqa: PLW2901
            e_island_inhabitant_probs[island_id] = probs / probs.sum()  # type: ignore[attr-defined]

        # crossover & mutation
        free_agents: dict[str, MOAIMIslandInhabitant] = {}
        for s_ind in mating_non_elites:
            # get a non-elite team from the tournament winners
            s_team = list(teams[s_ind].values())

            # pick one of the teammates at random for crossover and mutation
            crossover_partners = []
            for s_mate in s_team:
                if e_inhabitants := e_island_inhabitants[s_mate.island_id]:
                    e_mate = self._numpy_generator.choice(e_inhabitants, p=e_island_inhabitant_probs[s_mate.island_id])  # type: ignore[arg-type]
                else:
                    e_mate = MOAIMIslandInhabitant(
                        name=s_mate.name,
                        team_id=-1,
                        island_id=s_mate.island_id,
                        current_island_id=s_mate.current_island_id,
                        inhabitant_id=s_mate.inhabitant_id,
                        genome=s_mate.genome.copy(),
                        conspecific_utility_dict=s_mate.conspecific_utility_dict,
                    )
                crossover_partners.append((s_mate, e_mate))

            # Do some reshaping so we can use PyMOOs crossover methods
            # change [team_size, 2, genome] to [2, team_size, genome] since crossover needs -> [n_parents, n_matings, n_var]
            matings = np.asarray([[t.genome.copy().flatten() for t in team] for team in zip(*crossover_partners, strict=True)])
            name: Iterator[tuple[MOAIMIslandInhabitant, MOAIMIslandInhabitant]] = iter(crossover_partners)
            s_genomes = self.crossover._do(  # noqa: SLF001
                Problem(n_obj=len(problem.fitness_metric_keys), xl=-np.inf, xu=np.inf, n_var=matings.shape[-1]),
                matings,
            )[0]

            for teammate_idx, s_genome in enumerate(s_genomes):
                # add some happy little (random) noise to the mutated genes
                inhabitant_id = str(uuid4())
                free_agents[inhabitant_id] = MOAIMIslandInhabitant(
                    name=next(name)[1].name,
                    team_id=-1,
                    island_id=s_team[teammate_idx].island_id,
                    current_island_id=s_team[teammate_idx].current_island_id,
                    inhabitant_id=inhabitant_id,
                    genome=s_genome + self._numpy_generator.normal(0, self.config.mutation_noise, s_genome.shape),
                    conspecific_utility_dict=s_team[teammate_idx].conspecific_utility_dict,
                )
        if len(free_agents) != len({free_agent.inhabitant_id for free_agent in free_agents.values()}):
            counts: dict[str, int] = {
                ind.inhabitant_id: sum(1 for ind_id in [indv.inhabitant_id for indv in free_agents.values()] if ind_id == ind.inhabitant_id)
                for ind in free_agents.values()
            }
            counts = {ind_id: cnt for ind_id, cnt in counts.items() if cnt > 1}
            msg = f"Found free_agent individuals belonging to multiple teams: {counts}."
            raise RuntimeError(msg)
        return elites, non_elites, free_agents, results

    @property
    def fitness(self):  # noqa: D102
        if self._fitness is None:
            msg = "Must call `evolve` first"
            raise RuntimeError(msg)
        return self._fitness

    @staticmethod
    def _non_elite_id_generator(elites: "Sequence[InhabitantID]"):
        i = -1
        while True:
            i += 1
            if i not in elites:
                yield i
            else:
                continue

    def on_evolve_begin(  # type: ignore[override]
        self,
        pop: MOAIMMainlandPopulation,
        *,
        problem: MOAIMMainlandUDP,
        **kwargs,
    ) -> tuple[MOAIMMainlandPopulation, dict[str, Any]]:
        """
        Run operations before the current training epoch begins.

        Resets the teams by randomly reassigning all members on the island to a new team.

        Parameters
        ----------
        pop : MOAIMMainlandPopulation
            The population being evolved
        problem : MOAIMMainlandUDP
            The problem to use for evaluating fitness.
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        tuple[MOAIMMainlandPopulation, dict[str, Any]]
            Extra info returned by the callback
        """
        pop, results = super().on_evolve_begin(pop, problem=problem, **kwargs)
        return pop, results

    def on_evolve_step(  # type: ignore[override]
        self,
        pop: MOAIMMainlandPopulation,
        *,
        problem: MOAIMMainlandUDP,
        **kwargs,
    ) -> tuple[MOAIMMainlandPopulation, dict[str, Any]]:
        """
        Run operations at each step of the epoch.

        updates the populations teams using the data from the previous iteration and resets the data buffers.

        Parameters
        ----------
        pop : MOAIMMainlandPopulation
            The population being evolved.
        problem : MOAIMMainlandUDP
            The problem to use for evaluating fitness.
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        MOAIMMainlandPopulation
            The evolved population
        dict[str, Any]
            Any extra info returned by the callback.
        """
        pop, results = super().on_evolve_step(pop, problem=problem, **kwargs)
        pop.init_teams(reset=False, problem=problem)
        opposing_teams = pop.get_teams_for_competition(problem=problem, **kwargs)
        opponents: list[MOAIMIslandInhabitant] = []
        for team in opposing_teams.values():
            opponents.extend(team)
        opponents = opponents[: len(problem.opponent_agents)]
        if opponents:
            self.opposing_teams = opposing_teams
            self.opposing_team_islands = {opponent.current_island_id for opponent in opponents}
            problem.load_opponents([opponent.genome for opponent in opponents])  # type: ignore[attr-defined]
        else:
            self.opposing_teams = {}
            self.opposing_team_islands = set()
        return pop, results

    def on_evolve_end(  # type: ignore[override]
        self,
        pop: MOAIMMainlandPopulation,
        envs: list,
        *,
        problem: MOAIMMainlandUDP,
        **kwargs,
    ) -> tuple[MOAIMMainlandPopulation, dict[str, Any]]:
        """
        Run operations at the end of a training epoch.

        Calculate the cumulative conspecific utility and adds it to the info dict.

        Parameters
        ----------
        pop : MOAIMMainlandPopulation
            The population being evolved.
        envs : list
            The environments used for evaluation
        problem : MOAIMMainlandUDP
            The problem to use for evaluating fitness.
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        tuple[MOAIMMainlandPopulation, dict[str, Any]]
            Any extra info returned by the callback
        """
        pop, results = super().on_evolve_end(pop, problem=problem, envs=envs, **kwargs)
        results |= {"conspecific_utility": {f"island/{k}": v for k, v in self.get_and_reset_cumulative_conspecific_utility().items()}}
        pop.set_rollout_buffers(self.rollout_buffers)
        pop.set_elites_to_return()
        return pop, results

    def evolve(  # type: ignore[override]
        self,
        pop: MOAIMMainlandPopulation,
        envs: list,
        *,
        problem: MOAIMMainlandUDP,
        **kwargs,
    ) -> tuple[MOAIMMainlandPopulation, dict[str, Any]]:
        """
        Run one evolutionary step on the mainland.

        Rolls out the team and adds the rollout data to the buffer. The cumulative conspecific utility
        for each island is updated as well.

        The elites are evolved using the non-dominated-sorting, tournament, and crossover methods defined by :python:`__init__`.

        Parameters
        ----------
        pop : MOAIMMainlandPopulation
            The population being evolved.
        envs : list
            The environments used for evaluation
        problem : MOAIMMainlandUDP
            The problem to use for evaluating fitness.
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        MOAIMMainlandPopulation
            The evolved population
        dict[str, Any]
            Additional result info for this step
        """
        # with Timer("mainland_evolve", "mean"):
        # evaluate the team fitness
        teams = pop.get_teams_for_training(problem=problem, **kwargs)
        teams = ray.get(teams) if isinstance(teams, ray.ObjectRef) else teams
        pop_f: dict[TeamID, np.ndarray] = {}
        results = {}
        total_fitness: np.ndarray = 0.0  # type: ignore[assignment]
        max_fitness: np.ndarray = np.inf  # type: ignore[assignment]
        min_fitness: np.ndarray = -np.inf  # type: ignore[assignment]
        curr_ccu: dict[IslandID, float | np.ndarray] = defaultdict(float)
        team_infos = []
        for team_id, team in teams.items():
            f, team_info = problem.fitness(team, train=False, envs=envs, **kwargs)
            pop_f[team_id] = f
            total_fitness += pop_f[team_id]
            # use the opposite functions since the fitness is negated
            max_fitness = np.minimum(max_fitness, f)
            min_fitness = np.maximum(min_fitness, f)
            team_infos.append(team_info)
            for teammate_id, buffer in problem.teammate_buffers.items():
                teammate = team[teammate_id]
                self._cumulative_conspecific_utility.setdefault(teammate.island_id, np.array([0.0], dtype=np.float32))
                self._conspecific_utility_dict.setdefault(teammate.island_id, dict(teammate.conspecific_utility_dict.items()))
                c_util = np.stack([
                    np.stack([b[c_key].mean() for b in buffer]).mean(axis=0, keepdims=False) * c_weight
                    for c_key, c_weight in teammate.conspecific_utility_dict.items()
                ])
                curr_ccu[teammate.island_id] += c_util
                self._cumulative_conspecific_utility[teammate.island_id] = c_util + self._cumulative_conspecific_utility[teammate.island_id]
            self._rollout_buffers[team_id] = problem.teammate_buffers.copy()
        self._fitness = {team_id: f.copy() for team_id, f in pop_f.items()}  # type: ignore[assignment]

        pop.update_opponent_softmax(total_fitness.sum(), list(self.opposing_team_islands))

        s_subs, elite_indices, non_elite_indices, sel_results = self._do_selection(pop, teams, pop_f, problem=problem, **kwargs)
        elites, non_elites, free_agents, mate_results = self._do_mating(
            teams,
            elite_indices=elite_indices,
            non_elite_indices=non_elite_indices,
            mating_non_elites=s_subs,  # type: ignore[arg-type]
            problem=problem,
            **kwargs,
        )

        # gather some statistics
        for idx, metric in enumerate(problem.fitness_metric_keys):
            results[f"fitness/team/total/{metric}_mean"] = -total_fitness[idx] / len(teams)
            results[f"fitness/team/total/{metric}_max"] = -max_fitness[idx]
            results[f"fitness/team/total/{metric}_min"] = -min_fitness[idx]

        results["selection"] = sel_results
        results["mating"] = mate_results

        for isl, cum_ccu in self._cumulative_conspecific_utility.items():
            for idx, ccu_key in enumerate(self._conspecific_utility_dict[isl]):
                results[f"island/{isl}/conspecific_utility/cumulative/{ccu_key}"] = cum_ccu[idx]
                results[f"island/{isl}/conspecific_utility/current/{ccu_key}"] = (
                    curr_ccu[isl][idx] if isinstance(curr_ccu[isl], np.ndarray) else curr_ccu[isl]  # type: ignore[index]
                )

        results["info/team"] = list_to_stack(team_infos)

        # store the results for recall later
        pop.elites = elites.copy()
        pop.elite_teams = [tf.team_id for tf in elite_indices]
        pop.non_elites = non_elites.copy()
        pop.non_elite_teams = [tf.team_id for tf in non_elite_indices]
        for ind in free_agents.values():
            pop.replace_non_elite(ind, clear_buffer=True)
        for team_id, team_f in pop_f.items():
            pop.set_f(team_id, team_f.copy())
        return pop, {"evolve": results, **pop.init_teams(reset=True, problem=problem, envs=envs, **kwargs)}

    def run_evolve(  # type: ignore[override] # noqa: D102
        self,
        pop: MOAIMMainlandPopulation,
        iterations: int,
        envs: list,
        *,
        problem: MOAIMMainlandUDP,
        **kwargs,
    ) -> tuple[MOAIMMainlandPopulation, dict[str, Any]]:
        results: dict[str, Any] = {}
        pop, callback_results = self.on_evolve_begin(pop, problem=problem, envs=envs, **kwargs)
        results |= callback_results
        iter_results: list[dict] = []
        elite_teams_histogram = np.zeros(pop.n_teams)
        non_elite_teams_histogram = np.zeros(pop.n_teams)
        logger = logging.getLogger("ray")

        for it in range(iterations):
            logger.info(f"MainlandUDA run_evolve step {it}")

            # TODO: this makes more logical sense to use at IslandServer, but there it uses more memory.
            with ManagedEnvironmentsContext(env_list=envs) as mec_envs:
                pop, step_results = self.on_evolve_step(pop, problem=problem, envs=mec_envs, **kwargs)
                pop, evolve_results = self.evolve(pop, problem=problem, envs=mec_envs, **kwargs)
                init_teams_results = evolve_results.pop("init_teams", {})
                # keep a histogram of the times each team has appeared as an elite
                elite_teams_histogram[init_teams_results.pop("elite_teams/ids", [])] += 1
                non_elite_teams_histogram[init_teams_results.pop("non_elite_teams/ids", [])] += 1
                iter_results.extend((step_results, {"evolve": evolve_results.pop("evolve", {}), "init_teams": init_teams_results}))

        iter_results: dict[str, Any] = list_to_stack(iter_results)  # type: ignore[no-redef]
        for key, result in iter_results.items():  # type: ignore[attr-defined]
            if not isinstance(result, np.ndarray) or result.dtype == np.uint8:
                continue
            if result.ndim > 0:
                iter_results[key] = result.mean(axis=0)
        results |= iter_results  # type: ignore[arg-type]
        pop, end_results = self.on_evolve_end(pop, problem=problem, envs=envs, **kwargs)
        results |= end_results
        return pop, {
            "algorithm": results,
            "elite_teams/ids": elite_teams_histogram[None, :],
            "non_elite_teams/ids": non_elite_teams_histogram[None, :],
        }

    def run_evaluate(  # type: ignore[override] # noqa: PLR6301
        self,
        pop: MOAIMMainlandPopulation,
        envs: list,
        problem: MOAIMMainlandUDP,
        num_mainland_teams: int,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Evaluate the mainland's solutions for team composition problem.

        Parameters
        ----------
        pop : MOAIMMainlandPopulation
            The population to evaluate.
        envs : list
            The list of environments to use for evaluation.
        problem : MOAIMMainlandUDP
            The problem the population is working on.
        num_mainland_teams : int
            The number of mainland teams.
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        dict[str, Any]]
            The info gathered by evaluation callbacks.
        """
        pop.set_elites_to_return()
        # partially ordered teams when total elites is larger than non-dominating front
        teams: dict[TeamID, dict[AgentID, MOAIMIslandInhabitant]]
        teams = pop._teams_to_compete  # noqa: SLF001

        results: dict[str, Any] = {}
        team_to_rendered_episodes: dict[int, Any] = {}

        with ManagedEnvironmentsContext(env_list=envs) as mec_envs:
            for ix, (team_id, team) in enumerate(teams.items()):
                if ix + 1 > num_mainland_teams:
                    break

                fitness, infos = problem.fitness(team, envs=mec_envs, **kwargs)
                _ = infos.pop("teams", {})

                team_to_rendered_episodes[team_id] = infos.pop("rendered_episodes", [])
                results[f"fitness/team/{team_id}"] = fitness
                results[f"info/team/{team_id}"] = infos

        return {"algorithm": results, "iter_rendered_episode_map": [team_to_rendered_episodes]}

    @property
    def rollout_buffers(self) -> dict[TeamID, dict[AgentID, list[SampleBatch]]]:
        """
        The rollout buffer for each team.

        The rollout data for each teammate on the teamm is stored separately and can be retrieved
        using the teammate ID.

        Returns
        -------
        dict[TeamID, dict[AgentID, list[SampleBatch]]]
            A dictionary which maps a team ID to a mapping of rollout data for each teammate.
        """
        return self._rollout_buffers
