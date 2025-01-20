"""MPE utilities for HARL learners."""

import copy
import importlib
import logging

from collections import defaultdict
from collections.abc import Callable
from types import ModuleType
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

import gymnasium
import supersuit as ss

from pettingzoo.utils.env import ParallelEnv

from algatross.utils.merge_dicts import merge_dicts
from algatross.utils.types import AgentID, PlatformID

if TYPE_CHECKING:
    from pettingzoo.mpe._mpe_utils.core import World
    from pettingzoo.mpe.simple_tag.simple_tag import Scenario


T = TypeVar("T")


logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)

PZ_ENV_INIT_KWARGS = {"simple_spread": {"N": 3, "local_ratio": 0.5, "max_cycles": 25, "continuous_actions": False, "render_mode": None}}


class PettingZooMPEEnv:
    """
    A wrapper for PettingZoo MPE environments.

    Parameters
    ----------
    scenario : str
        The MPE scenario to use with this environment.
    `**kwargs`
        Additional keyword arguments.
    """

    args: dict
    """The keyword arguments passed to the constructor."""
    scenario: str
    """The scenario being used in this MPE environment."""
    discrete: bool
    """Whether the environment action space is discrete."""
    max_cycles: int
    """The maximum episode length."""
    cur_step: int
    """The current environment step."""
    module: ModuleType
    """The pettingzoo module itself."""
    env: ParallelEnv
    """The pettingzoo environment being used."""
    possible_platforms: list[str]
    """The possible agents in this environment."""
    n_agents: int
    """The number of possible agents in this environment."""
    share_observation_space: list[gymnasium.spaces.Space]
    """The shared observation space for agents."""
    observation_space: list[gymnasium.spaces.Space]
    """The list of each agents observation space."""
    action_space: list[gymnasium.spaces.Space]
    """The list of each agents action space."""
    extra_info_fn: Callable
    """The callable to use to gather extra info from the environment on a call to ``step``."""

    def __init__(self, scenario: str, **kwargs):
        self.args = copy.deepcopy(kwargs)
        self.scenario = scenario

        self.discrete = True
        if self.args.get("continuous_actions"):
            self.discrete = False
        self.args.setdefault("max_cycles", 25)
        self.max_cycles = self.args["max_cycles"]
        self.args["max_cycles"] += 1

        for sc, default in PZ_ENV_INIT_KWARGS.items():  # noqa: B007
            if sc in scenario:
                break
        else:
            msg = f"`{scenario}` scenario not supported. Must be: {list(PZ_ENV_INIT_KWARGS)}"
            raise ValueError(msg)

        self.cur_step = 0
        self.module = importlib.import_module("pettingzoo.mpe." + self.scenario)
        env_kwargs = {k: self.args.get(k, v) for k, v in default.items()}
        self.env = ss.pad_action_space_v0(ss.pad_observations_v0(self.module.parallel_env(**env_kwargs)))
        self.possible_platforms = sorted(self.env.possible_agents)
        self.n_agents = len(self.possible_platforms)
        self.share_observation_space = self.repeat(self.env.state_space)
        self.observation_space = self.unwrap(self.env.observation_spaces)
        self.action_space = self.unwrap(self.env.action_spaces)

        for key, fn in EXTRA_INFO_FUNCTIONS.items():
            if key in self.env.metadata["name"]:
                self.extra_info_fn = fn
                break
        else:
            self.extra_info_fn = lambda env: {}

    def step(
        self,
        actions: np.ndarray,
        trainable_agents: list,
        reward_map: dict,
        reward_gains: dict,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[list[float]], list[np.ndarray], list[np.ndarray], list[list[int]] | None]:
        """Return local_obs, global_state, rewards, dones, infos, available_actions.

        Parameters
        ----------
        actions : np.ndarray
            Actions to send to the environment.
        trainable_agents : list
            A list of agents which are trainable in the environment. This is used to determine which agent rewards should be collected into
            The total reward
        reward_map : dict
            A mapping from platform names to a list of reward names to use when determining the agents reward.
        reward_gains : dict
            A mapping from platform names to a sequence of scale factors to apply to the rewards.

        Returns
        -------
        list[np.ndarray]
            Observations for each platform
        list[np.ndarray]
            Global state observations
        list[list[float]]
            Shared rewards
        list[np.ndarray]
            Dones for each platform
        list[np.ndarray]
            Infos for each platform
        list[list[int]] | None
            Available actions for each platform
        """
        extra_info = self.extra_info_fn(self.env)
        if self.discrete:
            obs, rew, term, trunc, info = self.env.step(self.wrap(actions.flatten()))
        else:
            obs, rew, term, trunc, info = self.env.step(self.wrap(actions))
        info = merge_dicts(info, extra_info)

        self.cur_step += 1
        for platform in self.possible_platforms:
            info[platform]["original_rewards"] = rew[platform]
            info[platform]["step"] = self.cur_step

        if self.cur_step == self.max_cycles:
            trunc = dict.fromkeys(self.possible_platforms, True)
            for platform in self.possible_platforms:
                info[platform]["bad_transition"] = True
                info[platform]["truncated"] = True

        dones = {platform: term[platform] or trunc[platform] for platform in self.possible_platforms}
        s_obs: list[np.ndarray] = self.repeat(self.env.state())

        total_reward: float = sum(
            sum(info[agent][reward] * gain for reward, gain in zip(reward_map[agent], reward_gains[agent], strict=True))
            for agent in trainable_agents
        )
        rewards = [[total_reward]] * self.n_agents

        return (self.unwrap(obs), s_obs, rewards, self.unwrap(dones), self.unwrap(info), self.get_avail_actions())

    def reset(self) -> tuple[dict, list, list, list]:
        """Return initial observations and states.

        Returns
        -------
        dict
            The obs for the next step
        list
            The shared obs for the next step
        list
            The info for each agent in the next step
        list
            The list of available actions for the next step
        """
        self.cur_step = 0
        obs, info = self.env.reset()
        extra_info = self.extra_info_fn(self.env)
        info = merge_dicts(info, extra_info)
        info = merge_dicts(info, {agent: {"step": self.cur_step} for agent in self.possible_platforms})
        obs = self.unwrap(obs)
        s_obs = self.repeat(self.env.state())
        return obs, s_obs, self.unwrap(info), self.get_avail_actions()

    def get_avail_actions(self) -> list[list[int]] | None:
        """Get the available actions at the current timestep.

        Returns
        -------
        list[list[int]] | None
            A list of actions available to each agent, None if the environment is continuous.
        """
        if self.discrete:
            return [self.get_avail_agent_actions(agent_id) for agent_id in range(self.n_agents)]
        return None

    def get_avail_agent_actions(self, agent_id: AgentID) -> list[int]:
        """Return the available actions for agent_id.

        Parameters
        ----------
        agent_id : AgentID
            The agent for which to get available actions

        Returns
        -------
        list[int]
            The currently available actions for the agent
        """
        return [1] * self.action_space[agent_id].n  # type: ignore[attr-defined, index]

    def render(self):  # noqa: D102
        self.env.render()

    def close(self):  # noqa: D102
        self.env.close()

    def seed(self, seed):  # noqa: D102
        self._seed = seed
        self.env.reset(seed=self._seed)

    def wrap(self, array_list: np.ndarray | list[np.ndarray]) -> dict[PlatformID, np.ndarray]:
        """Wrap the array into a dictionary keyed by platform ID.

        Parameters
        ----------
        array_list : np.ndarray | list[np.ndarray]
            A list of arrays, one for each platform

        Returns
        -------
        dict[PlatformID, np.ndarray]
            A dictionary of arrays for each platform
        """
        return {platform: array_list[i] for i, platform in enumerate(self.possible_platforms)}

    def unwrap(self, array_dict: dict[str, T]) -> list[T]:
        """Unwrap a dict of arrays for each platform into a list of arrays.

        Parameters
        ----------
        array_dict : dict[str, T]
            A dictionary of arrays for each platform

        Returns
        -------
        list[T]
            A list of arrays, one for each platform
        """
        return [array_dict[platform] for platform in self.possible_platforms]

    def repeat(self, value: T) -> list[T]:
        """Construct a list by repeating a value for each platform.

        Parameters
        ----------
        value : T
            The value to be repeated

        Returns
        -------
        list[T]
            A list containing the value repeated for each platform
        """
        return [value for _ in range(self.n_agents)]


def simple_tag_extra_info(env: ParallelEnv) -> dict[PlatformID, Any]:  # noqa: PLR0915
    """Extra info to gather from the simple tag environment.

    Parameters
    ----------
    env : ParallelEnv
        The simple_tag environment

    Returns
    -------
    dict[PlatformID, Any]
        The extra info retrieved from the environment
    """
    world: World = env.unwrapped.world
    scenario: Scenario = env.unwrapped.scenario

    agent_sizes = {agent.name: agent.size for agent in world.agents}

    agent_collisions: dict[AgentID, int] = defaultdict(int)
    extra_agent_info = {}

    landmark_rel_dists = []
    for lm in world.landmarks:
        dists = {a.name: np.sqrt(np.sum(np.square(a.state.p_pos - lm.state.p_pos))) for a in world.agents}
        landmark_rel_dists.append(dists)

    # determine which agent this is
    for this_agent in world.agents:
        agent_info = {}
        agent_id = this_agent.name

        # determine the distances to other agents
        ally_distances = {}
        ally_speeds = {}
        adversary_distances = {}
        adversary_speeds = {}
        for a in world.agents:
            displacement = a.state.p_pos - this_agent.state.p_pos
            velocity = a.state.p_vel - this_agent.state.p_vel
            distance = np.sqrt(np.power(displacement, 2.0).sum())
            # determine if the agent is approaching or moving away
            speed = np.sqrt(np.power(velocity, 2.0).sum()) * np.sign(np.dot(displacement, velocity))
            if a.adversary == this_agent.adversary:
                ally_distances[a.name] = distance
                ally_speeds[a.name] = speed
            else:
                adversary_distances[a.name] = distance
                adversary_speeds[a.name] = speed

        # relative distances to landmarks
        landmark_rel_dist = [d[agent_id] for d in landmark_rel_dists]

        # agent velocity
        agent_vel = np.sqrt(np.power(this_agent.state.p_vel, 2.0).sum())

        bound_penalty = 0.0
        if not this_agent.adversary:
            for p in this_agent.state.p_pos:
                x = abs(p)
                if x < 0.9:  # noqa: PLR2004
                    bound_penalty += 0.0
                elif x < 1.0:
                    bound_penalty += (x - 0.9) * 10
                else:
                    bound_penalty += min(np.exp(2 * x - 2), 10)
        bound_penalty *= -1.0

        tagged = 0.0
        if this_agent.collide:
            for a in world.agents:
                if scenario.is_collision(a, this_agent) and this_agent.name != a.name:
                    agent_collisions[agent_id] += 1
                    tagged += 10 if this_agent.adversary != a.adversary else 0
            tagged *= 1 if this_agent.adversary else -1
            for lm in world.landmarks:
                if scenario.is_collision(lm, this_agent):
                    agent_collisions[agent_id] += 1
            agent_collisions[agent_id] = 0

        agent_info["boundary_penalty"] = bound_penalty

        agent_info["cum_ally_distance"] = sum(ally_distances.values())
        agent_info["cum_adversary_distance"] = sum(adversary_distances.values())

        agent_info["nearby_allies"] = sum(
            1 if d - agent_sizes[ally_id] - this_agent.size <= 1.25 * (this_agent.size + agent_sizes[ally_id]) else 0
            for ally_id, d in ally_distances.items()
        )
        agent_info["nearby_adversaries"] = sum(
            1 if d - d - agent_sizes[adv_id] - this_agent.size <= 1.25 * (this_agent.size + agent_sizes[adv_id]) else 0
            for adv_id, d in adversary_distances.items()
        )

        for ally_id, speed in ally_speeds.items():
            agent_info[f"ally_{ally_id}_speed"] = speed
        for adv_id, speed in adversary_speeds.items():
            agent_info[f"adversary_{adv_id}_speed"] = speed

        agent_info["minimum_ally_speed"] = min(ally_speeds.values())
        agent_info["maximum_ally_speed"] = max(ally_speeds.values())
        agent_info["minimum_adversary_speed"] = min(adversary_speeds.values())
        agent_info["maximum_adversary_speed"] = max(adversary_speeds.values())

        agent_info["closest_ally_distance"] = min(ally_distances.values())
        agent_info["furthest_ally_distance"] = max(ally_distances.values())
        agent_info["closest_adversary_distance"] = min(adversary_distances.values())
        agent_info["furthest_adversary_distance"] = max(adversary_distances.values())

        agent_info["closest_landmark_distance"] = min(landmark_rel_dist)
        agent_info["furthest_landmark_distance"] = max(landmark_rel_dist)
        agent_info["collisions"] = agent_collisions[agent_id]
        agent_info["velocity"] = agent_vel

        agent_info["tag_score"] = tagged

        extra_agent_info[agent_id] = agent_info

    return extra_agent_info


def simple_spread_extra_info(env: ParallelEnv) -> dict[PlatformID, Any]:
    """Extra info to gather from the simple spread environment.

    Parameters
    ----------
    env : ParallelEnv
        The simple spread environment

    Returns
    -------
    dict[PlatformID, Any]
        Extra info gathered from the environment
    """
    world: World = env.unwrapped.world
    scenario: Scenario = env.unwrapped.scenario
    agent_sizes = {agent.name: agent.size for agent in world.agents}
    n_agents, n_landmarks = len(world.agents), len(world.landmarks)
    extra_agent_info = {}

    agent_collisions: dict[AgentID, int] = defaultdict(int)
    agent_landmarks_occupied = {agent.name: [False] * n_landmarks for agent in world.agents}

    occupied_landmarks = 0
    landmark_rel_dists = []
    for lm in world.landmarks:
        dists = {a.name: np.sqrt(np.sum(np.square(a.state.p_pos - lm.state.p_pos))) for a in world.agents}
        landmark_rel_dists.append(dists)
        if min(dists.values()) < 0.1:  # noqa: PLR2004
            occupied_landmarks += 1

    total_collisions = 0
    checked_agents = set()
    for agent in world.agents:
        total_collisions += sum(
            1
            for other_agent in world.agents
            if other_agent.name != agent.name
            and other_agent.collide
            and other_agent.name not in checked_agents
            and scenario.is_collision(other_agent, agent)
        )
        checked_agents.add(agent.name)

    agent_rel_pos = {
        agent.name: np.concatenate(
            [other_agent.state.p_pos - agent.state.p_pos for other_agent in world.agents if other_agent.name != agent.name],
            axis=-1,
        )
        for agent in world.agents
    }
    agent_rel_vel = {
        agent.name: np.concatenate(
            [other_agent.state.p_vel - agent.state.p_vel for other_agent in world.agents if other_agent.name != agent.name],
            axis=-1,
        )
        for agent in world.agents
    }

    for agent in world.agents:
        agent_info = {}
        # penalty for not occupying a landmark
        time_penalty = -1

        # relative distances to landmarks
        landmark_rel_dist = [d[agent.name] for d in landmark_rel_dists]

        # relative distances to other agents
        agent_rel_dist = np.power(agent_rel_pos[agent.name], 2.0)
        agent_rel_dist = [np.sqrt(pos.sum()) for pos in np.split(agent_rel_dist, n_agents - 1)]

        # agent velocity
        agent_vel = np.power(agent_rel_vel[agent.name], 2.0)
        agent_vel = np.sqrt(agent_vel.sum())

        if agent.collide:
            for a in world.agents:
                if scenario.is_collision(a, agent) and agent != a:
                    agent_collisions[agent.name] += 1
        else:
            agent_collisions[agent.name] = 0

        # number of visited landmarks
        for lm in range(n_landmarks):
            lm_occupied = landmark_rel_dists[lm][agent.name] < 0.1  # noqa: PLR2004
            if time_penalty == -1 and lm_occupied:
                time_penalty = 0
            agent_landmarks_occupied[agent.name][lm] = lm_occupied
            agent_info[f"landmark_{lm}_occupied"] = lm_occupied

        agent_info["nearby_agents"] = sum(1 if d <= 1.5 * agent_sizes[agent.name] else 0 for d in agent_rel_dist)
        agent_info["closest_agent_distance"] = min(*agent_rel_dist)
        agent_info["closest_landmark_distance"] = min(*landmark_rel_dist)
        agent_info["collisions"] = agent_collisions[agent.name]
        agent_info["landmarks_occupied"] = sum(agent_landmarks_occupied[agent.name])
        agent_info["global_landmarks_occupied"] = occupied_landmarks
        agent_info["velocity"] = agent_vel
        agent_info["time_penalty"] = time_penalty
        agent_info["success"] = occupied_landmarks / n_agents

        extra_agent_info[agent.name] = agent_info
    return extra_agent_info


EXTRA_INFO_FUNCTIONS: dict[str, Callable[[ParallelEnv], dict]] = {
    "simple_spread": simple_spread_extra_info,
    "simple_tag": simple_tag_extra_info,
}
