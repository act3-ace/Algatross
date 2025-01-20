"""Wrappers for converting SMACv2 to pettingzoo API."""

import numpy as np

from ray.rllib import MultiAgentEnv

import gymnasium

from gymnasium.spaces import Box, Dict, Discrete
from smacv2.env.starcraft2.distributions import get_distribution
from smacv2.env.starcraft2.starcraft2 import CannotResetException, StarCraft2Env

from algatross.utils.random import seed_global


class SMACV2RllibEnv(MultiAgentEnv):
    """Create a new multi-agent StarCraft env compatible with RLlib.

    Parameters
    ----------
    `**kwargs`
        Arguments to pass to the underlying smac.env.starcraft.StarCraft2Env instance.

    Raises
    ------
    ValueError
        The keys of the ``capability_config`` don't match the ``distribution_config``

    Examples
    --------
    >>> from smac.examples.rllib import RLlibStarCraft2Env
    >>> env = RLlibStarCraft2Env(map_name="8m")
    >>> print(env.reset())
    """

    action_space: dict[str, gymnasium.spaces.Space]  # type: ignore[assignment]
    """The action space."""
    observation_space: dict[str, gymnasium.spaces.Space]  # type: ignore[assignment]
    """The observation space."""
    distribution_config: dict
    """The configuration to pass to the SMACv2 distribution."""
    env_key_to_distribution_map: dict
    """A mapping from environment keys to distributions."""
    env_info: dict
    """Global environment info dict."""

    def __init__(self, **kwargs):
        self._env = StarCraft2Env(**kwargs)
        self.distribution_config = kwargs["capability_config"]
        self.env_key_to_distribution_map = {}
        self._parse_distribution_config()
        if self.distribution_config.keys() != kwargs["capability_config"].keys():
            msg = "Must give distribution config and capability config the same keys"
            raise ValueError(msg)
        self._ready_agents = []
        self.env_info = {
            "state_shape": self._env.get_state_size(),
            "obs_shape": self._env.get_obs_size(),
            "cap_shape": self._env.get_cap_size(),
            "n_actions": self._env.get_total_actions(),
            "n_agents": self._env.n_agents,
            "episode_limit": self._env.episode_limit,
        }
        obs_shape = self.env_info["obs_shape"]
        allies = [f"allies_{ag_idx}" for ag_idx in range(self._env.n_agents)]
        actions = self._env.get_total_actions()
        self._agent_ids = set(allies)
        self.observation_space = {
            agent_id: Dict({"obs": Box(low=-np.inf, high=np.inf, shape=(obs_shape,)), "action_mask": Box(0, 1, shape=(actions,))})
            for agent_id in allies
        }
        self.action_space = {agent_id: Discrete(actions) for agent_id in allies}

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        """Reset the env and returns observations from ready agents.

        Parameters
        ----------
        seed : int | None, optional
            The new seed for the environments RNG, default is :data:`python:None`.
        options : dict | None, optional
            The options to pass to the environments reset function, default is :data:`python:None`.

        Returns
        -------
        obs : dict
            New observations for each ready agent.
        infos : dict
            Info for each ready agent
        """
        if seed is not None:
            self.seed(seed)
        try:
            reset_config = options or {}
            for distribution in self.env_key_to_distribution_map.values():
                reset_config = {**reset_config, **distribution.generate()}
            obs_list, state_list = self._env.reset(reset_config)
        except CannotResetException:
            # just retry
            obs_list, state_list = self._env.reset()
        return_obs = {}
        return_infos: dict[str, dict] = {f"allies_{i}": {} for i in range(len(obs_list))}
        return_infos.update(dict(zip(self._env.get_state_feature_names(), state_list, strict=True)))
        for i, obs in enumerate(obs_list):
            return_obs[f"allies_{i}"] = {"action_mask": np.array(self._env.get_avail_agent_actions(i)), "obs": obs}
            return_infos[f"allies_{i}"] = {}
        self._ready_agents = [f"allies_{i}" for i in range(len(obs_list))]
        return return_obs, return_infos

    def step(self, action_dict):
        """Return observations from ready agents.

        The returns are dicts mapping from agent_id strings to values. The
        number of agents in the env can vary over time.

        Parameters
        ----------
        action_dict : dict
            A dictionary of actions to send to the environment

        Returns
        -------
        obs : dict
            New observations for each ready agent.
        rewards : dict
            Reward values for each ready agent. If the episode is just started, the value will be None.
        dones : dict
            Done values for each ready agent. The special key
            "__all__" (required) is used to indicate env termination.
        truncateds : dict
            The mapping indicating which agent trajectories have been truncated
        infos : dict
            Optional info values for each agent id.

        Raises
        ------
        ValueError
            If an agent did not provide an action
        """
        actions = []
        for agent in self._ready_agents:
            if agent not in action_dict:
                msg = f"You must supply an action for agent: {agent}"
                raise ValueError(msg)
            actions.append(action_dict[agent])

        if len(actions) != len(self._ready_agents):
            msg = f"Unexpected number of actions: {action_dict}"
            raise ValueError(msg)

        rew, done, info = self._env.step(actions)
        obs_list = self._env.get_obs()
        return_obs = {}
        for i, obs in enumerate(obs_list):
            return_obs[f"allies_{i}"] = {"action_mask": np.array(self._env.get_avail_agent_actions(i), dtype=np.float32), "obs": obs}
        if "dead_enemies" not in info or "dead_alies" not in info:
            info.update(self._get_dead_agents())
        if "battle_won" not in info:
            info.update(self._get_battle_won())

        rews = {f"allies_{i}": rew / len(obs_list) for i in range(len(obs_list))}
        dones = dict.fromkeys((f"allies_{i}" for i in range(len(obs_list))), done)
        dones["__all__"] = done
        truncateds = dict.fromkeys((f"allies_{i}" for i in range(len(obs_list))), False)
        infos = dict.fromkeys((f"allies_{i}" for i in range(len(obs_list))), info)
        self._ready_agents = [f"allies_{i}" for i in range(len(obs_list))]

        return return_obs, rews, dones, truncateds, infos

    def _get_dead_agents(self):
        dead_allies, dead_enemies = 0, 0
        for al_unit in self._env.agents.values():
            if al_unit.health == 0:
                dead_allies += 1
        for e_unit in self._env.enemies.values():
            if e_unit.health == 0:
                dead_enemies += 1
        return {"dead_allies": dead_allies, "dead_enemies": dead_enemies}

    def _get_battle_status(self):
        n_ally_alive = 0
        n_enemy_alive = 0

        for al_unit in self._env.previous_ally_units.values():
            for unit in self._env._obs.observation.raw_data.units:  # noqa: SLF001
                if al_unit.tag == unit.tag:
                    n_ally_alive += 1
                    break

        for e_unit in self._env.previous_enemy_units.values():
            for unit in self._env._obs.observation.raw_data.units:  # noqa: SLF001
                if e_unit.tag == unit.tag:
                    n_enemy_alive += 1
                    break

        if (n_ally_alive == 0 and n_enemy_alive > 0) or self._env.only_medivac_left(ally=True):
            return -1  # lost
        if (n_ally_alive > 0 and n_enemy_alive == 0) or self._env.only_medivac_left(ally=False):
            return 1  # won
        if n_ally_alive == 0 and n_enemy_alive == 0:
            return 0

        return None

    def _get_battle_won(self):
        return {"battle_won": self._get_battle_status() == 1}

    def close(self):
        """Close the environment."""
        self._env.close()

    def seed(self, seed):  # noqa: D102, PLR6301
        seed_global(seed)
        # get_generators(seed=seed, seed_global=True)  # noqa: ERA001

    def _parse_distribution_config(self):
        for env_key, config in self.distribution_config.items():
            if env_key in {"n_units", "n_enemies"}:
                continue
            config["env_key"] = env_key
            # add n_units key
            config["n_units"] = self.distribution_config["n_units"]
            config["n_enemies"] = self.distribution_config["n_enemies"]
            distribution = get_distribution(config["dist_type"])(config)
            self.env_key_to_distribution_map[env_key] = distribution

    def get_obs(self):  # noqa: D102
        return self._env.get_obs()

    def get_obs_feature_names(self):  # noqa: D102
        return self._env.get_obs_feature_names()

    def get_state(self):  # noqa: D102
        return self._env.get_state()

    def get_state_feature_names(self):  # noqa: D102
        return self._env.get_state_feature_names()

    def get_avail_actions(self):  # noqa: D102
        return self._env.get_avail_actions()

    def get_env_info(self):  # noqa: D102
        return self._env.get_env_info()

    def get_obs_size(self):  # noqa: D102
        return self._env.get_obs_size()

    def get_state_size(self):  # noqa: D102
        return self._env.get_state_size()

    def get_total_actions(self):  # noqa: D102
        return self._env.get_total_actions()

    def get_capabilities(self):  # noqa: D102
        return self._env.get_capabilities()

    def get_obs_agent(self, agent_id):  # noqa: D102
        return self._env.get_obs_agent(agent_id)

    def get_avail_agent_actions(self, agent_id):  # noqa: D102
        return self._env.get_avail_agent_actions(agent_id)

    def render(self, mode="human"):  # noqa: D102
        return self._env.render(mode=mode)

    def get_stats(self):  # noqa: D102
        return self._env.get_stats()

    def full_restart(self):  # noqa: D102
        return self._env.full_restart()
