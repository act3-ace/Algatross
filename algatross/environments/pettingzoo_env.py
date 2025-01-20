"""PettingZoo wrappers for Rllib."""

import logging

from ray.rllib.env.wrappers.pettingzoo_env import (
    ParallelPettingZooEnv as _ParallelPettingZooEnv,
    PettingZooEnv,
)

import gymnasium

from pettingzoo.utils import ParallelEnv, aec_to_parallel, parallel_to_aec
from supersuit.multiagent_wrappers import pad_action_space_v0, pad_observations_v0

from algatross.environments.utilities import is_continuous_env
from algatross.environments.wrappers import SilentClipOutOfBoundsWrapper
from algatross.utils.types import ConstructorData


class AECPettingZooEnv(PettingZooEnv):
    """Simple wrapper class for AEC environments which constructs the environment from ConstructorData before calling super().

    Parameters
    ----------
    env : ConstructorData
        The construct data for the environment
    """

    def __init__(self, env: ConstructorData):
        super().__init__(env())


class ParallelPettingZooEnv(_ParallelPettingZooEnv):
    """
    Simple wrapper class for Parallel environments which constructs the environment from ConstructorData before calling super().

    Parameters
    ----------
    env : ConstructorData
        The construct data for the environment
    """

    def __init__(self, env: ConstructorData):
        # constructed = pad_observations_v0(pad_action_space_v0(env()))  # noqa: ERA001
        constructed = env()

        if not hasattr(constructed, "metadata"):
            constructed.metadata = {}

        # TODO: we may not always want to silence out of bounds warning from PZ, make it toggleable?
        silence_oob = True
        if silence_oob and is_continuous_env(constructed):
            if isinstance(constructed, ParallelEnv):
                # pettingzoo wrappers only work with AEC :(
                logger = logging.getLogger("ray")
                logger.warning("Got parallel environment but needed AEC, this will double-wrap the environment.")
                constructed = parallel_to_aec(constructed)

            constructed = SilentClipOutOfBoundsWrapper(constructed)
            # supersuit only works with parallel
            constructed = aec_to_parallel(constructed)

        constructed = pad_observations_v0(pad_action_space_v0(constructed))

        if not hasattr(constructed, "observation_spaces"):
            constructed.observation_spaces = gymnasium.spaces.Dict({
                agent_id: constructed.observation_space(agent_id) for agent_id in constructed.possible_agents
            })
        if not hasattr(constructed, "action_spaces"):
            constructed.action_spaces = gymnasium.spaces.Dict({
                agent_id: constructed.action_space(agent_id) for agent_id in constructed.possible_agents
            })
        super().__init__(constructed)
