"""Problem classes for running classic RL algorithms."""

import copy
import logging

from collections.abc import Mapping

import numpy as np

from algatross.agents.base import BaseAgent
from algatross.algorithms.genetic.mo_aim.problem import MOAIMIslandUDP
from algatross.utils.types import AgentID, MOAIMIslandInhabitant


class MOAIMRLUDP(MOAIMIslandUDP):
    """User-defined problem for running classic RL algorithms."""

    def load_team(self, team: Mapping[AgentID, MOAIMIslandInhabitant]) -> dict[AgentID, BaseAgent]:  # noqa: D102
        logger = logging.getLogger("ray")
        loaded_agent_map = copy.deepcopy({agent_id: self.agent_templates_map[agent_id] for agent_id in team})

        for agent_id, inhabitant in team.items():
            logger.debug(f"load team agent {agent_id}")
            loaded_agent_map[agent_id].load_flat_params(inhabitant.genome)
            self._solution_dim[agent_id] = np.prod(loaded_agent_map[agent_id].flat_parameters.shape)
            loaded_agent_map[agent_id].reset_optimizer()

        return loaded_agent_map
