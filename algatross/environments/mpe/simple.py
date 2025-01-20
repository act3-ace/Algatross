"""A module containing clean-rl style PPO agents, rollout, and training operations."""

from collections import defaultdict
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from pettingzoo.utils.env import AECEnv, ParallelEnv

from algatross.agents.base import BaseAgent
from algatross.environments.runners import BaseRunner
from algatross.utils.types import AgentID

if TYPE_CHECKING:
    from pettingzoo.mpe._mpe_utils.core import World
    from pettingzoo.mpe.simple.simple import Scenario


class MPESimpleRunner(BaseRunner):
    """Runner class for MPE Simple."""

    def gather_batch_data(  # noqa: D102, PLR6301
        self,
        env: AECEnv | ParallelEnv,
        sample_batch: dict[AgentID, defaultdict[str, list]],
        agent_map: dict[AgentID, BaseAgent],
        trainable_agents: Sequence[AgentID],
        opponent_agents: Sequence[AgentID] | None = None,  # noqa: ARG002
        reportable_agent: AgentID | None = None,  # noqa: ARG002
        **kwargs,
    ):
        world: World = env.unwrapped.world
        scenario: Scenario = env.unwrapped.scenario
        _agent_sizes = {agent.name: agent.size for agent in world.agents}
        _n_agents, n_landmarks = len(world.agents), len(world.landmarks)

        agent_collisions: dict[AgentID, int] = defaultdict(int)
        agent_landmarks_occupied = {agent_id: [False] * n_landmarks for agent_id in agent_map}

        occupied_landmarks, trainable_occupied_landmarks = 0, 0
        landmark_rel_dists = []
        for lm in world.landmarks:
            dists = {a.name: np.sqrt(np.sum(np.square(a.state.p_pos - lm.state.p_pos))) for a in world.agents}
            trainable_dists = {
                a.name: np.sqrt(np.sum(np.square(a.state.p_pos - lm.state.p_pos))) for a in world.agents if a.name in trainable_agents
            }
            landmark_rel_dists.append(dists)
            if min(dists.values()) < 0.1:  # noqa: PLR2004
                occupied_landmarks += 1
            if min(trainable_dists.values()) < 0.1:  # noqa: PLR2004
                trainable_occupied_landmarks += 1

        for agent in world.agents:
            # penalty for not occupying a landmark
            time_penalty = -1

            # relative distances to landmarks
            landmark_rel_dist = [d[agent.name] for d in landmark_rel_dists]

            if agent.collide:
                for a in world.agents:
                    if scenario.is_collision(a, agent) and agent != a:
                        agent_collisions[agent.name] += 1
            else:
                agent_collisions[agent.name] = 0

            # only store the rollouts for trainable agents
            if agent.name in trainable_agents:
                # number of visited landmarks
                for lm in range(n_landmarks):
                    lm_occupied = landmark_rel_dists[lm][agent.name] < 0.1  # noqa: PLR2004
                    if time_penalty == -1 and lm_occupied:
                        time_penalty = 0
                    agent_landmarks_occupied[agent.name][lm] = lm_occupied
                    sample_batch[agent.name][f"landmark_{lm}_occupied"].append([lm_occupied])

                sample_batch[agent.name]["closest_landmark_distance"].append([min(*landmark_rel_dist)])
                sample_batch[agent.name]["collisions"].append([agent_collisions[agent.name]])
                sample_batch[agent.name]["global_landmarks_occupied"].append([occupied_landmarks])
                sample_batch[agent.name]["success"].append([trainable_occupied_landmarks / len(trainable_agents)])
