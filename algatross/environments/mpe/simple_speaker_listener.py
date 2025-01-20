"""A module containing clean-rl style PPO agents, rollout, and training operations."""

from collections import defaultdict
from collections.abc import Sequence
from itertools import chain
from typing import TYPE_CHECKING

import numpy as np

from pettingzoo.utils.env import AECEnv, ParallelEnv

from algatross.agents.base import BaseAgent
from algatross.environments.runners import BaseRunner
from algatross.utils.types import AgentID

if TYPE_CHECKING:
    from pettingzoo.mpe._mpe_utils.core import World
    from pettingzoo.mpe.simple_speaker_listener.simple_speaker_listener import Scenario


class MPESimpleSpeakerListenerRunner(BaseRunner):
    """Runner class for MPE Simple Speaker Listenet."""

    def gather_batch_data(  # noqa: D102, PLR0915, PLR6301, PLR0914
        self,
        env: AECEnv | ParallelEnv,
        sample_batch: dict[AgentID, defaultdict[str, list]],
        agent_map: dict[AgentID, BaseAgent],  # noqa: ARG002
        trainable_agents: Sequence[AgentID],
        opponent_agents: Sequence[AgentID] | None = None,
        reportable_agent: AgentID | None = None,  # noqa: ARG002
        **kwargs,
    ):
        world: World = env.unwrapped.world
        scenario: Scenario = env.unwrapped.scenario
        agent_sizes = {agent.name: agent.size for agent in world.agents}
        _n_agents, n_landmarks = len(world.agents), len(world.landmarks)

        agent_collisions: dict[AgentID, int] = defaultdict(int)
        agent_landmarks_occupied = {agent_id: [False] * n_landmarks for agent_id in trainable_agents}

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

        total_collisions = 0
        checked_agents = set()
        for agent in world.agents:
            if agent.name in chain(trainable_agents, opponent_agents or []) and agent.collide:
                total_collisions += sum(
                    1
                    for other_agent in world.agents
                    if other_agent.name != agent.name
                    and other_agent.collide
                    and other_agent.name in trainable_agents
                    and other_agent.name not in checked_agents
                    and scenario.is_collision(other_agent, agent)
                )
            checked_agents.add(agent.name)

        agent_rel_pos = {
            agent.name: {
                other_agent.name: other_agent.state.p_pos - agent.state.p_pos
                for other_agent in world.agents
                if other_agent.name != agent.name
            }
            for agent in world.agents
        }
        agent_rel_vels = {
            agent.name: {
                other_agent.name: other_agent.state.p_vel - agent.state.p_vel
                for other_agent in world.agents
                if other_agent.name != agent.name
            }
            for agent in world.agents
        }

        speaker = world.agents[0]
        listener = world.agents[1]
        goal = speaker.goal_b
        goal_dist = np.sqrt(np.power(goal.state.p_pos - listener.state.p_pos, 2.0).sum)
        goal_vel = np.sqrt(np.power(goal.state.p_vel - listener.state.p_vel, 2.0).sum)
        success = goal_dist <= 1.05 * (listener.size + goal.size)

        for agent in world.agents:
            # penalty for not occupying a landmark
            time_penalty = -1 if not success else 0

            # relative distances to landmarks
            landmark_rel_dist = [d[agent.name] for d in landmark_rel_dists]

            # relative distances and velocities to other agents
            agent_rel_vel = {other_agent: np.sqrt(np.power(vel, 2.0).sum()) for other_agent, vel in agent_rel_vels[agent.name].items()}
            agent_rel_dist = {other_agent: np.sqrt(np.power(dist, 2.0).sum()) for other_agent, dist in agent_rel_pos[agent.name].items()}

            # agent velocity
            agent_vel = np.power(agent.state.p_vel, 2.0)
            agent_vel = np.sqrt(agent_vel.sum())

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
                    agent_landmarks_occupied[agent.name][lm] = lm_occupied
                    sample_batch[agent.name][f"landmark_{lm}_occupied"].append([lm_occupied])

                for other_agent, rel_vel in agent_rel_vel.items():
                    sample_batch[agent.name][f"{other_agent}_relative_velocity"].append([rel_vel])
                    sample_batch[agent.name][f"{other_agent}_relative_distance"].append([agent_rel_dist[other_agent]])
                sample_batch[agent.name][f"{agent.name}_relative_velocity"].append([0.0])
                sample_batch[agent.name][f"{agent.name}_relative_distance"].append([0.0])

                sample_batch[agent.name]["nearby_agents"].append([
                    sum(1 if d <= 1.5 * agent_sizes[agent.name] else 0 for d in agent_rel_dist.values()),
                ])
                sample_batch[agent.name]["goal_distance"].append([goal_dist])
                sample_batch[agent.name]["goal_velocity"].append([goal_vel])
                sample_batch[agent.name]["closest_agent_distance"].append([min(*agent_rel_dist)])
                sample_batch[agent.name]["closest_landmark_distance"].append([min(*landmark_rel_dist)])
                sample_batch[agent.name]["collisions"].append([agent_collisions[agent.name]])
                sample_batch[agent.name]["landmarks_occupied"].append([sum(agent_landmarks_occupied[agent.name])])
                sample_batch[agent.name]["global_landmarks_occupied"].append([occupied_landmarks])
                sample_batch[agent.name]["velocity"].append([agent_vel])
                sample_batch[agent.name]["time_penalty"].append([time_penalty])
                sample_batch[agent.name]["success"].append([success])
