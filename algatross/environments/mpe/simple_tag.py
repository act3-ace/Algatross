"""A module containing clean-rl style PPO agents, rollout, and training operations."""

from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import numpy as np

from algatross.agents.base import BaseAgent
from algatross.environments.pettingzoo_env import AECPettingZooEnv, ParallelPettingZooEnv
from algatross.environments.runners import BaseRunner
from algatross.utils.types import AgentID

if TYPE_CHECKING:
    from pettingzoo.mpe._mpe_utils.core import World
    from pettingzoo.mpe.simple_tag.simple_tag import Scenario


class MPESimpleTagRunner(BaseRunner):
    """Runner class for MPE Simple Spread."""

    def gather_batch_data(  # noqa: D102, PLR0915, PLR6301
        self,
        env: ParallelPettingZooEnv | AECPettingZooEnv,
        sample_batch: dict[str | int, defaultdict[str, list]],
        agent_map: dict[AgentID, BaseAgent],
        trainable_agents: Iterable[AgentID],  # noqa: ARG002
        opponent_agents: Sequence[AgentID] | None = None,  # noqa: ARG002
        reportable_agent: str | int | None = None,  # noqa: ARG002
        **kwargs,
    ):
        world: World = env.unwrapped.par_env.unwrapped.world  # type: ignore[attr-defined]
        scenario: Scenario = env.unwrapped.par_env.unwrapped.scenario  # type: ignore[attr-defined]

        agent_sizes = {agent.name: agent.size for agent in world.agents}

        agent_collisions: dict[AgentID, int] = defaultdict(int)

        landmark_rel_dists = []
        for lm in world.landmarks:
            dists = {a.name: np.sqrt(np.sum(np.square(a.state.p_pos - lm.state.p_pos))) for a in world.agents}
            landmark_rel_dists.append(dists)

        # determine which agent this is
        for this_agent in world.agents:
            agent_id = this_agent.name

            if agent_id in agent_map:
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

                sample_batch[agent_id]["boundary_penalty"].append([bound_penalty])

                sample_batch[agent_id]["cum_ally_distance"].append([sum(ally_distances.values())])
                sample_batch[agent_id]["cum_adversary_distance"].append([sum(adversary_distances.values())])

                sample_batch[agent_id]["nearby_allies"].append([
                    sum(
                        1 if d - agent_sizes[ally_id] - this_agent.size <= 1.25 * (this_agent.size + agent_sizes[ally_id]) else 0
                        for ally_id, d in ally_distances.items()
                    ),
                ])
                sample_batch[agent_id]["nearby_adversaries"].append([
                    sum(
                        1 if d - d - agent_sizes[adv_id] - this_agent.size <= 1.25 * (this_agent.size + agent_sizes[adv_id]) else 0
                        for adv_id, d in adversary_distances.items()
                    ),
                ])

                for ally_id, speed in ally_speeds.items():
                    sample_batch[agent_id][f"ally_{ally_id}_speed"].append([speed])
                for adv_id, speed in adversary_speeds.items():
                    sample_batch[agent_id][f"adversary_{adv_id}_speed"].append([speed])

                sample_batch[agent_id]["minimum_ally_speed"].append([min(ally_speeds.values())])
                sample_batch[agent_id]["maximum_ally_speed"].append([max(ally_speeds.values())])
                sample_batch[agent_id]["minimum_adversary_speed"].append([min(adversary_speeds.values())])
                sample_batch[agent_id]["maximum_adversary_speed"].append([max(adversary_speeds.values())])

                sample_batch[agent_id]["closest_ally_distance"].append([min(ally_distances.values())])
                sample_batch[agent_id]["furthest_ally_distance"].append([max(ally_distances.values())])
                sample_batch[agent_id]["closest_adversary_distance"].append([min(adversary_distances.values())])
                sample_batch[agent_id]["furthest_adversary_distance"].append([max(adversary_distances.values())])

                sample_batch[agent_id]["closest_landmark_distance"].append([min(landmark_rel_dist)])
                sample_batch[agent_id]["furthest_landmark_distance"].append([max(landmark_rel_dist)])
                sample_batch[agent_id]["collisions"].append([agent_collisions[agent_id]])
                sample_batch[agent_id]["velocity"].append([agent_vel])

                sample_batch[agent_id]["tag_score"].append([tagged])
