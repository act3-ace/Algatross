"""Classes of safe-autonomy-sims runners for "insepction" variant."""

from collections import defaultdict
from collections.abc import Sequence

from ray.rllib import MultiAgentEnv

from pettingzoo import AECEnv, ParallelEnv

from algatross.agents.base import BaseAgent
from algatross.environments.runners import BaseRunner
from algatross.utils.types import AgentID


class InspectionRunner(BaseRunner):
    """A runner for safe-autonomy-sims ``inspection`` variant."""

    def visualize_step(  # type: ignore[override] # noqa: PLR6301
        self,
        env: MultiAgentEnv | ParallelEnv | AECEnv,  # noqa: ARG002
        sample_batch: dict[AgentID, defaultdict[str, list]],
        agent_map: dict[AgentID, BaseAgent],  # noqa: ARG002
        trainable_agents: Sequence[AgentID],  # noqa: ARG002
        opponent_agents: Sequence[AgentID] | None = None,  # noqa: ARG002
        reportable_agent: AgentID | None = None,  # noqa: ARG002
        **kwargs,
    ) -> dict[AgentID, defaultdict[str, list]]:
        """
        Visualize the current environment to np array.

        Parameters
        ----------
        env : MultiAgentEnv | ParallelEnv | AECEnv
            The environment to visualize
        sample_batch : dict[AgentID, defaultdict[str, list]]
            Batches of agent data for the episode
        agent_map : dict[AgentID, BaseAgent]
            Mapping to agent modules
        trainable_agents : Iterable[AgentID]
            Iterable of trainable agent IDs
        opponent_agents : Sequence[AgentID] | None, optional
            Iterable of trainable agent IDs, default is None
        reportable_agent : AgentID | None, optional
            The agent ID to gather info for in the case of AEC api. Defaults to None causing data for all agents to be added to
            the batch (parallel env API).
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        dict[AgentID, defaultdict[str, list]]
            The mapping from agent ID to rollout data for the agent.
        """
        return sample_batch
