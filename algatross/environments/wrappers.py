"""Environment wrappers."""

from __future__ import annotations

import numpy as np

from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import AECEnv
from pettingzoo.utils.env_logger import EnvLogger

from algatross.environments.utilities import is_continuous_env


class SilentClipOutOfBoundsWrapper(BaseWrapper):
    """Clips the input action to fit in the continuous action space.

    Applied to continuous environments in pettingzoo.

    Parameters
    ----------
    env : AECEnv
        The environment to be wrapped.
    """

    def __init__(self, env: AECEnv):
        super().__init__(env)
        assert isinstance(env, AECEnv), "ClipOutOfBoundsWrapper is only compatible with AEC environments."  # noqa: S101
        agent_to_space_t = {agent: type(self.action_space(agent)) for agent in getattr(self, "possible_agents", [])}
        assert is_continuous_env(  # noqa: S101
            self,
        ), f"should only use SilentClipOutOfBoundsWrapper for Box spaces. Got: {agent_to_space_t}"

    def step(self, action: np.ndarray | None) -> None:  # noqa: D102
        space = self.action_space(self.agent_selection)
        if not (
            action is None and (self.terminations[self.agent_selection] or self.truncations[self.agent_selection])
        ) and not space.contains(action):
            if action is None or np.isnan(action).any():
                EnvLogger.error_nan_action()
            assert (  # noqa: S101
                space.shape == action.shape  # pyright: ignore[reportOptionalMemberAccess]
            ), f"action should have shape {space.shape}, has shape {action.shape}"  # pyright: ignore[reportOptionalMemberAccess]

            action = np.clip(
                action,  # pyright: ignore[reportGeneralTypeIssues]
                space.low,  # pyright: ignore[reportGeneralTypeIssues]
                space.high,  # pyright: ignore[reportGeneralTypeIssues]
            )

        super().step(action)

    def __str__(self) -> str:  # noqa: D105
        return str(self.env)
