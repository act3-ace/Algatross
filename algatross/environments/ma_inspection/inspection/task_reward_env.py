"""STARS MA-inspection task reward environments."""

from safe_autonomy_sims.pettingzoo import MultiInspectionEnv

import algatross.environments.ma_inspection.task_reward as tr


class TaskRewardMultiInspectionEnv(MultiInspectionEnv):
    r"""
    PZ MultiInspectionEnv with weighted task rewards.

    Parameters
    ----------
    num_agents : int, optional
        The number of agents in the environment, default is 2
    success_threshold : float, optional
        The minimum threshold (score) needed to consider the episode successful, default is 100
    crash_radius : float, optional
        The crash radius for the spacecraft, default is 15
    max_distance : float, optional
        The domain boundary around the chief beyond which the episode will terminate early, default is 800.
    max_time : float, optional
        The maximum sim time before the episode terminates, default is 1000
    dense_observed_points_weight : float, optional
        The weight factor for the dense observed points reward, default is 0.01
    dense_delta_v_weight : float, optional
        The weight factor for the dense :math:`\Delta v` reward, default is 1.0
    sparse_inspection_success_weight : float, optional
        The weight factor for the sparse inspection success reward, default is 1.0
    sparse_crash_weight : float, optional
        The weight factor for the sparse crash reward, default is 1.0
    """

    def __init__(
        self,
        num_agents: int = 2,
        success_threshold: float = 100,
        crash_radius: float = 15,
        max_distance: float = 800,
        max_time: float = 1000,
        # task weight
        dense_observed_points_weight: float = 0.01,
        dense_delta_v_weight: float = 1.0,
        sparse_inspection_success_weight: float = 1.0,
        sparse_crash_weight: float = 1.0,
    ):
        super().__init__(
            num_agents=num_agents,
            success_threshold=success_threshold,
            crash_radius=crash_radius,
            max_distance=max_distance,
            max_time=max_time,
        )
        self.dense_observed_points_weight = dense_observed_points_weight
        self.dense_delta_v_weight = dense_delta_v_weight
        self.sparse_inspection_success_weight = sparse_inspection_success_weight
        self.sparse_crash_weight = sparse_crash_weight

    # copied from safe_autonomy_sims.pettingzoo.inspection.reward
    def _get_reward(self, agent):
        reward = 0
        deputy = self.deputies[agent]

        # Dense rewards
        points_reward = tr.observed_points_reward(
            chief=self.chief,
            prev_num_inspected=self.prev_num_inspected,
            weight=self.dense_observed_points_weight,
        )
        self.reward_components[agent]["observed_points"] = points_reward
        reward += points_reward

        delta_v_reward = tr.delta_v_reward(v=deputy.velocity, prev_v=self.prev_state[agent][3:6], weight=self.dense_delta_v_weight)
        self.reward_components[agent]["delta_v"] = delta_v_reward
        reward += delta_v_reward

        # Sparse rewards
        success_reward = tr.inspection_success_reward(
            chief=self.chief,
            total_points=self.success_threshold,
            weight=self.sparse_inspection_success_weight,
        )
        self.reward_components[agent]["success"] = success_reward
        reward += success_reward

        crash_reward = tr.crash_reward(chief=self.chief, deputy=deputy, crash_radius=self.crash_radius, weight=self.sparse_crash_weight)
        self.reward_components[agent]["crash"] = crash_reward
        reward += crash_reward

        return reward
