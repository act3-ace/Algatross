"""Weighted task reward functions for the inspection tasks."""

import numpy as np

import safe_autonomy_simulation.sims.inspection as sim

from safe_autonomy_sims.pettingzoo.inspection.utils import delta_v, rel_dist


def observed_points_reward(chief: sim.Target, prev_num_inspected: int, weight: float = 0.01) -> float:
    """
    Calculate a dense reward which rewards the agent for inspecting.

    new points during each step of the episode.

    $r_t = 0.01 * (p_t - p_{t-1})$

    where $p_t$ is the total number of inspected points at
    time $t$.

    Parameters
    ----------
    chief : sim.Target
        Chief spacecraft under inspection
    prev_num_inspected : int
        Number of previously inspected points
    weight : float, optional
        The overall reward weight, default is 0.01.

    Returns
    -------
    float
        Reward value
    """
    current_num_inspected = chief.inspection_points.get_num_points_inspected()
    step_inspected = current_num_inspected - prev_num_inspected
    return weight * step_inspected


def weighted_observed_points_reward(chief: sim.Target, prev_weight_inspected: float, weight: float = 1.0) -> float:
    """
    Calculate a dense reward for inspecting new points during each step of the episode conditioned by individual point weights.

    $r_t = 1.0 * (w_t - w_{t-1})$

    where $w_t$ is the total weight of inspected points at time $t$.

    Parameters
    ----------
    chief : sim.Target
        Chief spacecraft under inspected
    prev_weight_inspected : float
        Weight of previously inspected points
    weight : float, optional
        The overall reward weight, default is 1.

    Returns
    -------
    float
        Reward value
    """
    current_weight_inspected = chief.inspection_points.get_total_weight_inspected()
    step_inspected = current_weight_inspected - prev_weight_inspected
    return weight * step_inspected


def delta_v_reward(v: np.ndarray, prev_v: np.ndarray, m: float = 12.0, b: float = 0.0, weight: float = 1) -> float:
    r"""
    Calculate a dense reward based on the deputy's fuel.

    use (change in velocity).

    $r_t = -((\deltav / m) + b)$

    where
    * $\deltav$ is the change in velocity
    * $m$ is the mass of the deputy
    * $b$ is a tunable bias term

    Parameters
    ----------
    v : np.ndarray
        Current velocity
    prev_v : np.ndarray
        Previous velocity
    m : float, optional
        Deputy mass, by default 12.0
    b : float, optional
        Bias term, by default 0.0
    weight : float, optional
        The overall reward weight, default is 1.

    Returns
    -------
    float
        Reward value
    """
    r = -((abs(delta_v(v=v, prev_v=prev_v)) / m) + b)
    return weight * r


def inspection_success_reward(chief: sim.Target, total_points: int, weight: float = 1.0) -> float:
    """
    Calculate a sparse reward applied when the agent successfully.

    inspects every point.

    $r_t = 1 if p_t == p_{total}, else 0$

    where $p_t$ is the number of inspected points at time
    $t$ and $p_{total}$ is the total number of points to be
    inspected.

    Parameters
    ----------
    chief : sim.Target
        Chief spacecraft under inspection
    total_points : int
        Total number of points to be inspected
    weight : float, optional
        The overall reward weight, default is 1.

    Returns
    -------
    float
        Reward value
    """
    num_inspected = chief.inspection_points.get_num_points_inspected()
    r = 1.0 if num_inspected == total_points else 0.0
    return weight * r


def crash_reward(chief: sim.Target, deputy: sim.Inspector, crash_radius: float, weight: float = 1.0) -> float:
    """Calculate a sparse reward that punishes the agent.

    for intersecting with the chief (crashing).

    $r_t = d_c < r$

    where $d_c$ is the distance between the deputy and
    the chief and $r$ is the radius of the crash region.

    Parameters
    ----------
    chief : sim.Target
        Chief spacecraft under inspection
    deputy : sim.Inspector
        Deputy spacecraft performing inspection
    crash_radius : float
        Distance from chief which triggers a crash
    weight : float, optional
        The overall reward weight, default is 1.

    Returns
    -------
    float
        Reward value
    """
    r = -1.0 if rel_dist(pos1=chief.position, pos2=deputy.position) < crash_radius else 0
    return weight * r
